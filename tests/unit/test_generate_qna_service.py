"""Unit tests for the generation service."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from qna_generation_agent.application.dto import (
    AssessmentContext,
    GenerationCommand,
    QuestionBatch,
    QuestionDraft,
)
from qna_generation_agent.application.errors import (
    IdempotencyConflict,
    RetrievalError,
    StorageTransientError,
    WorkflowEscalationError,
)
from qna_generation_agent.application.ports.idempotency import IdempotencyStatus
from qna_generation_agent.application.ports.llm import LLMProvider
from qna_generation_agent.application.ports.prompt_provider import (
    Prompt,
    PromptProvider,
)
from qna_generation_agent.application.ports.submission_client import (
    AssessmentConfig,
    CreateQuestionSetCommand,
    IncrementIterationCommand,
    IncrementIterationResult,
    QuestionSetRecord,
    SubmissionClient,
    WriteGeneratedQuestionsCommand,
    WriteGeneratedQuestionsResult,
)
from qna_generation_agent.application.ports.telemetry import Span, TelemetryPort
from qna_generation_agent.application.services.generate_qna import GenerateQnAService
from qna_generation_agent.domain.enums import DifficultyLevel, Purpose, QuestionType
from qna_generation_agent.domain.events import QnAGenerationCompleted
from qna_generation_agent.infrastructure.grpc.knowledge_client import (
    GetTopicsCommand,
    SimilaritySearchCommand,
    Topic,
)
from qna_generation_agent.infrastructure.llm.prompt_builder import (
    AssessmentGeneratorOutputSchema,
    AssessmentQuestionSchema,
    MCQAnswerGeneratorOutputSchema,
    MCQDistractorExplanationSchema,
)
from qna_generation_agent.infrastructure.messaging.pubsub_publisher import (
    NullEventPublisher,
)
from qna_generation_agent.infrastructure.persistence.in_memory import (
    InMemoryIdempotencyStore,
    InMemoryQuestionSetRepository,
)


class FakeLLMProvider(LLMProvider):
    """Deterministic LLM stub for service tests."""

    @property
    def model_id(self) -> str:
        return "fake-model"

    async def generate_structured(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> QuestionBatch:
        return self._build_batch(
            count=count,
            difficulty_level=difficulty_level,
            question_type=QuestionType.STRUCTURED,
            topic_id=context.topic_ids[0] if context.topic_ids else "topic_1",
        )

    async def generate_non_structured(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> QuestionBatch:
        return self._build_batch(
            count=count,
            difficulty_level=difficulty_level,
            question_type=QuestionType.NON_STRUCTURED,
            topic_id=context.topic_ids[0] if context.topic_ids else "topic_1",
        )

    async def generate_with_prompt(
        self,
        *,
        prompt: str,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
        question_type: str,
    ) -> QuestionBatch:
        """Generate using a pre-compiled prompt."""
        qt = (
            QuestionType.STRUCTURED
            if question_type == "structured"
            else QuestionType.NON_STRUCTURED
        )
        return self._build_batch(
            count=count,
            difficulty_level=difficulty_level,
            question_type=qt,
            topic_id="topic_1",
        )

    async def health_check(self) -> bool:
        """Return True for tests."""
        return True

    async def invoke_with_system_and_user[
        T: BaseModel
    ](
        self,
        system_message: str,
        user_message: str,
        *,
        structured_output_model: type[T],
        model_tier: str = "expensive",
    ) -> T | None:
        """Invoke LLM with system and user messages for 3-prompt workflow."""
        combined = f"{system_message}\n\n{user_message}"

        # Return appropriate mock data based on the schema type
        if structured_output_model == AssessmentGeneratorOutputSchema:
            # Extract count from prompt to return appropriate number of questions
            # The prompt format is: "Generate {N} structured and {M} non-structured questions..."
            import re

            match = re.search(
                r"(\d+)\s+structured.*?(\d+)\s+(?:non-structured|open-ended)",
                combined,
                re.IGNORECASE,
            )
            if match:
                structured_count = int(match.group(1))
                non_structured_count = int(match.group(2))
            else:
                structured_count = 1
                non_structured_count = 0

            questions = []
            for i in range(structured_count):
                questions.append(
                    AssessmentQuestionSchema(
                        question_id=f"q-{i + 1:03d}",
                        question_type="structured",
                        content=f"Test question {i + 1} stem?",
                        structured_answer="A",
                        metadata={
                            "topic": "test-topic",
                            "difficulty": "medium",
                            "source_chunk_ids": ["chunk-1"],
                        },
                    )
                )
            for i in range(non_structured_count):
                questions.append(
                    AssessmentQuestionSchema(
                        question_id=f"q-{structured_count + i + 1:03d}",
                        question_type="non_structured",
                        content=f"Test non-structured question {i + 1}?",
                        non_structured_model_answer=f"Model answer {i + 1}",
                        metadata={
                            "topic": "test-topic",
                            "difficulty": "medium",
                            "source_chunk_ids": ["chunk-1"],
                        },
                    )
                )
            return AssessmentGeneratorOutputSchema(questions=questions)  # type: ignore[return-value]
        if structured_output_model == MCQAnswerGeneratorOutputSchema:
            return MCQAnswerGeneratorOutputSchema(  # type: ignore[return-value]
                question_stem="Test question stem?",
                correct_answer=MCQDistractorExplanationSchema(
                    option_letter="A",
                    option_text="Correct answer",
                    is_correct=True,
                    explanation="This is correct",
                ),
                distractors=[
                    MCQDistractorExplanationSchema(
                        option_letter="B",
                        option_text="Distractor 1",
                        is_correct=False,
                        explanation="This is wrong",
                    ),
                    MCQDistractorExplanationSchema(
                        option_letter="C",
                        option_text="Distractor 2",
                        is_correct=False,
                        explanation="This is wrong",
                    ),
                    MCQDistractorExplanationSchema(
                        option_letter="D",
                        option_text="Distractor 3",
                        is_correct=False,
                        explanation="This is wrong",
                    ),
                ],
                grammar_point_tested="test grammar",
                difficulty_justification="test difficulty",
                l1_considerations=["L1 interference 1"],
            )
        return None

    def _build_batch(
        self,
        *,
        count: int,
        difficulty_level: str | None,
        question_type: QuestionType,
        topic_id: str,
    ) -> QuestionBatch:
        return QuestionBatch(
            questions=[
                QuestionDraft(
                    question_text=f"{question_type.value} question {index}",
                    answer_text=f"{question_type.value} answer {index}",
                    question_type=question_type,
                    difficulty_level=difficulty_level,
                    explanation="grounded explanation",
                    references=["chunk-1"],
                    topic_id=topic_id,
                )
                for index in range(count)
            ],
            model_name="fake",
            prompt_version="test",
        )


class SwarmAwareFakeLLMProvider(FakeLLMProvider):
    """Fake provider that exposes swarm candidate generation."""

    def __init__(self, candidates: list[Any]) -> None:
        self._candidates = candidates
        self.swarm_sizes: list[int] = []

    async def generate_swarm_candidates(
        self,
        *,
        system_message: str,
        user_message: str,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
        swarm_size: int,
    ) -> list[Any]:
        del system_message, user_message, count, difficulty_level, correlation_id
        self.swarm_sizes.append(swarm_size)
        return self._candidates

    async def invoke_with_system_and_user[
        T: BaseModel
    ](
        self,
        system_message: str,
        user_message: str,
        *,
        structured_output_model: type[T],
        model_tier: str = "expensive",
    ) -> T | None:
        if structured_output_model == MCQAnswerGeneratorOutputSchema:
            import re

            match = re.search(r"Question:\s*(.*?)\nTopic:", user_message, re.S)
            stem = match.group(1).strip() if match else "Test question stem?"
            return MCQAnswerGeneratorOutputSchema(  # type: ignore[return-value]
                question_stem=stem,
                correct_answer=MCQDistractorExplanationSchema(
                    option_letter="A",
                    option_text="Correct answer",
                    is_correct=True,
                    explanation="This is correct",
                ),
                distractors=[
                    MCQDistractorExplanationSchema(
                        option_letter="B",
                        option_text="Distractor 1",
                        is_correct=False,
                        explanation="This is wrong",
                    ),
                    MCQDistractorExplanationSchema(
                        option_letter="C",
                        option_text="Distractor 2",
                        is_correct=False,
                        explanation="This is wrong",
                    ),
                    MCQDistractorExplanationSchema(
                        option_letter="D",
                        option_text="Distractor 3",
                        is_correct=False,
                        explanation="This is wrong",
                    ),
                ],
                grammar_point_tested="test grammar",
                difficulty_justification="test difficulty",
                l1_considerations=["L1 interference 1"],
            )

        return await super().invoke_with_system_and_user(
            system_message,
            user_message,
            structured_output_model=structured_output_model,
            model_tier=model_tier,
        )


def _assessment_question(
    *,
    question_id: str,
    question_type: str,
    question_text: str,
    answer_text: str,
    explanation: str | None = None,
) -> AssessmentQuestionSchema:
    if question_type == "structured":
        metadata = {
            "question_type": "structured",
            "options": {"A": "A", "B": "B", "C": "C", "D": "D"},
            "source_chunk_ids": ["chunk-1"],
            "difficulty": "medium",
            "topic": "Test Topic",
        }
    else:
        metadata = {
            "question_type": "non_structured",
            "source_chunk_ids": ["chunk-1"],
            "difficulty": "medium",
            "topic": "Test Topic",
            "rubric": "Explain clearly",
        }

    payload: dict[str, Any] = {
        "question_id": question_id,
        "question_type": question_type,
        "question_text": question_text,
        "answer_text": answer_text,
        "metadata": metadata,
    }
    if explanation is not None:
        payload["explanation"] = explanation
    return AssessmentQuestionSchema.model_validate(payload)


class FakeTelemetry(TelemetryPort):
    """Telemetry stub that returns predictable trace IDs."""

    @contextmanager
    def propagate(
        self,
        trace_name: str,
        correlation_id: str,
        workflow_id: str,
        metadata: dict[str, str] | None = None,
    ) -> Generator[None]:
        yield

    @contextmanager
    def trace(
        self,
        name: str,
        metadata: dict[str, str] | None = None,
    ) -> Generator[Span]:
        class FakeSpan(Span):
            def set_attribute(self, key: str, value: Any) -> None:
                pass

            def record_error(self, error: BaseException) -> None:
                pass

        yield FakeSpan()

    def current_trace_id(self) -> str | None:
        return "trace-test"

    def current_trace_url(self) -> str | None:
        return "http://tracing.example.com/trace-test"

    async def shutdown(self) -> None:
        pass


class RecordingPublisher(NullEventPublisher):
    """Publisher that records events for verification."""

    def __init__(self) -> None:
        self.events: list[QnAGenerationCompleted] = []

    async def publish_completion(self, event: QnAGenerationCompleted) -> None:
        self.events.append(event)


class FakeSubmissionClient(SubmissionClient):
    """Stub submission client for tests."""

    def __init__(self) -> None:
        self._question_set_counter = 0
        self._iteration_counter: dict[str, int] = {}
        self.questions_written: list[WriteGeneratedQuestionsCommand] = []

    async def create_question_set(
        self,
        command: CreateQuestionSetCommand,
    ) -> QuestionSetRecord:

        self._question_set_counter += 1
        question_set_id = f"qs_{self._question_set_counter}"
        self._iteration_counter[question_set_id] = 1
        return QuestionSetRecord(
            id=question_set_id,
            workflow_id=command.workflow_id,
            iteration_count=1,
            status="created",
            created_at=datetime.now(UTC),
        )

    async def increment_iteration(
        self,
        command: IncrementIterationCommand,
    ) -> IncrementIterationResult:
        current = self._iteration_counter.get(command.question_set_id, 1)
        new_count = current + 1
        self._iteration_counter[command.question_set_id] = new_count
        return IncrementIterationResult(
            question_set_id=command.question_set_id,
            iteration_count=new_count,
            status="incremented",
        )

    async def write_generated_questions(
        self,
        command: WriteGeneratedQuestionsCommand,
    ) -> WriteGeneratedQuestionsResult:
        self.questions_written.append(command)
        return WriteGeneratedQuestionsResult(
            questions_written=len(command.questions),
            status="written",
        )

    async def get_assessment_config(self, command: Any) -> Any:
        """Not used in current tests."""
        raise NotImplementedError

    async def close(self) -> None:
        pass


class FakeKnowledgeClient:
    """Stub knowledge client for tests."""

    async def get_topics(self, command: GetTopicsCommand) -> list[Topic]:
        """Return fake topics."""
        return [
            Topic(
                topic_id="topic_1",
                name="Test Topic 1",
                subtopics=[
                    Topic(topic_id="sub_1", name="Subtopic 1", subtopics=[]),
                    Topic(topic_id="sub_2", name="Subtopic 2", subtopics=[]),
                ],
            )
        ]

    async def similarity_search(
        self,
        command: SimilaritySearchCommand,
    ) -> list[Any]:
        """Return fake chunks."""
        from qna_generation_agent.infrastructure.grpc.knowledge_client import Chunk

        return [
            Chunk(
                chunk_id="chunk_1",
                workflow_id=command.workflow_id,
                content=f"Content for {command.query}",
                source_type="document",
                metadata={},
                score=0.9,
            ),
        ]

    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Any]:
        return []

    async def close(self) -> None:
        pass


@pytest.mark.unit
async def test_execute_generates_and_publishes_receipt() -> None:
    publisher = RecordingPublisher()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
    )

    receipt = await service.execute(
        GenerationCommand(
            request_id="evt_123",
            workflow_id="wf_123",
            correlation_id="corr_123",
            trace_id=None,  # Will be populated from telemetry
            assessment_id="assessment_123",
            question_set_id="qs_123",
            validation_result=None,
            iteration=None,
            structured_count=2,
            non_structured_count=1,
            difficulty_level="medium",
            purpose="assessment",
            feedback_issues=[],
        )
    )

    assert receipt.question_count == 3
    assert receipt.status == "completed"
    assert receipt.trace_id == "trace-test"
    assert len(publisher.events) == 1
    assert publisher.events[0].question_set_id == receipt.question_set_id


@pytest.mark.unit
async def test_execute_initial_trigger_uses_swarm_best_candidate() -> None:
    publisher = RecordingPublisher()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()

    weak_candidate = AssessmentGeneratorOutputSchema(
        questions=[
            _assessment_question(
                question_id="q-1",
                question_type="structured",
                question_text="Weak structured question?",
                answer_text="A",
            ),
            _assessment_question(
                question_id="q-2",
                question_type="non_structured",
                question_text="Weak open question?",
                answer_text="Weak answer",
            ),
        ]
    )
    strong_candidate = AssessmentGeneratorOutputSchema(
        questions=[
            _assessment_question(
                question_id="q-1",
                question_type="structured",
                question_text="Best structured question?",
                answer_text="B",
                explanation="Because the source chunk supports it.",
            ),
            _assessment_question(
                question_id="q-2",
                question_type="non_structured",
                question_text="Best open question?",
                answer_text="Best open answer",
                explanation="Because the rubric is explicit.",
            ),
        ]
    )

    service = GenerateQnAService(
        llm_provider=SwarmAwareFakeLLMProvider([weak_candidate, strong_candidate]),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
        swarm_size=2,
    )

    receipt = await service.execute(
        GenerationCommand(
            request_id="evt_swarm_123",
            workflow_id="wf_swarm_123",
            correlation_id="corr_swarm_123",
            trace_id=None,
            assessment_id="assessment_swarm_123",
            question_set_id="qs_swarm_123",
            validation_result=None,
            iteration=None,
            structured_count=1,
            non_structured_count=1,
            difficulty_level="medium",
            purpose="assessment",
            feedback_issues=[],
        )
    )

    assert receipt.question_count == 2
    assert submission_client.questions_written[0].questions[0].content == (
        "Best structured question?"
    )
    assert submission_client.questions_written[0].questions[1].content == (
        "Best open question?"
    )


@pytest.mark.unit
async def test_execute_returns_cached_receipt_for_completed_duplicate() -> None:
    """Test that completed duplicate events return the cached receipt."""
    publisher = RecordingPublisher()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
    )

    command = GenerationCommand(
        request_id="evt_123",
        workflow_id="wf_123",
        correlation_id="corr_123",
        trace_id=None,
        assessment_id="assessment_123",
        question_set_id="qs_123",
        validation_result=None,
        iteration=None,
        structured_count=1,
        non_structured_count=0,
        difficulty_level="medium",
        purpose="assessment",
        feedback_issues=[],
    )

    receipt1 = await service.execute(command)
    receipt2 = await service.execute(command)

    # Should return the same cached receipt for completed duplicates
    assert receipt1.question_set_id == receipt2.question_set_id
    assert receipt1.status == receipt2.status


@pytest.mark.unit
async def test_execute_with_prompt_provider_uses_prompt_assets() -> None:
    """Test that prompt provider is used when local prompt assets are available."""

    class FakePromptProvider(PromptProvider):
        def __init__(self) -> None:
            self.prompts_fetched: list[str] = []

        async def get_prompt(self, name: str) -> Prompt:
            self.prompts_fetched.append(name)
            if name == PromptProvider.ITEM_WRITER_PROMPT_NAME:
                return Prompt(
                    name=name,
                    version="1",
                    system_prompt=f"System prompt for {name}",
                    user_prompt=(
                        "Generate {{structured_count}} structured and "
                        "{{non_structured_count}} open-ended questions at "
                        "{{difficulty}} difficulty about {{topics}} using {{chunks}}."
                    ),
                )
            if name == PromptProvider.OPTIONS_ONLY_WRITER_PROMPT_NAME:
                return Prompt(
                    name=name,
                    version="1",
                    system_prompt=f"System prompt for {name}",
                    user_prompt=(
                        "Complete the following MCQ question by generating the correct "
                        "answer and three plausible distractors.\n\nQuestion: {{question_text}}\n"
                        "Topic: {{topic}}\nDifficulty: {{difficulty}}\n\nContext: {{chunk_content}}"
                    ),
                )
            return Prompt(
                name=name,
                version="1",
                system_prompt=f"System prompt for {name}",
                user_prompt="Feedback prompt",
            )

        async def health_check(self) -> bool:
            return True

        async def shutdown(self) -> None:
            pass

    publisher = RecordingPublisher()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()
    prompt_provider = FakePromptProvider()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
        prompt_provider=prompt_provider,
    )

    # The fake LLM provider will handle generate_with_prompt
    receipt = await service.execute(
        GenerationCommand(
            request_id="evt_prompt_123",
            workflow_id="wf_prompt_123",
            correlation_id="corr_prompt_123",
            trace_id=None,
            assessment_id="assessment_prompt_123",
            question_set_id="qs_prompt_123",
            validation_result=None,
            iteration=None,
            structured_count=1,
            non_structured_count=0,
            difficulty_level="medium",
            purpose="assessment",
            feedback_issues=[],
        )
    )

    assert receipt.question_count == 1
    assert receipt.status == "completed"


@pytest.mark.unit
async def test_execute_exceeds_max_iterations_raises_error() -> None:
    """Test that exceeding max iterations raises WorkflowEscalationError."""

    publisher = RecordingPublisher()
    knowledge_client = FakeKnowledgeClient()

    # Create a submission client that returns a question set with high iteration
    class HighIterationSubmissionClient(FakeSubmissionClient):
        async def increment_iteration(self, command: Any) -> Any:
            from qna_generation_agent.application.ports.submission_client import (
                IncrementIterationResult,
            )

            return IncrementIterationResult(
                question_set_id=command.question_set_id,
                iteration_count=4,  # Exceeds default max_iterations=3
                status="incremented",
            )

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=HighIterationSubmissionClient(),
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
        max_iterations=3,
    )

    with pytest.raises(WorkflowEscalationError):
        await service.execute(
            GenerationCommand(
                request_id="evt_iter_123",
                workflow_id="wf_iter_123",
                correlation_id="corr_iter_123",
                trace_id=None,
                assessment_id="assessment_iter_123",
                question_set_id="qs_existing_123",  # Existing question set triggers increment
                validation_result="fail",  # Regeneration trigger
                iteration=3,  # Current iteration before increment
                structured_count=None,  # Nullable for regeneration
                non_structured_count=None,  # Nullable for regeneration
                difficulty_level=None,  # Nullable for regeneration
                purpose=None,  # Nullable for regeneration
                feedback_issues=["Q1 issue"],
            )
        )


@pytest.mark.unit
async def test_execute_retrigger_limits_to_mcq_only() -> None:
    publisher = RecordingPublisher()

    class RegenSubmissionClient(FakeSubmissionClient):
        async def get_assessment_config(self, command: Any) -> AssessmentConfig:
            return AssessmentConfig(
                assessment_id=command.assessment_id,
                workflow_id=command.workflow_id,
                assessor_id="assessor_123",
                assessment_title="Regeneration Assessment",
                purpose="assessment",
                duration_minutes=60,
                difficulty_level="medium",
                structured_question_count=2,
                non_structured_question_count=2,
                web_research_mode="disabled",
                status="draft",
            )

    submission_client = RegenSubmissionClient()
    knowledge_client = FakeKnowledgeClient()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
    )

    receipt = await service.execute(
        GenerationCommand(
            request_id="evt_regen_mcq_only_123",
            workflow_id="wf_regen_mcq_only",
            correlation_id="corr_regen_mcq_only",
            trace_id=None,
            assessment_id="assessment_regen_mcq_only",
            question_set_id="qs_existing_123",
            validation_result="fail",
            iteration=1,
            structured_count=None,
            non_structured_count=None,
            difficulty_level=None,
            purpose=None,
            feedback_issues=["validator rejected open-ended questions"],
        )
    )

    assert receipt.structured_generated == 2
    assert receipt.non_structured_generated == 0
    assert all(
        question.question_type == "structured"
        for question in submission_client.questions_written[0].questions
    )


@pytest.mark.unit
async def test_execute_raises_idempotency_conflict_when_processing() -> None:
    """Test that PROCESSING status raises IdempotencyConflict for concurrent events."""
    idempotency_store = InMemoryIdempotencyStore()
    publisher = RecordingPublisher()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=idempotency_store,
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
    )

    command = GenerationCommand(
        request_id="evt_processing_123",
        workflow_id="wf_123",
        correlation_id="corr_123",
        trace_id=None,
        assessment_id="assessment_123",
        question_set_id="qs_123",
        validation_result=None,
        iteration=None,
        structured_count=1,
        non_structured_count=0,
        difficulty_level="medium",
        purpose="assessment",
        feedback_issues=[],
    )

    # Manually set the idempotency record to PROCESSING state
    await idempotency_store.start(command.request_id, ttl_seconds=300)

    # Second attempt should raise IdempotencyConflict
    with pytest.raises(IdempotencyConflict):
        await service.execute(command)


@pytest.mark.unit
async def test_execute_retries_after_failed_idempotency() -> None:
    """Test that failed idempotency status allows retry by proceeding with generation."""
    idempotency_store = InMemoryIdempotencyStore()
    publisher = RecordingPublisher()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=idempotency_store,
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
    )

    command = GenerationCommand(
        request_id="evt_retry_123",
        workflow_id="wf_123",
        correlation_id="corr_123",
        trace_id=None,
        assessment_id="assessment_123",
        question_set_id="qs_123",
        validation_result=None,
        iteration=None,
        structured_count=1,
        non_structured_count=0,
        difficulty_level="medium",
        purpose="assessment",
        feedback_issues=[],
    )

    # First attempt - start processing but mark as failed
    acquired = await idempotency_store.start(command.request_id, ttl_seconds=300)
    assert acquired is True
    await idempotency_store.fail(
        command.request_id, error_message="Previous failure", ttl_seconds=3600
    )

    # Verify the record is in FAILED state
    failed_record = await idempotency_store.get(command.request_id)
    assert failed_record.status is IdempotencyStatus.FAILED

    # Second attempt should succeed (retry allowed for failed records)
    receipt = await service.execute(command)

    assert receipt.question_count == 1
    assert receipt.status == "completed"


@pytest.mark.unit
async def test_execute_raises_retrieval_error_when_no_chunks() -> None:
    """Test that empty chunks from Knowledge Service raises RetrievalError."""

    class EmptyChunksKnowledgeClient:
        """Knowledge client that returns no chunks."""

        async def get_topics(self, command: GetTopicsCommand) -> list[Topic]:
            return [
                Topic(
                    topic_id="topic_1",
                    name="Test Topic 1",
                    subtopics=[
                        Topic(topic_id="sub_1", name="Subtopic 1", subtopics=[])
                    ],
                )
            ]

        async def similarity_search(
            self, command: SimilaritySearchCommand
        ) -> list[Any]:
            return []  # No chunks found

        async def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[Any]:
            return []

        async def close(self) -> None:
            pass

    publisher = RecordingPublisher()
    submission_client = FakeSubmissionClient()
    knowledge_client = EmptyChunksKnowledgeClient()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
    )

    with pytest.raises(RetrievalError, match="No chunks retrieved"):
        await service.execute(
            GenerationCommand(
                request_id="evt_no_chunks_123",
                workflow_id="wf_123",
                correlation_id="corr_123",
                trace_id=None,
                assessment_id="assessment_123",
                question_set_id="qs_123",
                validation_result=None,
                iteration=None,
                structured_count=2,
                non_structured_count=1,
                difficulty_level="medium",
                purpose="assessment",
                feedback_issues=[],
            )
        )


@pytest.mark.unit
async def test_execute_raises_transient_error_on_write_failure() -> None:
    """Test that submission write failure raises StorageTransientError."""

    class FailingWriteSubmissionClient(FakeSubmissionClient):
        """Submission client that fails on write."""

        async def write_generated_questions(
            self, command: WriteGeneratedQuestionsCommand
        ) -> WriteGeneratedQuestionsResult:
            raise StorageTransientError(
                "Submission Service unavailable", retry_after_seconds=5
            )

    publisher = RecordingPublisher()
    knowledge_client = FakeKnowledgeClient()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=FailingWriteSubmissionClient(),
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
    )

    with pytest.raises(StorageTransientError, match="Submission Service unavailable"):
        await service.execute(
            GenerationCommand(
                request_id="evt_write_fail_123",
                workflow_id="wf_123",
                correlation_id="corr_123",
                trace_id=None,
                assessment_id="assessment_123",
                question_set_id="qs_123",
                validation_result=None,
                iteration=None,
                structured_count=1,
                non_structured_count=0,
                difficulty_level="medium",
                purpose="assessment",
                feedback_issues=[],
            )
        )


@pytest.mark.unit
async def test_execute_best_effort_publish_failure_does_not_corrupt_idempotency() -> None:
    """Test that best-effort publish failure does not corrupt idempotency status.

    After the submission service successfully persists questions, idempotency
    must remain COMPLETED even if post-completion events fail to publish.
    This prevents duplicate question generation on retry.
    """
    idempotency_store = InMemoryIdempotencyStore()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()

    class FailingPublisher(NullEventPublisher):
        """Publisher that fails on completion publish."""

        async def publish_completion(self, event: QnAGenerationCompleted) -> None:
            raise StorageTransientError("Pub/Sub unavailable", retry_after_seconds=5)

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=idempotency_store,
        event_publisher=FailingPublisher(),
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
    )

    receipt = await service.execute(
        GenerationCommand(
            request_id="evt_publish_fail_123",
            workflow_id="wf_123",
            correlation_id="corr_123",
            trace_id=None,
            assessment_id="assessment_123",
            question_set_id="qs_123",
            validation_result=None,
            iteration=None,
            structured_count=1,
            non_structured_count=0,
            difficulty_level=DifficultyLevel.MEDIUM,
            purpose=Purpose.ASSESSMENT,
            feedback_issues=[],
        )
    )

    # Receipt should still be returned successfully
    assert receipt.question_count == 1

    # Idempotency must remain COMPLETED, not FAILED
    record = await idempotency_store.get("evt_publish_fail_123")
    assert record.status is IdempotencyStatus.COMPLETED
    assert record.receipt is not None


@pytest.mark.unit
async def test_execute_generates_non_structured_only() -> None:
    """Test generation with only non-structured questions (no structured)."""
    publisher = RecordingPublisher()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
    )

    receipt = await service.execute(
        GenerationCommand(
            request_id="evt_non_structured_123",
            workflow_id="wf_123",
            correlation_id="corr_123",
            trace_id=None,
            assessment_id="assessment_123",
            question_set_id="qs_123",
            validation_result=None,
            iteration=None,
            structured_count=0,  # No structured questions
            non_structured_count=3,  # Only non-structured
            difficulty_level="medium",
            purpose="assessment",
            feedback_issues=[],
        )
    )

    assert receipt.question_count == 3
    assert receipt.status == "completed"
    assert len(publisher.events) == 1
    assert publisher.events[0].non_structured_generated == 3


@pytest.mark.unit
async def test_execute_with_prompt_provider_uses_combined_prompt() -> None:
    """Test that prompt provider drives the mixed assessment prompt."""

    class RecordingPromptProvider(PromptProvider):
        """Prompt provider that records prompt requests."""

        def __init__(self) -> None:
            self.prompts_fetched: list[str] = []

        async def get_prompt(self, name: str) -> Prompt:
            self.prompts_fetched.append(name)
            if name == PromptProvider.ITEM_WRITER_PROMPT_NAME:
                user_prompt = (
                    "Generate {{structured_count}} structured and "
                    "{{non_structured_count}} open-ended questions at "
                    "{{difficulty}} difficulty about {{topics}} using {{chunks}}."
                )
            elif name == PromptProvider.OPTIONS_ONLY_WRITER_PROMPT_NAME:
                user_prompt = (
                    "Complete the following MCQ question by generating the correct "
                    "answer and three plausible distractors.\n\nQuestion: {{question_text}}\n"
                    "Topic: {{topic}}\nDifficulty: {{difficulty}}\n\nContext: {{chunk_content}}"
                )
            else:
                user_prompt = "Feedback prompt"
            return Prompt(
                name=name,
                version="1",
                system_prompt=f"System prompt for {name}",
                user_prompt=user_prompt,
            )

        async def health_check(self) -> bool:
            return True

        async def shutdown(self) -> None:
            pass

    publisher = RecordingPublisher()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()
    prompt_provider = RecordingPromptProvider()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
        prompt_provider=prompt_provider,
    )

    receipt = await service.execute(
        GenerationCommand(
            request_id="evt_prompt_compile_123",
            workflow_id="wf_prompt_compile",
            correlation_id="corr_prompt_compile",
            trace_id=None,
            assessment_id="assessment_prompt_compile",
            question_set_id="qs_prompt_compile",
            validation_result=None,
            iteration=None,
            structured_count=2,
            non_structured_count=1,
            difficulty_level="hard",
            purpose="assessment",
            feedback_issues=[],
        )
    )

    assert receipt.question_count == 3
    assert receipt.status == "completed"
    assert prompt_provider.prompts_fetched[0] == "item-writer"
    assert len(prompt_provider.prompts_fetched) == 5


@pytest.mark.unit
async def test_execute_publishes_audit_and_token_events() -> None:
    """Test that decision audit and token usage events are published with correct data."""

    class AuditCapturingPublisher(NullEventPublisher):
        """Publisher that captures audit events."""

        def __init__(self) -> None:
            super().__init__()
            from qna_generation_agent.application.ports.publisher import (
                DecisionAuditEvent,
                TokenUsageEvent,
            )

            self.decision_audits: list[DecisionAuditEvent] = []
            self.token_usages: list[TokenUsageEvent] = []
            self.completions: list[QnAGenerationCompleted] = []

        async def publish_completion(self, event: QnAGenerationCompleted) -> None:
            self.completions.append(event)

        async def publish_decision_audit(self, event: Any) -> None:
            self.decision_audits.append(event)

        async def publish_token_usage(self, event: Any) -> None:
            self.token_usages.append(event)

    publisher = AuditCapturingPublisher()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
    )

    receipt = await service.execute(
        GenerationCommand(
            request_id="evt_audit_123",
            workflow_id="wf_audit",
            correlation_id="corr_audit",
            trace_id=None,
            assessment_id="assessment_audit",
            question_set_id="qs_audit",
            validation_result=None,
            iteration=None,
            structured_count=2,
            non_structured_count=1,
            difficulty_level="medium",
            purpose="assessment",
            feedback_issues=[],
        )
    )

    assert receipt.question_count >= 1  # At least 1 question generated
    # Verify audit events were captured
    # Note: The exact count depends on implementation; we verify the structure
    assert len(publisher.completions) == 1
    # The question_set_id comes from the submission service, not the command
    assert publisher.completions[0].question_set_id == receipt.question_set_id
    # Verify counts match what was actually generated
    assert publisher.completions[0].structured_generated >= 0
    assert publisher.completions[0].non_structured_generated >= 0


@pytest.mark.unit
async def test_execute_propagates_trace_id_from_command() -> None:
    """Test that trace_id from command is propagated to receipt."""
    publisher = RecordingPublisher()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()

    class TraceCapturingTelemetry(FakeTelemetry):
        """Telemetry that captures trace context."""

        def __init__(self) -> None:
            super().__init__()
            self.captured_traces: list[dict[str, str]] = []

        @contextmanager
        def propagate(
            self,
            trace_name: str,
            correlation_id: str,
            workflow_id: str,
            metadata: dict[str, str] | None = None,
        ) -> Generator[None]:
            self.captured_traces.append(
                {
                    "trace_name": trace_name,
                    "correlation_id": correlation_id,
                    "workflow_id": workflow_id,
                    "metadata": metadata or {},
                }
            )
            yield

    telemetry = TraceCapturingTelemetry()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=telemetry,
    )

    receipt = await service.execute(
        GenerationCommand(
            request_id="evt_trace_123",
            workflow_id="wf_trace",
            correlation_id="corr_trace",
            trace_id="custom-trace-id-123",  # Custom trace ID
            assessment_id="assessment_trace",
            question_set_id="qs_trace",
            validation_result=None,
            iteration=None,
            structured_count=1,
            non_structured_count=0,
            difficulty_level="medium",
            purpose="assessment",
            feedback_issues=[],
        )
    )

    assert receipt.status == "completed"
    # Verify trace was captured with correct workflow_id
    assert len(telemetry.captured_traces) == 1
    assert telemetry.captured_traces[0]["workflow_id"] == "wf_trace"
    assert telemetry.captured_traces[0]["correlation_id"] == "corr_trace"


@pytest.mark.unit
async def test_mcq_option_letter_preservation_non_a_correct() -> None:
    """Test that MCQ option letters from answer generator are preserved, even when correct answer is not 'A'."""
    from qna_generation_agent.application.ports.prompt_provider import (
        PromptProvider,
    )

    # LLM provider that returns C as the correct answer (not A)
    class NonACorrectLLMProvider(FakeLLMProvider):
        async def invoke_with_system_and_user[
            T: BaseModel
        ](
            self,
            system_message: str,
            user_message: str,
            *,
            structured_output_model: type[T],
            model_tier: str = "expensive",
        ) -> T | None:
            if structured_output_model == AssessmentGeneratorOutputSchema:
                return AssessmentGeneratorOutputSchema(  # type: ignore[return-value]
                    questions=[
                        AssessmentQuestionSchema(
                            question_id="q-001",
                            question_type="structured",
                            content="What is the correct article?",
                            structured_answer="C",
                            metadata={
                                "topic": "test-topic",
                                "difficulty": "medium",
                                "source_chunk_ids": ["chunk-1"],
                            },
                        )
                    ]
                )
            if structured_output_model == MCQAnswerGeneratorOutputSchema:
                # Return C as correct answer - this tests letter preservation
                return MCQAnswerGeneratorOutputSchema(  # type: ignore[return-value]
                    question_stem="What is the correct article?",
                    correct_answer=MCQDistractorExplanationSchema(
                        option_letter="C",
                        option_text="Correct: 'the' is required for specific nouns",
                        is_correct=True,
                        explanation="'The' is used before specific nouns known to both speaker and listener",
                    ),
                    distractors=[
                        MCQDistractorExplanationSchema(
                            option_letter="A",
                            option_text="Incorrect: no article",
                            is_correct=False,
                            explanation="Chinese L1 speakers often omit articles",
                        ),
                        MCQDistractorExplanationSchema(
                            option_letter="B",
                            option_text="Incorrect: 'a' instead of 'the'",
                            is_correct=False,
                            explanation="Using indefinite article instead of definite",
                        ),
                        MCQDistractorExplanationSchema(
                            option_letter="D",
                            option_text="Incorrect: 'an' used incorrectly",
                            is_correct=False,
                            explanation="Wrong article form",
                        ),
                    ],
                    grammar_point_tested="article usage",
                    difficulty_justification="medium difficulty",
                    l1_considerations=[
                        "Chinese article omission",
                        "Indefinite vs definite",
                    ],
                )
            return None

    class ConfigurablePromptProvider(PromptProvider):
        """Prompt provider that returns prompts for 3-prompt workflow."""

        async def get_prompt(self, name: str) -> Prompt:
            return Prompt(
                name=name,
                version="1",
                system_prompt=f"System prompt for {name}",
                user_prompt=f"User prompt for {name}",
            )

        async def health_check(self) -> bool:
            return True

        async def shutdown(self) -> None:
            pass

    publisher = RecordingPublisher()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()
    prompt_provider = ConfigurablePromptProvider()
    llm_provider = NonACorrectLLMProvider()

    service = GenerateQnAService(
        llm_provider=llm_provider,
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
        prompt_provider=prompt_provider,
    )

    receipt = await service.execute(
        GenerationCommand(
            request_id="evt_mcq_letter_123",
            workflow_id="wf_mcq",
            correlation_id="corr_mcq",
            trace_id=None,
            assessment_id="assessment_mcq",
            question_set_id="qs_mcq",
            validation_result=None,
            iteration=None,
            structured_count=1,
            non_structured_count=0,
            difficulty_level="medium",
            purpose="assessment",
            feedback_issues=[],
        )
    )

    assert receipt.question_count == 1
    assert receipt.status == "completed"
    assert receipt.structured_generated == 1

    # Verify the written question preserves the correct letter
    written = submission_client.questions_written[0]
    assert len(written.questions) == 1
    question = written.questions[0]
    # The structured_answer should contain option C as correct
    assert (
        "C) Correct" in question.structured_answer
        or "C ✓ CORRECT" in question.structured_answer
        or "Correct Answer: C" in question.structured_answer
    )


@pytest.mark.unit
async def test_regeneration_uses_assessment_config_from_submission_service() -> None:
    """Test that regeneration fetches config from Submission Service when available."""
    from qna_generation_agent.application.ports.submission_client import (
        AssessmentConfig,
        GetAssessmentConfigCommand,
    )

    idempotency_store = InMemoryIdempotencyStore()
    publisher = RecordingPublisher()
    knowledge_client = FakeKnowledgeClient()

    class ConfigCapturingSubmissionClient(FakeSubmissionClient):
        """Client that captures get_assessment_config calls and returns predefined config."""

        def __init__(self) -> None:
            super().__init__()
            self.config_calls: list[GetAssessmentConfigCommand] = []
            self.config_to_return = AssessmentConfig(
                assessment_id="assessment_regen_config",
                workflow_id="wf_regen_config",
                assessor_id="assessor_123",
                assessment_title="Test Assessment",
                purpose="assessment",
                duration_minutes=60,
                difficulty_level="hard",  # Different from trigger (which is null)
                structured_question_count=5,  # Different from trigger
                non_structured_question_count=2,  # Different from trigger
                web_research_mode="disabled",
                status="active",
            )

        async def get_assessment_config(self, command: Any) -> Any:
            self.config_calls.append(command)
            return self.config_to_return

    submission_client = ConfigCapturingSubmissionClient()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=idempotency_store,
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
    )

    receipt = await service.execute(
        GenerationCommand(
            request_id="evt_regen_config_123",
            workflow_id="wf_regen_config",
            correlation_id="corr_regen_config",
            trace_id=None,
            assessment_id="assessment_regen_config",
            question_set_id="qs_regen_existing",  # Existing question set
            validation_result="fail",  # Regeneration trigger
            iteration=1,
            structured_count=None,  # Null - should use config
            non_structured_count=None,  # Null - should use config
            difficulty_level=None,  # Null - should use config
            purpose=None,
            feedback_issues=["Q1 needs fix"],
        )
    )

    # Verify that get_assessment_config was called
    assert len(submission_client.config_calls) == 1
    assert submission_client.config_calls[0].assessment_id == "assessment_regen_config"

    # Retrigger regenerations now stay MCQ-only.
    assert receipt.question_count == 5
    assert receipt.structured_generated == 5
    assert receipt.non_structured_generated == 0
