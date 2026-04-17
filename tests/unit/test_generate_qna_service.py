"""Unit tests for the generation service."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest

from qna_generation_agent.application.dto import (
    AssessmentContext,
    GenerationCommand,
    QuestionBatch,
    QuestionDraft,
)
from qna_generation_agent.application.ports.llm import LLMProvider
from qna_generation_agent.application.ports.submission_client import (
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
from qna_generation_agent.domain.enums import QuestionType
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

    async def invoke_with_schema(
        self, prompt: str, *, structured_output_model: type[Any]
    ) -> Any | None:
        """Invoke LLM with structured output schema for 3-prompt workflow."""
        # Return appropriate mock data based on the schema type
        if structured_output_model == AssessmentGeneratorOutputSchema:
            return AssessmentGeneratorOutputSchema(
                questions=[
                    AssessmentQuestionSchema(
                        question_id="q-001",
                        question_type="structured",
                        content="Test question stem?",
                        structured_answer="A",
                        metadata={
                            "topic": "test-topic",
                            "difficulty": "medium",
                            "source_chunk_ids": ["chunk-1"],
                        },
                    )
                ]
            )
        if structured_output_model == MCQAnswerGeneratorOutputSchema:
            return MCQAnswerGeneratorOutputSchema(
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
        from datetime import UTC, datetime

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
async def test_execute_with_prompt_provider_uses_langfuse() -> None:
    """Test that prompt provider triggers Langfuse path when available."""
    from qna_generation_agent.application.ports.prompt_provider import (
        Prompt,
        PromptProvider,
    )

    class FakePromptProvider(PromptProvider):
        async def get_prompt(
            self, name: str, *, label: str | None = None, version: int | None = None
        ) -> Prompt:
            return Prompt(
                name=name,
                version=1,
                prompt_text="Test prompt with {structured_generated} questions",
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
    from qna_generation_agent.application.errors import WorkflowEscalationError

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
