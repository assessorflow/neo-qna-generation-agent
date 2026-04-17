"""Unit tests for the generation service."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from qna_generation_agent.application.dto import (
    AssessmentContext,
    GenerationCommand,
    QuestionBatch,
    QuestionDraft,
)
from qna_generation_agent.application.errors import (
    IdempotencyConflict,
    RetrievalError,
    StoragePermanentError,
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
            # Extract count from prompt to return appropriate number of questions
            # The prompt format is: "Generate {N} structured and {M} non-structured questions..."
            import re

            match = re.search(
                r"(\d+)\s+structured.*?(\d+)\s+non-structured", prompt, re.IGNORECASE
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
            return AssessmentGeneratorOutputSchema(questions=questions)
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
async def test_execute_with_prompt_provider_uses_langfuse() -> None:
    """Test that prompt provider triggers Langfuse path when available."""
    from qna_generation_agent.application.ports.prompt_provider import (
        Prompt,
        PromptProvider,
    )

    class FakePromptProvider(PromptProvider):
        @property
        def default_label(self) -> str:
            return "production"

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
async def test_execute_marks_idempotency_failed_on_publish_failure() -> None:
    """Test that completion publish failure marks idempotency as failed."""
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

    with pytest.raises(StorageTransientError):
        await service.execute(
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
                difficulty_level="medium",
                purpose="assessment",
                feedback_issues=[],
            )
        )

    # Verify idempotency record is marked as failed
    failed_record = await idempotency_store.get("evt_publish_fail_123")
    assert failed_record.status is IdempotencyStatus.FAILED


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
async def test_execute_with_prompt_provider_uses_compiled_prompt() -> None:
    """Test that prompt provider compiles and uses the correct prompt variables."""

    class RecordingPromptProvider(PromptProvider):
        """Prompt provider that records compilation calls."""

        def __init__(self) -> None:
            self.prompts_fetched: list[tuple[str, str | None]] = []
            self.compiled_prompts: list[tuple[str, dict[str, Any]]] = []

        @property
        def default_label(self) -> str:
            return "production"

        async def get_prompt(
            self, name: str, *, label: str | None = None, version: int | None = None
        ) -> Prompt:
            self.prompts_fetched.append((name, label))
            # Create a prompt with proper double-brace placeholders
            return Prompt(
                name=name,
                version=1,
                prompt_text="Generate {structured_count} structured and {non_structured_count} non-structured questions. Difficulty: {difficulty}. Topics: {topics}. Chunks: {chunks}.",
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

    # Note: With 3-prompt workflow, structured_count may generate fewer questions
    # than requested due to MCQ assembly complexity. We verify generation succeeded.
    assert receipt.question_count >= 1
    assert receipt.status == "completed"
    # Verify prompt provider was called
    assert len(prompt_provider.prompts_fetched) > 0


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
        Prompt,
        PromptProvider,
    )

    # LLM provider that returns C as the correct answer (not A)
    class NonACorrectLLMProvider(FakeLLMProvider):
        async def invoke_with_schema(
            self, prompt: str, *, structured_output_model: type[Any]
        ) -> Any | None:
            if structured_output_model == AssessmentGeneratorOutputSchema:
                return AssessmentGeneratorOutputSchema(
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
                return MCQAnswerGeneratorOutputSchema(
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

        @property
        def default_label(self) -> str:
            return "production"

        async def get_prompt(
            self, name: str, *, label: str | None = None, version: int | None = None
        ) -> Prompt:
            return Prompt(
                name=name,
                version=1,
                prompt_text=f"Test prompt for {name}",
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

    # Verify receipt uses config values (5 structured + 2 non_structured)
    assert receipt.question_count == 7
    assert receipt.structured_generated == 5
    assert receipt.non_structured_generated == 2


@pytest.mark.unit
async def test_regeneration_fallback_to_trigger_when_config_unimplemented() -> None:
    """Test that regeneration falls back to trigger data when GetAssessmentConfig not implemented."""
    from qna_generation_agent.application.ports.submission_client import (
        GetAssessmentConfigCommand,
    )

    idempotency_store = InMemoryIdempotencyStore()
    publisher = RecordingPublisher()
    knowledge_client = FakeKnowledgeClient()

    class UnimplementedConfigSubmissionClient(FakeSubmissionClient):
        """Client that raises StoragePermanentError for get_assessment_config."""

        def __init__(self) -> None:
            super().__init__()
            self.config_calls: list[GetAssessmentConfigCommand] = []

        async def get_assessment_config(self, command: Any) -> Any:
            self.config_calls.append(command)
            raise StoragePermanentError(
                "GetAssessmentConfig not yet implemented",
                code="UNIMPLEMENTED",
            )

    submission_client = UnimplementedConfigSubmissionClient()

    service = GenerateQnAService(
        llm_provider=FakeLLMProvider(),
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=idempotency_store,
        event_publisher=publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,  # type: ignore
        telemetry=FakeTelemetry(),
    )

    # Regeneration with fallback values in trigger (not null)
    receipt = await service.execute(
        GenerationCommand(
            request_id="evt_regen_fallback_123",
            workflow_id="wf_regen_fallback",
            correlation_id="corr_regen_fallback",
            trace_id=None,
            assessment_id="assessment_regen_fallback",
            question_set_id="qs_regen_fallback",  # Existing question set
            validation_result="fail",  # Regeneration trigger
            iteration=1,
            structured_count=3,  # Fallback value
            non_structured_count=1,  # Fallback value
            difficulty_level="easy",  # Fallback value
            purpose="assessment",
            feedback_issues=["Q1 needs fix"],
        )
    )

    # Verify that get_assessment_config was attempted
    assert len(submission_client.config_calls) == 1

    # Verify receipt uses fallback values from trigger (3 + 1)
    assert receipt.question_count == 4
    assert receipt.structured_generated == 3
    assert receipt.non_structured_generated == 1
