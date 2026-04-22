"""QnA generation application service."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from qna_generation_agent.app.json import dumps
from qna_generation_agent.app.logging import bind_context, get_logger
from qna_generation_agent.application.dto import (
    AssessmentContext,
    GenerationCommand,
    GenerationReceipt,
)
from qna_generation_agent.application.errors import (
    GenerationError,
    IdempotencyConflict,
    LLMPermanentError,
    LLMTransientError,
    RetrievalError,
    StoragePermanentError,
    StorageTransientError,
    WorkflowEscalationError,
)
from qna_generation_agent.application.ports.idempotency import (
    IdempotencyStatus,
    IdempotencyStore,
)
from qna_generation_agent.application.ports.knowledge_client import (
    GetTopicsCommand,
    KnowledgeClient,
    SimilaritySearchCommand,
)
from qna_generation_agent.application.ports.llm import LLMProvider
from qna_generation_agent.application.ports.prompt_provider import (
    PromptProvider,
)
from qna_generation_agent.application.ports.publisher import (
    DecisionAuditEvent,
    EventPublisher,
    TokenUsageEvent,
)
from qna_generation_agent.application.ports.repository import QuestionSetRepository
from qna_generation_agent.application.ports.submission_client import (
    CreateQuestionSetCommand,
    GetAssessmentConfigCommand,
    IncrementIterationCommand,
    Question,
    SubmissionClient,
    WriteGeneratedQuestionsCommand,
)
from qna_generation_agent.application.ports.telemetry import TelemetryPort
from qna_generation_agent.domain.entities import (
    Answer,
    GenerationRequest,
    QuestionSet,
)
from qna_generation_agent.domain.entities import (
    Question as DomainQuestion,
)
from qna_generation_agent.domain.enums import (
    DifficultyLevel,
    GenerationStage,
    GenerationStatus,
    QuestionType,
    ValidationResult,
)
from qna_generation_agent.domain.events import QnAGenerationCompleted
from qna_generation_agent.domain.value_objects import AnswerId, QuestionId

logger = get_logger(__name__)


@runtime_checkable
class SwarmCapableLLMProvider(Protocol):
    """LLM provider extension for swarm-based candidate generation."""

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
        """Generate swarm candidates for initial trigger workflows."""


DEFAULT_ASSESSMENT_SYSTEM_PROMPT = (
    "You are an expert assessment creator for English language proficiency tests. "
    "Generate a mixed set of structured multiple-choice questions and open-ended "
    "questions grounded in the source material. Follow the requested counts "
    "exactly and return valid JSON matching the schema."
)

# Overall generation timeout per stage (in seconds)
# This is higher than the LLM timeout to allow for retries and overhead
GENERATION_STAGE_TIMEOUT_SECONDS = 180


@dataclass(frozen=True, slots=True)
class Subtopic:
    """Selected subtopic for question generation."""

    topic_id: str
    name: str


@dataclass(frozen=True, slots=True)
class RegenerationQuestionSource:
    """Local replay source for regeneration flows."""

    question_id: str
    question_type: str
    question_text: str
    answer_text: str
    metadata: dict[str, Any]


class GenerateQnAService:
    """Use-case service for question generation per spec workflow."""

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        question_set_repo: QuestionSetRepository,
        idempotency_store: IdempotencyStore,
        event_publisher: EventPublisher,
        submission_client: SubmissionClient,
        knowledge_client: KnowledgeClient,
        telemetry: TelemetryPort | None = None,
        max_iterations: int = 3,
        swarm_size: int = 3,
        prompt_provider: PromptProvider | None = None,
        cheap_llm_provider: LLMProvider | None = None,
        expensive_llm_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize the generation service with required ports."""
        self._llm_provider = llm_provider
        self._question_set_repo = question_set_repo
        self._idempotency_store = idempotency_store
        self._event_publisher = event_publisher
        self._submission_client = submission_client
        self._knowledge_client = knowledge_client
        self._telemetry = telemetry
        self._max_iterations = max_iterations
        self._swarm_size = swarm_size
        self._prompt_provider = prompt_provider
        self._cheap_llm_provider = cheap_llm_provider
        self._expensive_llm_provider = expensive_llm_provider or llm_provider

    def _get_provider_for_stage(self, stage: GenerationStage) -> LLMProvider:
        """Get the appropriate LLM provider based on generation stage.

        Args:
            stage: The generation stage identifier.

        Returns:
            LLMProvider appropriate for the stage (cheap or expensive).
        """
        expensive_stages = {
            GenerationStage.ASSESSMENT_GENERATOR,
            GenerationStage.QUESTION_GENERATION,
        }
        cheap_stages = {
            GenerationStage.MCQ_ANSWER_GENERATOR,
            GenerationStage.MCQ_EXPLANATION_GENERATOR,
        }

        if stage in expensive_stages:
            provider = self._expensive_llm_provider
        elif stage in cheap_stages:
            provider = self._cheap_llm_provider or self._expensive_llm_provider
        else:
            provider = self._expensive_llm_provider

        logger.debug("using_llm_provider", stage=stage, model=provider.model_id)
        return provider

    async def execute(self, command: GenerationCommand) -> GenerationReceipt:
        """Execute the generation workflow per spec."""
        bind_context(
            event_id=command.request_id,
            workflow_id=command.workflow_id,
            correlation_id=command.correlation_id,
            assessment_id=command.assessment_id,
        )
        root_context = (
            self._telemetry.propagate(
                trace_name="qna_generation",
                correlation_id=command.correlation_id,
                workflow_id=command.workflow_id,
                metadata={"assessment_id": command.assessment_id},
            )
            if self._telemetry is not None
            else nullcontext()
        )

        with root_context:
            acquired = await self._idempotency_store.start(
                command.request_id, ttl_seconds=300
            )
            if not acquired:
                existing = await self._idempotency_store.get(command.request_id)
                if (
                    existing.status is IdempotencyStatus.COMPLETED
                    and existing.receipt is not None
                ):
                    # Return cached receipt for completed duplicate events
                    logger.info(
                        "returning_cached_receipt",
                        event_id=command.request_id,
                        question_set_id=existing.receipt.question_set_id,
                    )
                    return existing.receipt
                if existing.status is IdempotencyStatus.PROCESSING:
                    raise IdempotencyConflict(
                        "Generation is already in progress",
                        event_id=command.request_id,
                    )
                # Failed or other states: allow retry by proceeding

            try:
                receipt = await self._generate(command)
                # Idempotency is marked complete inside _generate() immediately after
                # successful write to prevent duplicate-write hazard on retry
                return receipt
            except (
                StorageTransientError,
                StoragePermanentError,
                GenerationError,
                RetrievalError,
                WorkflowEscalationError,
                LLMTransientError,
                LLMPermanentError,
            ) as error:
                # Preserve typed errors for proper ack/nack decisions
                await self._idempotency_store.fail(
                    command.request_id,
                    error_message=str(error),
                    ttl_seconds=3600,
                )
                raise
            except Exception as error:
                # Unexpected error - still mark as failed but log as error
                logger.exception(
                    "unexpected_generation_error", request_id=command.request_id
                )
                await self._idempotency_store.fail(
                    command.request_id,
                    error_message=f"Unexpected error: {error}",
                    ttl_seconds=3600,
                )
                raise

    async def _generate(self, command: GenerationCommand) -> GenerationReceipt:
        """Execute spec-compliant generation workflow."""
        logger.info(
            "generation_started",
            assessment_id=command.assessment_id,
            workflow_id=command.workflow_id,
            structured_count=command.structured_count,
            non_structured_count=command.non_structured_count,
            difficulty=command.difficulty_level,
            iteration=command.iteration,
        )

        if (
            command.question_set_id
            and command.validation_result == ValidationResult.FAIL
        ):
            next_iteration = (command.iteration or 1) + 1
            if next_iteration > self._max_iterations:
                raise WorkflowEscalationError(
                    "Max regeneration iterations exceeded",
                    iteration=next_iteration,
                    max_iterations=self._max_iterations,
                )

        # Step 1: GetTopics from Knowledge Service
        logger.info(
            "knowledge_service_get_topics_request",
            workflow_id=command.workflow_id,
        )
        try:
            topics = await self._knowledge_client.get_topics(
                GetTopicsCommand(workflow_id=command.workflow_id)
            )
            topic_count = len(topics)
            logger.info(
                "knowledge_service_get_topics_response",
                workflow_id=command.workflow_id,
                topic_count=topic_count,
                topics=[getattr(t, "name", str(t)) for t in topics[:5]],
            )
        except (StorageTransientError, StoragePermanentError):
            # Preserve typed errors for proper retry semantics
            raise
        except Exception as error:
            logger.error(
                "knowledge_service_get_topics_failed",
                workflow_id=command.workflow_id,
                error=str(error),
            )
            raise RetrievalError(
                "Failed to retrieve topics from Knowledge Service",
                workflow_id=command.workflow_id,
            ) from error

        # Step 2: Select subtopics based on question count and difficulty
        # For regeneration, fetch authoritative config from Submission Service
        generation_config = await self._resolve_generation_config(command)
        total_count = generation_config["total_count"]
        difficulty_level = generation_config["difficulty_level"]
        structured_count = generation_config["structured_count"]
        non_structured_count = generation_config["non_structured_count"]

        subtopics = self._select_subtopics(
            topics,
            total_count=total_count,
            difficulty_level=difficulty_level,
        )
        logger.info(
            "subtopics_selected",
            subtopic_count=len(subtopics),
            subtopics=[s.name for s in subtopics],
        )

        # Step 4: SimilaritySearch per subtopic
        all_chunks: list[str] = []
        all_chunk_ids: list[str] = []  # Track actual chunk IDs for audit
        subtopic_names = []
        for subtopic in subtopics:
            try:
                logger.debug(
                    "knowledge_service_similarity_search_request",
                    subtopic=subtopic.name,
                    query=subtopic.name,
                    kb_type="document",
                    top_k=5,
                )
                chunks = await self._knowledge_client.similarity_search(
                    SimilaritySearchCommand(
                        workflow_id=command.workflow_id,
                        query=subtopic.name,
                        kb_type="document",
                        top_k=5,
                    )
                )
                chunk_count = len(chunks)
                logger.debug(
                    "knowledge_service_similarity_search_response",
                    subtopic=subtopic.name,
                    chunks_retrieved=chunk_count,
                    chunk_ids=[c.chunk_id for c in chunks[:5]],
                )
                for chunk in chunks:
                    all_chunks.append(chunk.content)
                    all_chunk_ids.append(chunk.chunk_id)  # Collect chunk IDs
                subtopic_names.append(subtopic.name)
            except (StorageTransientError, StoragePermanentError):
                # Preserve typed errors - don't swallow gRPC failures
                raise
            except Exception as error:
                logger.warning(
                    "similarity_search_failed",
                    subtopic=subtopic.name,
                    error=str(error),
                )

        logger.info(
            "chunks_aggregation_complete",
            total_chunks=len(all_chunks),
            subtopics_with_chunks=subtopic_names,
        )

        if not all_chunks:
            raise RetrievalError(
                "No chunks retrieved for any subtopic",
                subtopics=[s.name for s in subtopics],
            )

        iteration = command.iteration or 1
        if (
            command.question_set_id
            and command.validation_result == ValidationResult.FAIL
        ):
            # Regeneration flow: increment iteration via gRPC, then generate
            question_set_id = command.question_set_id
            logger.info(
                "increment_iteration_request",
                question_set_id=question_set_id,
                current_iteration=command.iteration,
            )

            try:
                increment_result = await self._submission_client.increment_iteration(
                    IncrementIterationCommand(question_set_id=question_set_id)
                )
                iteration = increment_result.iteration_count
                logger.info(
                    "increment_iteration_response",
                    question_set_id=question_set_id,
                    new_iteration=iteration,
                    status=increment_result.status,
                )
            except (StorageTransientError, StoragePermanentError):
                # Preserve typed errors for proper retry semantics
                raise
            except Exception as error:
                logger.error(
                    "increment_iteration_failed",
                    question_set_id=question_set_id,
                    error=str(error),
                )
                raise RetrievalError(
                    "Failed to increment iteration",
                    question_set_id=question_set_id,
                ) from error

            # Check max iterations against server-returned value
            if iteration > self._max_iterations:
                raise WorkflowEscalationError(
                    "Max regeneration iterations exceeded",
                    iteration=iteration,
                    max_iterations=self._max_iterations,
                )
        else:
            # New generation: create question set
            logger.info(
                "submission_service_create_question_set_request",
                workflow_id=command.workflow_id,
            )
            try:
                record = await self._submission_client.create_question_set(
                    CreateQuestionSetCommand(workflow_id=command.workflow_id)
                )
                question_set_id = record.id
                logger.info(
                    "submission_service_create_question_set_response",
                    question_set_id=question_set_id,
                    workflow_id=command.workflow_id,
                )
            except (StorageTransientError, StoragePermanentError):
                # Preserve typed errors for proper retry semantics
                raise
            except Exception as error:
                logger.error(
                    "submission_service_create_question_set_failed",
                    workflow_id=command.workflow_id,
                    error=str(error),
                )
                raise RetrievalError(
                    "Failed to create question set",
                    workflow_id=command.workflow_id,
                ) from error

        # Create local question set for tracking
        # Use resolved generation config (loaded from Submission Service for regeneration)
        request = GenerationRequest(
            id=command.request_id,
            workflow_id=command.workflow_id,
            correlation_id=command.correlation_id,
            assessment_id=command.assessment_id,
            validation_result=command.validation_result,
            iteration=iteration,
            structured_count=structured_count,
            non_structured_count=non_structured_count,
            difficulty_level=difficulty_level,
            purpose=command.purpose,
        )

        question_set = QuestionSet(
            id=question_set_id,
            assessment_id=request.assessment_id,
            iteration=iteration,
            purpose=request.purpose,
        )
        question_set.mark_in_progress()

        # Step 6: Generate questions via LLM
        context = AssessmentContext(
            assessment_id=request.assessment_id,
            title="",  # Not available from trigger; could be enriched later
            topic_ids=[s.topic_id for s in subtopics],
            chunks=all_chunks,
        )
        logger.info(
            "llm_generation_context_prepared",
            assessment_id=request.assessment_id,
            chunk_count=len(context.chunks),
            topic_ids=context.topic_ids,
        )

        # Include feedback in generation if present (logged for now, TODO: pass to LLM)
        if command.feedback_issues:
            logger.info(
                "regeneration_feedback_received",
                issues=command.feedback_issues,
            )
        assessment_questions: list[Any] = []
        prompt_version = "qa-gen/unknown@v0"

        with (
            self._telemetry.trace(
                "generate_question_batches",
                metadata={"assessment_id": request.assessment_id},
            )
            if self._telemetry is not None
            else nullcontext()
        ):
            if (
                command.question_set_id
                and command.validation_result == ValidationResult.FAIL
            ):
                (
                    assessment_questions,
                    prompt_version,
                ) = await self._generate_regeneration_assessment_questions(
                    context=context,
                    question_set_id=question_set_id,
                    structured_count=structured_count,
                    non_structured_count=non_structured_count,
                    difficulty_level=difficulty_level,
                    correlation_id=request.correlation_id,
                )
            else:
                (
                    assessment_questions,
                    prompt_version,
                ) = await self._generate_initial_assessment_questions(
                    context=context,
                    structured_count=structured_count,
                    non_structured_count=non_structured_count,
                    difficulty_level=difficulty_level,
                    correlation_id=request.correlation_id,
                )

            for idx, question in enumerate(assessment_questions):
                if question.question_type == "structured":
                    final_question = (
                        await self._generate_structured_question_three_prompt(
                            question=question,
                            difficulty_level=request.difficulty_level,
                            question_index=idx,
                        )
                    )
                    if final_question is not None:
                        question_set.add_question(final_question)
                    continue

                if question.question_type == "non_structured":
                    question_set.add_question(
                        self._assessment_question_to_domain_question(
                            question,
                            difficulty_level=request.difficulty_level,
                        )
                    )
                    continue

                logger.warning(
                    "skipping_question_unknown_type",
                    question_id=getattr(question, "question_id", None),
                    question_type=getattr(question, "question_type", None),
                )

        structured_generated = sum(
            1
            for q in question_set.questions
            if q.question_type is QuestionType.STRUCTURED
        )
        non_structured_generated = sum(
            1
            for q in question_set.questions
            if q.question_type is QuestionType.NON_STRUCTURED
        )
        logger.info(
            "all_questions_generated",
            structured_count=structured_generated,
            non_structured_count=non_structured_generated,
            total_questions=len(question_set.questions),
        )

        # Step 7: WriteGeneratedQuestions to Submission Service
        questions_for_submission = [
            Question(
                question_id=q.id.value,
                question_type="structured"
                if q.question_type is QuestionType.STRUCTURED
                else "non_structured",
                content=q.text,
                structured_answer=q.answer.text
                if q.question_type is QuestionType.STRUCTURED
                else "",
                non_structured_model_answer=q.answer.text
                if q.question_type is QuestionType.NON_STRUCTURED
                else "",
                metadata_json=dumps(q.metadata).decode("utf-8"),
                topic_id=q.topic_id or "",
                iteration=iteration,
                sort_order=idx + 1,
            )
            for idx, q in enumerate(question_set.questions)
        ]
        logger.info(
            "submission_service_write_generated_questions_request",
            question_set_id=question_set_id,
            question_count=len(questions_for_submission),
        )

        # Build receipt early for idempotency completion immediately after write
        # Use trace_id from command (propagated from inbound event) first,
        # then fall back to telemetry if available
        trace_id = command.trace_id or (
            self._telemetry.current_trace_id() if self._telemetry is not None else None
        )
        receipt = GenerationReceipt(
            question_set_id=question_set.id,
            assessment_id=question_set.assessment_id,
            structured_generated=structured_generated,
            non_structured_generated=non_structured_generated,
            iteration=question_set.iteration,
            question_count=len(question_set.questions),
            status=GenerationStatus.COMPLETED,
            trace_id=trace_id,
            trace_url=(
                self._telemetry.current_trace_url()
                if self._telemetry is not None
                else None
            ),
        )

        try:
            await self._submission_client.write_generated_questions(
                WriteGeneratedQuestionsCommand(
                    question_set_id=question_set_id,
                    questions=questions_for_submission,
                )
            )
            logger.info(
                "submission_service_write_generated_questions_response",
                question_set_id=question_set_id,
                success=True,
            )
        except (StorageTransientError, StoragePermanentError):
            # Preserve typed errors for proper retry semantics
            raise
        except Exception as error:
            logger.error(
                "submission_service_write_generated_questions_failed",
                question_set_id=question_set_id,
                error=str(error),
                error_type=type(error).__name__,
            )
            raise GenerationError(
                "Failed to write generated questions",
                question_set_id=question_set_id,
            ) from error

        # CRITICAL: Mark idempotency complete IMMEDIATELY after successful write.
        # This prevents duplicate writes on retry if subsequent operations fail.
        # All operations after this point are best-effort (events, audit logs).
        await self._idempotency_store.complete(
            command.request_id,
            receipt=receipt,
            ttl_seconds=86400,
        )
        logger.info(
            "idempotency_marked_complete",
            request_id=command.request_id,
            question_set_id=question_set_id,
        )

        # ------------------------------------------------------------------
        # EVERYTHING BELOW IS BEST-EFFORT ONLY.
        # Failures here MUST NOT propagate out of _generate(), otherwise the
        # caller will mark idempotency as FAILED even though the submission
        # service already persisted the generated questions.
        # ------------------------------------------------------------------
        try:
            question_set.mark_completed()
            await self._question_set_repo.save(question_set)
            logger.info(
                "question_set_marked_completed",
                question_set_id=question_set_id,
                assessment_id=question_set.assessment_id,
            )

            # Step 8: Publish completion + audit events
            # Duplicates here are acceptable
            logger.info(
                "generation_receipt_created",
                question_set_id=receipt.question_set_id,
                question_count=receipt.question_count,
                structured=receipt.structured_generated,
                non_structured=receipt.non_structured_generated,
                status=receipt.status,
                trace_id=trace_id,
            )

            # Publish completion event per spec 5.11
            completion_event = QnAGenerationCompleted(
                event_id=f"evt_{uuid.uuid4().hex[:12]}",
                workflow_id=request.workflow_id,
                assessment_id=request.assessment_id,
                question_set_id=question_set.id,
                structured_generated=structured_generated,
                non_structured_generated=non_structured_generated,
                iteration=question_set.iteration,
                correlation_id=request.correlation_id,
                trace_id=trace_id,
            )
            logger.info(
                "publishing_completion_event",
                event_id=completion_event.event_id,
                workflow_id=completion_event.workflow_id,
                question_set_id=completion_event.question_set_id,
            )
            await self._event_publisher.publish_completion(completion_event)
            logger.info(
                "completion_event_published",
                event_id=completion_event.event_id,
            )

            # Publish audit events (best-effort)
            actual_prompt_version = prompt_version

            try:
                await self._event_publisher.publish_decision_audit(
                    DecisionAuditEvent(
                        workflow_id=command.workflow_id,
                        input_summary={
                            "question_set_id": question_set.id,
                            "iteration": question_set.iteration,
                        },
                        output_summary={
                            "result": "pass",  # Hardcoded - validation happens in evaluator
                            "issues_found": 0,  # Hardcoded - no self-detected issues
                        },
                        reasoning_steps=[
                            f"Retrieved {len(all_chunks)} chunks across {len(subtopics)} subtopics",
                            f"Generated {structured_generated} structured and {non_structured_generated} non-structured questions",
                            f"Used prompt version: {actual_prompt_version}",
                        ],
                        confidence_score=0.88,  # HARDCODED - would need LLM to return this
                        prompt_version=actual_prompt_version,
                        model_id=self._llm_provider.model_id,
                        grounding_sources=all_chunk_ids[:20]
                        if all_chunk_ids
                        else [],  # Dynamic - actual chunk IDs
                    )
                )
            except Exception as e:
                # Audit events are best-effort; log but don't fail
                logger.warning("decision_audit_publish_failed", error=str(e))

            try:
                # Estimate tokens (rough approximation)
                prompt_tokens = len(all_chunks) * 200  # Rough estimate
                completion_tokens = (
                    structured_generated + non_structured_generated
                ) * 100
                await self._event_publisher.publish_token_usage(
                    TokenUsageEvent(
                        workflow_id=command.workflow_id,
                        model_id=self._llm_provider.model_id,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                        estimated_cost_usd=(prompt_tokens + completion_tokens)
                        * 0.000002,  # Rough estimate for gpt-4o-mini
                        prompt_version=actual_prompt_version,
                    )
                )
            except Exception as e:
                # Audit events are best-effort; log but don't fail
                logger.warning("token_usage_publish_failed", error=str(e))
        except Exception as error:
            logger.warning(
                "post_completion_best_effort_failed",
                request_id=command.request_id,
                question_set_id=question_set_id,
                error=str(error),
                error_type=type(error).__name__,
            )

        return receipt

    async def _resolve_generation_config(
        self, command: GenerationCommand
    ) -> dict[str, Any]:
        """Resolve generation configuration for initial or regeneration flow.

        For regeneration (is_regeneration=True), fetches authoritative config
        from Submission Service. For initial generation, uses trigger payload.

        Returns:
            Dict with structured_count, non_structured_count, total_count,
            and difficulty_level.
        """
        is_regeneration = (
            command.question_set_id is not None
            and command.question_set_id != ""
            and command.validation_result == ValidationResult.FAIL
        )

        if is_regeneration:
            logger.info(
                "regeneration_config_fetch",
                assessment_id=command.assessment_id,
                workflow_id=command.workflow_id,
            )
            try:
                config = await self._submission_client.get_assessment_config(
                    GetAssessmentConfigCommand(
                        assessment_id=command.assessment_id,
                        workflow_id=command.workflow_id,
                    )
                )
                structured_count = config.structured_question_count
                non_structured_count = config.non_structured_question_count
                difficulty_level = config.difficulty_level
                logger.info(
                    "regeneration_config_loaded",
                    assessment_id=command.assessment_id,
                    structured_count=structured_count,
                    non_structured_count=non_structured_count,
                    difficulty_level=difficulty_level,
                )
            except StorageTransientError:
                raise  # Retryable errors should be retried
        else:
            # Initial generation: use trigger payload
            structured_count = command.structured_count or 0
            non_structured_count = command.non_structured_count or 0
            difficulty_level = command.difficulty_level

        if is_regeneration and non_structured_count > 0:
            logger.info(
                "regeneration_open_ended_questions_disabled",
                requested_non_structured_count=non_structured_count,
            )
            non_structured_count = 0

        total_count = structured_count + non_structured_count

        return {
            "structured_count": structured_count,
            "non_structured_count": non_structured_count,
            "total_count": total_count,
            "difficulty_level": difficulty_level,
        }

    def _select_subtopics(
        self,
        topics: list[Any],
        total_count: int,
        difficulty_level: DifficultyLevel | None,
    ) -> list[Subtopic]:
        """Select subtopics based on question count and difficulty per spec."""
        # Flatten all subtopics recursively
        all_subtopics: list[Subtopic] = []

        def collect_subtopics(topic: object, parent_name: str = "") -> None:
            """Recursively collect subtopics."""
            if hasattr(topic, "subtopics") and topic.subtopics:
                for st in topic.subtopics:
                    all_subtopics.append(
                        Subtopic(
                            topic_id=getattr(st, "topic_id", ""),
                            name=getattr(st, "name", ""),
                        )
                    )
                    collect_subtopics(st, getattr(st, "name", ""))

        for topic in topics:
            collect_subtopics(topic)

        # Select subtopics to cover the question count
        # Strategy: distribute questions across subtopics
        if not all_subtopics:
            return []

        # Select up to total_count subtopics (or all if fewer)
        selected_count = min(total_count, len(all_subtopics))
        return all_subtopics[:selected_count]

    def _assessment_question_to_domain_question(
        self,
        question: Any,
        *,
        difficulty_level: DifficultyLevel | None,
    ) -> DomainQuestion:
        """Convert a non-structured assessment question into a domain question."""
        question_text = question.question_text or question.content
        answer_text = (
            question.answer_text
            or question.structured_answer
            or question.non_structured_model_answer
        )

        if not question_text or not answer_text:
            raise GenerationError(
                "Assessment Generator returned invalid non-structured question",
                model=self._llm_provider.model_id,
                question_id=getattr(question, "question_id", None),
            )

        metadata_source = question.metadata or {}
        metadata: dict[str, str] = {
            key: str(value) for key, value in metadata_source.items()
        }
        topic_value = metadata_source.get("topic") or metadata_source.get(
            "topic_id", ""
        )
        source_chunk_ids = metadata_source.get("source_chunk_ids", [])
        references = (
            [str(chunk_id) for chunk_id in source_chunk_ids]
            if isinstance(source_chunk_ids, list)
            else []
        )
        source_question_id = getattr(question, "question_id", None)

        return DomainQuestion(
            id=QuestionId(source_question_id)
            if source_question_id
            else QuestionId.generate(),
            text=question_text,
            question_type=QuestionType.NON_STRUCTURED,
            difficulty_level=difficulty_level,
            topic_id=str(topic_value) if topic_value else None,
            metadata=metadata,
            answer=Answer(
                id=AnswerId.generate(),
                text=answer_text,
                explanation=question.explanation,
                references=references,
            ),
        )

    def _score_question_batch(
        self,
        batch: Any,
        *,
        expected_count: int,
        structured_count: int,
        non_structured_count: int,
    ) -> int:
        """Score a candidate batch for hybrid swarm selection."""
        score = 0
        questions = getattr(batch, "questions", []) or []
        actual_count = len(questions)
        score -= abs(expected_count - actual_count) * 20
        score += min(actual_count, expected_count) * 10

        structured_seen = 0
        non_structured_seen = 0
        for question in questions:
            question_type = getattr(question, "question_type", None)
            if question_type == "structured":
                structured_seen += 1
            elif question_type == "non_structured":
                non_structured_seen += 1

            if getattr(question, "question_text", ""):
                score += 4
            if getattr(question, "answer_text", ""):
                score += 3
            if getattr(question, "explanation", None):
                score += 2
            if getattr(question, "references", None):
                score += 2
            if getattr(question, "topic_id", None):
                score += 1

        score -= abs(structured_count - structured_seen) * 8
        score -= abs(non_structured_count - non_structured_seen) * 4
        if (
            structured_seen == structured_count
            and non_structured_seen == non_structured_count
        ):
            score += 15
        return score

    async def _generate_initial_assessment_questions(
        self,
        *,
        context: AssessmentContext,
        structured_count: int,
        non_structured_count: int,
        difficulty_level: DifficultyLevel | None,
        correlation_id: str,
    ) -> tuple[list[Any], str]:
        """Generate initial questions with swarm-based candidate selection."""
        if self._swarm_size > 1 and isinstance(
            self._llm_provider, SwarmCapableLLMProvider
        ):
            from qna_generation_agent.infrastructure.llm.prompt_builder import (
                format_chunks_for_prompt,
            )
            from qna_generation_agent.infrastructure.llm.user_prompt_builder import (
                UserPromptBuilder,
            )

            prompt_version = "item-writer@legacy"
            if self._prompt_provider is not None:
                prompt = await self._prompt_provider.get_item_writer_prompt()
                compiled_prompt = prompt.compile(
                    structured_count=structured_count,
                    non_structured_count=non_structured_count,
                    difficulty=difficulty_level or "medium",
                    topics=", ".join(context.topic_ids),
                    chunks=format_chunks_for_prompt(context.chunks),
                )
                system_msg = compiled_prompt.system_prompt
                user_msg = compiled_prompt.user_prompt
                prompt_version = compiled_prompt.version_string
            else:
                system_msg = DEFAULT_ASSESSMENT_SYSTEM_PROMPT
                user_prompt_builder = UserPromptBuilder()
                user_msg = user_prompt_builder.build_assessment_generator_prompt(
                    structured_count=structured_count,
                    non_structured_count=non_structured_count,
                    difficulty=difficulty_level or "medium",
                    topics=", ".join(context.topic_ids),
                    chunks=format_chunks_for_prompt(context.chunks),
                )

            candidates = await self._llm_provider.generate_swarm_candidates(
                system_message=system_msg,
                user_message=user_msg,
                count=structured_count + non_structured_count,
                difficulty_level=difficulty_level,
                correlation_id=correlation_id,
                swarm_size=self._swarm_size,
            )

            if candidates:
                scored_candidates = [
                    (
                        index,
                        self._score_question_batch(
                            batch,
                            expected_count=structured_count + non_structured_count,
                            structured_count=structured_count,
                            non_structured_count=non_structured_count,
                        ),
                        batch,
                    )
                    for index, batch in enumerate(candidates)
                ]
                best_index, best_score, best_batch = max(
                    scored_candidates, key=lambda item: item[1]
                )
                prompt_version = f"{prompt_version}+hybrid-swarm#{best_index + 1}"
                logger.info(
                    "swarm_candidate_selected",
                    candidate_count=len(candidates),
                    selected_index=best_index,
                    best_score=best_score,
                    prompt_version=prompt_version,
                )
                return best_batch.questions, prompt_version

            logger.warning(
                "swarm_candidate_generation_returned_no_batches",
                swarm_size=self._swarm_size,
            )

        return await self._generate_assessment_questions(
            context=context,
            structured_count=structured_count,
            non_structured_count=non_structured_count,
            difficulty_level=difficulty_level,
            correlation_id=correlation_id,
        )

    async def _generate_regeneration_assessment_questions(
        self,
        *,
        context: AssessmentContext,
        question_set_id: str,
        structured_count: int,
        non_structured_count: int,
        difficulty_level: DifficultyLevel | None,
        correlation_id: str,
    ) -> tuple[list[Any], str]:
        """Replay stored questions for retrigger regeneration."""
        stored_question_set = await self._question_set_repo.get_by_id(question_set_id)
        if stored_question_set is None:
            logger.warning(
                "regeneration_question_set_not_found",
                question_set_id=question_set_id,
            )
            return await self._generate_assessment_questions(
                context=context,
                structured_count=structured_count,
                non_structured_count=non_structured_count,
                difficulty_level=difficulty_level,
                correlation_id=correlation_id,
            )

        if self._prompt_provider is not None:
            options_prompt = (
                await self._prompt_provider.get_options_only_writer_prompt()
            )
            feedback_prompt = await self._prompt_provider.get_feedback_writer_prompt()
            prompt_version = (
                f"{options_prompt.version_string}+{feedback_prompt.version_string}"
            )
        else:
            prompt_version = "retrigger-local-state@v1"

        replay_questions: list[RegenerationQuestionSource] = []
        for question in stored_question_set.questions:
            replay_questions.append(
                RegenerationQuestionSource(
                    question_id=question.id.value,
                    question_type=question.question_type.value,
                    question_text=question.text,
                    answer_text=question.answer.text,
                    metadata=dict(question.metadata),
                )
            )

        logger.info(
            "regeneration_question_set_replayed",
            question_set_id=question_set_id,
            question_count=len(replay_questions),
            prompt_version=prompt_version,
        )
        return replay_questions, prompt_version

    async def _generate_assessment_questions(
        self,
        *,
        context: AssessmentContext,
        structured_count: int,
        non_structured_count: int,
        difficulty_level: DifficultyLevel | None,
        correlation_id: str,
    ) -> tuple[list[Any], str]:
        """Generate a mixed batch of assessment questions."""
        from qna_generation_agent.infrastructure.llm.prompt_builder import (
            AssessmentGeneratorOutputSchema,
            format_chunks_for_prompt,
        )
        from qna_generation_agent.infrastructure.llm.user_prompt_builder import (
            UserPromptBuilder,
        )

        if structured_count == 0 and non_structured_count == 0:
            logger.info("skipping_question_generation", count=0)
            return [], "qa-gen/unknown@v0"

        prompt_version = (
            "item-writer@legacy" if self._prompt_provider is None else "unknown"
        )

        if self._prompt_provider is not None:
            prompt = await self._prompt_provider.get_item_writer_prompt()
            compiled_prompt = prompt.compile(
                structured_count=structured_count,
                non_structured_count=non_structured_count,
                difficulty=difficulty_level or "medium",
                topics=", ".join(context.topic_ids),
                chunks=format_chunks_for_prompt(context.chunks),
            )
            system_msg = compiled_prompt.system_prompt
            user_msg = compiled_prompt.user_prompt
            prompt_version = compiled_prompt.version_string
        else:
            system_msg = DEFAULT_ASSESSMENT_SYSTEM_PROMPT
            user_prompt_builder = UserPromptBuilder()
            user_msg = user_prompt_builder.build_assessment_generator_prompt(
                structured_count=structured_count,
                non_structured_count=non_structured_count,
                difficulty=difficulty_level or "medium",
                topics=", ".join(context.topic_ids),
                chunks=format_chunks_for_prompt(context.chunks),
            )

        logger.info(
            "llm_generate_assessment_started",
            structured_count=structured_count,
            non_structured_count=non_structured_count,
            difficulty=difficulty_level,
            correlation_id=correlation_id,
            model_tier="expensive",
        )
        logger.debug(
            "llm_generate_assessment_prompt_built",
            prompt_length=len(user_msg),
            model_tier="expensive",
        )

        expensive_provider = self._get_provider_for_stage(
            GenerationStage.ASSESSMENT_GENERATOR
        )

        # Wrap the LLM call with an overall stage timeout to prevent
        # indefinite hanging in the generation flow
        try:
            async with asyncio.timeout(GENERATION_STAGE_TIMEOUT_SECONDS):
                result = await expensive_provider.invoke_with_system_and_user(
                    system_message=system_msg,
                    user_message=user_msg,
                    structured_output_model=AssessmentGeneratorOutputSchema,
                    model_tier="expensive",
                )
        except TimeoutError as error:
            logger.error(
                "assessment_generation_stage_timeout",
                timeout_seconds=GENERATION_STAGE_TIMEOUT_SECONDS,
                correlation_id=correlation_id,
            )
            raise LLMTransientError(
                "Assessment generation stage timed out",
                retry_after_seconds=30,
                model=expensive_provider.model_id,
            ) from error

        if not result or not result.questions:
            raise GenerationError(
                "Assessment Generator returned no questions",
                model=expensive_provider.model_id,
            )

        logger.info(
            "llm_generate_assessment_complete",
            generated_count=len(result.questions),
            question_types=[q.question_type for q in result.questions],
        )
        return result.questions, prompt_version

    async def _generate_structured_question_three_prompt(
        self,
        *,
        question: Any,
        difficulty_level: DifficultyLevel | None,
        question_index: int | None = None,
    ) -> DomainQuestion | None:
        """Complete one structured question using the 3-prompt workflow."""
        from qna_generation_agent.infrastructure.llm.prompt_builder import (
            MCQAnswerGeneratorOutputSchema,
            MCQExplanationOutputSchema,
            build_system_prompt,
        )
        from qna_generation_agent.infrastructure.llm.user_prompt_builder import (
            UserPromptBuilder,
        )

        user_prompt_builder = UserPromptBuilder()

        logger.debug(
            "processing_structured_question",
            question_index=question_index,
            question_id=question.question_id,
        )

        question_stem = question.question_text or question.content
        if not question_stem:
            logger.warning(
                "skipping_question_no_stem",
                question_id=question.question_id,
            )
            return None

        logger.debug(
            "prompt_2_mcq_answer_generator_started",
            question_index=question_index,
            model_tier="cheap",
        )

        metadata_source = question.metadata or {}

        if self._prompt_provider is not None:
            answer_prompt = await self._prompt_provider.get_options_only_writer_prompt()
            compiled_answer_prompt = answer_prompt.compile(
                question_text=question_stem,
                topic=str(metadata_source.get("topic", "general grammar")),
                difficulty=difficulty_level or "medium",
                chunk_content="Mixed",
            )
            answer_system_msg = compiled_answer_prompt.system_prompt
            answer_user_msg = compiled_answer_prompt.user_prompt
        else:
            answer_system_msg = build_system_prompt(QuestionType.STRUCTURED)
            answer_user_msg = user_prompt_builder.build_mcq_answer_generator_prompt(
                question_text=question_stem,
                topic=str(metadata_source.get("topic", "general grammar")),
                difficulty=difficulty_level or "medium",
                chunk_content="Mixed",
            )

        cheap_provider = self._get_provider_for_stage(
            GenerationStage.MCQ_ANSWER_GENERATOR
        )

        # Wrap MCQ Answer Generator with stage timeout
        try:
            async with asyncio.timeout(GENERATION_STAGE_TIMEOUT_SECONDS):
                answer_result = await cheap_provider.invoke_with_system_and_user(
                    system_message=answer_system_msg,
                    user_message=answer_user_msg,
                    structured_output_model=MCQAnswerGeneratorOutputSchema,
                    model_tier="cheap",
                )
        except TimeoutError:
            logger.warning(
                "mcq_answer_generator_timeout",
                question_index=question_index,
                question_id=question.question_id,
                timeout_seconds=GENERATION_STAGE_TIMEOUT_SECONDS,
            )
            return None

        if not answer_result:
            logger.warning(
                "mcq_answer_generator_failed_for_question",
                question_index=question_index,
                question_id=question.question_id,
            )
            return None

        logger.debug(
            "prompt_2_mcq_answer_generator_complete",
            question_index=question_index,
            correct_letter=answer_result.correct_answer.option_letter,
            distractor_count=len(answer_result.distractors),
        )

        logger.debug(
            "prompt_3_mcq_explanation_generator_started",
            question_index=question_index,
            model_tier="cheap",
        )

        options_dict: dict[str, str] = {}
        correct_letter = answer_result.correct_answer.option_letter
        options_dict[correct_letter] = answer_result.correct_answer.option_text
        for distractor in answer_result.distractors:
            options_dict[distractor.option_letter] = distractor.option_text

        if self._prompt_provider is not None:
            explanation_prompt = (
                await self._prompt_provider.get_feedback_writer_prompt()
            )
            compiled_explanation_prompt = explanation_prompt.compile(
                question_text=answer_result.question_stem,
                topic="mixed L1 students",
                correct_answer=answer_result.correct_answer.option_letter,
                option_a=options_dict.get("A", ""),
                option_b=options_dict.get("B", ""),
                option_c=options_dict.get("C", ""),
                option_d=options_dict.get("D", ""),
                chunk_content="mixed L1 students",
            )
            explanation_system_msg = compiled_explanation_prompt.system_prompt
            explanation_user_msg = compiled_explanation_prompt.user_prompt
        else:
            explanation_system_msg = build_system_prompt(QuestionType.STRUCTURED)
            explanation_user_msg = (
                user_prompt_builder.build_mcq_explanation_generator_prompt(
                    question_text=answer_result.question_stem,
                    topic="mixed L1 students",
                    correct_answer=answer_result.correct_answer.option_letter,
                    option_a=options_dict.get("A", ""),
                    option_b=options_dict.get("B", ""),
                    option_c=options_dict.get("C", ""),
                    option_d=options_dict.get("D", ""),
                    chunk_content="mixed L1 students",
                )
            )

        explanation_provider = self._get_provider_for_stage(
            GenerationStage.MCQ_EXPLANATION_GENERATOR
        )

        # Wrap MCQ Explanation Generator with stage timeout
        try:
            async with asyncio.timeout(GENERATION_STAGE_TIMEOUT_SECONDS):
                explanation_result = (
                    await explanation_provider.invoke_with_system_and_user(
                        system_message=explanation_system_msg,
                        user_message=explanation_user_msg,
                        structured_output_model=MCQExplanationOutputSchema,
                        model_tier="cheap",
                    )
                )
        except TimeoutError:
            logger.warning(
                "mcq_explanation_generator_timeout",
                question_index=question_index,
                question_id=question.question_id,
                timeout_seconds=GENERATION_STAGE_TIMEOUT_SECONDS,
            )
            explanation_result = None

        if explanation_result:
            logger.debug(
                "prompt_3_mcq_explanation_generator_complete",
                question_index=question_index,
                cefr_level=explanation_result.cefr_level,
                option_count=len(explanation_result.option_explanations),
            )
        else:
            logger.warning(
                "mcq_explanation_generator_failed_for_question",
                question_index=question_index,
            )

        combined_answer_parts = [
            f"Question: {answer_result.question_stem}",
            "",
            "Options:",
        ]
        for letter in sorted(options_dict.keys()):
            text = options_dict[letter]
            marker = " ✓ CORRECT" if letter == correct_letter else ""
            combined_answer_parts.append(f"  {letter}) {text}{marker}")
        combined_answer_parts.append("")
        combined_answer_parts.append(f"Correct Answer: {correct_letter}")
        combined_answer = "\n".join(combined_answer_parts)

        explanation_parts: list[str] = []
        if explanation_result:
            explanation_parts.append(
                f"Analysis: {explanation_result.question_analysis}"
            )
            explanation_parts.append(f"CEFR Level: {explanation_result.cefr_level}")
            explanation_parts.append(f"Teaching Tip: {explanation_result.teaching_tip}")
            for opt_exp in explanation_result.option_explanations:
                explanation_parts.append(
                    f"{opt_exp.option_letter}: {opt_exp.explanation}"
                )
        else:
            explanation_parts.append(answer_result.correct_answer.explanation)
            for distractor in answer_result.distractors:
                explanation_parts.append(
                    f"{distractor.option_letter}: {distractor.explanation}"
                )

        combined_explanation = "\n".join(explanation_parts)

        topic_value = metadata_source.get("topic", "")
        topic_id_str = str(topic_value) if topic_value else None

        metadata_dict: dict[str, str] = {
            "original_question_id": question.question_id,
            "grammar_point": answer_result.grammar_point_tested,
            "l1_considerations": ", ".join(answer_result.l1_considerations),
        }
        for key, value in metadata_source.items():
            if key not in metadata_dict and isinstance(value, (str, int, float, bool)):
                metadata_dict[key] = str(value)

        source_question_id = getattr(question, "question_id", None)

        return DomainQuestion(
            id=QuestionId(source_question_id)
            if source_question_id
            else QuestionId.generate(),
            text=answer_result.question_stem,
            question_type=QuestionType.STRUCTURED,
            difficulty_level=difficulty_level,
            topic_id=topic_id_str,
            metadata=metadata_dict,
            answer=Answer(
                id=AnswerId.generate(),
                text=combined_answer,
                explanation=combined_explanation,
                references=[],
            ),
        )
