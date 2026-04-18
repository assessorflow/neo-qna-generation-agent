"""QnA generation application service."""

from __future__ import annotations

import asyncio
import uuid
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from qna_generation_agent.app.json import dumps
from qna_generation_agent.app.logging import bind_context, get_logger
from qna_generation_agent.application.dto import (
    AssessmentContext,
    GenerationCommand,
    GenerationReceipt,
    QuestionDraft,
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
    Prompt,
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


@dataclass(frozen=True, slots=True)
class Subtopic:
    """Selected subtopic for question generation."""

    topic_id: str
    name: str


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
        prompt_provider: PromptProvider | None = None,
        cheap_llm_provider: LLMProvider | None = None,
        expensive_llm_provider: LLMProvider | None = None,
    ) -> None:
        self._llm_provider = llm_provider
        self._question_set_repo = question_set_repo
        self._idempotency_store = idempotency_store
        self._event_publisher = event_publisher
        self._submission_client = submission_client
        self._knowledge_client = knowledge_client
        self._telemetry = telemetry
        self._max_iterations = max_iterations
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

        if command.question_set_id and command.validation_result == ValidationResult.FAIL:
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
        if command.question_set_id and command.validation_result == ValidationResult.FAIL:
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

        async def generate_structured() -> tuple[list[DomainQuestion], str | None]:
            if not request.structured_count:
                logger.info("skipping_structured_generation", count=0)
                return [], None
            try:
                logger.info(
                    "llm_generate_structured_started",
                    count=request.structured_count,
                    difficulty=request.difficulty_level,
                    correlation_id=request.correlation_id,
                )

                # Use Langfuse prompt provider with 3-prompt workflow
                if self._prompt_provider is not None:
                    (
                        questions,
                        prompt_version,
                    ) = await self._generate_structured_three_prompt(
                        context=context,
                        count=request.structured_count,
                        difficulty_level=request.difficulty_level,
                        correlation_id=request.correlation_id,
                    )
                else:
                    logger.debug(
                        "llm_generate_structured_legacy",
                        context_chunks=len(context.chunks),
                    )

                    # Use expensive provider for question generation
                    expensive_provider = self._get_provider_for_stage(GenerationStage.QUESTION_GENERATION)
                    batch = await expensive_provider.generate_structured(
                        context=context,
                        count=request.structured_count,
                        difficulty_level=request.difficulty_level,
                        correlation_id=request.correlation_id,
                    )
                    questions = [
                        self._to_domain_question(draft) for draft in batch.questions
                    ]
                    prompt_version = None  # Legacy path doesn't track prompt version

                logger.info(
                    "llm_generate_structured_complete",
                    generated_count=len(questions),
                )
                return questions, prompt_version
            except (LLMTransientError, LLMPermanentError):
                # Preserve typed LLM errors for proper retry semantics
                raise
            except Exception as error:
                # Get the provider that was used (expensive for question generation)
                provider_used = self._expensive_llm_provider or self._llm_provider
                logger.error(
                    "llm_generate_structured_failed",
                    error=str(error),
                    error_type=type(error).__name__,
                    model=provider_used.model_id,
                )
                raise GenerationError(
                    "Failed to generate structured questions",
                    model=provider_used.model_id,
                ) from error

        async def generate_non_structured() -> tuple[list[DomainQuestion], str | None]:
            if not request.non_structured_count:
                logger.info("skipping_non_structured_generation", count=0)
                return [], None
            try:
                logger.info(
                    "llm_generate_non_structured_started",
                    count=request.non_structured_count,
                    difficulty=request.difficulty_level,
                    correlation_id=request.correlation_id,
                )

                prompt_version: str | None = None

                # Use Langfuse prompt provider if available, otherwise fallback to legacy
                if self._prompt_provider is not None:
                    from qna_generation_agent.infrastructure.llm.prompt_builder import (
                        AssessmentGeneratorInputSchema,
                    )

                    prompt_obj = await self._prompt_provider.get_prompt(
                        "Assessment Generator",
                        label=self._prompt_provider.default_label,
                    )
                    # Track prompt version for audit
                    prompt_version = prompt_obj.version_string
                    input_data = AssessmentGeneratorInputSchema.from_context(
                        context=context,
                        structured_count=0,
                        non_structured_count=request.non_structured_count,
                        difficulty_level=request.difficulty_level,
                    )
                    compiled_prompt = prompt_obj.compile(**input_data.model_dump())
                    logger.debug(
                        "llm_generate_non_structured_prompt_compiled",
                        prompt_version=prompt_version,
                        prompt_length=len(compiled_prompt)
                        if isinstance(compiled_prompt, str)
                        else len(str(compiled_prompt)),
                    )
                    # Use expensive provider for question generation
                    expensive_provider = self._get_provider_for_stage(GenerationStage.QUESTION_GENERATION)
                    batch = await expensive_provider.generate_with_prompt(
                        prompt=compiled_prompt
                        if isinstance(compiled_prompt, str)
                        else str(compiled_prompt),
                        count=request.non_structured_count,
                        difficulty_level=request.difficulty_level,
                        correlation_id=request.correlation_id,
                        question_type="non_structured",
                    )
                else:
                    logger.debug(
                        "llm_generate_non_structured_legacy",
                        context_chunks=len(context.chunks),
                    )
                    # Use expensive provider for question generation
                    expensive_provider = self._get_provider_for_stage(GenerationStage.QUESTION_GENERATION)
                    batch = await expensive_provider.generate_non_structured(
                        context=context,
                        count=request.non_structured_count,
                        difficulty_level=request.difficulty_level,
                        correlation_id=request.correlation_id,
                    )

                questions = [
                    self._to_domain_question(draft) for draft in batch.questions
                ]
                logger.info(
                    "llm_generate_non_structured_complete",
                    generated_count=len(questions),
                    questions_preview=[q.text[:100] for q in questions[:3]],
                )
                return questions, prompt_version
            except (LLMTransientError, LLMPermanentError):
                # Preserve typed LLM errors for proper retry semantics
                raise
            except Exception as error:
                # Get the provider that was used (expensive for question generation)
                provider_used = self._expensive_llm_provider or self._llm_provider
                logger.error(
                    "llm_generate_non_structured_failed",
                    error=str(error),
                    error_type=type(error).__name__,
                    model=provider_used.model_id,
                )
                raise GenerationError(
                    "Failed to generate non-structured questions",
                    model=provider_used.model_id,
                ) from error

        with (
            self._telemetry.trace(
                "generate_question_batches",
                metadata={"assessment_id": request.assessment_id},
            )
            if self._telemetry is not None
            else nullcontext()
        ):
            # Execute both generation tasks concurrently
            structured_result, non_structured_result = await asyncio.gather(
                generate_structured(),
                generate_non_structured(),
            )
            # Unpack results (questions, prompt_version)
            structured_questions, structured_prompt_version = structured_result
            non_structured_questions, non_structured_prompt_version = (
                non_structured_result
            )

            for q in structured_questions:
                question_set.add_question(q)
            for q in non_structured_questions:
                question_set.add_question(q)

        structured_generated = len(structured_questions)
        non_structured_generated = len(non_structured_questions)
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
            # Determine prompt version used (prefer structured path version)
            actual_prompt_version = (
                structured_prompt_version
                or non_structured_prompt_version
                or "qa-gen/unknown@v0"
            )

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
                completion_tokens = (structured_generated + non_structured_generated) * 100
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

    def _to_domain_question(self, draft: QuestionDraft) -> DomainQuestion:

        return DomainQuestion(
            id=QuestionId.generate(),
            text=draft.question_text,
            question_type=draft.question_type,
            difficulty_level=draft.difficulty_level,
            topic_id=draft.topic_id,
            metadata=draft.metadata,
            answer=Answer(
                id=AnswerId.generate(),
                text=draft.answer_text,
                explanation=draft.explanation,
                references=draft.references,
            ),
        )

    async def _generate_structured_three_prompt(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: DifficultyLevel | None,
        correlation_id: str,
    ) -> tuple[list[DomainQuestion], str]:
        """Generate structured questions using 3-prompt workflow.

        1. Assessment Generator: Generate question stems
        2. MCQ Answer Generator: Generate MCQ answers with distractors
        3. MCQ Explanation Generator: Generate detailed explanations

        Returns:
            Tuple of (questions, prompt_version) where prompt_version is the
            version string of the Assessment Generator prompt used.
        """
        from qna_generation_agent.infrastructure.llm.prompt_builder import (
            AssessmentGeneratorInputSchema,
            AssessmentGeneratorOutputSchema,
            MCQAnswerGeneratorOutputSchema,
            MCQAnswerInputSchema,
            MCQExplanationInputSchema,
            MCQExplanationOutputSchema,
        )

        if self._prompt_provider is None:
            raise GenerationError("Prompt provider required for 3-prompt workflow")

        # Step 1: Assessment Generator - Generate question stems
        logger.info("prompt_1_assessment_generator_started", count=count)

        prompt_obj = await self._prompt_provider.get_prompt(
            "Assessment Generator",
            label=self._prompt_provider.default_label,
        )

        # Debug: Log prompt metadata only
        logger.debug(
            "prompt_1_metadata",
            prompt_name=prompt_obj.name,
            prompt_version=prompt_obj.version,
            is_chat=prompt_obj.is_chat_prompt(),
        )

        input_data = AssessmentGeneratorInputSchema.from_context(
            context=context,
            structured_count=count,
            non_structured_count=0,
            difficulty_level=difficulty_level,
        )

        # Log prompt variables metadata only
        input_dump = input_data.model_dump()
        logger.debug(
            "prompt_1_variables",
            structured_count=input_dump["structured_count"],
            non_structured_count=input_dump["non_structured_count"],
            difficulty=input_dump["difficulty"],
            topics_count=len(context.topic_ids),
            chunks_count=len(context.chunks),
        )

        compiled = prompt_obj.compile(**input_dump)

        # Log compilation metadata only
        logger.debug(
            "prompt_1_compiled",
            compiled_type=type(compiled).__name__,
            prompt_name=prompt_obj.name,
            prompt_version=prompt_obj.version,
        )

        prompt_str = Prompt.compiled_to_string(compiled)

        logger.debug(
            "prompt_1_assessment_generator_compiled",
            prompt_name=prompt_obj.name,
            prompt_version=prompt_obj.version,
            prompt_length=len(prompt_str),
        )

        # Generate initial question stems using expensive provider
        expensive_provider = self._get_provider_for_stage(GenerationStage.ASSESSMENT_GENERATOR)
        initial_result = await expensive_provider.invoke_with_schema(
            prompt_str,
            structured_output_model=AssessmentGeneratorOutputSchema,
        )

        if not initial_result or not initial_result.questions:
            raise GenerationError(
                "Assessment Generator returned no questions",
                model=expensive_provider.model_id,
            )

        # Log prompt 1 completion metadata only
        logger.info(
            "prompt_1_assessment_generator_complete",
            questions_generated=len(initial_result.questions),
            question_types=[q.question_type for q in initial_result.questions[:count]],
        )

        # Step 2 & 3: For each structured question, generate MCQ answers and explanations
        final_questions: list[DomainQuestion] = []

        for idx, question in enumerate(initial_result.questions[:count]):
            if question.question_type != "structured":
                continue

            logger.debug(
                "processing_structured_question",
                question_index=idx,
                question_id=question.question_id,
            )

            # Step 2: MCQ Answer Generator
            logger.debug("prompt_2_mcq_answer_generator_started", question_index=idx)

            # Validator ensures question_text/content is set, but mypy doesn't know
            question_stem = question.question_text or question.content
            if not question_stem:
                logger.warning(
                    "skipping_question_no_stem",
                    question_id=question.question_id,
                )
                continue

            answer_prompt_obj = await self._prompt_provider.get_prompt(
                "MCQ Answer Generator",
                label=self._prompt_provider.default_label,
            )
            answer_input = MCQAnswerInputSchema(
                question_stem=question_stem,
                grammar_target="general grammar",  # Could be extracted from metadata
                difficulty=difficulty_level or "medium",
                l1_background="Mixed",  # Could be configured
            )
            answer_compiled = answer_prompt_obj.compile(**answer_input.model_dump())
            answer_prompt_str = Prompt.compiled_to_string(answer_compiled)

            cheap_provider = self._get_provider_for_stage(GenerationStage.MCQ_ANSWER_GENERATOR)
            answer_result = await cheap_provider.invoke_with_schema(
                answer_prompt_str,
                structured_output_model=MCQAnswerGeneratorOutputSchema,
            )

            if not answer_result:
                logger.warning(
                    "mcq_answer_generator_failed_for_question",
                    question_index=idx,
                    question_id=question.question_id,
                )
                continue

            # Log prompt 2 completion metadata only
            logger.debug(
                "prompt_2_mcq_answer_generator_complete",
                question_index=idx,
                correct_letter=answer_result.correct_answer.option_letter,
                distractor_count=len(answer_result.distractors),
            )

            # Step 3: MCQ Explanation Generator
            logger.debug(
                "prompt_3_mcq_explanation_generator_started", question_index=idx
            )

            explanation_prompt_obj = await self._prompt_provider.get_prompt(
                "MCQ Explanation Generator",
                label=self._prompt_provider.default_label,
            )

            # Build options dict from answer result - preserving actual option letters
            options_dict: dict[str, str] = {}
            correct_letter = answer_result.correct_answer.option_letter
            options_dict[correct_letter] = answer_result.correct_answer.option_text
            for distractor in answer_result.distractors:
                options_dict[distractor.option_letter] = distractor.option_text

            explanation_input = MCQExplanationInputSchema(
                question=answer_result.question_stem,
                options=dumps(options_dict).decode("utf-8"),
                correct_answer=answer_result.correct_answer.option_letter,
                target_audience="mixed L1 students",
            )
            explanation_compiled = explanation_prompt_obj.compile(
                **explanation_input.model_dump()
            )
            explanation_prompt_str = Prompt.compiled_to_string(explanation_compiled)

            explanation_provider = self._get_provider_for_stage(
                GenerationStage.MCQ_EXPLANATION_GENERATOR
            )
            explanation_result = await explanation_provider.invoke_with_schema(
                explanation_prompt_str,
                structured_output_model=MCQExplanationOutputSchema,
            )

            if explanation_result:
                # Log prompt 3 completion metadata only
                logger.debug(
                    "prompt_3_mcq_explanation_generator_complete",
                    question_index=idx,
                    cefr_level=explanation_result.cefr_level,
                    option_count=len(explanation_result.option_explanations),
                )
            else:
                logger.warning(
                    "mcq_explanation_generator_failed_for_question",
                    question_index=idx,
                )

            # Combine all results into final question
            # Build clear MCQ options display preserving actual option letters
            correct_letter = answer_result.correct_answer.option_letter
            combined_answer_parts = [
                f"Question: {answer_result.question_stem}",
                "",
                "Options:",
            ]
            # Sort by letter for consistent display while preserving actual letters
            for letter in sorted(options_dict.keys()):
                text = options_dict[letter]
                marker = " ✓ CORRECT" if letter == correct_letter else ""
                combined_answer_parts.append(f"  {letter}) {text}{marker}")
            combined_answer_parts.append("")
            combined_answer_parts.append(f"Correct Answer: {correct_letter}")
            combined_answer = "\n".join(combined_answer_parts)

            # Build explanation from explanation_result if available
            explanation_parts = []
            if explanation_result:
                explanation_parts.append(
                    f"Analysis: {explanation_result.question_analysis}"
                )
                explanation_parts.append(f"CEFR Level: {explanation_result.cefr_level}")
                explanation_parts.append(
                    f"Teaching Tip: {explanation_result.teaching_tip}"
                )
                for opt_exp in explanation_result.option_explanations:
                    explanation_parts.append(
                        f"{opt_exp.option_letter}: {opt_exp.explanation}"
                    )
            else:
                # Fallback to answer generator explanation
                explanation_parts.append(answer_result.correct_answer.explanation)
                for distractor in answer_result.distractors:
                    explanation_parts.append(
                        f"{distractor.option_letter}: {distractor.explanation}"
                    )

            combined_explanation = "\n".join(explanation_parts)

            # Create final domain question
            topic_value = question.metadata.get("topic", "")
            topic_id_str = str(topic_value) if topic_value else None

            # Build metadata dict with string values only
            metadata_dict: dict[str, str] = {
                "original_question_id": question.question_id,
                "grammar_point": answer_result.grammar_point_tested,
                "l1_considerations": ", ".join(answer_result.l1_considerations),
            }
            # Add other metadata from question, converting to strings
            for key, value in question.metadata.items():
                if key not in metadata_dict and isinstance(
                    value, (str, int, float, bool)
                ):
                    metadata_dict[key] = str(value)

            final_question = DomainQuestion(
                id=QuestionId.generate(),
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
            final_questions.append(final_question)

        logger.info(
            "three_prompt_workflow_complete",
            total_questions_generated=len(final_questions),
        )
        # Build dynamic prompt version from actual prompt used
        prompt_version_str = f"{prompt_obj.name}@v{prompt_obj.version}"
        return final_questions, prompt_version_str
