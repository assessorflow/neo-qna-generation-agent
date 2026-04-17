"""Application bootstrap and dependency graph."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qna_generation_agent.infrastructure.grpc.knowledge_client import (
        GrpcKnowledgeClient,
    )

from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.app.settings import Settings
from qna_generation_agent.application.commands import HandleGenerationTrigger
from qna_generation_agent.application.ports.idempotency import IdempotencyStore
from qna_generation_agent.application.ports.llm import LLMProvider
from qna_generation_agent.application.ports.prompt_provider import PromptProvider
from qna_generation_agent.application.ports.publisher import EventPublisher
from qna_generation_agent.application.ports.repository import QuestionSetRepository
from qna_generation_agent.application.ports.submission_client import SubmissionClient
from qna_generation_agent.application.ports.telemetry import TelemetryPort
from qna_generation_agent.application.services.generate_qna import GenerateQnAService
from qna_generation_agent.errors import ConfigurationError
from qna_generation_agent.infrastructure.grpc.knowledge_client import (
    GrpcKnowledgeClient,
)
from qna_generation_agent.infrastructure.grpc.submission_client import (
    GrpcSubmissionClient,
)
from qna_generation_agent.infrastructure.llm.langfuse_prompt_provider import (
    LangfusePromptProvider,
)
from qna_generation_agent.infrastructure.llm.strands_provider import StrandsLLMProvider
from qna_generation_agent.infrastructure.llm.workflow_provider import (
    WorkflowLLMProvider,
)
from qna_generation_agent.infrastructure.messaging.pubsub_publisher import (
    NullEventPublisher,
    PubSubCompletionPublisher,
)
from qna_generation_agent.infrastructure.messaging.pubsub_subscriber import (
    PubSubSubscriptionWorker,
    SubscriptionConfig,
)
from qna_generation_agent.infrastructure.persistence.in_memory import (
    InMemoryIdempotencyStore,
    InMemoryQuestionSetRepository,
)
from qna_generation_agent.infrastructure.telemetry.langfuse_client import (
    LangfuseTelemetry,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class ApplicationContainer:
    """Factory pattern: process-scoped dependency graph."""

    settings: Settings
    llm_provider: LLMProvider | None
    question_set_repo: QuestionSetRepository
    idempotency_store: IdempotencyStore
    event_publisher: EventPublisher
    telemetry: TelemetryPort | None
    subscriber: PubSubSubscriptionWorker | None
    submission_client: SubmissionClient
    knowledge_client: GrpcKnowledgeClient
    prompt_provider: PromptProvider | None = None
    generate_service: GenerateQnAService | None = None

    async def startup(self) -> None:
        """Initialize external clients."""
        if self.subscriber is not None:
            await self.subscriber.start()
        logger.info(
            "container_started",
            telemetry=bool(self.telemetry),
            subscriber=bool(self.subscriber),
            submission_client=bool(self.submission_client),
            prompt_provider=bool(self.prompt_provider),
        )

    async def shutdown(self) -> None:
        """Dispose external clients with isolated cleanup."""
        clients: list[tuple[str, Callable[[], Awaitable[None]] | None]] = [
            ("subscriber", self.subscriber.shutdown if self.subscriber else None),
            ("telemetry", self.telemetry.shutdown if self.telemetry else None),
            (
                "submission_client",
                self.submission_client.close if self.submission_client else None,
            ),
            (
                "knowledge_client",
                self.knowledge_client.close if self.knowledge_client else None,
            ),
            ("event_publisher", self.event_publisher.close),
            (
                "prompt_provider",
                self.prompt_provider.shutdown if self.prompt_provider else None,
            ),
        ]
        for name, close_fn in clients:
            if close_fn:
                try:
                    await close_fn()
                except Exception:
                    logger.exception("shutdown_failed", client=name)
        logger.info("container_stopped")


def _build_submission_client(settings: Settings) -> GrpcSubmissionClient:
    return GrpcSubmissionClient(
        target=settings.submission_service_url,
        timeout_seconds=30.0,
        tls_enabled=settings.grpc_tls_enabled,
        tls_cert_path=settings.grpc_tls_cert_path,
    )


def _build_knowledge_client(settings: Settings) -> GrpcKnowledgeClient:
    return GrpcKnowledgeClient(
        target=settings.knowledge_service_url,
        timeout_seconds=30.0,
        tls_enabled=settings.grpc_tls_enabled,
        tls_cert_path=settings.grpc_tls_cert_path,
    )


def _build_telemetry(settings: Settings) -> TelemetryPort | None:
    if not settings.langfuse_enabled:
        return None

    return LangfuseTelemetry(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_base_url,
        environment=settings.environment.value,
        release=settings.release,
    )


def _build_llm(
    settings: Settings,
    prompt_provider: PromptProvider | None = None,
) -> LLMProvider:
    """Build LLM provider based on configuration.

    Supports two modes:
    - Standard: StrandsLLMProvider with sequential 3-prompt workflow
    - Workflow: WorkflowLLMProvider with parallel multi-level workflow

    Args:
        settings: Application settings
        prompt_provider: Optional prompt provider for Langfuse integration.
            Required when llm_workflow_mode is True.
    """
    if settings.llm_workflow_mode:
        if prompt_provider is None:
            logger.warning(
                "workflow_mode_requires_langfuse",
                reason="PromptProvider required for WorkflowLLMProvider but Langfuse not configured",
                action="falling_back_to_standard_llm_provider",
            )
        else:
            logger.info("using_workflow_llm_provider")
            return WorkflowLLMProvider(
                model_provider="openai",
                model_id=settings.model_id,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout_seconds=settings.llm_timeout_seconds,
                prompt_provider=prompt_provider,
            )

    logger.info("using_standard_llm_provider")
    return StrandsLLMProvider(
        model_provider="openai",
        model_id=settings.model_id,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def _build_prompt_provider(settings: Settings) -> PromptProvider | None:
    """Build Langfuse prompt provider if credentials are available."""
    if not settings.langfuse_enabled:
        return None

    return LangfusePromptProvider(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_base_url,
        environment=settings.environment.value,
        release=settings.release,
    )


def _build_subscriber(
    settings: Settings,
    generate_service: GenerateQnAService,
) -> PubSubSubscriptionWorker | None:
    if not settings.pubsub_enabled or not settings.worker_ready:
        return None

    handler = HandleGenerationTrigger(generate_service)
    return PubSubSubscriptionWorker(
        SubscriptionConfig(
            project_id=settings.pubsub_project_id or "",
            subscription_id=settings.pubsub_subscription_trigger or "",
            max_messages=settings.pubsub_max_workers,
        ),
        handler.handle,
    )


def _build_container(settings: Settings) -> ApplicationContainer:
    # Dependency injection pattern: build all adapters in one place.
    # Build prompt_provider first (needed by WorkflowLLMProvider)
    telemetry = _build_telemetry(settings)
    prompt_provider = _build_prompt_provider(settings)
    llm_provider = _build_llm(settings, prompt_provider=prompt_provider)
    submission_client = _build_submission_client(settings)

    # Always use in-memory adapters
    question_set_repo: QuestionSetRepository = InMemoryQuestionSetRepository()
    idempotency_store: IdempotencyStore = InMemoryIdempotencyStore()

    if (
        settings.pubsub_enabled
        and settings.pubsub_project_id
        and settings.pubsub_topic_complete
    ):
        event_publisher: EventPublisher = PubSubCompletionPublisher(
            project_id=settings.pubsub_project_id,
            topic_id=settings.pubsub_topic_complete,
            decision_audit_topic_id=settings.pubsub_topic_decision_audit,
            token_usage_topic_id=settings.pubsub_topic_token_usage,
        )
    else:
        event_publisher = NullEventPublisher()

    # Build gRPC clients before subscriber (subscriber needs them)
    knowledge_client = _build_knowledge_client(settings)

    # Create the generation service (used by both subscriber and HTTP tests)
    generate_service = GenerateQnAService(
        llm_provider=llm_provider,
        question_set_repo=question_set_repo,
        idempotency_store=idempotency_store,
        event_publisher=event_publisher,
        submission_client=submission_client,
        knowledge_client=knowledge_client,
        telemetry=telemetry,
        max_iterations=settings.qa_gen_max_iterations,
        prompt_provider=prompt_provider,
    )

    subscriber: PubSubSubscriptionWorker | None = None
    if llm_provider is not None and settings.pubsub_enabled and settings.worker_ready:
        subscriber = _build_subscriber(
            settings=settings,
            generate_service=generate_service,
        )

    return ApplicationContainer(
        settings=settings,
        llm_provider=llm_provider,
        question_set_repo=question_set_repo,
        idempotency_store=idempotency_store,
        event_publisher=event_publisher,
        telemetry=telemetry,
        subscriber=subscriber,
        submission_client=submission_client,
        knowledge_client=knowledge_client,
        prompt_provider=prompt_provider,
        generate_service=generate_service,
    )


def bootstrap_serve(settings: Settings) -> ApplicationContainer:
    """Bootstrap dependencies for the serve process.

    Args:
        settings: Pre-loaded and validated settings. Logging must be
            configured by the caller before invoking this function.
    """
    settings.validate_settings()
    container = _build_container(settings)

    if container.llm_provider is None:
        raise ConfigurationError("Serve runtime requires an LLM provider")

    if container.subscriber is None:
        raise ConfigurationError("Serve runtime requires a configured subscriber")

    return container
