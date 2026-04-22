"""Unit tests for runtime helpers and bootstrap wiring."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import BaseModel

from qna_generation_agent.app import bootstrap as bootstrap_module
from qna_generation_agent.app.bootstrap import ApplicationContainer
from qna_generation_agent.app.json import dumps, loads
from qna_generation_agent.app.settings import LogLevel, RuntimeEnvironment, Settings
from qna_generation_agent.application.commands import HandleGenerationTrigger
from qna_generation_agent.application.dto import GenerationReceipt
from qna_generation_agent.application.ports.llm import LLMProvider
from qna_generation_agent.application.ports.publisher import EventPublisher
from qna_generation_agent.application.ports.telemetry import Span, TelemetryPort
from qna_generation_agent.domain.enums import DifficultyLevel, Purpose, QuestionType
from qna_generation_agent.domain.events import (
    QnAGenerationCompleted,
    QnAGenerationTriggered,
)
from qna_generation_agent.errors import AppError, ConfigurationError
from qna_generation_agent.infrastructure.messaging.pubsub_publisher import (
    NullEventPublisher,
)
from qna_generation_agent.infrastructure.persistence.in_memory import (
    InMemoryIdempotencyStore,
    InMemoryQuestionSetRepository,
)


@dataclass(slots=True)
class ExamplePayload:
    """Example dataclass serialized by the JSON helpers."""

    timestamp: datetime
    mode: QuestionType


class ExampleModel(BaseModel):
    """Example pydantic model serialized by the JSON helpers."""

    value: str


class FakeSpan(Span):
    """Minimal span used by runtime tests."""

    def set_attribute(self, key: str, value: Any) -> None:
        del key, value

    def record_error(self, error: BaseException) -> None:
        del error


class FakeTelemetry(TelemetryPort):
    """Simple telemetry stub."""

    def __init__(self) -> None:
        self.shutdown_called = False

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> Iterator[Span]:
        del name, metadata
        yield FakeSpan()

    @contextmanager
    def propagate(
        self,
        *,
        trace_name: str,
        correlation_id: str,
        workflow_id: str,
        metadata: dict[str, str] | None = None,
    ) -> Iterator[None]:
        del trace_name, correlation_id, workflow_id, metadata
        yield None

    def current_trace_id(self) -> str | None:
        return "trace-runtime"

    def current_trace_url(self) -> str | None:
        return "https://langfuse.example/trace-runtime"

    async def shutdown(self) -> None:
        self.shutdown_called = True


class FakePublisher(EventPublisher):
    """Publisher stub that records lifecycle calls."""

    def __init__(self) -> None:
        self.closed = False

    async def publish_completion(self, event: QnAGenerationCompleted) -> None:
        del event

    async def publish_decision_audit(self, event: Any) -> None:
        del event

    async def publish_token_usage(self, event: Any) -> None:
        del event

    async def close(self) -> None:
        self.closed = True


class FakeLLMProvider(LLMProvider):
    """LLM stub used only for bootstrap selection tests."""

    @property
    def model_id(self) -> str:
        return "fake-model"

    async def generate_structured(self, **kwargs: Any) -> Any:
        del kwargs
        raise NotImplementedError

    async def generate_non_structured(self, **kwargs: Any) -> Any:
        del kwargs
        raise NotImplementedError

    async def generate_with_prompt(self, **kwargs: Any) -> Any:
        del kwargs
        raise NotImplementedError

    async def health_check(self) -> bool:
        """Return True for bootstrap tests."""
        return True

    async def invoke_with_system_and_user[T: BaseModel](
        self,
        system_message: str,
        user_message: str,
        *,
        structured_output_model: type[T],
        model_tier: str = "expensive",
    ) -> T | None:
        """Invoke LLM with system and user messages - not used in bootstrap tests."""
        del system_message, user_message, model_tier
        return None


class FakeSubmissionClient:
    """Submission client stub for container lifecycle tests."""

    def __init__(self) -> None:
        self.closed = False

    async def create_question_set(self, command: Any) -> Any:
        del command
        raise NotImplementedError

    async def close(self) -> None:
        self.closed = True


class FakeKnowledgeClient:
    """Knowledge client stub for container lifecycle tests."""

    def __init__(self) -> None:
        self.closed = False

    async def similarity_search(self, command: Any) -> Any:
        del command
        raise NotImplementedError

    async def get_chunks_by_ids(self, chunk_ids: Any) -> Any:
        del chunk_ids
        raise NotImplementedError

    async def close(self) -> None:
        self.closed = True


class FakeService:
    """Service stub used by the command handler test."""

    def __init__(self) -> None:
        self.command: Any = None

    async def execute(self, command: Any) -> GenerationReceipt:
        self.command = command
        return GenerationReceipt(
            question_set_id="qs_123",
            assessment_id=command.assessment_id,
            structured_generated=command.structured_count or 0,
            non_structured_generated=command.non_structured_count or 0,
            iteration=1,
            question_count=(command.structured_count or 0)
            + (command.non_structured_count or 0),
            status="completed",
        )


def _settings(**overrides: Any) -> Settings:
    base: dict[str, Any] = {
        "environment": RuntimeEnvironment.LOCAL,
        "host": "127.0.0.1",
        "port": 8080,
        "workers": 1,
        "log_level": LogLevel.DEBUG,
        "pubsub_project_id": None,
        "pubsub_subscription_trigger": None,
        "pubsub_topic_complete": None,
        "pubsub_max_workers": 10,
        "langfuse_public_key": None,
        "langfuse_secret_key": None,
        "langfuse_base_url": "https://cloud.langfuse.com",
        "model_id": "gpt-4o-mini",
        "llm_api_key": "test-key",
        "llm_base_url": None,
        "submission_service_url": "localhost:50052",
        "knowledge_service_url": "localhost:50053",
        "release": "test",
    }
    base.update(overrides)
    return Settings.model_construct(**base)


def test_json_helpers_round_trip_dataclass_and_enum() -> None:
    payload = ExamplePayload(
        timestamp=datetime(2026, 4, 15, tzinfo=UTC),
        mode=QuestionType.STRUCTURED,
    )

    decoded = loads(dumps(payload))

    assert decoded["mode"] == "structured"
    assert decoded["timestamp"] == "2026-04-15T00:00:00+00:00"


def test_json_helpers_support_pydantic_and_reject_unknown_types() -> None:
    assert loads(dumps(ExampleModel(value="ok"))) == {"value": "ok"}

    with pytest.raises(TypeError):
        dumps(object())


def test_app_error_string_includes_context() -> None:
    error = AppError("boom", foo="bar")
    assert str(error) == "boom (foo='bar')"


def test_settings_validate_rejects_missing_configuration() -> None:
    with pytest.raises(ConfigurationError):
        _settings().validate_settings()


async def test_handle_generation_trigger_translates_event_to_command() -> None:
    service = FakeService()
    handler = HandleGenerationTrigger(service)  # type: ignore[arg-type]

    receipt = await handler.handle(
        QnAGenerationTriggered(
            event_id="evt_123",
            workflow_id="wf_123",
            assessment_id="assessment_123",
            question_set_id="qs_123",
            validation_result=None,
            iteration=1,
            structured_count=2,
            non_structured_count=1,
            difficulty_level=DifficultyLevel.MEDIUM,
            purpose=Purpose.ASSESSMENT,
            feedback_issues=[],
            correlation_id="corr_123",
        )
    )

    assert service.command.request_id == "evt_123"
    assert receipt.question_count == 3


async def test_application_container_startup_and_shutdown() -> None:
    publisher = FakePublisher()
    telemetry = FakeTelemetry()
    submission_client = FakeSubmissionClient()
    knowledge_client = FakeKnowledgeClient()
    container = ApplicationContainer(
        settings=_settings(),
        llm_provider=None,
        question_set_repo=InMemoryQuestionSetRepository(),
        idempotency_store=InMemoryIdempotencyStore(),
        event_publisher=publisher,
        telemetry=telemetry,
        subscriber=None,
        submission_client=submission_client,  # type: ignore[arg-type]
        knowledge_client=knowledge_client,  # type: ignore[arg-type]
    )

    await container.startup()
    await container.shutdown()
    assert knowledge_client.closed is True

    assert publisher.closed is True
    assert telemetry.shutdown_called is True
    assert submission_client.closed is True


def test_build_container_uses_in_memory_adapters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bootstrap_module, "_build_telemetry", lambda settings: None)
    monkeypatch.setattr(
        bootstrap_module,
        "_build_llm",
        lambda settings, prompt_provider=None: FakeLLMProvider(),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "_build_submission_client",
        lambda settings: FakeSubmissionClient(),
    )
    monkeypatch.setattr(
        bootstrap_module,
        "_build_knowledge_client",
        lambda settings: FakeKnowledgeClient(),
    )

    container = bootstrap_module._build_container(_settings())

    assert isinstance(container.question_set_repo, InMemoryQuestionSetRepository)
    assert isinstance(container.idempotency_store, InMemoryIdempotencyStore)
    assert isinstance(container.event_publisher, NullEventPublisher)
