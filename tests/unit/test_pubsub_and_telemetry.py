"""Unit tests for Pub/Sub and Langfuse adapters."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

import pytest

from qna_generation_agent.app.json import dumps
from qna_generation_agent.application.dto import GenerationReceipt
from qna_generation_agent.application.errors import StoragePermanentError
from qna_generation_agent.domain.enums import DifficultyLevel, Purpose
from qna_generation_agent.domain.events import QnAGenerationCompleted
from qna_generation_agent.infrastructure.messaging.pubsub_publisher import (
    PubSubCompletionPublisher,
)
from qna_generation_agent.infrastructure.messaging.pubsub_subscriber import (
    PubSubSubscriptionWorker,
    SubscriptionConfig,
)
from qna_generation_agent.infrastructure.telemetry import (
    langfuse_client as telemetry_module,
)
from qna_generation_agent.infrastructure.telemetry.langfuse_client import (
    LangfuseTelemetry,
)


class FakePublishFuture:
    """Fake Pub/Sub future."""

    def result(self, timeout: float) -> str:
        del timeout
        return "message-id"


class FakePublisherTransport:
    """Fake transport with a close hook."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakePublisherClient:
    """Publisher client stub."""

    should_raise: BaseException | None = None

    def __init__(self) -> None:
        self.transport = FakePublisherTransport()
        self.published: list[tuple[str, bytes, dict[str, str]]] = []

    def topic_path(self, project_id: str, topic_id: str) -> str:
        return f"projects/{project_id}/topics/{topic_id}"

    def publish(
        self, topic_path: str, payload: bytes, **attributes: str
    ) -> FakePublishFuture:
        if self.should_raise is not None:
            raise self.should_raise
        self.published.append((topic_path, payload, attributes))
        return FakePublishFuture()


class FakeStreamingFuture:
    """Streaming pull future stub."""

    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    def result(self, timeout: float | None = None) -> None:
        del timeout
        return None


class FakeSubscriberClient:
    """Subscriber client stub."""

    def __init__(self) -> None:
        self.closed = False
        self.future = FakeStreamingFuture()

    def subscription_path(self, project_id: str, subscription_id: str) -> str:
        return f"projects/{project_id}/subscriptions/{subscription_id}"

    def subscribe(self, *args: Any, **kwargs: Any) -> FakeStreamingFuture:
        del args, kwargs
        return self.future

    def close(self) -> None:
        self.closed = True


class FakeMessage:
    """Pub/Sub message stub."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.message_id = "msg-123"
        self.acked = False
        self.nacked = False

    def ack(self) -> None:
        self.acked = True

    def nack(self) -> None:
        self.nacked = True


class FakeLangfuseClient:
    """Langfuse stub used by telemetry tests."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.flush_called = False
        self.shutdown_called = False
        self.updated_metadata: list[dict[str, str]] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any) -> Iterator[dict[str, Any]]:
        yield kwargs

    def update_current_span(
        self, *, metadata: dict[str, str], status_message: str | None = None
    ) -> None:
        del status_message
        self.updated_metadata.append(metadata)

    def get_current_trace_id(self) -> str:
        return "trace-123"

    def get_trace_url(self) -> str:
        return "https://langfuse.example/trace-123"

    def flush(self) -> None:
        self.flush_called = True

    def shutdown(self) -> None:
        self.shutdown_called = True


async def test_pubsub_completion_publisher_publishes_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    client = FakePublisherClient()
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_publisher.PublisherClient",
        lambda: client,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_publisher.asyncio.to_thread",
        immediate_to_thread,
    )

    publisher = PubSubCompletionPublisher(project_id="project-id", topic_id="topic-id")
    await publisher.publish_completion(
        QnAGenerationCompleted(
            event_id="evt_123",
            workflow_id="wf_123",
            assessment_id="assessment_123",
            question_set_id="qs_123",
            structured_generated=2,
            non_structured_generated=1,
            iteration=1,
            correlation_id="corr_123",
            trace_id="trace_123",
        )
    )
    await publisher.close()

    assert client.transport.closed is True
    assert client.published[0][0] == "projects/project-id/topics/topic-id"


async def test_pubsub_completion_publisher_maps_permission_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    FakePublisherClient.should_raise = PermissionError("denied")
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_publisher.PublisherClient",
        FakePublisherClient,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_publisher.asyncio.to_thread",
        immediate_to_thread,
    )

    publisher = PubSubCompletionPublisher(project_id="project-id", topic_id="topic-id")

    with pytest.raises(StoragePermanentError):
        await publisher.publish_completion(
            QnAGenerationCompleted(
                event_id="evt_123",
                workflow_id="wf_123",
                assessment_id="assessment_123",
                question_set_id="qs_123",
                structured_generated=1,
                non_structured_generated=0,
                iteration=1,
                correlation_id="corr_123",
                trace_id=None,
            )
        )

    FakePublisherClient.should_raise = None


async def test_pubsub_completion_publisher_maps_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    from qna_generation_agent.application.errors import StorageTransientError

    FakePublisherClient.should_raise = ConnectionError("connection reset")
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_publisher.PublisherClient",
        FakePublisherClient,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_publisher.asyncio.to_thread",
        immediate_to_thread,
    )

    publisher = PubSubCompletionPublisher(project_id="project-id", topic_id="topic-id")

    with pytest.raises(StorageTransientError):
        await publisher.publish_completion(
            QnAGenerationCompleted(
                event_id="evt_123",
                workflow_id="wf_123",
                assessment_id="assessment_123",
                question_set_id="qs_123",
                structured_generated=1,
                non_structured_generated=0,
                iteration=1,
                correlation_id="corr_123",
                trace_id=None,
            )
        )

    FakePublisherClient.should_raise = None


async def test_pubsub_completion_publisher_skips_audit_topics_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    client = FakePublisherClient()
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_publisher.PublisherClient",
        lambda: client,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_publisher.asyncio.to_thread",
        immediate_to_thread,
    )

    from qna_generation_agent.application.ports.publisher import (
        DecisionAuditEvent,
        TokenUsageEvent,
    )

    publisher = PubSubCompletionPublisher(
        project_id="project-id",
        topic_id="topic-id",
        decision_audit_topic_id=None,
        token_usage_topic_id=None,
    )

    # These should return early without error
    await publisher.publish_decision_audit(
        DecisionAuditEvent(
            workflow_id="wf_123",
            input_summary={},
            output_summary={},
            reasoning_steps=[],
            confidence_score=0.5,
            prompt_version="v1",
            model_id="gpt-4",
            grounding_sources=[],
        )
    )
    await publisher.publish_token_usage(
        TokenUsageEvent(
            workflow_id="wf_123",
            model_id="gpt-4",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.001,
            prompt_version="v1",
        )
    )

    # No calls should have been made
    assert len(client.published) == 0

    await publisher.close()


async def test_pubsub_subscription_worker_processes_valid_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that valid messages are processed and acked."""

    async def immediate_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    subscriber = FakeSubscriberClient()
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_subscriber.SubscriberClient",
        lambda: subscriber,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_subscriber.asyncio.to_thread",
        immediate_to_thread,
    )

    handler_called = False

    async def handler(event: Any) -> GenerationReceipt:
        nonlocal handler_called
        handler_called = True
        return GenerationReceipt(
            question_set_id="qs_123",
            assessment_id="assessment_123",
            structured_generated=1,
            non_structured_generated=0,
            iteration=1,
            question_count=1,
            status="completed",
        )

    worker = PubSubSubscriptionWorker(
        SubscriptionConfig(
            project_id="project-id",
            subscription_id="subscription-id",
            max_messages=5,
        ),
        handler,
    )
    await worker.start()

    valid_message = FakeMessage(
        dumps(
            {
                "event_id": "evt_123",
                "event_type": "assessorflow.qa-generation.trigger",
                "workflow_id": "wf_123",
                "timestamp": datetime.now(UTC).isoformat(),
                "source_agent": "workflow-agent",
                "correlation_id": "corr_123",
                "trace_id": "trace_123",
                "payload": {
                    "assessment_id": "assessment_123",
                    "question_set_id": "qs_123",
                    "validation_result": None,
                    "iteration": 1,
                    "structured_count": 1,
                    "non_structured_count": 0,
                    "difficulty_level": DifficultyLevel.MEDIUM.value,
                    "purpose": Purpose.ASSESSMENT.value,
                },
            }
        )
    )

    await worker._handle_message(valid_message)
    await worker.shutdown()

    # Verify handler was called and message was acked
    assert handler_called is True
    assert valid_message.acked is True
    assert subscriber.future.cancelled is True
    assert subscriber.closed is True


async def test_pubsub_subscription_worker_rejects_malformed_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that malformed envelopes are rejected (acked without processing)."""

    async def immediate_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    subscriber = FakeSubscriberClient()
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_subscriber.SubscriberClient",
        lambda: subscriber,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_subscriber.asyncio.to_thread",
        immediate_to_thread,
    )

    handler_called = False

    async def handler(event: Any) -> GenerationReceipt:
        nonlocal handler_called
        handler_called = True
        return GenerationReceipt(
            question_set_id="qs_123",
            assessment_id="assessment_123",
            structured_generated=1,
            non_structured_generated=0,
            iteration=1,
            question_count=1,
            status="completed",
        )

    worker = PubSubSubscriptionWorker(
        SubscriptionConfig(
            project_id="project-id",
            subscription_id="subscription-id",
            max_messages=5,
        ),
        handler,
    )
    await worker.start()

    # Malformed message - missing required fields
    invalid_message = FakeMessage(b'{"invalid":true}')

    await worker._handle_message(invalid_message)
    await worker.shutdown()

    # Handler should NOT be called for malformed messages
    assert handler_called is False
    # But message should be acked (to prevent retry of bad messages)
    assert invalid_message.acked is True


async def test_langfuse_telemetry_wraps_client_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeLangfuseClient()

    @contextmanager
    def fake_propagate_attributes(**kwargs: Any) -> Iterator[dict[str, Any]]:
        yield kwargs

    monkeypatch.setattr(telemetry_module, "Langfuse", lambda **kwargs: fake_client)
    monkeypatch.setattr(
        telemetry_module, "propagate_attributes", fake_propagate_attributes
    )

    telemetry = LangfuseTelemetry(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
        release="sha-123",
    )

    with telemetry.trace("child-span") as span:
        span.set_attribute("foo", "bar")

    with telemetry.propagate(
        trace_name="root-trace",
        correlation_id="corr_123",
        workflow_id="wf_123",
        metadata={"assessment_id": "assessment_123"},
    ):
        pass

    await telemetry.shutdown()

    assert telemetry.current_trace_id() == "trace-123"
    assert telemetry.current_trace_url() == "https://langfuse.example/trace-123"
    assert fake_client.flush_called is True
    assert fake_client.shutdown_called is True


async def test_pubsub_subscription_worker_nacks_during_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    subscriber = FakeSubscriberClient()
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_subscriber.SubscriberClient",
        lambda: subscriber,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_subscriber.asyncio.to_thread",
        immediate_to_thread,
    )

    async def handler(event: Any) -> GenerationReceipt:
        del event
        return GenerationReceipt(
            question_set_id="qs_123",
            assessment_id="assessment_123",
            structured_generated=1,
            non_structured_generated=0,
            iteration=1,
            question_count=1,
            status="completed",
        )

    worker = PubSubSubscriptionWorker(
        SubscriptionConfig(
            project_id="project-id",
            subscription_id="subscription-id",
            max_messages=5,
        ),
        handler,
    )
    await worker.start()

    # Set shutting down flag
    worker._shutting_down = True

    # Message should be nacked immediately when shutting down
    message = FakeMessage(b'{"event_id": "evt_123"}')
    worker._callback(message)

    assert message.nacked is True
    assert message.acked is False

    await worker.shutdown()


async def test_pubsub_subscription_worker_handles_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def immediate_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    subscriber = FakeSubscriberClient()
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_subscriber.SubscriberClient",
        lambda: subscriber,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.messaging.pubsub_subscriber.asyncio.to_thread",
        immediate_to_thread,
    )

    call_count = 0

    async def failing_handler(event: Any) -> GenerationReceipt:
        del event
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Transient error")

    worker = PubSubSubscriptionWorker(
        SubscriptionConfig(
            project_id="project-id",
            subscription_id="subscription-id",
            max_messages=5,
        ),
        failing_handler,
    )
    await worker.start()

    message = FakeMessage(
        dumps(
            {
                "event_id": "evt_123",
                "event_type": "assessorflow.qa-generation.trigger",
                "workflow_id": "wf_123",
                "timestamp": datetime.now(UTC).isoformat(),
                "source_agent": "workflow-agent",
                "correlation_id": "corr_123",
                "payload": {
                    "assessment_id": "assessment_123",
                    "question_set_id": "qs_123",
                    "validation_result": None,
                    "iteration": 1,
                    "structured_count": 1,
                    "non_structured_count": 0,
                    "difficulty_level": DifficultyLevel.MEDIUM.value,
                    "purpose": Purpose.ASSESSMENT.value,
                },
            }
        )
    )

    await worker._handle_message(message)
    await worker.shutdown()

    assert call_count == 1
    assert message.nacked is True
    assert message.acked is False


async def test_null_event_publisher() -> None:
    from qna_generation_agent.infrastructure.messaging.pubsub_publisher import (
        NullEventPublisher,
    )

    publisher = NullEventPublisher()

    # Should do nothing without error
    await publisher.publish_completion(
        QnAGenerationCompleted(
            event_id="evt_123",
            workflow_id="wf_123",
            assessment_id="assessment_123",
            question_set_id="qs_123",
            structured_generated=1,
            non_structured_generated=0,
            iteration=1,
            correlation_id="corr_123",
            trace_id=None,
        )
    )
    await publisher.close()
