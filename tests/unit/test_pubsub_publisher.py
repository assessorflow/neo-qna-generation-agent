"""Unit tests for Pub/Sub publisher."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import PermissionDenied

from qna_generation_agent.application.errors import (
    StoragePermanentError,
    StorageTransientError,
)
from qna_generation_agent.application.ports.publisher import (
    DecisionAuditEvent,
    TokenUsageEvent,
)
from qna_generation_agent.domain.events import QnAGenerationCompleted
from qna_generation_agent.infrastructure.messaging.pubsub_publisher import (
    NullEventPublisher,
    PubSubCompletionPublisher,
)


class FakePublishFuture:
    """Fake publish future for testing."""

    def __init__(self, result: Any = None, exception: Exception | None = None) -> None:
        self._result = result
        self._exception = exception

    def result(self, timeout: float | None = None) -> Any:
        if self._exception:
            raise self._exception
        return self._result


class FakePublisherClient:
    """Fake PublisherClient for testing."""

    def __init__(self) -> None:
        self.published: list[dict[str, Any]] = []
        self._transport = MagicMock()

    def topic_path(self, project_id: str, topic_id: str) -> str:
        return f"projects/{project_id}/topics/{topic_id}"

    def publish(
        self,
        topic: str,
        data: bytes,
        **attrs: str,
    ) -> FakePublishFuture:
        self.published.append(
            {
                "topic": topic,
                "data": data,
                "attrs": attrs,
            }
        )
        return FakePublishFuture("message_id_123")


class TestNullEventPublisher:
    """Tests for NullEventPublisher."""

    @pytest.mark.unit
    async def test_publish_completion_returns_none(self) -> None:
        """Test that publish_completion returns None."""
        publisher = NullEventPublisher()
        event = QnAGenerationCompleted(
            event_id="evt_123",
            workflow_id="wf_123",
            assessment_id="assessment_123",
            question_set_id="qs_123",
            structured_generated=2,
            non_structured_generated=1,
            iteration=1,
            correlation_id="corr_123",
            trace_id=None,
        )
        await publisher.publish_completion(event)

    @pytest.mark.unit
    async def test_publish_decision_audit_returns_none(self) -> None:
        """Test that publish_decision_audit returns None."""
        publisher = NullEventPublisher()
        event = DecisionAuditEvent(
            workflow_id="wf_123",
            input_summary={"key": "value"},
            output_summary={"key": "value"},
            reasoning_steps=["step1"],
            confidence_score=0.9,
            prompt_version="v1",
            model_id="gpt-4o",
            grounding_sources=["chunk1"],
        )
        await publisher.publish_decision_audit(event)

    @pytest.mark.unit
    async def test_publish_token_usage_returns_none(self) -> None:
        """Test that publish_token_usage returns None."""
        publisher = NullEventPublisher()
        event = TokenUsageEvent(
            workflow_id="wf_123",
            model_id="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.001,
            prompt_version="v1",
        )
        await publisher.publish_token_usage(event)

    @pytest.mark.unit
    async def test_close_returns_none(self) -> None:
        """Test that close returns None."""
        publisher = NullEventPublisher()
        await publisher.close()


class TestPubSubCompletionPublisher:
    """Tests for PubSubCompletionPublisher."""

    @pytest.mark.unit
    async def test_publish_completion_success(self) -> None:
        """Test successful completion event publishing."""
        publisher = PubSubCompletionPublisher(
            project_id="test-project",
            topic_id="test-topic",
        )
        fake_client = FakePublisherClient()
        publisher._client = fake_client

        event = QnAGenerationCompleted(
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
        fake_client = FakePublisherClient()
        publisher._client = fake_client

        event = QnAGenerationCompleted(
            event_id="evt_123",
            workflow_id="wf_123",
            assessment_id="assessment_123",
            question_set_id="qs_123",
            structured_generated=2,
            non_structured_generated=1,
            iteration=1,
            correlation_id="corr_123",
            trace_id=None,
        )

        await publisher.publish_completion(event)

        assert len(fake_client.published) == 1
        assert (
            fake_client.published[0]["attrs"]["event_type"]
            == "assessorflow.qa-generation.complete"
        )
        assert fake_client.published[0]["attrs"]["workflow_id"] == "wf_123"

    @pytest.mark.unit
    async def test_publish_completion_permission_denied(self) -> None:
        """Test that PermissionDenied raises StoragePermanentError."""
        publisher = PubSubCompletionPublisher(
            project_id="test-project",
            topic_id="test-topic",
        )
        fake_client = FakePublisherClient()

        def raise_permission_denied(*args: Any, **kwargs: Any) -> FakePublishFuture:
            raise PermissionDenied("Permission denied")  # type: ignore[no-untyped-call]

        fake_client.publish = raise_permission_denied  # type: ignore[method-assign]
        publisher._client = fake_client

        event = QnAGenerationCompleted(
            event_id="evt_123",
            workflow_id="wf_123",
            assessment_id="assessment_123",
            question_set_id="qs_123",
            structured_generated=2,
            non_structured_generated=1,
            iteration=1,
            correlation_id="corr_123",
            trace_id=None,
        )

        with pytest.raises(StoragePermanentError):
            await publisher.publish_completion(event)

    @pytest.mark.unit
    async def test_publish_completion_other_error(self) -> None:
        """Test that other errors raise StorageTransientError."""
        publisher = PubSubCompletionPublisher(
            project_id="test-project",
            topic_id="test-topic",
        )
        fake_client = FakePublisherClient()

        def raise_error(*args: Any, **kwargs: Any) -> FakePublishFuture:
            raise RuntimeError("Network error")

        fake_client.publish = raise_error  # type: ignore[method-assign]
        publisher._client = fake_client

        event = QnAGenerationCompleted(
            event_id="evt_123",
            workflow_id="wf_123",
            assessment_id="assessment_123",
            question_set_id="qs_123",
            structured_generated=2,
            non_structured_generated=1,
            iteration=1,
            correlation_id="corr_123",
            trace_id=None,
        )

        with pytest.raises(StorageTransientError):
            await publisher.publish_completion(event)

    @pytest.mark.unit
    async def test_publish_decision_audit_success(self) -> None:
        """Test successful decision audit publishing."""
        publisher = PubSubCompletionPublisher(
            project_id="test-project",
            topic_id="test-topic",
            decision_audit_topic_id="audit-topic",
        )
        fake_client = FakePublisherClient()
        publisher._client = fake_client

        event = DecisionAuditEvent(
            workflow_id="wf_123",
            input_summary={
                "subtopics_selected": ["topic1"],
                "chunks_retrieved": 5,
                "difficulty": "medium",
            },
            output_summary={
                "structured_generated": 3,
                "non_structured_generated": 2,
            },
            reasoning_steps=["step1", "step2"],
            confidence_score=0.9,
            prompt_version="v1",
            model_id="gpt-4o",
            grounding_sources=["chunk1", "chunk2"],
        )

        await publisher.publish_decision_audit(event)

        assert len(fake_client.published) == 1
        assert (
            fake_client.published[0]["attrs"]["event_type"]
            == "assessorflow.audit.decision"
        )

    @pytest.mark.unit
    async def test_publish_decision_audit_skips_if_no_topic(self) -> None:
        """Test that decision audit is skipped if no topic configured."""
        publisher = PubSubCompletionPublisher(
            project_id="test-project",
            topic_id="test-topic",
        )
        fake_client = FakePublisherClient()
        publisher._client = fake_client

        event = DecisionAuditEvent(
            workflow_id="wf_123",
            input_summary={"key": "value"},
            output_summary={"key": "value"},
            reasoning_steps=["step1"],
            confidence_score=0.9,
            prompt_version="v1",
            model_id="gpt-4o",
            grounding_sources=["chunk1"],
        )

        await publisher.publish_decision_audit(event)

        assert len(fake_client.published) == 0

    @pytest.mark.unit
    async def test_publish_decision_audit_logs_errors(self) -> None:
        """Test that decision audit errors are logged but not raised."""
        publisher = PubSubCompletionPublisher(
            project_id="test-project",
            topic_id="test-topic",
            decision_audit_topic_id="audit-topic",
        )
        fake_client = FakePublisherClient()

        def raise_error(*args: Any, **kwargs: Any) -> FakePublishFuture:
            raise RuntimeError("Publish error")

        fake_client.publish = raise_error  # type: ignore[method-assign]
        publisher._client = fake_client

        event = DecisionAuditEvent(
            workflow_id="wf_123",
            input_summary={"key": "value"},
            output_summary={"key": "value"},
            reasoning_steps=["step1"],
            confidence_score=0.9,
            prompt_version="v1",
            model_id="gpt-4o",
            grounding_sources=["chunk1"],
        )

        # Should not raise
        await publisher.publish_decision_audit(event)

    @pytest.mark.unit
    async def test_publish_token_usage_success(self) -> None:
        """Test successful token usage publishing."""
        publisher = PubSubCompletionPublisher(
            project_id="test-project",
            topic_id="test-topic",
            token_usage_topic_id="token-topic",
        )
        fake_client = FakePublisherClient()
        publisher._client = fake_client

        event = TokenUsageEvent(
            workflow_id="wf_123",
            model_id="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.001,
            prompt_version="v1",
        )

        await publisher.publish_token_usage(event)

        assert len(fake_client.published) == 1
        assert (
            fake_client.published[0]["attrs"]["event_type"]
            == "assessorflow.audit.token-usage"
        )

    @pytest.mark.unit
    async def test_publish_token_usage_skips_if_no_topic(self) -> None:
        """Test that token usage is skipped if no topic configured."""
        publisher = PubSubCompletionPublisher(
            project_id="test-project",
            topic_id="test-topic",
        )
        fake_client = FakePublisherClient()
        publisher._client = fake_client

        event = TokenUsageEvent(
            workflow_id="wf_123",
            model_id="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.001,
            prompt_version="v1",
        )

        await publisher.publish_token_usage(event)

        assert len(fake_client.published) == 0

    @pytest.mark.unit
    async def test_publish_token_usage_logs_errors(self) -> None:
        """Test that token usage errors are logged but not raised."""
        publisher = PubSubCompletionPublisher(
            project_id="test-project",
            topic_id="test-topic",
            token_usage_topic_id="token-topic",
        )
        fake_client = FakePublisherClient()

        def raise_error(*args: Any, **kwargs: Any) -> FakePublishFuture:
            raise RuntimeError("Publish error")

        fake_client.publish = raise_error  # type: ignore[method-assign]
        publisher._client = fake_client

        event = TokenUsageEvent(
            workflow_id="wf_123",
            model_id="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            estimated_cost_usd=0.001,
            prompt_version="v1",
        )

        # Should not raise
        await publisher.publish_token_usage(event)

    @pytest.mark.unit
    async def test_close_closes_transport(self) -> None:
        """Test that close closes the transport."""
        publisher = PubSubCompletionPublisher(
            project_id="test-project",
            topic_id="test-topic",
        )
        fake_transport = MagicMock()
        fake_client = MagicMock()
        fake_client.transport = fake_transport
        publisher._client = fake_client

        with patch(
            "qna_generation_agent.infrastructure.messaging.pubsub_publisher.asyncio.to_thread",
            return_value=None,
        ):
            await publisher.close()

        assert publisher._client is None

    @pytest.mark.unit
    async def test_get_client_lazy_initialization(self) -> None:
        """Test that client is lazily initialized."""
        publisher = PubSubCompletionPublisher(
            project_id="test-project",
            topic_id="test-topic",
        )

        with patch(
            "qna_generation_agent.infrastructure.messaging.pubsub_publisher.asyncio.to_thread",
            return_value=FakePublisherClient(),
        ):
            client = await publisher._get_client()

        assert client is not None
        assert publisher._client is not None
