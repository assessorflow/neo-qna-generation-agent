"""Unit tests for Pub/Sub subscriber."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from qna_generation_agent.application.dto import GenerationReceipt
from qna_generation_agent.application.errors import (
    LLMTransientError,
    PermanentError,
    StorageTransientError,
    TransientError,
)
from qna_generation_agent.application.errors import (
    ValidationError as AppValidationError,
)
from qna_generation_agent.domain.errors import ValidationError as DomainValidationError
from qna_generation_agent.infrastructure.messaging.pubsub_subscriber import (
    PubSubSubscriptionWorker,
    SubscriptionConfig,
)


class FakeMessage:
    """Fake Pub/Sub message for testing."""

    def __init__(
        self,
        message_id: str = "msg_123",
        data: bytes | None = None,
        acked: bool = False,
        nacked: bool = False,
    ) -> None:
        self.message_id = message_id
        self.data = data or b"{}"
        self._acked = acked
        self._nacked = nacked

    def ack(self) -> None:
        self._acked = True

    def nack(self) -> None:
        self._nacked = True


class FakeStreamingPullFuture:
    """Fake StreamingPullFuture for testing."""

    def __init__(self) -> None:
        self._cancelled = False
        self._result_called = False

    def cancel(self) -> None:
        self._cancelled = True

    def result(self) -> None:
        self._result_called = True
        if self._cancelled:
            from concurrent.futures import CancelledError as FutureCancelledError

            raise FutureCancelledError()


@pytest.fixture
def subscription_config() -> SubscriptionConfig:
    return SubscriptionConfig(
        project_id="test-project",
        subscription_id="test-subscription",
        max_messages=1,
    )


@pytest.fixture
def valid_trigger_event() -> dict[str, Any]:
    return {
        "event_id": "evt_123",
        "event_type": "assessorflow.qa-generation.trigger",
        "workflow_id": "wf_123",
        "timestamp": "2025-01-01T00:00:00+00:00",
        "source_agent": "test-agent",
        "correlation_id": "corr_123",
        "payload": {
            "assessment_id": "assessment_123",
            "question_set_id": "qs_123",
            "validation_result": None,
            "iteration": 1,
            "structured_generated": 2,
            "non_structured_generated": 1,
            "difficulty": "medium",
            "purpose": "assessment",
        },
    }


class TestPubSubSubscriptionWorker:
    """Tests for PubSubSubscriptionWorker."""

    @pytest.mark.unit
    async def test_callback_nacks_during_shutdown(
        self,
        subscription_config: SubscriptionConfig,
    ) -> None:
        """Test that callback nacks messages during shutdown."""
        handler = AsyncMock()
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        worker._shutting_down = True

        message = FakeMessage(message_id="msg_123")

        worker._callback(message)

        assert message._nacked

    @pytest.mark.unit
    async def test_handle_message_acks_on_success(
        self,
        subscription_config: SubscriptionConfig,
        valid_trigger_event: dict[str, Any],
    ) -> None:
        """Test that message is acked on successful processing."""
        receipt = GenerationReceipt(
            question_set_id="qs_123",
            assessment_id="assessment_123",
            structured_generated=2,
            non_structured_generated=1,
            iteration=1,
            question_count=3,
            status="completed",
        )
        handler = AsyncMock(return_value=receipt)
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        import orjson

        message = FakeMessage(
            message_id="msg_123",
            data=orjson.dumps(valid_trigger_event),
        )

        await worker._handle_message(message)

        assert message._acked

    @pytest.mark.unit
    async def test_handle_message_acks_permanent_errors(
        self,
        subscription_config: SubscriptionConfig,
        valid_trigger_event: dict[str, Any],
    ) -> None:
        """Test that message is acked for permanent errors."""
        handler = AsyncMock(side_effect=PermanentError("Permanent failure"))
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        import orjson

        message = FakeMessage(
            message_id="msg_123",
            data=orjson.dumps(valid_trigger_event),
        )

        await worker._handle_message(message)

        assert message._acked

    @pytest.mark.unit
    async def test_handle_message_nacks_transient_errors(
        self,
        subscription_config: SubscriptionConfig,
        valid_trigger_event: dict[str, Any],
    ) -> None:
        """Test that message is nacked for transient errors."""
        handler = AsyncMock(
            side_effect=TransientError("Transient failure", retry_after_seconds=5)
        )
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        import orjson

        message = FakeMessage(
            message_id="msg_123",
            data=orjson.dumps(valid_trigger_event),
        )

        await worker._handle_message(message)

        assert message._nacked

    @pytest.mark.unit
    async def test_handle_message_nacks_unexpected_errors(
        self,
        subscription_config: SubscriptionConfig,
        valid_trigger_event: dict[str, Any],
    ) -> None:
        """Test that message is nacked for unexpected errors."""
        handler = AsyncMock(side_effect=RuntimeError("Unexpected failure"))
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        import orjson

        message = FakeMessage(
            message_id="msg_123",
            data=orjson.dumps(valid_trigger_event),
        )

        await worker._handle_message(message)

        assert message._nacked

    @pytest.mark.unit
    async def test_handle_message_nacks_cancelled_error(
        self,
        subscription_config: SubscriptionConfig,
        valid_trigger_event: dict[str, Any],
    ) -> None:
        """Test that message is nacked and re-raises CancelledError."""
        handler = AsyncMock(side_effect=asyncio.CancelledError())
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        import orjson

        message = FakeMessage(
            message_id="msg_123",
            data=orjson.dumps(valid_trigger_event),
        )

        with pytest.raises(asyncio.CancelledError):
            await worker._handle_message(message)

        assert message._nacked

    @pytest.mark.unit
    async def test_handle_message_acks_validation_errors(
        self,
        subscription_config: SubscriptionConfig,
    ) -> None:
        """Test that message is acked for validation errors."""
        handler = AsyncMock()
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        # Invalid data that will fail Pydantic validation
        message = FakeMessage(
            message_id="msg_123",
            data=b'{"invalid": "data"}',
        )

        await worker._handle_message(message)

        assert message._acked

    @pytest.mark.unit
    async def test_handle_message_acks_pydantic_validation_error(
        self,
        subscription_config: SubscriptionConfig,
    ) -> None:
        """Test that message is acked for Pydantic validation errors."""
        handler = AsyncMock()
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        # Data missing required fields
        message = FakeMessage(
            message_id="msg_123",
            data=b'{"event_id": "evt_123"}',  # Missing required fields
        )

        await worker._handle_message(message)

        assert message._acked

    @pytest.mark.unit
    async def test_handle_message_acks_domain_validation_error(
        self,
        subscription_config: SubscriptionConfig,
        valid_trigger_event: dict[str, Any],
    ) -> None:
        """Test that message is acked for domain validation errors."""
        handler = AsyncMock(side_effect=DomainValidationError("Domain error"))
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        import orjson

        message = FakeMessage(
            message_id="msg_123",
            data=orjson.dumps(valid_trigger_event),
        )

        await worker._handle_message(message)

        assert message._acked

    @pytest.mark.unit
    async def test_handle_message_acks_app_validation_error(
        self,
        subscription_config: SubscriptionConfig,
        valid_trigger_event: dict[str, Any],
    ) -> None:
        """Test that message is acked for app validation errors."""
        handler = AsyncMock(side_effect=AppValidationError("App error"))
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        import orjson

        message = FakeMessage(
            message_id="msg_123",
            data=orjson.dumps(valid_trigger_event),
        )

        await worker._handle_message(message)

        assert message._acked

    @pytest.mark.unit
    async def test_handle_message_nacks_llm_transient_error(
        self,
        subscription_config: SubscriptionConfig,
        valid_trigger_event: dict[str, Any],
    ) -> None:
        """Test that message is nacked for LLM transient errors."""
        handler = AsyncMock(side_effect=LLMTransientError("LLM transient error"))
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        import orjson

        message = FakeMessage(
            message_id="msg_123",
            data=orjson.dumps(valid_trigger_event),
        )

        await worker._handle_message(message)

        assert message._nacked

    @pytest.mark.unit
    async def test_handle_message_nacks_storage_transient_error(
        self,
        subscription_config: SubscriptionConfig,
        valid_trigger_event: dict[str, Any],
    ) -> None:
        """Test that message is nacked for storage transient errors."""
        handler = AsyncMock(
            side_effect=StorageTransientError("Storage transient error")
        )
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        import orjson

        message = FakeMessage(
            message_id="msg_123",
            data=orjson.dumps(valid_trigger_event),
        )

        await worker._handle_message(message)

        assert message._nacked
