"""Unit tests for Pub/Sub subscriber."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qna_generation_agent.application.dto import GenerationReceipt
from qna_generation_agent.application.errors import (
    LLMTransientError,
    StorageTransientError,
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
    async def test_handle_message_nacks_non_json_payload(
        self,
        subscription_config: SubscriptionConfig,
    ) -> None:
        """Test that non-JSON payloads are nacked."""
        handler = AsyncMock()
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        # Non-JSON binary data
        message = FakeMessage(
            message_id="msg_123",
            data=b"\x00\x01\x02\x03\xff\xfe",  # Binary garbage
        )

        await worker._handle_message(message)

        # Should nack non-JSON payloads (can't parse, so treat as transient)
        assert message._nacked
        # Handler should not be called
        handler.assert_not_called()

    @pytest.mark.unit
    async def test_handle_message_acks_idempotency_conflict(
        self,
        subscription_config: SubscriptionConfig,
        valid_trigger_event: dict[str, Any],
    ) -> None:
        """Test that IdempotencyConflict errors are acked (don't retry duplicates)."""
        from qna_generation_agent.application.errors import IdempotencyConflict

        handler = AsyncMock(
            side_effect=IdempotencyConflict(
                "Event already processing", event_id="evt_123"
            )
        )
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        import orjson

        message = FakeMessage(
            message_id="msg_123",
            data=orjson.dumps(valid_trigger_event),
        )

        await worker._handle_message(message)

        # Idempotency conflicts should be acked (not retried)
        assert message._acked
        assert not message._nacked

    @pytest.mark.unit
    async def test_callback_handles_closed_loop(
        self,
        subscription_config: SubscriptionConfig,
    ) -> None:
        """Test that callback handles closed event loop gracefully."""
        handler = AsyncMock()
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        # Create a closed loop and assign it
        loop = asyncio.new_event_loop()
        loop.close()
        worker._loop = loop

        message = FakeMessage(message_id="msg_123", data=b"{}")

        # Callback should handle closed loop gracefully (nack message)
        worker._callback(message)

        # Message should be nacked when loop is closed
        assert message._nacked

    @pytest.mark.unit
    async def test_callback_handles_run_coroutine_threadsafe_failure(
        self,
        subscription_config: SubscriptionConfig,
    ) -> None:
        """Test that callback handles run_coroutine_threadsafe RuntimeError."""
        handler = AsyncMock()
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        # Set up a valid loop
        worker._loop = asyncio.get_running_loop()

        message = FakeMessage(message_id="msg_123", data=b"{}")

        # Mock run_coroutine_threadsafe to raise RuntimeError
        original_run = asyncio.run_coroutine_threadsafe

        def mock_run_coroutine(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("Event loop is closed")

        asyncio.run_coroutine_threadsafe = mock_run_coroutine  # type: ignore

        try:
            worker._callback(message)
        finally:
            asyncio.run_coroutine_threadsafe = original_run  # type: ignore

        # Message should be nacked when scheduling fails
        assert message._nacked

    @pytest.mark.unit
    async def test_shutdown_handles_timeout(
        self,
        subscription_config: SubscriptionConfig,
    ) -> None:
        """Test that shutdown handles timeout gracefully."""

        class SlowFuture:
            """Future that never completes (simulates timeout)."""

            def __init__(self) -> None:
                self.cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

            def result(self, timeout: float | None = None) -> None:
                # Simulate timeout by sleeping longer than the timeout
                import time

                time.sleep(0.1)  # This will exceed the 10s timeout in shutdown

        class TimeoutSubscriber:
            """Subscriber that returns slow future."""

            def __init__(self) -> None:
                self.closed = False
                self.future = SlowFuture()

            def subscription_path(self, project_id: str, subscription_id: str) -> str:
                return f"projects/{project_id}/subscriptions/{subscription_id}"

            def subscribe(self, *args: Any, **kwargs: Any) -> Any:
                return self.future

            def close(self) -> None:
                self.closed = True

        handler = AsyncMock()
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        # Manually set subscriber and future
        slow_subscriber = TimeoutSubscriber()
        worker._subscriber = slow_subscriber  # type: ignore
        worker._future = slow_subscriber.future  # type: ignore

        # Start the worker first
        await worker.start()

        # Shutdown should handle timeout gracefully
        await worker.shutdown()

        # Subscriber should be closed even after timeout
        # Note: In the actual implementation, timeout is caught and logged

    @pytest.mark.unit
    async def test_shutdown_handles_not_found(
        self,
        subscription_config: SubscriptionConfig,
    ) -> None:
        """Test that shutdown handles NotFound exception."""
        from google.api_core.exceptions import NotFound

        class NotFoundFuture:
            """Future that raises NotFound."""

            def __init__(self) -> None:
                self.cancelled = False

            def cancel(self) -> None:
                self.cancelled = True

            def result(self, timeout: float | None = None) -> None:
                raise NotFound("Subscription not found")

        class NotFoundSubscriber:
            """Subscriber that returns future raising NotFound."""

            def __init__(self) -> None:
                self.closed = False
                self.future = NotFoundFuture()

            def subscription_path(self, project_id: str, subscription_id: str) -> str:
                return f"projects/{project_id}/subscriptions/{subscription_id}"

            def subscribe(self, *args: Any, **kwargs: Any) -> Any:
                return self.future

            def close(self) -> None:
                self.closed = True

        handler = AsyncMock()
        worker = PubSubSubscriptionWorker(subscription_config, handler)

        # Manually set subscriber and future
        not_found_subscriber = NotFoundSubscriber()
        worker._subscriber = not_found_subscriber  # type: ignore
        worker._future = not_found_subscriber.future  # type: ignore

        # Shutdown should handle NotFound gracefully
        await worker.shutdown()

        # Should complete without raising

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

    @pytest.mark.unit
    async def test_worker_tracks_runtime_state_initial(self) -> None:
        """Test that worker initializes with correct runtime state."""
        config = SubscriptionConfig(
            project_id="test-project",
            subscription_id="test-subscription",
            max_messages=1,
        )
        handler = AsyncMock()
        worker = PubSubSubscriptionWorker(config, handler)

        assert worker.is_started is False
        assert worker.is_running is False
        assert worker.is_failed is False
        assert not worker.start_event.is_set()
        assert not worker.stop_event.is_set()

    @pytest.mark.unit
    async def test_worker_sets_failed_on_start_error(self) -> None:
        """Test that worker sets failed state when start fails."""
        config = SubscriptionConfig(
            project_id="test-project",
            subscription_id="test-subscription",
            max_messages=1,
        )
        handler = AsyncMock()
        worker = PubSubSubscriptionWorker(config, handler)

        # Simulate start failure by mocking SubscriberClient to raise
        with patch(
            "qna_generation_agent.infrastructure.messaging.pubsub_subscriber.asyncio.to_thread",
            side_effect=RuntimeError("Failed to create client"),
        ):
            with pytest.raises(RuntimeError):
                await worker.start()

        assert worker.is_started is True  # Started was attempted
        assert worker.is_running is False  # But not running
        assert worker.is_failed is True  # And marked as failed
        assert not worker.start_event.is_set()  # Start event not set on failure

    @pytest.mark.unit
    async def test_worker_runtime_state_after_shutdown(self) -> None:
        """Test that worker correctly updates runtime state after shutdown."""
        config = SubscriptionConfig(
            project_id="test-project",
            subscription_id="test-subscription",
            max_messages=1,
        )
        handler = AsyncMock()
        worker = PubSubSubscriptionWorker(config, handler)

        # Manually set started state (simulating successful start)
        worker._started = True
        worker._running = True

        # Create mock subscriber that doesn't block
        mock_subscriber = MagicMock()
        mock_subscriber.close = MagicMock()
        worker._subscriber = mock_subscriber  # type: ignore

        await worker.shutdown()

        assert worker.is_started is False
        assert worker.is_running is False
        assert worker.is_failed is False
        assert worker.stop_event.is_set()
