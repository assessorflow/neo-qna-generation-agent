"""Pub/Sub subscriber adapter using async patterns with SubscriberClient."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from concurrent.futures import CancelledError as FutureCancelledError
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

from google.api_core.exceptions import Cancelled as GrpcCancelledError
from google.api_core.exceptions import NotFound
from google.cloud.pubsub_v1 import SubscriberClient
from google.cloud.pubsub_v1.types import FlowControl
from pydantic import ValidationError as PydanticValidationError

if TYPE_CHECKING:
    from google.cloud.pubsub_v1.subscriber.futures import StreamingPullFuture

from qna_generation_agent.app.json import loads
from qna_generation_agent.app.logging import bind_context, clear_context, get_logger
from qna_generation_agent.application.dto import GenerationReceipt
from qna_generation_agent.application.errors import (
    IdempotencyConflict,
    LLMTransientError,
    PermanentError,
    StorageTransientError,
    TransientError,
)
from qna_generation_agent.application.errors import (
    ValidationError as AppValidationError,
)
from qna_generation_agent.domain.errors import ValidationError as DomainValidationError
from qna_generation_agent.domain.events import QnAGenerationTriggered
from qna_generation_agent.infrastructure.messaging.envelope_models import (
    TriggerEnvelope,
)

logger = get_logger(__name__)
TriggerHandler = Callable[[QnAGenerationTriggered], Awaitable[GenerationReceipt]]


@dataclass(frozen=True, slots=True)
class SubscriptionConfig:
    """Immutable subscriber configuration."""

    project_id: str
    subscription_id: str
    max_messages: int


class PubSubSubscriptionWorker:
    """Async subscriber runtime using SubscriberClient with asyncio futures."""

    def __init__(self, config: SubscriptionConfig, handler: TriggerHandler) -> None:
        """Initialize the Pub/Sub subscription worker.

        Args:
            config: Subscription configuration.
            handler: Async handler for triggered events.
        """
        self._config = config
        self._handler = handler
        self._subscriber: SubscriberClient | None = None
        self._future: StreamingPullFuture | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._shutdown_lock = asyncio.Lock()
        self._shutting_down = False

        # Runtime state tracking for health checks
        self._started = False
        self._running = False
        self._failed = False
        self._start_event = asyncio.Event()
        self._stop_event = asyncio.Event()

    async def start(self) -> None:
        """Start the subscriber using sync client with async threading wrapper."""
        self._started = True
        self._failed = False
        self._loop = asyncio.get_running_loop()

        try:
            self._subscriber = await asyncio.to_thread(SubscriberClient)
            if self._subscriber is None:
                raise RuntimeError("Failed to create SubscriberClient")
            flow_control = FlowControl(max_messages=self._config.max_messages)
            subscription_path = self._subscriber.subscription_path(
                self._config.project_id,
                self._config.subscription_id,
            )

            # Subscribe in thread to avoid blocking
            self._future = await asyncio.to_thread(
                self._subscriber.subscribe,
                subscription_path,
                callback=self._callback,
                flow_control=flow_control,
                await_callbacks_on_shutdown=True,
            )
            self._running = True
            self._start_event.set()
            logger.info(
                "subscription_started",
                subscription_id=self._config.subscription_id,
            )
        except Exception as error:
            self._failed = True
            self._running = False
            logger.error(
                "subscription_start_failed",
                subscription_id=self._config.subscription_id,
                error=str(error),
                error_type=type(error).__name__,
            )
            raise

    async def shutdown(self) -> None:
        """Gracefully shutdown the subscriber."""
        async with self._shutdown_lock:
            self._shutting_down = True
            self._running = False
            self._stop_event.set()

        logger.info(
            "subscriber_shutdown_initiated",
            subscription_id=self._config.subscription_id,
        )

        try:
            if self._future is not None:
                await asyncio.to_thread(self._future.cancel)
                try:
                    await asyncio.wait_for(
                        asyncio.to_thread(self._future.result),
                        timeout=10.0,
                    )
                except GrpcCancelledError:
                    logger.debug(
                        "subscription_cancelled",
                        subscription_id=self._config.subscription_id,
                    )
                except FutureCancelledError:
                    logger.debug(
                        "subscription_cancelled_cleanly",
                        subscription_id=self._config.subscription_id,
                    )
                except NotFound:
                    logger.warning(
                        "subscription_not_found",
                        subscription_id=self._config.subscription_id,
                    )
                except TimeoutError:
                    logger.warning(
                        "subscriber_shutdown_timeout",
                        subscription_id=self._config.subscription_id,
                    )
        finally:
            if self._subscriber is not None:
                await asyncio.to_thread(self._subscriber.close)

            self._started = False
            logger.info(
                "subscription_stopped",
                subscription_id=self._config.subscription_id,
            )

    @property
    def is_started(self) -> bool:
        """Return True if subscriber has been started."""
        return self._started

    @property
    def is_running(self) -> bool:
        """Return True if subscriber is actively consuming messages."""
        return self._running and not self._shutting_down

    @property
    def is_failed(self) -> bool:
        """Return True if subscriber failed to start or crashed."""
        return self._failed

    @property
    def start_event(self) -> asyncio.Event:
        """Event that is set when subscriber successfully starts."""
        return self._start_event

    @property
    def stop_event(self) -> asyncio.Event:
        """Event that is set when subscriber is shutting down."""
        return self._stop_event

    def _callback(self, message: object) -> None:
        """Synchronous callback invoked by the subscriber thread.

        Schedules async message handling in the event loop.
        Checks shutdown state safely and handles event loop lifecycle edge cases.
        """
        if self._shutting_down:
            nack_method = getattr(message, "nack", None)
            if nack_method:
                nack_method()
            logger.debug(
                "message_nacked_during_shutdown",
                message_id=getattr(message, "message_id", "unknown"),
            )
            return

        if self._loop is None or self._loop.is_closed():
            nack_method = getattr(message, "nack", None)
            if nack_method:
                nack_method()
            logger.warning(
                "message_nacked_no_event_loop",
                message_id=getattr(message, "message_id", "unknown"),
            )
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._handle_message(message), self._loop
            )
        except RuntimeError as e:
            nack_method = getattr(message, "nack", None)
            if nack_method:
                nack_method()
            logger.warning(
                "message_nacked_event_loop_error",
                message_id=getattr(message, "message_id", "unknown"),
                error=str(e),
            )
            return

        try:
            future.result()
        except (FutureCancelledError, asyncio.CancelledError):
            raise
        except Exception:
            logger.exception(
                "message_handler_failed_in_callback",
                message_id=getattr(message, "message_id", "unknown"),
            )

    async def _extend_ack_deadline(
        self, message: object, interval: int = 30
    ) -> asyncio.Task[None] | None:
        """Create a background task to extend ack deadline periodically.

        Args:
            message: The Pub/Sub message to extend deadline for.
            interval: Seconds between deadline extensions (default 30).

        Returns:
            Task that extends deadline until cancelled, or None if extension
            is not available for this message type.
        """
        extend_method = getattr(message, "modify_ack_deadline", None)
        if not extend_method:
            return None

        msg_id = getattr(message, "message_id", "unknown")

        async def extend_loop() -> None:
            """Background loop to extend message ack deadlines."""
            while True:
                await asyncio.sleep(interval)
                try:
                    # Extend by 60 seconds each time
                    await asyncio.to_thread(extend_method, 60)
                    logger.debug(
                        "ack_deadline_extended",
                        message_id=msg_id,
                        extension_seconds=60,
                    )
                except Exception:
                    logger.warning(
                        "ack_deadline_extension_failed",
                        message_id=msg_id,
                        exc_info=True,
                    )

        return asyncio.create_task(extend_loop())

    async def _handle_message(self, message: object) -> None:
        """Process a single message with proper ack/nack handling."""
        clear_context()
        msg_id = getattr(message, "message_id", "unknown")
        bind_context(message_id=msg_id)

        extend_task = await self._extend_ack_deadline(message)

        try:
            data = getattr(message, "data", b"")
            payload = loads(data)
            envelope = TriggerEnvelope.model_validate(payload)
            bind_context(
                event_id=envelope.event_id,
                workflow_id=envelope.workflow_id,
                correlation_id=envelope.correlation_id,
                source_agent=envelope.source_agent,
            )
            receipt = await self._handler(envelope.to_domain_event())

            ack_method = getattr(message, "ack", None)
            if ack_method:
                await asyncio.to_thread(ack_method)
            logger.info(
                "message_processed",
                question_set_id=receipt.question_set_id,
                question_count=receipt.question_count,
            )
        except (
            PydanticValidationError,
            AppValidationError,
            DomainValidationError,
            PermanentError,
            IdempotencyConflict,
        ) as error:
            ack_method = getattr(message, "ack", None)
            if ack_method:
                await asyncio.to_thread(ack_method)
            logger.info(
                "message_rejected_or_duplicate",
                error=str(error),
                error_type=error.__class__.__name__,
            )
        except asyncio.CancelledError:
            nack_method = getattr(message, "nack", None)
            if nack_method:
                await asyncio.to_thread(nack_method)
            raise
        except (TransientError, StorageTransientError, LLMTransientError) as error:
            nack_method = getattr(message, "nack", None)
            if nack_method:
                await asyncio.to_thread(nack_method)
            logger.warning(
                "message_failed_transient",
                error=str(error),
                error_type=error.__class__.__name__,
                retry_after_seconds=getattr(error, "retry_after_seconds", None),
            )
        except Exception:
            nack_method = getattr(message, "nack", None)
            if nack_method:
                await asyncio.to_thread(nack_method)
            logger.exception("message_failed_unexpected")
        finally:
            if extend_task is not None:
                extend_task.cancel()
                with suppress(asyncio.CancelledError):
                    await extend_task
            clear_context()
