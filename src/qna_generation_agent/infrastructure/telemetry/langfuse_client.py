"""Langfuse telemetry adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse, propagate_attributes

from qna_generation_agent.app.logging import bind_context, get_logger
from qna_generation_agent.application.ports.telemetry import Span, TelemetryPort

logger = get_logger(__name__)


class _LangfuseSpan(Span):
    """Span wrapper used by the telemetry port."""

    def __init__(self, client: Langfuse) -> None:
        self._client = client

    def set_attribute(self, key: str, value: Any) -> None:
        self._client.update_current_span(metadata={key: str(value)})

    def record_error(self, error: BaseException) -> None:
        self._client.update_current_span(
            metadata={"error_type": type(error).__name__},
            status_message=str(error),
        )


class LangfuseTelemetry(TelemetryPort):
    """Adapter for the Langfuse Python SDK."""

    def __init__(
        self,
        *,
        public_key: str | None,
        secret_key: str | None,
        host: str,
        environment: str,
        release: str | None,
    ) -> None:
        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            environment=environment,
            release=release,
        )

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> Generator[Span]:
        with self._client.start_as_current_observation(
            name=name,
            metadata=metadata or {},
        ):
            # Bind trace_id to structlog context for correlation
            trace_id = self.current_trace_id()
            if trace_id:
                bind_context(trace_id=trace_id)
            yield _LangfuseSpan(self._client)

    @contextmanager
    def propagate(
        self,
        *,
        trace_name: str,
        correlation_id: str,
        workflow_id: str,
        metadata: dict[str, str] | None = None,
    ) -> Generator[None]:
        merged_metadata = {"workflow_id": workflow_id, **(metadata or {})}
        with propagate_attributes(
            trace_name=trace_name,
            session_id=correlation_id,
            metadata=merged_metadata,
        ):
            with self._client.start_as_current_observation(
                name=trace_name,
                metadata=merged_metadata,
            ):
                # Bind trace_id to structlog context for correlation
                trace_id = self.current_trace_id()
                if trace_id:
                    bind_context(trace_id=trace_id)
                yield

    def current_trace_id(self) -> str | None:
        return self._client.get_current_trace_id()

    def current_trace_url(self) -> str | None:
        return self._client.get_trace_url()

    async def shutdown(self) -> None:
        """Flush and shutdown the Langfuse client.

        Per Langfuse 4.2 SDK, flush() must be called to ensure all
        traces/spans are persisted before process exit.
        """
        try:
            await asyncio.to_thread(self._client.flush)
        except (OSError, TimeoutError, ConnectionError) as e:
            # Network-level errors during flush - may indicate data loss
            logger.error("langfuse_flush_failed_network", error=str(e))
        except Exception as e:
            # Other flush failures - log as warning since we're shutting down
            logger.warning(
                "langfuse_flush_failed", error=str(e), error_type=type(e).__name__
            )
        try:
            await asyncio.to_thread(self._client.shutdown)
        except Exception as e:
            logger.warning(
                "langfuse_shutdown_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
