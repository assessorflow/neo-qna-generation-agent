"""Telemetry port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from typing import Any


class Span(ABC):
    """Active span handle."""

    @abstractmethod
    def set_attribute(self, key: str, value: Any) -> None:
        """Attach structured metadata to the current span."""

    @abstractmethod
    def record_error(self, error: BaseException) -> None:
        """Attach error metadata to the current span."""


class TelemetryPort(ABC):
    """Port used by the application layer for tracing."""

    @abstractmethod
    def trace(
        self,
        name: str,
        *,
        metadata: dict[str, str] | None = None,
    ) -> AbstractContextManager[Span]:
        """Create a child span."""

    @abstractmethod
    def propagate(
        self,
        *,
        trace_name: str,
        correlation_id: str,
        workflow_id: str,
        metadata: dict[str, str] | None = None,
    ) -> AbstractContextManager[None]:
        """Create the root trace context."""

    @abstractmethod
    def current_trace_id(self) -> str | None:
        """Return the current trace identifier."""

    @abstractmethod
    def current_trace_url(self) -> str | None:
        """Return the current trace URL when available."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Flush and close any pending telemetry state."""
