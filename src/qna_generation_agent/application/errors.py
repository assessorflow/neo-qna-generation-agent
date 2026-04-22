"""Typed application-layer errors."""

from __future__ import annotations

from typing import Any

from qna_generation_agent.errors import AppError


class ValidationError(AppError):
    """Raised when a boundary object or command is invalid."""


class TransientError(AppError):
    """Raised for retryable failures."""

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: int | None = None,
        **context: Any,
    ) -> None:
        """Initialize with retry guidance."""
        super().__init__(message, retry_after_seconds=retry_after_seconds, **context)
        self.retry_after_seconds = retry_after_seconds


class PermanentError(AppError):
    """Raised for non-retryable failures."""


class IdempotencyConflict(AppError):
    """Raised when a completed event is replayed."""

    def __init__(self, message: str, *, event_id: str, **context: Any) -> None:
        """Initialize with the conflicting event identifier."""
        super().__init__(message, event_id=event_id, **context)
        self.event_id = event_id


# Spec-compliant exception classes
class InvalidTriggerPayloadError(PermanentError):
    """Raised when a Pub/Sub trigger message is malformed."""


class RetrievalError(TransientError):
    """Raised when grounding material cannot be retrieved."""


class GenerationError(TransientError):
    """Raised when model output is unusable."""


class WorkflowEscalationError(PermanentError):
    """Raised when max regeneration iterations are exceeded."""

    def __init__(
        self, message: str, *, iteration: int, max_iterations: int, **context: Any
    ) -> None:
        """Initialize with iteration context for escalation."""
        super().__init__(
            message, iteration=iteration, max_iterations=max_iterations, **context
        )
        self.iteration = iteration
        self.max_iterations = max_iterations


class LLMTransientError(TransientError):
    """Raised when the LLM provider fails temporarily."""


class LLMPermanentError(PermanentError):
    """Raised when the LLM provider fails permanently."""


class StorageTransientError(TransientError):
    """Raised when storage or messaging fails temporarily."""


class StoragePermanentError(PermanentError):
    """Raised when storage or messaging fails permanently."""
