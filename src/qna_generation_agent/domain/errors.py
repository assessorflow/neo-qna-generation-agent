"""Domain error hierarchy."""

from __future__ import annotations

from qna_generation_agent.errors import AppError

__all__ = [
    "DomainError",
    "InconsistentStateError",
    "InvalidStateTransitionError",
    "ValidationError",
]


class DomainError(AppError):
    """Base class for domain invariants."""


class ValidationError(DomainError):
    """Raised when a domain invariant is violated."""


class InconsistentStateError(DomainError):
    """Raised when an entity would enter an invalid state."""


class InvalidStateTransitionError(InconsistentStateError):
    """Raised when an entity transition is not allowed."""

    def __init__(
        self,
        message: str,
        *,
        from_state: str,
        to_state: str,
        entity_id: str,
    ) -> None:
        super().__init__(
            message,
            from_state=from_state,
            to_state=to_state,
            entity_id=entity_id,
        )
