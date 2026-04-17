"""Idempotency port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from qna_generation_agent.application.dto import GenerationReceipt


class IdempotencyStatus(StrEnum):
    """State of an idempotency key."""

    NEW = "new"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """Typed idempotency record."""

    event_id: str
    status: IdempotencyStatus
    receipt: GenerationReceipt | None = None
    error_message: str | None = None


class IdempotencyStore(ABC):
    """Repository pattern: durable duplicate-detection boundary."""

    @abstractmethod
    async def get(self, event_id: str) -> IdempotencyRecord:
        """Fetch an idempotency record."""

    @abstractmethod
    async def start(self, event_id: str, *, ttl_seconds: int) -> bool:
        """Attempt to acquire the processing lock."""

    @abstractmethod
    async def complete(
        self,
        event_id: str,
        *,
        receipt: GenerationReceipt,
        ttl_seconds: int,
    ) -> None:
        """Persist a completed receipt."""

    @abstractmethod
    async def fail(
        self,
        event_id: str,
        *,
        error_message: str,
        ttl_seconds: int,
    ) -> None:
        """Persist a failed processing attempt."""
