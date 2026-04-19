"""In-memory adapters used by tests and local development."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from qna_generation_agent.application.dto import GenerationReceipt
from qna_generation_agent.application.ports.idempotency import (
    IdempotencyRecord,
    IdempotencyStatus,
    IdempotencyStore,
)
from qna_generation_agent.application.ports.repository import QuestionSetRepository
from qna_generation_agent.domain.entities import QuestionSet


class InMemoryIdempotencyStore(IdempotencyStore):
    """In-memory idempotency adapter.

    **NOTE: This adapter is for testing and local development only.**
    It does not implement TTL expiration - records persist for the lifetime
    of the process. For production use, a persistent store with TTL support
    (e.g., Redis, Cloud Memorystore) is required.

    The ttl_seconds parameter is accepted for interface compatibility but
    is ignored by this implementation.
    """

    def __init__(self) -> None:
        """Initialize the in-memory idempotency store."""
        self._records: dict[str, IdempotencyRecord] = {}
        self._lock = asyncio.Lock()

    async def get(self, event_id: str) -> IdempotencyRecord:
        """Retrieve the idempotency record for an event."""
        async with self._lock:
            return self._records.get(
                event_id,
                IdempotencyRecord(event_id=event_id, status=IdempotencyStatus.NEW),
            )

    async def start(self, event_id: str, *, ttl_seconds: int) -> bool:
        """Acquire idempotency lock. Return False if already processing or completed.

        Uses asyncio.Lock to prevent race conditions in concurrent scenarios.

        Note: ttl_seconds is ignored in this in-memory adapter. Records persist
        for the lifetime of the process.
        """
        async with self._lock:
            current = self._records.get(
                event_id,
                IdempotencyRecord(event_id=event_id, status=IdempotencyStatus.NEW),
            )
            if current.status in (
                IdempotencyStatus.PROCESSING,
                IdempotencyStatus.COMPLETED,
            ):
                return False
            self._records[event_id] = IdempotencyRecord(
                event_id=event_id,
                status=IdempotencyStatus.PROCESSING,
            )
            return True

    async def complete(
        self,
        event_id: str,
        *,
        receipt: GenerationReceipt,
        ttl_seconds: int,
    ) -> None:
        """Mark event as completed with receipt.

        Note: ttl_seconds is ignored in this in-memory adapter. Records persist
        for the lifetime of the process.
        """
        async with self._lock:
            self._records[event_id] = IdempotencyRecord(
                event_id=event_id,
                status=IdempotencyStatus.COMPLETED,
                receipt=receipt,
            )

    async def fail(
        self,
        event_id: str,
        *,
        error_message: str,
        ttl_seconds: int,
    ) -> None:
        """Mark event as failed with error message.

        Note: ttl_seconds is ignored in this in-memory adapter. Records persist
        for the lifetime of the process.
        """
        async with self._lock:
            self._records[event_id] = IdempotencyRecord(
                event_id=event_id,
                status=IdempotencyStatus.FAILED,
                error_message=error_message,
            )


class InMemoryQuestionSetRepository(QuestionSetRepository):
    """In-memory question-set adapter."""

    def __init__(self) -> None:
        """Initialize the in-memory question set repository."""
        self._question_sets: dict[str, QuestionSet] = {}
        self._lock = asyncio.Lock()

    async def next_iteration(self, *, assessment_id: str) -> int:
        """Compute the next iteration number for an assessment."""
        iterations = [
            question_set.iteration
            for question_set in self._question_sets.values()
            if question_set.assessment_id == assessment_id
        ]
        return (max(iterations) + 1) if iterations else 1

    async def save(self, question_set: QuestionSet) -> None:
        """Store a question set (deep-copied)."""
        async with self._lock:
            self._question_sets[question_set.id] = replace(question_set)

    async def get_by_id(self, question_set_id: str) -> QuestionSet | None:
        """Retrieve a question set by ID."""
        async with self._lock:
            question_set = self._question_sets.get(question_set_id)
            if question_set is None:
                return None
            return replace(question_set)
