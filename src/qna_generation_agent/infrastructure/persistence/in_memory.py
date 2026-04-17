"""In-memory adapters used by tests and local development."""

from __future__ import annotations

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
    """In-memory idempotency adapter."""

    def __init__(self) -> None:
        self._records: dict[str, IdempotencyRecord] = {}

    async def get(self, event_id: str) -> IdempotencyRecord:
        return self._records.get(
            event_id,
            IdempotencyRecord(event_id=event_id, status=IdempotencyStatus.NEW),
        )

    async def start(self, event_id: str, *, ttl_seconds: int) -> bool:
        """Acquire idempotency lock. Return False if already processing or completed."""
        current = await self.get(event_id)
        if current.status in (IdempotencyStatus.PROCESSING, IdempotencyStatus.COMPLETED):
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
        self._records[event_id] = IdempotencyRecord(
            event_id=event_id,
            status=IdempotencyStatus.FAILED,
            error_message=error_message,
        )


class InMemoryQuestionSetRepository(QuestionSetRepository):
    """In-memory question-set adapter."""

    def __init__(self) -> None:
        self._question_sets: dict[str, QuestionSet] = {}

    async def next_iteration(self, *, assessment_id: str) -> int:
        iterations = [
            question_set.iteration
            for question_set in self._question_sets.values()
            if question_set.assessment_id == assessment_id
        ]
        return (max(iterations) + 1) if iterations else 1

    async def save(self, question_set: QuestionSet) -> None:
        self._question_sets[question_set.id] = replace(question_set)

    async def get_by_id(self, question_set_id: str) -> QuestionSet | None:
        question_set = self._question_sets.get(question_set_id)
        if question_set is None:
            return None
        return replace(question_set)
