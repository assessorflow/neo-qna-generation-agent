"""Persistence ports."""

from __future__ import annotations

from abc import ABC, abstractmethod

from qna_generation_agent.application.dto import AssessmentContext
from qna_generation_agent.domain.entities import QuestionSet


class AssessmentContextRepository(ABC):
    """Repository pattern: load generation context."""

    @abstractmethod
    async def get_context(
        self,
        *,
        assessment_id: str,
        topic_ids: list[str],
    ) -> AssessmentContext:
        """Load the context used to ground generation."""


class QuestionSetRepository(ABC):
    """Repository pattern: persist generated question sets."""

    @abstractmethod
    async def next_iteration(self, *, assessment_id: str) -> int:
        """Return the next iteration number for an assessment."""

    @abstractmethod
    async def save(self, question_set: QuestionSet) -> None:
        """Persist a generated question set."""

    @abstractmethod
    async def get_by_id(self, question_set_id: str) -> QuestionSet | None:
        """Fetch a persisted question set by identifier."""
