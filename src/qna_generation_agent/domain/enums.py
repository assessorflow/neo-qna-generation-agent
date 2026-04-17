"""Domain enums for QnA Generation Agent."""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "DifficultyLevel",
    "GenerationStatus",
    "Purpose",
    "QuestionType",
]


class DifficultyLevel(StrEnum):
    """Difficulty levels for questions."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionType(StrEnum):
    """Types of questions that can be generated."""

    STRUCTURED = "structured"
    NON_STRUCTURED = "non_structured"


class Purpose(StrEnum):
    """Purpose of the question set per spec."""

    ASSESSMENT = "assessment"
    PRACTICE = "practice"
    REVIEW = "review"
    TOPIC_REVISION = "topic_revision"


class GenerationStatus(StrEnum):
    """Status of a generation attempt."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
