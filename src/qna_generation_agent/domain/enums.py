"""Domain enums for QnA Generation Agent."""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "DifficultyLevel",
    "GenerationStage",
    "GenerationStatus",
    "Purpose",
    "QuestionType",
    "ValidationResult",
]


class DifficultyLevel(StrEnum):
    """Difficulty levels for questions."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class GenerationStage(StrEnum):
    """Generation stages for LLM provider routing."""

    ASSESSMENT_GENERATOR = "assessment_generator"
    QUESTION_GENERATION = "question_generation"
    MCQ_ANSWER_GENERATOR = "mcq_answer_generator"
    MCQ_EXPLANATION_GENERATOR = "mcq_explanation_generator"


class ValidationResult(StrEnum):
    """Validation result values."""

    PASS = "pass"
    FAIL = "fail"


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
