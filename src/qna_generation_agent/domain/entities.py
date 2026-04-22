"""Domain entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from qna_generation_agent.domain.enums import (
    DifficultyLevel,
    GenerationStatus,
    Purpose,
    QuestionType,
    ValidationResult,
)
from qna_generation_agent.domain.errors import (
    InvalidStateTransitionError,
    ValidationError,
)
from qna_generation_agent.domain.value_objects import AnswerId, QuestionId

__all__ = [
    "Answer",
    "GenerationRequest",
    "Question",
    "QuestionSet",
]


@dataclass(slots=True)
class Answer:
    """Represents a model answer with optional explanation and references."""

    id: AnswerId
    text: str
    explanation: str | None = None
    references: list[str] = field(default_factory=list)
    confidence_score: float | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Validate answer invariants."""
        if not self.text.strip():
            raise ValidationError("Answer text cannot be empty")
        if (
            self.confidence_score is not None
            and not 0.0 <= self.confidence_score <= 1.0
        ):
            raise ValidationError("Confidence score must be between 0.0 and 1.0")


@dataclass(slots=True)
class Question:
    """Represents an assessment question with its answer and metadata."""

    id: QuestionId
    text: str
    question_type: QuestionType
    difficulty_level: DifficultyLevel | None
    answer: Answer
    topic_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Validate question invariants."""
        if not self.text.strip():
            raise ValidationError("Question text cannot be empty")


@dataclass(slots=True)
class QuestionSet:
    """Aggregate root representing a collection of questions for an assessment."""

    id: str
    assessment_id: str
    iteration: int
    purpose: Purpose | None
    questions: list[Question] = field(default_factory=list)
    status: GenerationStatus = GenerationStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Validate question set invariants."""
        if not self.id.strip():
            raise ValidationError("QuestionSet id cannot be empty")
        if not self.assessment_id.strip():
            raise ValidationError("QuestionSet assessment_id cannot be empty")
        if self.iteration < 1:
            raise ValidationError("QuestionSet iteration must be >= 1")

    def _transition_to(
        self,
        target: GenerationStatus,
        valid_from: set[GenerationStatus],
        message: str,
    ) -> None:
        """Transition to target state if current state is valid."""
        if self.status not in valid_from:
            raise InvalidStateTransitionError(
                message,
                from_state=self.status.value,
                to_state=target.value,
                entity_id=self.id,
            )
        self.status = target

    def mark_in_progress(self) -> None:
        """Transition the question set to in-progress status."""
        self._transition_to(
            GenerationStatus.IN_PROGRESS,
            {GenerationStatus.PENDING},
            "QuestionSet can only start from pending",
        )

    def mark_completed(self) -> None:
        """Transition the question set to completed status."""
        self._transition_to(
            GenerationStatus.COMPLETED,
            {GenerationStatus.IN_PROGRESS},
            "QuestionSet can only complete from in_progress",
        )

    def mark_failed(self) -> None:
        """Transition the question set to failed status."""
        self._transition_to(
            GenerationStatus.FAILED,
            {GenerationStatus.PENDING, GenerationStatus.IN_PROGRESS},
            "QuestionSet can only fail from pending or in_progress",
        )

    def add_question(self, question: Question) -> None:
        """Add a question to the set with state and duplicate guards.

        Raises:
            InvalidStateTransitionError: If QuestionSet is in a terminal state.
            ValidationError: If question with same ID already exists.
        """
        if self.status in {GenerationStatus.COMPLETED, GenerationStatus.FAILED}:
            raise InvalidStateTransitionError(
                "Cannot add questions to terminal state",
                from_state=self.status.value,
                to_state="modified",
                entity_id=self.id,
            )
        if any(q.id == question.id for q in self.questions):
            raise ValidationError(
                f"Question with id {question.id} already exists in set",
                question_id=question.id.value,
                question_set_id=self.id,
            )
        self.questions.append(question)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Represents a normalized generation request per spec.

    Supports two flows:
    1. Initial generation: structured_count, non_structured_count, difficulty_level, purpose required
    2. Regeneration: validation_result, iteration, feedback_issues present; generation params nullable
    """

    id: str
    assessment_id: str
    validation_result: ValidationResult | None
    iteration: int | None  # Nullable for initial generation
    structured_count: int | None  # Nullable for regeneration
    non_structured_count: int | None  # Nullable for regeneration
    difficulty_level: DifficultyLevel | None  # Nullable for regeneration
    purpose: Purpose | None  # Nullable for regeneration
    correlation_id: str
    workflow_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Validate generation request invariants."""
        if not self.id.strip():
            raise ValidationError("GenerationRequest id cannot be empty")
        if not self.assessment_id.strip():
            raise ValidationError("GenerationRequest assessment_id cannot be empty")
        if not self.correlation_id.strip():
            raise ValidationError("GenerationRequest correlation_id cannot be empty")
        if not self.workflow_id.strip():
            raise ValidationError("GenerationRequest workflow_id cannot be empty")
