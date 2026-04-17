"""Immutable domain events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

__all__ = [
    "QnAGenerationCompleted",
    "QnAGenerationTriggered",
]


@dataclass(frozen=True)
class QnAGenerationTriggered:
    """Represents an inbound generation request.

    Supports two flows:
    1. Initial generation: structured_count, non_structured_count, difficulty_level, purpose required
    2. Regeneration: validation_result, iteration, feedback_issues present; generation params fetched from prior
    """

    event_id: str
    workflow_id: str
    assessment_id: str
    question_set_id: str  # Required for all flows
    validation_result: str | None  # "pass" or "fail"
    iteration: int | None  # Nullable for initial generation
    structured_count: int | None  # Nullable for regeneration
    non_structured_count: int | None  # Nullable for regeneration
    difficulty_level: str | None  # Nullable for regeneration
    purpose: str | None  # Nullable for regeneration
    feedback_issues: list[str]  # For regeneration feedback
    correlation_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True)
class QnAGenerationCompleted:
    """Represents a successfully completed generation run.

    Spec 5.11: Q&A Generation Complete
    - structured_generated: Number of MCQ questions actually generated
    - non_structured_generated: Number of open-ended questions actually generated
    - iteration: Feedback loop iteration count
    """

    event_id: str
    workflow_id: str
    assessment_id: str
    question_set_id: str
    structured_generated: int
    non_structured_generated: int
    iteration: int
    correlation_id: str
    trace_id: str | None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
