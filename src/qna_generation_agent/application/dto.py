"""Application-layer data transfer objects."""

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


@dataclass(frozen=True, slots=True)
class AssessmentContext:
    """Context used to ground a generation request."""

    assessment_id: str
    title: str
    topic_ids: list[str]
    chunks: list[str]


@dataclass(frozen=True, slots=True)
class QuestionDraft:
    """Generated question payload returned by the LLM adapter."""

    question_text: str
    answer_text: str
    question_type: QuestionType
    difficulty_level: DifficultyLevel | None
    explanation: str | None = None
    references: list[str] = field(default_factory=list)
    topic_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QuestionBatch:
    """Batch of question drafts returned by the LLM adapter."""

    questions: list[QuestionDraft]
    model_name: str
    prompt_version: str


@dataclass(frozen=True, slots=True)
class GenerationCommand:
    """Normalized command for the generation use case per spec.

    Supports two flows:
    1. Initial generation: structured_count, non_structured_count, difficulty_level, purpose required
    2. Regeneration: validation_result, iteration, feedback_issues present; generation params nullable
    """

    request_id: str
    workflow_id: str
    correlation_id: str
    trace_id: str | None  # For distributed tracing propagation
    assessment_id: str
    question_set_id: str  # Required for all flows
    validation_result: ValidationResult | None
    iteration: int | None  # Nullable for initial generation
    structured_count: int | None  # Nullable for regeneration
    non_structured_count: int | None  # Nullable for regeneration
    difficulty_level: DifficultyLevel | None  # Nullable for regeneration
    purpose: Purpose | None  # Nullable for regeneration
    feedback_issues: list[str]  # For regeneration feedback


@dataclass(frozen=True, slots=True)
class GenerationReceipt:
    """Result returned by the generation use case per spec.

    Spec 5.11: Q&A Generation Complete outbound event fields
    - structured_generated: Number of MCQ questions actually generated
    - non_structured_generated: Number of open-ended questions actually generated
    - iteration: Feedback loop iteration count
    """

    question_set_id: str
    assessment_id: str
    structured_generated: int
    non_structured_generated: int
    iteration: int
    question_count: int
    status: GenerationStatus
    trace_id: str | None = None
    trace_url: str | None = None
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
