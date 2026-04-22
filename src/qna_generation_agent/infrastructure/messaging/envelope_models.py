"""Pydantic models for inbound and outbound event envelopes."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from qna_generation_agent.domain.events import (
    QnAGenerationCompleted,
    QnAGenerationTriggered,
)

# Event types per spec (not the internal workflow names)
TRIGGER_EVENT_TYPE: Final = "assessorflow.qa-generation.trigger"
COMPLETED_EVENT_TYPE: Final = "assessorflow.qa-generation.complete"
DECISION_AUDIT_EVENT_TYPE: Final = "assessorflow.audit.decision"
TOKEN_USAGE_AUDIT_EVENT_TYPE: Final = "assessorflow.audit.token-usage"


class FeedbackPayload(BaseModel):
    """Feedback for regeneration (Phase 7a).

    Spec: Quality Validation Failed - feedback.issues and feedback.action
    """

    model_config = ConfigDict(strict=True)

    issues: list[str] = Field(default_factory=list)
    action: str | None = Field(default=None, pattern="^(regenerate|)$")


class TriggerPayload(BaseModel):
    """Inbound trigger payload per spec.

    Supports two shapes:
    1. Initial generation: assessment_id, question_set_id, structured_count, non_structured_count, difficulty_level, purpose
    2. Regeneration: assessment_id, question_set_id, validation_result, iteration, feedback
    """

    model_config = ConfigDict(strict=True)

    assessment_id: str
    question_set_id: str  # Required for all flows
    validation_result: str | None = Field(default=None, pattern="^(pass|fail|)$")
    iteration: int | None = Field(default=None, ge=1)
    structured_count: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices("structured_count", "structured_generated"),
    )
    non_structured_count: int | None = Field(
        default=None,
        ge=0,
        validation_alias=AliasChoices(
            "non_structured_count", "non_structured_generated"
        ),
    )
    # Constrained to known difficulty values per spec
    difficulty_level: str | None = Field(
        default=None,
        pattern="^(easy|medium|hard|)$",
        validation_alias=AliasChoices("difficulty_level", "difficulty"),
    )
    # Constrained to known purpose values per spec
    purpose: str | None = Field(
        default=None, pattern="^(assessment|practice|review|topic_revision|)$"
    )
    feedback: FeedbackPayload | None = None


class TriggerEnvelope(BaseModel):
    """Inbound trigger envelope."""

    model_config = ConfigDict(strict=True)

    event_id: str
    event_type: Literal["assessorflow.qa-generation.trigger"]
    workflow_id: str
    timestamp: datetime
    source_agent: str
    correlation_id: str
    trace_id: str | None = None  # For distributed tracing propagation
    payload: TriggerPayload

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, value: datetime | str) -> datetime:
        """Parse timestamp from string if needed.

        Pub/Sub messages encode timestamps as ISO format strings.
        This validator accepts both datetime objects and ISO strings.
        """
        if isinstance(value, str):
            # Handle ISO format with or without timezone
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value)
        return value

    def to_domain_event(self) -> QnAGenerationTriggered:
        """Convert the envelope to a domain event."""
        return QnAGenerationTriggered(
            event_id=self.event_id,
            workflow_id=self.workflow_id,
            assessment_id=self.payload.assessment_id,
            question_set_id=self.payload.question_set_id,
            validation_result=self.payload.validation_result,
            iteration=self.payload.iteration,
            structured_count=self.payload.structured_count,
            non_structured_count=self.payload.non_structured_count,
            difficulty_level=self.payload.difficulty_level,
            purpose=self.payload.purpose,
            feedback_issues=self.payload.feedback.issues
            if self.payload.feedback
            else [],
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
            timestamp=self.timestamp,
        )


class CompletionPayload(BaseModel):
    """Outbound completion payload per spec 5.11.

    Spec 5.11: Q&A Generation Complete
    - assessment_id: References assessment_configs.id
    - question_set_id: References question_sets.id
    - structured_generated: Number of MCQ questions actually generated
    - non_structured_generated: Number of open-ended questions actually generated
    - iteration: Feedback loop iteration count (starts at 1)
    """

    model_config = ConfigDict(strict=True)

    assessment_id: str
    question_set_id: str
    structured_generated: int = Field(ge=0)
    non_structured_generated: int = Field(ge=0)
    iteration: int = Field(ge=1)


class CompletionEnvelope(BaseModel):
    """Outbound completion envelope."""

    model_config = ConfigDict(strict=True)

    event_id: str
    event_type: Literal["assessorflow.qa-generation.complete"]
    workflow_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    source_agent: str = "qa-generation-agent"
    correlation_id: str
    payload: CompletionPayload

    @classmethod
    def from_domain_event(cls, event: QnAGenerationCompleted) -> CompletionEnvelope:
        """Create an envelope from a domain completion event."""
        return cls(
            event_id=event.event_id,
            event_type=COMPLETED_EVENT_TYPE,
            workflow_id=event.workflow_id,
            correlation_id=event.correlation_id,
            timestamp=event.timestamp,
            payload=CompletionPayload(
                assessment_id=event.assessment_id,
                question_set_id=event.question_set_id,
                structured_generated=event.structured_generated,
                non_structured_generated=event.non_structured_generated,
                iteration=event.iteration,
            ),
        )


class DecisionAuditInputSummary(BaseModel):
    """Input summary for decision audit."""

    model_config = ConfigDict(strict=True)

    question_set_id: str
    iteration: int


class DecisionAuditOutputSummary(BaseModel):
    """Output summary for decision audit."""

    model_config = ConfigDict(strict=True)

    result: str
    issues_found: int


class DecisionAuditPayload(BaseModel):
    """Payload for assessorflow.audit.decision event."""

    model_config = ConfigDict(strict=True)

    workflow_id: str
    agent_name: str = "qa-generation-agent"
    decision_type: str = "question_generation"
    input_summary: DecisionAuditInputSummary
    output_summary: DecisionAuditOutputSummary
    reasoning_steps: list[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    prompt_version: str
    model_id: str
    grounding_sources: list[str]


class DecisionAuditEnvelope(BaseModel):
    """Decision audit event envelope."""

    model_config = ConfigDict(strict=True)

    event_type: Literal["assessorflow.audit.decision"] = DECISION_AUDIT_EVENT_TYPE
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: DecisionAuditPayload


class TokenUsagePayload(BaseModel):
    """Payload for assessorflow.audit.token-usage event."""

    model_config = ConfigDict(strict=True)

    workflow_id: str
    agent_name: str = "qa-generation-agent"
    model_id: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)
    prompt_version: str


class TokenUsageEnvelope(BaseModel):
    """Token usage audit event envelope."""

    model_config = ConfigDict(strict=True)

    event_type: Literal["assessorflow.audit.token-usage"] = TOKEN_USAGE_AUDIT_EVENT_TYPE
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: TokenUsagePayload
