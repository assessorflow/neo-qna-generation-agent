"""Contract tests for Pub/Sub event schemas per spec."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from qna_generation_agent.domain.events import QnAGenerationCompleted
from qna_generation_agent.infrastructure.messaging.envelope_models import (
    COMPLETED_EVENT_TYPE,
    TRIGGER_EVENT_TYPE,
    CompletionEnvelope,
    FeedbackPayload,
    TriggerEnvelope,
    TriggerPayload,
)


@pytest.mark.contract
def test_trigger_envelope_maps_validation_failed_to_domain_event() -> None:
    """Test Spec 5.13: Quality Validation Failed event mapping.

    The trigger envelope now receives validation_result and iteration
    from the Evaluator Agent via Orchestrator.
    """
    envelope = TriggerEnvelope(
        event_id="evt_123",
        event_type=TRIGGER_EVENT_TYPE,
        workflow_id="wf_123",
        timestamp=datetime.now(UTC),
        source_agent="evaluator-agent",
        correlation_id="corr_123",
        payload=TriggerPayload(
            assessment_id="assessment_123",
            question_set_id="qs_123",  # For regeneration
            validation_result="fail",
            iteration=1,
            structured_count=2,
            non_structured_count=1,
            difficulty_level="medium",
            purpose="assessment",
            feedback=FeedbackPayload(
                issues=["Q3 not grounded in source material"],
                action="regenerate",
            ),
        ),
    )

    event = envelope.to_domain_event()

    assert event.event_id == "evt_123"
    assert event.assessment_id == "assessment_123"
    assert event.question_set_id == "qs_123"
    assert event.validation_result == "fail"
    assert event.iteration == 1
    assert event.structured_count == 2
    assert event.non_structured_count == 1
    assert event.feedback_issues == ["Q3 not grounded in source material"]


@pytest.mark.contract
def test_trigger_payload_without_validation_result() -> None:
    """Test initial generation trigger (no validation_result yet)."""
    envelope = TriggerEnvelope(
        event_id="evt_124",
        event_type=TRIGGER_EVENT_TYPE,
        workflow_id="wf_124",
        timestamp=datetime.now(UTC),
        source_agent="orchestrator",
        correlation_id="corr_124",
        payload=TriggerPayload(
            assessment_id="assessment_124",
            question_set_id="qs_124",
            structured_count=6,
            non_structured_count=2,
            difficulty_level="easy",
            purpose="topic_revision",
        ),
    )

    event = envelope.to_domain_event()

    assert event.validation_result is None
    assert event.iteration is None
    assert event.question_set_id == "qs_124"


@pytest.mark.contract
def test_completion_envelope_serializes_domain_event() -> None:
    """Test Spec 5.11: Q&A Generation Complete event serialization.

    Uses structured_generated and non_structured_generated per spec.
    """
    envelope = CompletionEnvelope.from_domain_event(
        QnAGenerationCompleted(
            event_id="evt_999",
            workflow_id="wf_999",
            assessment_id="assessment_999",
            question_set_id="qs_999",
            structured_generated=3,
            non_structured_generated=2,
            iteration=1,
            correlation_id="corr_999",
            trace_id="trace_999",
        )
    )

    assert envelope.event_type == COMPLETED_EVENT_TYPE
    assert envelope.payload.question_set_id == "qs_999"
    assert envelope.payload.structured_generated == 3
    assert envelope.payload.non_structured_generated == 2
    assert envelope.payload.iteration == 1
