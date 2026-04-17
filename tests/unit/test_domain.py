"""Unit tests for domain invariants."""

from __future__ import annotations

import pytest

from qna_generation_agent.domain.entities import (
    Answer,
    GenerationRequest,
    Question,
    QuestionSet,
)
from qna_generation_agent.domain.enums import (
    GenerationStatus,
    QuestionType,
)
from qna_generation_agent.domain.errors import InvalidStateTransitionError
from qna_generation_agent.domain.value_objects import AnswerId, ContentHash, QuestionId


def test_question_id_generate_uses_prefix() -> None:
    identifier = QuestionId.generate(prefix="custom")
    assert identifier.value.startswith("custom_")


def test_content_hash_is_stable() -> None:
    left = ContentHash.from_content("stable content")
    right = ContentHash.from_content("stable content")
    assert left == right
    assert len(left.value) == 64


def test_question_set_transitions_follow_state_machine() -> None:
    question_set = QuestionSet(
        id="qs_123",
        assessment_id="assessment_123",
        iteration=1,
        purpose="assessment",
    )

    question_set.mark_in_progress()
    question_set.mark_completed()

    assert question_set.status is GenerationStatus.COMPLETED


def test_question_set_rejects_invalid_transition() -> None:
    question_set = QuestionSet(
        id="qs_123",
        assessment_id="assessment_123",
        iteration=1,
        purpose="assessment",
    )

    with pytest.raises(InvalidStateTransitionError):
        question_set.mark_completed()


def test_generation_request_valid_with_nullable_fields() -> None:
    # GenerationRequest now allows nullable generation params for regeneration flow
    request = GenerationRequest(
        id="evt_123",
        workflow_id="wf_123",
        correlation_id="corr_123",
        assessment_id="assessment_123",
        validation_result="fail",
        iteration=1,
        structured_count=None,  # Nullable for regeneration
        non_structured_count=None,  # Nullable for regeneration
        difficulty_level=None,  # Nullable for regeneration
        purpose=None,  # Nullable for regeneration
    )
    assert request.assessment_id == "assessment_123"


def test_question_and_answer_validate_text() -> None:
    answer = Answer(id=AnswerId.generate(), text="42")
    question = Question(
        id=QuestionId.generate(),
        text="What is the answer?",
        question_type=QuestionType.STRUCTURED,
        difficulty_level="easy",
        answer=answer,
    )
    assert question.text == "What is the answer?"
