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
    DifficultyLevel,
    GenerationStatus,
    Purpose,
    QuestionType,
)
from qna_generation_agent.domain.errors import (
    InvalidStateTransitionError,
    ValidationError,
)
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
        difficulty_level=DifficultyLevel.EASY,
        answer=answer,
    )
    assert question.text == "What is the answer?"


def test_question_set_rejects_iteration_zero() -> None:
    """QuestionSet requires iteration >= 1."""
    with pytest.raises(ValidationError) as exc_info:
        QuestionSet(
            id="qs_123",
            assessment_id="assessment_123",
            iteration=0,  # Invalid: must be >= 1
            purpose=Purpose.ASSESSMENT,
        )
    assert "iteration must be >= 1" in str(exc_info.value)


def test_question_set_requires_iteration_at_least_one() -> None:
    """QuestionSet accepts iteration >= 1."""
    question_set = QuestionSet(
        id="qs_123",
        assessment_id="assessment_123",
        iteration=1,
        purpose=Purpose.ASSESSMENT,
    )
    assert question_set.iteration == 1


def test_question_set_add_question_rejects_terminal_state() -> None:
    """add_question rejects when QuestionSet is in terminal state."""
    question_set = QuestionSet(
        id="qs_123",
        assessment_id="assessment_123",
        iteration=1,
        purpose=Purpose.ASSESSMENT,
    )
    question_set.mark_in_progress()
    question_set.mark_completed()

    answer = Answer(id=AnswerId.generate(), text="42")
    question = Question(
        id=QuestionId.generate(),
        text="New question?",
        question_type=QuestionType.STRUCTURED,
        difficulty_level=DifficultyLevel.EASY,
        answer=answer,
    )

    with pytest.raises(InvalidStateTransitionError) as exc_info:
        question_set.add_question(question)
    assert "terminal state" in str(exc_info.value)


def test_question_set_add_question_rejects_duplicate() -> None:
    """add_question rejects duplicate question IDs."""
    question_set = QuestionSet(
        id="qs_123",
        assessment_id="assessment_123",
        iteration=1,
        purpose=Purpose.ASSESSMENT,
    )
    question_set.mark_in_progress()

    qid = QuestionId.generate()
    answer = Answer(id=AnswerId.generate(), text="42")
    question = Question(
        id=qid,
        text="Original question?",
        question_type=QuestionType.STRUCTURED,
        difficulty_level=DifficultyLevel.EASY,
        answer=answer,
    )
    question_set.add_question(question)

    # Attempt to add same question ID again
    duplicate = Question(
        id=qid,
        text="Duplicate question?",
        question_type=QuestionType.STRUCTURED,
        difficulty_level=DifficultyLevel.HARD,
        answer=answer,
    )

    with pytest.raises(ValidationError) as exc_info:
        question_set.add_question(duplicate)
    assert "already exists" in str(exc_info.value)


def test_generation_request_rejects_empty_correlation_id() -> None:
    """GenerationRequest requires non-empty correlation_id."""
    with pytest.raises(ValidationError) as exc_info:
        GenerationRequest(
            id="evt_123",
            workflow_id="wf_123",
            correlation_id="",  # Invalid: must not be empty
            assessment_id="assessment_123",
            validation_result=None,
            iteration=None,
            structured_count=5,
            non_structured_count=3,
            difficulty_level=DifficultyLevel.MEDIUM,
            purpose=Purpose.ASSESSMENT,
        )
    assert "correlation_id cannot be empty" in str(exc_info.value)


def test_generation_request_rejects_empty_workflow_id() -> None:
    """GenerationRequest requires non-empty workflow_id."""
    with pytest.raises(ValidationError) as exc_info:
        GenerationRequest(
            id="evt_123",
            workflow_id="",  # Invalid: must not be empty
            correlation_id="corr_123",
            assessment_id="assessment_123",
            validation_result=None,
            iteration=None,
            structured_count=5,
            non_structured_count=3,
            difficulty_level=DifficultyLevel.MEDIUM,
            purpose=Purpose.ASSESSMENT,
        )
    assert "workflow_id cannot be empty" in str(exc_info.value)
