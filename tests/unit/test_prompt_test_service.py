"""Unit tests for the prompt test service."""

from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from qna_generation_agent.application.ports.llm import LLMProvider
from qna_generation_agent.application.ports.prompt_provider import PromptProvider
from qna_generation_agent.application.services.prompt_test_service import (
    PromptTestService,
)
from qna_generation_agent.infrastructure.llm.prompt_builder import (
    AssessmentGeneratorOutputSchema,
    AssessmentQuestionSchema,
    MCQAnswerGeneratorOutputSchema,
    MCQDistractorExplanationSchema,
    MCQExplanationOutputSchema,
)


class FakeLLMProvider(LLMProvider):
    """Fake LLM provider with controllable latency and responses."""

    response: ClassVar[Any | None] = None
    sleep_seconds: ClassVar[float] = 0.0
    cancelled: ClassVar[bool] = False
    call_count: ClassVar[int] = 0
    last_model_tier: ClassVar[str] = ""

    @property
    def model_id(self) -> str:
        return "fake-model"

    async def generate_structured(
        self,
        *,
        context: Any,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> Any:
        raise NotImplementedError

    async def generate_non_structured(
        self,
        *,
        context: Any,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> Any:
        raise NotImplementedError

    async def generate_with_prompt(
        self,
        *,
        prompt: str,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
        question_type: str,
    ) -> Any:
        raise NotImplementedError

    async def health_check(self) -> bool:
        return True

    async def invoke_with_system_and_user[
        T: BaseModel
    ](
        self,
        system_message: str,
        user_message: str,
        *,
        structured_output_model: type[T],
        model_tier: str = "expensive",
    ) -> T | None:
        type(self).call_count += 1
        type(self).last_model_tier = model_tier
        try:
            if type(self).sleep_seconds > 0:
                await asyncio.sleep(type(self).sleep_seconds)
        except asyncio.CancelledError:
            type(self).cancelled = True
            raise
        assert type(self).response is not None
        return type(self).response


class FakePromptProvider(PromptProvider):
    """Fake prompt provider returning static prompts."""

    @property
    def default_label(self) -> str:
        return "production"

    async def get_system_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        version: int | None = None,
    ) -> str:
        del label, version
        return f"System prompt for {name}"

    async def health_check(self) -> bool:
        return True

    async def shutdown(self) -> None:
        pass


def _reset_fake_llm() -> None:
    FakeLLMProvider.response = AssessmentGeneratorOutputSchema(
        questions=[
            AssessmentQuestionSchema(
                question_id="q-001",
                question_type="structured",
                content="Test question?",
                structured_answer="A",
            )
        ]
    )
    FakeLLMProvider.sleep_seconds = 0.0
    FakeLLMProvider.cancelled = False
    FakeLLMProvider.call_count = 0
    FakeLLMProvider.last_model_tier = ""


def _build_service(timeout_seconds: float) -> PromptTestService:
    return PromptTestService(
        llm_provider=FakeLLMProvider(),
        prompt_provider=FakePromptProvider(),
        timeout_seconds=timeout_seconds,
    )


@pytest.mark.unit
async def test_assessment_generator_returns_success_result() -> None:
    _reset_fake_llm()
    service = _build_service(timeout_seconds=0.01)

    result = await service.test_assessment_generator(
        structured_count=1,
        non_structured_count=0,
        difficulty="medium",
        topics="Grammar",
        chunks=["Chunk 1"],
    )

    assert result.error is None
    assert result.prompt_version == "Assessment Generator@production"
    assert result.result is not None
    assert result.result["questions"][0]["question_id"] == "q-001"
    assert FakeLLMProvider.call_count == 1
    assert FakeLLMProvider.last_model_tier == "expensive"
    assert FakeLLMProvider.cancelled is False


@pytest.mark.unit
async def test_mcq_answer_generator_returns_success_result() -> None:
    _reset_fake_llm()
    FakeLLMProvider.response = MCQAnswerGeneratorOutputSchema(
        question_stem="Test stem?",
        correct_answer=MCQDistractorExplanationSchema(
            option_letter="A",
            option_text="Correct",
            is_correct=True,
            explanation="This is correct",
        ),
        distractors=[
            MCQDistractorExplanationSchema(
                option_letter="B",
                option_text="Distractor 1",
                is_correct=False,
                explanation="Wrong",
            ),
            MCQDistractorExplanationSchema(
                option_letter="C",
                option_text="Distractor 2",
                is_correct=False,
                explanation="Wrong",
            ),
            MCQDistractorExplanationSchema(
                option_letter="D",
                option_text="Distractor 3",
                is_correct=False,
                explanation="Wrong",
            ),
        ],
        grammar_point_tested="test grammar",
        difficulty_justification="test difficulty",
        l1_considerations=["L1 interference 1"],
    )
    service = _build_service(timeout_seconds=0.01)

    result = await service.test_mcq_answer_generator(
        question_text="Question stem",
        topic="past tense",
        difficulty="medium",
        chunk_content="Chinese",
    )

    assert result.error is None
    assert result.prompt_version == "MCQ Answer Generator@production"
    assert result.result is not None
    assert FakeLLMProvider.call_count == 1
    assert FakeLLMProvider.last_model_tier == "cheap"


@pytest.mark.unit
async def test_mcq_explanation_generator_returns_success_result() -> None:
    _reset_fake_llm()
    FakeLLMProvider.response = MCQExplanationOutputSchema(
        question_analysis="Tests present simple",
        option_explanations=[
            MCQDistractorExplanationSchema(
                option_letter="A",
                option_text="Correct",
                is_correct=True,
                explanation="This is correct",
            ),
            MCQDistractorExplanationSchema(
                option_letter="B",
                option_text="Distractor 1",
                is_correct=False,
                explanation="Wrong",
            ),
            MCQDistractorExplanationSchema(
                option_letter="C",
                option_text="Distractor 2",
                is_correct=False,
                explanation="Wrong",
            ),
            MCQDistractorExplanationSchema(
                option_letter="D",
                option_text="Distractor 3",
                is_correct=False,
                explanation="Wrong",
            ),
        ],
        teaching_tip="Emphasize the -s pattern",
        cefr_level="A1",
    )
    service = _build_service(timeout_seconds=0.01)

    result = await service.test_mcq_explanation_generator(
        question_text="Question stem",
        topic="English learners",
        option_a="Option A",
        option_b="Option B",
        option_c="Option C",
        option_d="Option D",
        correct_answer="A",
        chunk_content="English learners",
    )

    assert result.error is None
    assert result.prompt_version == "MCQ Explanation Generator@production"
    assert result.result is not None
    assert FakeLLMProvider.call_count == 1
    assert FakeLLMProvider.last_model_tier == "cheap"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("method_name", "kwargs", "prompt_name"),
    [
        (
            "test_assessment_generator",
            {
                "structured_count": 1,
                "non_structured_count": 0,
                "difficulty": "medium",
                "topics": "Grammar",
                "chunks": ["Chunk 1"],
            },
            "Assessment Generator",
        ),
        (
            "test_mcq_answer_generator",
            {
                "question_text": "Question stem",
                "topic": "past tense",
                "difficulty": "medium",
                "chunk_content": "Chinese",
            },
            "MCQ Answer Generator",
        ),
        (
            "test_mcq_explanation_generator",
            {
                "question_text": "Question stem",
                "topic": "English learners",
                "option_a": "Option A",
                "option_b": "Option B",
                "option_c": "Option C",
                "option_d": "Option D",
                "correct_answer": "A",
                "chunk_content": "English learners",
            },
            "MCQ Explanation Generator",
        ),
    ],
)
async def test_prompt_test_service_applies_timeout(
    method_name: str,
    kwargs: dict[str, Any],
    prompt_name: str,
) -> None:
    _reset_fake_llm()
    FakeLLMProvider.sleep_seconds = 0.1
    service = _build_service(timeout_seconds=0.01)

    method = getattr(service, method_name)
    result = await method(**kwargs)

    assert result.result is None
    assert result.error is not None
    assert "timed out" in result.error.lower()
    assert result.prompt_version == f"{prompt_name}@production"
    assert FakeLLMProvider.call_count == 1
    assert FakeLLMProvider.cancelled is True
