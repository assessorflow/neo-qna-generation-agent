"""Unit tests for the prompt test service."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from qna_generation_agent.application.ports.prompt_provider import Prompt
from qna_generation_agent.application.services import prompt_test_service as module
from qna_generation_agent.application.services.prompt_test_service import (
    PromptTestService,
)
from qna_generation_agent.infrastructure.llm.prompt_builder import (
    AssessmentGeneratorOutputSchema,
    AssessmentQuestionSchema,
)


class FakeOpenAIModel:
    """Fake model constructor used to avoid external dependencies."""

    def __init__(
        self,
        client_args: dict[str, str | None],
        model_id: str,
        params: dict[str, object] | None = None,
    ) -> None:
        self.client_args = client_args
        self.model_id = model_id
        self.params = params or {}


class FakeAgent:
    """Fake Strands agent with controllable latency."""

    response: ClassVar[Any | None] = None
    sleep_seconds: ClassVar[float] = 0.0
    cancelled: ClassVar[bool] = False
    call_count: ClassVar[int] = 0

    def __init__(self, model: object, system_prompt: str) -> None:
        del model, system_prompt

    async def invoke_async(
        self, prompt: str, structured_output_model: object
    ) -> object:
        del prompt, structured_output_model
        type(self).call_count += 1
        try:
            if type(self).sleep_seconds > 0:
                await asyncio.sleep(type(self).sleep_seconds)
        except asyncio.CancelledError:
            type(self).cancelled = True
            raise
        assert type(self).response is not None
        return type(self).response


class FakePromptProvider:
    """Fake prompt provider returning static prompts."""

    def __init__(self) -> None:
        self.prompts = {
            "Assessment Generator": Prompt(
                name="Assessment Generator",
                version=3,
                prompt_text="Test prompt",
            ),
            "MCQ Answer Generator": Prompt(
                name="MCQ Answer Generator",
                version=3,
                prompt_text="Test prompt",
            ),
            "MCQ Explanation Generator": Prompt(
                name="MCQ Explanation Generator",
                version=3,
                prompt_text="Test prompt",
            ),
        }

    @property
    def default_label(self) -> str:
        return "production"

    async def get_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        version: int | None = None,
    ) -> Prompt:
        del label, version
        return self.prompts[name]


def _reset_fake_agent() -> None:
    FakeAgent.response = SimpleNamespace(
        structured_output=AssessmentGeneratorOutputSchema(
            questions=[
                AssessmentQuestionSchema(
                    question_id="q-001",
                    question_type="structured",
                    content="Test question?",
                    structured_answer="A",
                )
            ]
        )
    )
    FakeAgent.sleep_seconds = 0.0
    FakeAgent.cancelled = False
    FakeAgent.call_count = 0


def _build_service(
    monkeypatch: pytest.MonkeyPatch,
    timeout_seconds: float,
) -> PromptTestService:
    monkeypatch.setattr(module, "Agent", FakeAgent)
    monkeypatch.setattr(module, "OpenAIModel", FakeOpenAIModel)
    return PromptTestService(
        model_id="gpt-4o-mini",
        api_key="test-key",
        base_url=None,
        timeout_seconds=timeout_seconds,
        prompt_provider=FakePromptProvider(),
    )


@pytest.mark.unit
async def test_assessment_generator_returns_success_result(
    monkeypatch: pytest.MonkeyPatch,
    ) -> None:
    _reset_fake_agent()
    service = _build_service(monkeypatch, timeout_seconds=0.01)

    result = await service.test_assessment_generator(
        structured_count=1,
        non_structured_count=0,
        difficulty="medium",
        topics="Grammar",
        chunks=["Chunk 1"],
    )

    assert result.error is None
    assert result.prompt_version == "Assessment Generator@v3"
    assert result.result is not None
    assert result.result["questions"][0]["question_id"] == "q-001"
    assert FakeAgent.call_count == 1
    assert FakeAgent.cancelled is False


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
                "question_stem": "Question stem",
                "grammar_target": "past tense",
                "difficulty": "medium",
                "l1_background": "Chinese",
            },
            "MCQ Answer Generator",
        ),
        (
            "test_mcq_explanation_generator",
            {
                "question": "Question stem",
                "options": {
                    "A": "Option A",
                    "B": "Option B",
                    "C": "Option C",
                    "D": "Option D",
                },
                "correct_answer": "A",
                "target_audience": "English learners",
            },
            "MCQ Explanation Generator",
        ),
    ],
)
async def test_prompt_test_service_applies_timeout(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    kwargs: dict[str, Any],
    prompt_name: str,
) -> None:
    _reset_fake_agent()
    FakeAgent.sleep_seconds = 0.1
    service = _build_service(monkeypatch, timeout_seconds=0.01)

    method = getattr(service, method_name)
    result = await method(**kwargs)

    assert result.result is None
    assert result.error is not None
    assert "timed out" in result.error.lower()
    assert result.prompt_version == f"{prompt_name}@v3"
    assert FakeAgent.call_count == 1
    assert FakeAgent.cancelled is True
