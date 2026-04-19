"""Unit tests for the Strands LLM provider."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from qna_generation_agent.application.dto import AssessmentContext
from qna_generation_agent.application.errors import LLMPermanentError, LLMTransientError
from qna_generation_agent.domain.enums import DifficultyLevel, QuestionType
from qna_generation_agent.infrastructure.llm import strands_provider as provider_module
from qna_generation_agent.infrastructure.llm.prompt_builder import (
    AssessmentGeneratorOutputSchema,
    GeneratedQuestionBatchSchema,
)
from qna_generation_agent.infrastructure.llm.strands_provider import StrandsLLMProvider


class FakeAgent:
    """Fake Strands agent with configurable async behavior."""

    responses: ClassVar[list[Any]] = []

    def __init__(
        self, model: object, system_prompt: str, callback_handler: object = None
    ) -> None:
        del model, callback_handler
        self.system_prompt = system_prompt

    async def invoke_async(
        self, prompt: str, structured_output_model: object
    ) -> object:
        del prompt, structured_output_model
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, tuple) and response[0] == "sleep":
            await asyncio.sleep(response[1])
            return SimpleNamespace(structured_output=None)
        return response


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


def _provider(monkeypatch: pytest.MonkeyPatch) -> StrandsLLMProvider:
    monkeypatch.setattr(provider_module, "Agent", FakeAgent)
    monkeypatch.setattr(provider_module, "OpenAIModel", FakeOpenAIModel)
    return StrandsLLMProvider(
        model_provider="openai",
        model_id="gpt-4o-mini",
        api_key="test-key",
        base_url=None,
        timeout_seconds=1,
    )


def _context() -> AssessmentContext:
    return AssessmentContext(
        assessment_id="assessment_123",
        title="Assessment Title",
        topic_ids=["topic_a"],
        chunks=["chunk one"],
    )


async def test_generate_structured_returns_question_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeAgent.responses = [
        SimpleNamespace(
            structured_output=GeneratedQuestionBatchSchema.model_validate(
                {
                    "questions": [
                        {
                            "question_text": "Question 1",
                            "answer_text": "Answer 1",
                            "explanation": "Because",
                            "references": ["chunk-1"],
                            "topic_id": "topic_a",
                            "metadata": {"kind": "demo"},
                        }
                    ]
                }
            )
        ),
    ]
    provider = _provider(monkeypatch)

    batch = await provider.generate_structured(
        context=_context(),
        count=1,
        difficulty_level=DifficultyLevel.MEDIUM,
        correlation_id="corr_123",
    )

    assert batch.model_name == "gpt-4o-mini"
    assert batch.prompt_version == "structured_v2"
    assert batch.questions[0].question_type is QuestionType.STRUCTURED


def test_provider_rejects_unsupported_backend() -> None:
    with pytest.raises(LLMPermanentError):
        StrandsLLMProvider(
            model_provider="bedrock",
            model_id="model",
            api_key="key",
            base_url=None,
            timeout_seconds=1,
        )


async def test_generate_raises_when_count_is_wrong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeAgent.responses = [
        SimpleNamespace(
            structured_output=GeneratedQuestionBatchSchema.model_validate(
                {"questions": []}
            )
        ),
    ]
    provider = _provider(monkeypatch)

    with pytest.raises(LLMPermanentError):
        await provider.generate_structured(
            context=_context(),
            count=1,
            difficulty_level=DifficultyLevel.MEDIUM,
            correlation_id="corr_123",
        )


async def test_generate_maps_timeout_to_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeAgent.responses = [("sleep", 0.05)]
    provider = _provider(monkeypatch)
    provider._timeout_seconds = 0

    with pytest.raises(LLMTransientError):
        await provider.generate_non_structured(
            context=_context(),
            count=1,
            difficulty_level=DifficultyLevel.MEDIUM,
            correlation_id="corr_123",
        )


@pytest.mark.unit
async def test_generate_with_prompt_uses_compiled_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that generate_with_prompt uses the provided prompt text."""
    from qna_generation_agent.infrastructure.llm.prompt_builder import (
        GeneratedQuestionSchema,
    )

    FakeAgent.responses = [
        SimpleNamespace(
            structured_output=GeneratedQuestionBatchSchema(
                questions=[
                    GeneratedQuestionSchema(
                        question_text="Test question?",
                        answer_text="Test answer",
                        explanation="Test explanation",
                        references=["ref1"],
                        topic_id="topic_1",
                        metadata={},
                    )
                ]
            )
        )
    ]
    provider = _provider(monkeypatch)

    batch = await provider.generate_with_prompt(
        prompt="Compiled prompt from Langfuse",
        count=1,
        difficulty_level=DifficultyLevel.MEDIUM,
        correlation_id="corr_123",
        question_type="structured",
    )

    assert len(batch.questions) == 1
    assert batch.questions[0].question_text == "Test question?"
    assert batch.prompt_version == "langfuse_managed"


@pytest.mark.unit
async def test_generate_with_prompt_handles_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that generate_with_prompt raises LLMTransientError on timeout."""
    FakeAgent.responses = [("sleep", 0.05)]
    provider = _provider(monkeypatch)
    provider._timeout_seconds = 0

    with pytest.raises(LLMTransientError):
        await provider.generate_with_prompt(
            prompt="Compiled prompt",
            count=1,
            difficulty_level=DifficultyLevel.MEDIUM,
            correlation_id="corr_123",
            question_type="structured",
        )


@pytest.mark.unit
async def test_shutdown_closes_health_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that shutdown closes the AsyncOpenAI health client."""
    provider = _provider(monkeypatch)

    # Create a mock health client
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    provider._health_client = mock_client

    await provider.shutdown()

    mock_client.close.assert_called_once()


@pytest.mark.unit
async def test_shutdown_handles_errors_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that shutdown handles client close errors gracefully."""
    provider = _provider(monkeypatch)

    # Create a mock health client that raises on close
    mock_client = MagicMock()
    mock_client.close = AsyncMock(side_effect=RuntimeError("Close failed"))
    provider._health_client = mock_client

    # Should not raise
    await provider.shutdown()


@pytest.mark.unit
async def test_health_check_returns_true_when_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that health_check returns True when models.list succeeds."""
    provider = _provider(monkeypatch)

    # Create a mock health client
    mock_client = MagicMock()
    mock_models = MagicMock()
    mock_models.list = AsyncMock(return_value=[])
    mock_client.models = mock_models
    provider._health_client = mock_client

    result = await provider.health_check()

    assert result is True
    mock_models.list.assert_called_once()


@pytest.mark.unit
async def test_health_check_returns_false_on_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that health_check returns False on authentication error."""
    from openai import AuthenticationError

    provider = _provider(monkeypatch)

    # Create a mock health client that raises AuthenticationError
    mock_client = MagicMock()
    mock_models = MagicMock()
    # AuthenticationError requires response and body kwargs
    auth_error = AuthenticationError(
        "Invalid API key",
        response=MagicMock(),
        body={"error": {"message": "Invalid API key"}},
    )
    mock_models.list = AsyncMock(side_effect=auth_error)
    mock_client.models = mock_models
    provider._health_client = mock_client

    result = await provider.health_check()

    assert result is False


@pytest.mark.unit
async def test_generate_maps_rate_limit_to_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that rate limit errors are mapped to LLMTransientError."""
    FakeAgent.responses = [RuntimeError("Rate limit exceeded, retry after 30s")]
    provider = _provider(monkeypatch)

    with pytest.raises(LLMTransientError) as exc_info:
        await provider.generate_structured(
            context=_context(),
            count=1,
            difficulty_level=DifficultyLevel.MEDIUM,
            correlation_id="corr_123",
        )

    assert exc_info.value.retry_after_seconds == 30


@pytest.mark.unit
async def test_generate_maps_generic_error_to_permanent_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that generic errors are mapped to LLMPermanentError."""
    FakeAgent.responses = [RuntimeError("Some other error")]
    provider = _provider(monkeypatch)

    with pytest.raises(LLMPermanentError):
        await provider.generate_structured(
            context=_context(),
            count=1,
            difficulty_level=DifficultyLevel.MEDIUM,
            correlation_id="corr_123",
        )


# ============================================================================
# Invalid Structured Output Tests
# ============================================================================


@pytest.mark.unit
async def test_generate_raises_on_none_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that generate raises error when structured output is None."""
    FakeAgent.responses = [SimpleNamespace(structured_output=None)]
    provider = _provider(monkeypatch)

    with pytest.raises(LLMPermanentError, match="Structured output was not returned"):
        await provider.generate_structured(
            context=_context(),
            count=1,
            difficulty_level=DifficultyLevel.MEDIUM,
            correlation_id="corr_123",
        )


@pytest.mark.unit
async def test_generate_raises_on_invalid_output_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that generate raises error when output type is wrong."""
    FakeAgent.responses = [SimpleNamespace(structured_output={"invalid": "data"})]
    provider = _provider(monkeypatch)

    with pytest.raises(LLMPermanentError, match="Structured output was not returned"):
        await provider.generate_structured(
            context=_context(),
            count=1,
            difficulty_level=DifficultyLevel.MEDIUM,
            correlation_id="corr_123",
        )


# ============================================================================
# invoke_with_system_and_user Tests
# ============================================================================


@pytest.mark.unit
async def test_invoke_with_system_and_user_returns_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path: invoke with system_message, user_message, and schema returns parsed structured output."""
    FakeAgent.responses = [
        SimpleNamespace(
            structured_output=AssessmentGeneratorOutputSchema.model_validate(
                {
                    "questions": [
                        {
                            "question_id": "q-1",
                            "question_type": "structured",
                            "question_text": "What is the capital?",
                            "answer_text": "Paris",
                            "metadata": {
                                "question_type": "structured",
                                "options": {
                                    "A": "London",
                                    "B": "Paris",
                                    "C": "Berlin",
                                    "D": "Madrid",
                                },
                                "source_chunk_ids": ["chunk-1"],
                                "difficulty": "easy",
                                "topic": "Geography",
                            },
                        }
                    ]
                }
            )
        ),
    ]
    provider = _provider(monkeypatch)

    result = await provider.invoke_with_system_and_user(
        system_message="You are a test generator",
        user_message="Generate one question",
        structured_output_model=AssessmentGeneratorOutputSchema,
    )

    assert result is not None
    assert len(result.questions) == 1
    assert result.questions[0].question_text == "What is the capital?"


@pytest.mark.unit
async def test_invoke_with_system_and_user_uses_cheap_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When model_tier='cheap', uses the cheap model."""
    FakeAgent.responses = [
        SimpleNamespace(
            structured_output=AssessmentGeneratorOutputSchema.model_validate(
                {
                    "questions": [
                        {
                            "question_id": "q-1",
                            "question_type": "structured",
                            "question_text": "Q?",
                            "answer_text": "A",
                            "metadata": {
                                "question_type": "structured",
                                "options": {
                                    "A": "A",
                                    "B": "B",
                                    "C": "C",
                                    "D": "D",
                                },
                                "source_chunk_ids": ["c1"],
                                "difficulty": "easy",
                                "topic": "Test",
                            },
                        }
                    ]
                }
            )
        ),
    ]
    monkeypatch.setattr(provider_module, "Agent", FakeAgent)
    monkeypatch.setattr(provider_module, "OpenAIModel", FakeOpenAIModel)
    provider = StrandsLLMProvider(
        model_provider="openai",
        model_id="gpt-4o-mini",
        api_key="test-key",
        base_url=None,
        timeout_seconds=1,
        cheap_model_id="gpt-3.5-turbo",
        expensive_model_id="gpt-4o",
    )

    await provider.invoke_with_system_and_user(
        system_message="System prompt",
        user_message="User prompt",
        structured_output_model=AssessmentGeneratorOutputSchema,
        model_tier="cheap",
    )

    cache_keys = list(provider._agent_cache.keys())
    assert len(cache_keys) == 1
    assert cache_keys[0].startswith("gpt-3.5-turbo:")


@pytest.mark.unit
async def test_invoke_with_system_and_user_uses_expensive_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When model_tier='expensive', uses the expensive model."""
    FakeAgent.responses = [
        SimpleNamespace(
            structured_output=AssessmentGeneratorOutputSchema.model_validate(
                {
                    "questions": [
                        {
                            "question_id": "q-1",
                            "question_type": "structured",
                            "question_text": "Q?",
                            "answer_text": "A",
                            "metadata": {
                                "question_type": "structured",
                                "options": {
                                    "A": "A",
                                    "B": "B",
                                    "C": "C",
                                    "D": "D",
                                },
                                "source_chunk_ids": ["c1"],
                                "difficulty": "easy",
                                "topic": "Test",
                            },
                        }
                    ]
                }
            )
        ),
    ]
    monkeypatch.setattr(provider_module, "Agent", FakeAgent)
    monkeypatch.setattr(provider_module, "OpenAIModel", FakeOpenAIModel)
    provider = StrandsLLMProvider(
        model_provider="openai",
        model_id="gpt-4o-mini",
        api_key="test-key",
        base_url=None,
        timeout_seconds=1,
        cheap_model_id="gpt-3.5-turbo",
        expensive_model_id="gpt-4o",
    )

    await provider.invoke_with_system_and_user(
        system_message="System prompt",
        user_message="User prompt",
        structured_output_model=AssessmentGeneratorOutputSchema,
        model_tier="expensive",
    )

    cache_keys = list(provider._agent_cache.keys())
    assert len(cache_keys) == 1
    assert cache_keys[0].startswith("gpt-4o:")


@pytest.mark.unit
async def test_invoke_with_system_and_user_uses_default_tier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no tier specified, uses the default (expensive) model."""
    FakeAgent.responses = [
        SimpleNamespace(
            structured_output=AssessmentGeneratorOutputSchema.model_validate(
                {
                    "questions": [
                        {
                            "question_id": "q-1",
                            "question_type": "structured",
                            "question_text": "Q?",
                            "answer_text": "A",
                            "metadata": {
                                "question_type": "structured",
                                "options": {
                                    "A": "A",
                                    "B": "B",
                                    "C": "C",
                                    "D": "D",
                                },
                                "source_chunk_ids": ["c1"],
                                "difficulty": "easy",
                                "topic": "Test",
                            },
                        }
                    ]
                }
            )
        ),
    ]
    monkeypatch.setattr(provider_module, "Agent", FakeAgent)
    monkeypatch.setattr(provider_module, "OpenAIModel", FakeOpenAIModel)
    provider = StrandsLLMProvider(
        model_provider="openai",
        model_id="gpt-4o-mini",
        api_key="test-key",
        base_url=None,
        timeout_seconds=1,
        cheap_model_id="gpt-3.5-turbo",
        expensive_model_id="gpt-4o",
    )

    await provider.invoke_with_system_and_user(
        system_message="System prompt",
        user_message="User prompt",
        structured_output_model=AssessmentGeneratorOutputSchema,
    )

    cache_keys = list(provider._agent_cache.keys())
    assert len(cache_keys) == 1
    assert cache_keys[0].startswith("gpt-4o:")


@pytest.mark.unit
async def test_invoke_with_system_and_user_raises_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When agent takes too long, raises LLMTransientError."""
    FakeAgent.responses = [("sleep", 0.05)]
    provider = _provider(monkeypatch)
    provider._timeout_seconds = 0

    with pytest.raises(LLMTransientError):
        await provider.invoke_with_system_and_user(
            system_message="System prompt",
            user_message="User prompt",
            structured_output_model=AssessmentGeneratorOutputSchema,
        )


@pytest.mark.unit
async def test_invoke_with_system_and_user_raises_on_invalid_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When result lacks structured_output attribute, raises LLMPermanentError."""
    FakeAgent.responses = [SimpleNamespace()]
    provider = _provider(monkeypatch)

    with pytest.raises(LLMPermanentError):
        await provider.invoke_with_system_and_user(
            system_message="System prompt",
            user_message="User prompt",
            structured_output_model=AssessmentGeneratorOutputSchema,
        )
