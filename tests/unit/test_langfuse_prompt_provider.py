"""Unit tests for Langfuse prompt provider."""

from __future__ import annotations

from typing import Any

import pytest
from langfuse.api.commons.errors import (
    NotFoundError,
)

from qna_generation_agent.application.errors import (
    StoragePermanentError,
)
from qna_generation_agent.application.ports.prompt_provider import Prompt
from qna_generation_agent.infrastructure.llm.langfuse_prompt_provider import (
    LangfusePromptProvider,
)


class FakeLangfusePrompt:
    """Fake Langfuse prompt for testing."""

    def __init__(
        self,
        name: str = "Test Prompt",
        version: int = 1,
        prompt_text: str = "Test prompt {variable}",
        chat_messages: list[dict[str, str]] | None = None,
        labels: list[str] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.version = version
        self.prompt = prompt_text
        self.chat_messages = chat_messages
        self.labels = labels or ["production"]
        self.config = config or {}


class FakeLangfuseClient:
    """Fake Langfuse client for testing."""

    def __init__(self, **kwargs: Any) -> None:
        self.public_key = kwargs.get("public_key")
        self.secret_key = kwargs.get("secret_key")
        self.host = kwargs.get("host")
        self.environment = kwargs.get("environment")
        self.prompt_error = kwargs.get("prompt_error")
        self.prompts: dict[str, FakeLangfusePrompt] = {}
        self.flushed = False

    def get_prompt(self, name: str, **kwargs: Any) -> FakeLangfusePrompt:
        if self.prompt_error is not None:
            raise self.prompt_error
        if name not in self.prompts:
            raise NotFoundError(
                {"error": {"message": f"Prompt '{name}' not found"}}
            )
        return self.prompts[name]

    def create_prompt(
        self,
        name: str,
        prompt: str,
        **kwargs: Any,
    ) -> FakeLangfusePrompt:
        lp = FakeLangfusePrompt(name=name, prompt_text=prompt)
        self.prompts[name] = lp
        return lp

    def flush(self) -> None:
        self.flushed = True

    def get_current_trace_id(self) -> str | None:
        return "trace-123"


@pytest.mark.unit
async def test_prompt_provider_handles_auth_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_client(**kwargs: Any) -> FakeLangfuseClient:
        raise RuntimeError("unauthorized: invalid credentials")

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.llm.langfuse_prompt_provider.Langfuse",
        failing_client,
    )

    with pytest.raises(RuntimeError):
        LangfusePromptProvider(
            public_key="pk-invalid",
            secret_key="sk-invalid",
            host="https://langfuse.example",
            environment="test",
        )


@pytest.mark.unit
async def test_prompt_provider_health_check_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeLangfuseClient(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )
    fake_client.create_prompt(
        name="Assessment Generator",
        prompt="Test prompt",
    )

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.llm.langfuse_prompt_provider.Langfuse",
        lambda **kwargs: fake_client,
    )

    provider = LangfusePromptProvider(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )

    result = await provider.health_check()
    assert result is True


@pytest.mark.unit
async def test_prompt_provider_health_check_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeLangfuseClient(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.llm.langfuse_prompt_provider.Langfuse",
        lambda **kwargs: fake_client,
    )

    provider = LangfusePromptProvider(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )

    result = await provider.health_check()
    assert result is False


@pytest.mark.unit
async def test_prompt_provider_convenience_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = FakeLangfuseClient(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )
    fake_client.create_prompt(
        name="Assessment Generator",
        prompt="Generate questions",
    )
    fake_client.create_prompt(
        name="MCQ Explanation Generator",
        prompt="Explain MCQ",
    )
    fake_client.create_prompt(
        name="MCQ Answer Generator",
        prompt="Generate answers",
    )

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.llm.langfuse_prompt_provider.Langfuse",
        lambda **kwargs: fake_client,
    )

    provider = LangfusePromptProvider(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )

    assessment_prompt = await provider.get_assessment_generator_prompt(
        label="production"
    )
    assert assessment_prompt.name == "Assessment Generator"

    explanation_prompt = await provider.get_mcq_explanation_generator_prompt()
    assert explanation_prompt.name == "MCQ Explanation Generator"

    answer_prompt = await provider.get_mcq_answer_generator_prompt(version=1)
    assert answer_prompt.name == "MCQ Answer Generator"


@pytest.mark.unit
def test_prompt_compile_chat_messages() -> None:
    """Test that chat prompts compile correctly."""
    prompt = Prompt(
        name="Chat Prompt",
        version=1,
        prompt_text=None,
        chat_messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello {{name}}"},
        ],
    )

    compiled = prompt.compile(name="World")
    assert isinstance(compiled, list)
    assert compiled[0]["role"] == "system"
    assert compiled[1]["content"] == "Hello World"


@pytest.mark.unit
def test_prompt_compile_text() -> None:
    """Test that text prompts compile correctly."""
    prompt = Prompt(
        name="Text Prompt",
        version=1,
        prompt_text="Hello {{name}}, welcome to {{place}}",
    )

    compiled = prompt.compile(name="Alice", place="Wonderland")
    assert isinstance(compiled, str)
    assert compiled == "Hello Alice, welcome to Wonderland"


@pytest.mark.unit
async def test_prompt_is_chat_prompt() -> None:
    """Test chat prompt detection."""
    chat_prompt = Prompt(
        name="Chat",
        version=1,
        chat_messages=[{"role": "user", "content": "Hello"}],
    )
    assert chat_prompt.is_chat_prompt() is True

    text_prompt = Prompt(
        name="Text",
        version=1,
        prompt_text="Hello",
    )
    assert text_prompt.is_chat_prompt() is False


@pytest.mark.unit
async def test_get_system_prompt_text_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_system_prompt returns prompt_text for text prompts."""
    fake_client = FakeLangfuseClient(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )
    fake_client.create_prompt(
        name="Assessment Generator",
        prompt="You are an expert assessment generator.",
    )

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.llm.langfuse_prompt_provider.Langfuse",
        lambda **kwargs: fake_client,
    )

    provider = LangfusePromptProvider(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )

    system_prompt = await provider.get_system_prompt("Assessment Generator")
    assert system_prompt == "You are an expert assessment generator."


@pytest.mark.unit
async def test_get_system_prompt_chat_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_system_prompt extracts system message from chat prompts."""
    fake_client = FakeLangfuseClient(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )
    fake_client.create_prompt(
        name="Chat Prompt",
        prompt="ignored text",
    )
    fake_client.prompts["Chat Prompt"].chat_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.llm.langfuse_prompt_provider.Langfuse",
        lambda **kwargs: fake_client,
    )

    provider = LangfusePromptProvider(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )

    system_prompt = await provider.get_system_prompt("Chat Prompt")
    assert system_prompt == "You are a helpful assistant."


@pytest.mark.unit
async def test_get_system_prompt_chat_no_system_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test get_system_prompt falls back to first message if no system role."""
    fake_client = FakeLangfuseClient(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )
    fake_client.create_prompt(
        name="Chat Prompt",
        prompt="ignored",
    )
    fake_client.prompts["Chat Prompt"].chat_messages = [
        {"role": "user", "content": "First message content."},
    ]

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.llm.langfuse_prompt_provider.Langfuse",
        lambda **kwargs: fake_client,
    )

    provider = LangfusePromptProvider(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )

    system_prompt = await provider.get_system_prompt("Chat Prompt")
    assert system_prompt == "First message content."


@pytest.mark.unit
async def test_get_system_prompt_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test get_system_prompt raises StoragePermanentError for missing prompt."""
    fake_client = FakeLangfuseClient(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.llm.langfuse_prompt_provider.Langfuse",
        lambda **kwargs: fake_client,
    )

    provider = LangfusePromptProvider(
        public_key="pk-test",
        secret_key="sk-test",
        host="https://langfuse.example",
        environment="test",
    )

    with pytest.raises(StoragePermanentError) as exc_info:
        await provider.get_system_prompt("Non-existent Prompt")

    assert "not found" in str(exc_info.value).lower()
