"""Unit tests for the prompt provider."""

from __future__ import annotations

import pytest

from qna_generation_agent.application.ports.prompt_provider import Prompt
from qna_generation_agent.domain.errors import ValidationError


@pytest.mark.unit
def test_prompt_is_chat_detection() -> None:
    """Test that chat prompts are correctly identified."""
    text_prompt = Prompt(
        name="test-text",
        version=1,
        prompt_text="Hello {{name}}!",
        chat_messages=None,
    )
    assert not text_prompt.is_chat_prompt()

    chat_prompt = Prompt(
        name="test-chat",
        version=1,
        prompt_text=None,
        chat_messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
        ],
    )
    assert chat_prompt.is_chat_prompt()


@pytest.mark.unit
def test_prompt_compile_text() -> None:
    """Test compiling a text prompt with variables."""
    prompt = Prompt(
        name="test",
        version=1,
        prompt_text="Hello {{name}}! Your score is {{score}}.",
    )

    result = prompt.compile(name="Alice", score=100)

    assert isinstance(result, str)
    assert result == "Hello Alice! Your score is 100."


@pytest.mark.unit
def test_prompt_compile_chat() -> None:
    """Test compiling a chat prompt with variables."""
    prompt = Prompt(
        name="test",
        version=1,
        chat_messages=[
            {"role": "system", "content": "You are {{role}}."},
            {"role": "user", "content": "Help me with {{task}}."},
        ],
    )

    result = prompt.compile(role="an expert", task="Python")

    assert isinstance(result, list)
    assert result[0]["role"] == "system"
    assert result[0]["content"] == "You are an expert."
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "Help me with Python."


@pytest.mark.unit
def test_prompt_compile_no_variables() -> None:
    """Test compiling a prompt without variables."""
    prompt = Prompt(
        name="test",
        version=1,
        prompt_text="Hello World!",
    )

    result = prompt.compile()

    assert result == "Hello World!"


@pytest.mark.unit
def test_prompt_empty_chat_messages() -> None:
    """Test that empty chat_messages list is treated as text prompt."""
    prompt = Prompt(
        name="test",
        version=1,
        prompt_text="Hello!",
        chat_messages=[],
    )

    # Empty list should not be considered a chat prompt
    assert not prompt.is_chat_prompt()
    result = prompt.compile()
    assert result == "Hello!"


@pytest.mark.unit
def test_prompt_compile_rejects_unresolved_variables_text() -> None:
    """Text prompt compilation fails on unresolved placeholders."""
    prompt = Prompt(
        name="test",
        version=1,
        prompt_text="Hello {{name}}! Missing: {{missing_var}}.",
    )

    with pytest.raises(ValidationError) as exc_info:
        prompt.compile(name="Alice")
    assert "Unresolved prompt variables" in str(exc_info.value)
    assert "missing_var" in str(exc_info.value)


@pytest.mark.unit
def test_prompt_compile_rejects_unresolved_variables_chat() -> None:
    """Chat prompt compilation fails on unresolved placeholders."""
    prompt = Prompt(
        name="test",
        version=1,
        chat_messages=[
            {"role": "system", "content": "You are {{role}}."},
            {"role": "user", "content": "Missing: {{unresolved}}."},
        ],
    )

    with pytest.raises(ValidationError) as exc_info:
        prompt.compile(role="an expert")
    assert "Unresolved prompt variables" in str(exc_info.value)
    assert "unresolved" in str(exc_info.value)


@pytest.mark.unit
def test_prompt_compile_rejects_null_text() -> None:
    """Text prompt compilation fails when prompt_text is None."""
    prompt = Prompt(
        name="test",
        version=1,
        prompt_text=None,
    )

    with pytest.raises(ValidationError) as exc_info:
        prompt.compile()
    assert "prompt_text is None" in str(exc_info.value)


@pytest.mark.unit
def test_prompt_compile_rejects_null_chat_messages() -> None:
    """Chat prompt compilation fails when chat_messages is None.

    Note: When both prompt_text and chat_messages are None, the prompt
    is treated as text (is_chat_prompt returns False), so the error
    comes from _compile_text instead.
    """
    prompt = Prompt(
        name="test",
        version=1,
        prompt_text=None,
        chat_messages=None,
    )

    with pytest.raises(ValidationError) as exc_info:
        prompt.compile()
    # When both are None, is_chat_prompt() returns False, so _compile_text runs
    assert "prompt_text is None" in str(exc_info.value)


@pytest.mark.unit
def test_prompt_compile_preserves_string_values() -> None:
    """String values are substituted without coercion."""
    prompt = Prompt(
        name="test",
        version=1,
        prompt_text="Value: {{value}}",
    )

    result = prompt.compile(value="hello world")
    assert result == "Value: hello world"


@pytest.mark.unit
def test_prompt_compile_converts_non_string_values() -> None:
    """Non-string values are converted to string."""
    prompt = Prompt(
        name="test",
        version=1,
        prompt_text="Count: {{count}}, Active: {{active}}",
    )

    result = prompt.compile(count=42, active=True)
    assert result == "Count: 42, Active: True"
