"""Unit tests for the prompt model."""

from __future__ import annotations

import pytest

from qna_generation_agent.application.ports.prompt_provider import Prompt
from qna_generation_agent.domain.errors import ValidationError


@pytest.mark.unit
def test_prompt_compile_text_sections() -> None:
    """Test compiling a prompt with variables in both sections."""
    prompt = Prompt(
        name="item-writer",
        version="1",
        system_prompt="You are {{role}}.",
        user_prompt="Write {{count}} items for {{topic}}.",
    )

    compiled = prompt.compile(role="an expert", count=3, topic="grammar")

    assert compiled.system_prompt == "You are an expert."
    assert compiled.user_prompt == "Write 3 items for grammar."


@pytest.mark.unit
def test_prompt_compile_rejects_unresolved_variables() -> None:
    """Compilation fails when placeholders remain in either section."""
    prompt = Prompt(
        name="item-writer",
        version="1",
        system_prompt="You are {{role}}.",
        user_prompt="Write {{count}} items for {{topic}}.",
    )

    with pytest.raises(ValidationError) as exc_info:
        prompt.compile(role="an expert", count=3)

    assert "Unresolved prompt variables" in str(exc_info.value)
    assert "topic" in str(exc_info.value)


@pytest.mark.unit
def test_prompt_compile_preserves_string_values() -> None:
    """String values are substituted without coercion."""
    prompt = Prompt(
        name="item-writer",
        version="1",
        system_prompt="System",
        user_prompt="Value: {{value}}",
    )

    compiled = prompt.compile(value="hello world")
    assert compiled.user_prompt == "Value: hello world"


@pytest.mark.unit
def test_prompt_compile_converts_non_string_values() -> None:
    """Non-string values are converted to string."""
    prompt = Prompt(
        name="item-writer",
        version="1",
        system_prompt="System",
        user_prompt="Count: {{count}}, Active: {{active}}",
    )

    compiled = prompt.compile(count=42, active=True)
    assert compiled.user_prompt == "Count: 42, Active: True"


@pytest.mark.unit
def test_prompt_version_string_property() -> None:
    """version_string returns formatted name@vversion."""
    prompt = Prompt(
        name="item-writer",
        version="5",
        system_prompt="System",
        user_prompt="User",
    )

    assert prompt.version_string == "item-writer@v5"
