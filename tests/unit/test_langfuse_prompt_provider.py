"""Unit tests for the local prompt provider."""

from __future__ import annotations

import pytest

from qna_generation_agent.application.errors import PermanentError
from qna_generation_agent.infrastructure.llm.local_prompt_provider import (
    LocalPromptProvider,
)


@pytest.mark.unit
def test_local_prompt_provider_loads_packaged_assets() -> None:
    provider = LocalPromptProvider()

    assert provider.ITEM_WRITER_PROMPT_NAME == "item-writer"
    assert provider.OPTIONS_ONLY_WRITER_PROMPT_NAME == "options-only-writer"
    assert provider.FEEDBACK_WRITER_PROMPT_NAME == "feedback-writer"


@pytest.mark.unit
async def test_local_prompt_provider_resolves_aliases() -> None:
    provider = LocalPromptProvider()

    item_prompt = await provider.get_item_writer_prompt()
    options_prompt = await provider.get_options_only_writer_prompt()
    feedback_prompt = await provider.get_feedback_writer_prompt()

    assert item_prompt.name == "item-writer"
    assert item_prompt.version_string == "item-writer@v1"
    assert "Assessment Generator" in item_prompt.aliases
    assert item_prompt.metadata["description"].startswith(
        "Generate a mixed assessment"
    )

    assert options_prompt.name == "options-only-writer"
    assert options_prompt.version_string == "options-only-writer@v1"
    assert "MCQ Answer Generator" in options_prompt.aliases

    assert feedback_prompt.name == "feedback-writer"
    assert feedback_prompt.version_string == "feedback-writer@v1"
    assert "MCQ Explanation Generator" in feedback_prompt.aliases


@pytest.mark.unit
async def test_local_prompt_provider_compiles_prompt_sections() -> None:
    provider = LocalPromptProvider()

    prompt = await provider.get_item_writer_prompt()
    compiled = prompt.compile(
        structured_count=2,
        non_structured_count=1,
        difficulty="medium",
        topics="Grammar",
        chunks="[Chunk 1] Content",
    )

    assert compiled.name == "item-writer"
    assert "structured_count: 2" in compiled.user_prompt
    assert "Grammar" in compiled.user_prompt
    assert "[Chunk 1] Content" in compiled.user_prompt
    assert compiled.system_prompt.startswith(
        "You are an expert ELP item writer for AssessorFlow Singapore."
    )


@pytest.mark.unit
async def test_local_prompt_provider_system_prompt_lookup() -> None:
    provider = LocalPromptProvider()

    assessment_prompt = await provider.get_system_prompt("Assessment Generator")
    mcq_prompt = await provider.get_system_prompt("MCQ Answer Generator")

    assert assessment_prompt.startswith(
        "You are an expert ELP item writer for AssessorFlow Singapore."
    )
    assert mcq_prompt.startswith(
        "You are an expert item writer for Singapore ELP assessments."
    )


@pytest.mark.unit
async def test_local_prompt_provider_raises_for_missing_prompt() -> None:
    provider = LocalPromptProvider()

    with pytest.raises(PermanentError):
        await provider.get_prompt("missing-prompt")
