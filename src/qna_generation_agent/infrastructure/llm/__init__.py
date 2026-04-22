"""LLM infrastructure adapters."""

from __future__ import annotations

from qna_generation_agent.infrastructure.llm.local_prompt_provider import (
    LocalPromptProvider,
)
from qna_generation_agent.infrastructure.llm.strands_provider import StrandsLLMProvider

__all__ = ["LocalPromptProvider", "StrandsLLMProvider"]
