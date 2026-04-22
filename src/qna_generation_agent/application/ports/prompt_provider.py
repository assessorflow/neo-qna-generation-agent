"""Prompt provider port for local prompt assets."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from qna_generation_agent.domain.errors import ValidationError


def _format_variable(value: Any) -> str:
    """Format a variable value for prompt substitution.

    Preserves string values as-is. Other types are converted to string.
    """
    if isinstance(value, str):
        return value
    return str(value)


@dataclass(frozen=True, slots=True)
class Prompt:
    """A local prompt asset with version metadata."""

    name: str
    version: str
    system_prompt: str
    user_prompt: str
    aliases: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def version_string(self) -> str:
        """Return formatted version string (name@v<version>)."""
        return f"{self.name}@v{self.version}"

    def compile(self, **variables: Any) -> Prompt:
        """Compile the prompt with variables substituted into both sections."""
        return Prompt(
            name=self.name,
            version=self.version,
            system_prompt=_compile_text(
                self.system_prompt, prompt_name=self.name, version=self.version, **variables
            ),
            user_prompt=_compile_text(
                self.user_prompt, prompt_name=self.name, version=self.version, **variables
            ),
            aliases=self.aliases,
            metadata=self.metadata,
        )


def _detect_unresolved(content: str) -> list[str]:
    """Detect unresolved {{variable}} placeholders in content."""
    return re.findall(r"\{\{[\w.]+\}\}", content)


def _compile_text(content: str, *, prompt_name: str, version: str, **variables: Any) -> str:
    """Compile a prompt section with variable substitution."""
    result = content
    for key, value in variables.items():
        result = result.replace(f"{{{{{key}}}}}", _format_variable(value))
    unresolved = _detect_unresolved(result)
    if unresolved:
        raise ValidationError(
            "Unresolved prompt variables after compilation",
            prompt_name=prompt_name,
            prompt_version=version,
            unresolved_variables=unresolved,
        )
    return result


class PromptProvider(ABC):
    """Port for fetching local prompt assets."""

    ITEM_WRITER_PROMPT_NAME = "item-writer"
    OPTIONS_ONLY_WRITER_PROMPT_NAME = "options-only-writer"
    FEEDBACK_WRITER_PROMPT_NAME = "feedback-writer"

    @abstractmethod
    async def get_prompt(self, name: str) -> Prompt:
        """Fetch a local prompt asset by name or alias."""

    async def get_system_prompt(self, name: str) -> str:
        """Return the system prompt text for a prompt asset."""
        return (await self.get_prompt(name)).system_prompt

    async def get_user_prompt(self, name: str) -> str:
        """Return the user prompt text for a prompt asset."""
        return (await self.get_prompt(name)).user_prompt

    async def get_item_writer_prompt(self) -> Prompt:
        """Return the item writer prompt asset."""
        return await self.get_prompt(self.ITEM_WRITER_PROMPT_NAME)

    async def get_options_only_writer_prompt(self) -> Prompt:
        """Return the options-only writer prompt asset."""
        return await self.get_prompt(self.OPTIONS_ONLY_WRITER_PROMPT_NAME)

    async def get_feedback_writer_prompt(self) -> Prompt:
        """Return the feedback writer prompt asset."""
        return await self.get_prompt(self.FEEDBACK_WRITER_PROMPT_NAME)

    @abstractmethod
    async def health_check(self) -> bool:
        """Check that the prompt provider is ready."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Close any resources held by the provider."""
