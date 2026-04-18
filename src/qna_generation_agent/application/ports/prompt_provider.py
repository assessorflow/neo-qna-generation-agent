"""Prompt provider port for fetching prompts from Langfuse."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
    """A fetched prompt with metadata.

    This dataclass represents a complete prompt from a prompt management
    system such as Langfuse. It may contain either text or chat messages.

    For static system prompts, use PromptProvider.get_system_prompt(), which
    returns plain text directly.
    """

    name: str
    version: int
    prompt_text: str | None = None
    chat_messages: list[dict[str, Any]] | None = None
    labels: list[str] | None = None
    config: dict[str, Any] | None = None

    def is_chat_prompt(self) -> bool:
        """Check if this is a chat-style prompt."""
        return self.chat_messages is not None and len(self.chat_messages) > 0

    @property
    def version_string(self) -> str:
        """Return formatted version string (name@version)."""
        return f"{self.name}@v{self.version}"

    @staticmethod
    def compiled_to_string(compiled: str | list[dict[str, Any]]) -> str:
        """Convert compiled prompt (text or chat format) to string.

        Args:
            compiled: Either a string (text prompt) or list of message dicts
                     (chat prompt).

        Returns:
            String representation of the compiled prompt.
        """
        if isinstance(compiled, str):
            return compiled
        elif isinstance(compiled, list) and len(compiled) > 0:
            return "\n\n".join(
                msg.get("content", "") for msg in compiled if isinstance(msg, dict)
            )
        return str(compiled)

    def compile(self, **variables: Any) -> str | list[dict[str, Any]]:
        """Compile the prompt with variables.

        For text prompts: returns compiled string.
        For chat prompts: returns list of messages with variables substituted.

        Raises:
            ValidationError: If required prompt content is missing or
                unresolved placeholders remain after compilation.
        """
        if self.is_chat_prompt():
            return self._compile_chat(**variables)
        return self._compile_text(**variables)

    def _detect_unresolved(self, content: str) -> list[str]:
        """Detect unresolved {{variable}} placeholders in content."""
        pattern = r"\{\{\w+\}\}"
        return re.findall(pattern, content)

    def _compile_text(self, **variables: Any) -> str:
        """Compile text prompt with variable substitution.

        Raises:
            ValidationError: If prompt_text is None or unresolved placeholders remain.
        """
        if self.prompt_text is None:
            raise ValidationError(
                "Cannot compile prompt: prompt_text is None",
                prompt_name=self.name,
                prompt_version=self.version,
            )
        result = self.prompt_text
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", _format_variable(value))
        unresolved = self._detect_unresolved(result)
        if unresolved:
            raise ValidationError(
                "Unresolved prompt variables after compilation",
                prompt_name=self.name,
                prompt_version=self.version,
                unresolved_variables=unresolved,
            )
        return result

    def _compile_chat(self, **variables: Any) -> list[dict[str, Any]]:
        """Compile chat prompt with variable substitution.

        Raises:
            ValidationError: If chat_messages is None or unresolved placeholders remain.
        """
        if self.chat_messages is None:
            raise ValidationError(
                "Cannot compile prompt: chat_messages is None",
                prompt_name=self.name,
                prompt_version=self.version,
            )
        compiled: list[dict[str, Any]] = []
        unresolved_all: list[str] = []
        for msg in self.chat_messages:
            content = msg.get("content", "")
            for key, value in variables.items():
                placeholder = f"{{{{{key}}}}}"
                if placeholder in content:
                    content = content.replace(placeholder, _format_variable(value))
            unresolved = self._detect_unresolved(content)
            if unresolved:
                unresolved_all.extend(unresolved)
            compiled.append({"role": msg.get("role"), "content": content})
        if unresolved_all:
            raise ValidationError(
                "Unresolved prompt variables after compilation",
                prompt_name=self.name,
                prompt_version=self.version,
                unresolved_variables=unresolved_all,
            )
        return compiled


class PromptProvider(ABC):
    """Port for fetching system prompts from a prompt management system."""

    @property
    @abstractmethod
    def default_label(self) -> str:
        """Return the default label for prompt fetches (e.g., 'production')."""

    @abstractmethod
    async def get_system_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        version: int | None = None,
    ) -> str:
        """Fetch a system prompt from Langfuse.

        System prompts are static text without variables. They define the
        role, output format rules, and quality guidelines for the LLM.

        For chat prompts, extracts the first 'system' role message content.
        For text prompts, returns the prompt_text directly.

        Args:
            name: The prompt name/identifier.
            label: Optional label (e.g., "production", "latest").
            version: Optional specific version number.

        Returns:
            The system prompt text (plain string, no variable substitution).

        Raises:
            StoragePermanentError: If the prompt doesn't exist.
            StorageTransientError: If the fetch fails temporarily.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check connectivity to the prompt provider."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Flush and close any resources held by the provider."""
