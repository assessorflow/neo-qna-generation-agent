"""Prompt provider port for fetching prompts from Langfuse."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Prompt:
    """A fetched prompt with metadata."""

    name: str
    version: int
    prompt_text: str | None = None
    chat_messages: list[dict[str, Any]] | None = None
    labels: list[str] | None = None
    config: dict[str, Any] | None = None

    def is_chat_prompt(self) -> bool:
        """Check if this is a chat-style prompt."""
        return self.chat_messages is not None and len(self.chat_messages) > 0

    def compile(self, **variables: Any) -> str | list[dict[str, Any]]:
        """Compile the prompt with variables.

        For text prompts: returns compiled string.
        For chat prompts: returns list of messages with variables substituted.
        """
        if self.is_chat_prompt():
            return self._compile_chat(**variables)
        return self._compile_text(**variables)

    def _compile_text(self, **variables: Any) -> str:
        """Compile text prompt with variable substitution."""
        if self.prompt_text is None:
            return ""
        result = self.prompt_text
        for key, value in variables.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

    def _compile_chat(self, **variables: Any) -> list[dict[str, Any]]:
        """Compile chat prompt with variable substitution."""
        if self.chat_messages is None:
            return []
        compiled: list[dict[str, Any]] = []
        for msg in self.chat_messages:
            content = msg.get("content", "")
            for key, value in variables.items():
                placeholder = f"{{{{{key}}}}}"
                if placeholder in content:
                    content = content.replace(placeholder, str(value))
            compiled.append({"role": msg.get("role"), "content": content})
        return compiled


class PromptProvider(ABC):
    """Port for fetching prompts from a prompt management system."""

    @abstractmethod
    async def get_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        version: int | None = None,
    ) -> Prompt:
        """Fetch a prompt by name.

        Args:
            name: The prompt name/identifier.
            label: Optional label (e.g., "production", "latest").
            version: Optional specific version number.

        Returns:
            The fetched prompt.

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
