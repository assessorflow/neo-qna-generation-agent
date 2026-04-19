"""Langfuse prompt provider implementation."""

from __future__ import annotations

import asyncio
from typing import Any

from langfuse import Langfuse
from langfuse.api.commons.errors import (
    AccessDeniedError,
    NotFoundError,
    UnauthorizedError,
)

from qna_generation_agent.app.logging import bind_context, get_logger
from qna_generation_agent.application.errors import (
    StoragePermanentError,
    StorageTransientError,
)
from qna_generation_agent.application.ports.prompt_provider import (
    Prompt,
    PromptProvider,
)

logger = get_logger(__name__)


class LangfusePromptProvider(PromptProvider):
    """Fetches prompts from Langfuse Prompt Management."""

    # Prompt name constants
    ASSESSMENT_GENERATOR = "Assessment Generator"
    MCQ_EXPLANATION_GENERATOR = "MCQ Explanation Generator"
    MCQ_ANSWER_GENERATOR = "MCQ Answer Generator"

    def __init__(
        self,
        *,
        public_key: str | None,
        secret_key: str | None,
        host: str,
        environment: str,
        release: str | None = None,
        default_label: str = "production",
    ) -> None:
        """Initialize the Langfuse prompt provider.

        Args:
            public_key: Langfuse public key.
            secret_key: Langfuse secret key.
            host: Langfuse host URL.
            environment: Runtime environment name.
            release: Optional release version.
            default_label: Default prompt label (e.g., "production").
        """
        self._default_label = default_label
        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            environment=environment,
            release=release,
        )
        logger.info(
            "langfuse_prompt_provider_initialized",
            host=host,
            environment=environment,
            default_label=default_label,
        )

    async def _fetch_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        version: int | None = None,
    ) -> Prompt:
        """Internal prompt fetch without deprecation warning."""
        trace_id = self._client.get_current_trace_id()
        if trace_id:
            bind_context(trace_id=trace_id)

        try:
            kwargs: dict[str, Any] = {}
            if label is not None:
                kwargs["label"] = label
            if version is not None:
                kwargs["version"] = version

            def _fetch_prompt(n: str, kw: dict[str, Any]) -> Any:
                return self._client.get_prompt(n, **kw)

            langfuse_prompt = await asyncio.to_thread(_fetch_prompt, name, kwargs)

            chat_messages = None
            prompt_text = None

            if (
                hasattr(langfuse_prompt, "chat_messages")
                and langfuse_prompt.chat_messages
            ):
                chat_messages = [
                    {"role": msg.get("role"), "content": msg.get("content")}
                    for msg in langfuse_prompt.chat_messages
                ]
            elif hasattr(langfuse_prompt, "prompt"):
                raw_prompt = langfuse_prompt.prompt
                if isinstance(raw_prompt, list):
                    if len(raw_prompt) > 0 and isinstance(raw_prompt[0], dict):
                        chat_messages = [
                            {"role": msg.get("role"), "content": msg.get("content")}
                            for msg in raw_prompt
                        ]
                    else:
                        prompt_text = "\n".join(str(p) for p in raw_prompt)
                else:
                    prompt_text = str(raw_prompt)

            labels = None
            if hasattr(langfuse_prompt, "labels"):
                labels = (
                    list(langfuse_prompt.labels) if langfuse_prompt.labels else None
                )

            config = None
            if hasattr(langfuse_prompt, "config"):
                config = langfuse_prompt.config

            version_num = 1
            if hasattr(langfuse_prompt, "version"):
                version_num = langfuse_prompt.version

            prompt = Prompt(
                name=name,
                version=version_num,
                prompt_text=prompt_text,
                chat_messages=chat_messages,
                labels=labels,
                config=config,
            )

            # Log prompt metadata only (no raw content)
            logger.info(
                "langfuse_prompt_fetched",
                name=name,
                version=version_num,
                label=label,
                is_chat=prompt.is_chat_prompt(),
            )

            return prompt
        except NotFoundError as error:
            raise StoragePermanentError(
                f"Prompt '{name}' not found in Langfuse",
                prompt_name=name,
            ) from error
        except (UnauthorizedError, AccessDeniedError) as error:
            raise StoragePermanentError(
                "Authentication failed with Langfuse",
                prompt_name=name,
            ) from error
        except Exception as error:
            raise StorageTransientError(
                f"Failed to fetch prompt '{name}' from Langfuse",
                prompt_name=name,
                error=str(error),
            ) from error

    @property
    def default_label(self) -> str:
        """Return the default label for prompt fetches."""
        return self._default_label

    async def shutdown(self) -> None:
        """Flush and close the Langfuse client."""
        try:
            await asyncio.to_thread(self._client.flush)
        except (OSError, TimeoutError, ConnectionError) as e:
            # Network-level errors during flush - may indicate data loss
            logger.error("prompt_provider_flush_failed_network", error=str(e))
        except Exception as e:
            # Other flush failures - log as warning since we're shutting down
            logger.warning(
                "prompt_provider_flush_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
        try:
            await asyncio.to_thread(self._client.shutdown)
        except Exception as e:
            logger.warning(
                "prompt_provider_shutdown_failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    async def health_check(self) -> bool:
        """Check connectivity by fetching a known prompt."""
        try:
            await asyncio.to_thread(self._client.get_prompt, self.ASSESSMENT_GENERATOR)
            return True
        except Exception as error:
            logger.warning(
                "langfuse_prompt_health_check_failed",
                error=str(error),
            )
            return False

    async def get_assessment_generator_prompt(
        self,
        *,
        label: str | None = None,
        version: int | None = None,
    ) -> Prompt:
        """Fetch the Assessment Generator prompt from Langfuse."""
        return await self._fetch_prompt(
            self.ASSESSMENT_GENERATOR,
            label=label,
            version=version,
        )

    async def get_mcq_explanation_generator_prompt(
        self,
        *,
        label: str | None = None,
        version: int | None = None,
    ) -> Prompt:
        """Fetch the MCQ Explanation Generator prompt from Langfuse."""
        return await self._fetch_prompt(
            self.MCQ_EXPLANATION_GENERATOR,
            label=label,
            version=version,
        )

    async def get_mcq_answer_generator_prompt(
        self,
        *,
        label: str | None = None,
        version: int | None = None,
    ) -> Prompt:
        """Fetch the MCQ Answer Generator prompt from Langfuse."""
        return await self._fetch_prompt(
            self.MCQ_ANSWER_GENERATOR,
            label=label,
            version=version,
        )

    async def get_system_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        version: int | None = None,
    ) -> str:
        """Fetch a system prompt from Langfuse.

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
        prompt = await self._fetch_prompt(name, label=label, version=version)

        result: str
        if prompt.is_chat_prompt():
            for msg in prompt.chat_messages or []:
                if msg.get("role") == "system":
                    result = msg.get("content", "")
                    break
            else:
                if prompt.chat_messages:
                    result = prompt.chat_messages[0].get("content", "")
                else:
                    result = ""
        else:
            result = prompt.prompt_text or ""

        logger.info(
            "system_prompt_fetched",
            name=name,
            version=prompt.version,
            label=label,
            is_chat=prompt.is_chat_prompt(),
        )
        return result
