"""Langfuse prompt provider implementation."""

from __future__ import annotations

import asyncio
from typing import Any

from langfuse import Langfuse, observe

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
    ) -> None:
        """Initialize the Langfuse client for prompt management.

        Args:
            public_key: Langfuse public key.
            secret_key: Langfuse secret key.
            host: Langfuse host URL.
            environment: Deployment environment.
            release: Optional release version.
        """
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
        )

    @observe(name="get_prompt", as_type="generation")
    async def get_prompt(
        self,
        name: str,
        *,
        label: str | None = None,
        version: int | None = None,
    ) -> Prompt:
        """Fetch a prompt from Langfuse.

        Args:
            name: The prompt name.
            label: Optional label (production, latest, etc.).
            version: Optional specific version.

        Returns:
            The fetched prompt.
        """
        # Bind trace_id to structlog context for correlation
        trace_id = self._client.get_current_trace_id()
        if trace_id:
            bind_context(trace_id=trace_id)

        try:
            # Build kwargs for get_prompt
            kwargs: dict[str, Any] = {}
            if label is not None:
                kwargs["label"] = label
            if version is not None:
                kwargs["version"] = version

            def _fetch_prompt(n: str, kw: dict[str, Any]) -> Any:
                return self._client.get_prompt(n, **kw)

            langfuse_prompt = await asyncio.to_thread(_fetch_prompt, name, kwargs)

            # Determine if it's a chat or text prompt
            chat_messages = None
            prompt_text = None

            # Check for chat messages attribute
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
                # Ensure prompt_text is always a string
                if isinstance(raw_prompt, list):
                    # Convert list to string (join or take first element)
                    if len(raw_prompt) > 0 and isinstance(raw_prompt[0], dict):
                        # It's a list of message dicts - convert to chat format
                        chat_messages = [
                            {"role": msg.get("role"), "content": msg.get("content")}
                            for msg in raw_prompt
                        ]
                    else:
                        # It's a list of strings - join them
                        prompt_text = "\n".join(str(p) for p in raw_prompt)
                else:
                    prompt_text = str(raw_prompt)

            # Get labels if available
            labels = None
            if hasattr(langfuse_prompt, "labels"):
                labels = (
                    list(langfuse_prompt.labels) if langfuse_prompt.labels else None
                )

            # Get config if available
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

            # Debug: Log prompt format and raw content
            logger.info(
                "langfuse_prompt_fetched",
                name=name,
                version=version_num,
                label=label,
                is_chat=prompt.is_chat_prompt(),
                has_prompt_text=prompt_text is not None and len(prompt_text) > 0
                if prompt_text
                else False,
                prompt_text_preview=prompt_text[:200] if prompt_text else None,
                has_chat_messages=chat_messages is not None and len(chat_messages) > 0
                if chat_messages
                else False,
                chat_messages_count=len(chat_messages) if chat_messages else 0,
            )

            return prompt

        except Exception as error:
            error_msg = str(error).lower()
            if "not found" in error_msg or "does not exist" in error_msg:
                raise StoragePermanentError(
                    f"Prompt '{name}' not found in Langfuse",
                    prompt_name=name,
                ) from error
            if "unauthorized" in error_msg or "authentication" in error_msg:
                raise StoragePermanentError(
                    "Authentication failed with Langfuse",
                    prompt_name=name,
                ) from error
            raise StorageTransientError(
                f"Failed to fetch prompt '{name}' from Langfuse",
                prompt_name=name,
                error=str(error),
            ) from error

    async def shutdown(self) -> None:
        """Flush and shutdown the Langfuse client.

        Per Langfuse 4.2 SDK, flush() must be called to ensure all
        telemetry (prompt fetch traces) is persisted before process exit.
        shutdown() must also be called to release resources.
        """
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
        """Check connectivity to Langfuse prompt management.

        Attempts to fetch a known prompt to verify connectivity.
        Returns True if successful.
        """
        try:
            # Try to fetch Assessment Generator as health check
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
        """Fetch the Assessment Generator prompt.

        Args:
            label: Optional label (production, latest, etc.).
            version: Optional specific version.

        Returns:
            The Assessment Generator prompt.
        """
        return await self.get_prompt(
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
        """Fetch the MCQ Explanation Generator prompt.

        Args:
            label: Optional label (production, latest, etc.).
            version: Optional specific version.

        Returns:
            The MCQ Explanation Generator prompt.
        """
        return await self.get_prompt(
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
        """Fetch the MCQ Answer Generator prompt.

        Args:
            label: Optional label (production, latest, etc.).
            version: Optional specific version.

        Returns:
            The MCQ Answer Generator prompt.
        """
        return await self.get_prompt(
            self.MCQ_ANSWER_GENERATOR,
            label=label,
            version=version,
        )
