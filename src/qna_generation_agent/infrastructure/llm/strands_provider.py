"""Strands-based LLM provider.

This module uses the Strands framework for structured LLM interactions.
Strands provides type-safe, schema-validated LLM calls which is intentional
for ensuring consistent question generation output.
"""

from __future__ import annotations

import asyncio
from typing import cast

from openai import AsyncOpenAI, AuthenticationError, NotFoundError, OpenAIError
from pydantic import BaseModel
from strands import Agent
from strands.models.openai import OpenAIModel

from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.application.dto import (
    AssessmentContext,
    QuestionBatch,
    QuestionDraft,
)
from qna_generation_agent.application.errors import LLMPermanentError, LLMTransientError
from qna_generation_agent.application.ports.llm import LLMProvider
from qna_generation_agent.domain.enums import QuestionType
from qna_generation_agent.infrastructure.llm.prompt_builder import (
    GeneratedQuestionBatchSchema,
    build_system_prompt,
    build_user_prompt,
)

logger = get_logger(__name__)


class StrandsLLMProvider(LLMProvider):
    """Strategy pattern: Strands implementation of the generation port."""

    def __init__(
        self,
        *,
        model_provider: str,
        model_id: str,
        api_key: str,
        base_url: str | None,
        timeout_seconds: int,
    ) -> None:
        if model_provider != "openai":
            raise LLMPermanentError(
                "Only the OpenAI Strands backend is configured",
                provider=model_provider,
            )

        self._model_id = model_id
        self._timeout_seconds = timeout_seconds

        # Store client args for health check reuse
        self._api_key = api_key
        self._base_url = base_url

        # Initialize Strands OpenAIModel
        self._model = OpenAIModel(
            client_args={
                "api_key": api_key,
                "base_url": base_url,
            },
            model_id=model_id,
            params={
                "max_tokens": 2000,
                "temperature": 0.7,
            },
        )
        self._structured_agent = Agent(
            model=self._model,
            system_prompt=build_system_prompt(QuestionType.STRUCTURED),
        )
        self._non_structured_agent = Agent(
            model=self._model,
            system_prompt=build_system_prompt(QuestionType.NON_STRUCTURED),
        )

        # Create shared AsyncOpenAI client for health checks
        # This reuses the same connection pool as Strands' internal client
        self._health_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    @property
    def model(self) -> OpenAIModel:
        """Expose the underlying Strands model for direct use."""
        return self._model

    @property
    def model_id(self) -> str:
        """Return the model ID."""
        return self._model_id

    @property
    def timeout_seconds(self) -> int:
        """Return the timeout setting."""
        return self._timeout_seconds

    async def invoke_with_schema[T: BaseModel](
        self,
        prompt: str,
        *,
        structured_output_model: type[T],
    ) -> T | None:
        """Invoke the LLM with a custom schema for testing.

        Args:
            prompt: The prompt to send to the LLM.
            structured_output_model: The Pydantic model for the response.

        Returns:
            The structured output or None if parsing failed.
        """
        try:
            async with asyncio.timeout(self._timeout_seconds):
                result = await self._structured_agent.invoke_async(
                    prompt,
                    structured_output_model=structured_output_model,
                )
            return cast(T | None, result.structured_output)
        except TimeoutError:
            logger.warning("invoke_with_schema_timeout")
            return None
        except (AuthenticationError, NotFoundError) as error:
            # Permanent errors - don't retry
            logger.error(
                "invoke_with_schema_permanent_error",
                error_type=type(error).__name__,
                error=str(error),
            )
            return None
        except OpenAIError as error:
            # Transient OpenAI errors - worth retrying but we're in test mode
            logger.warning(
                "invoke_with_schema_openai_error",
                error_type=type(error).__name__,
                error=str(error),
            )
            return None
        except Exception:
            logger.exception("invoke_with_schema_unexpected_error")
            return None

    async def generate_structured(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> QuestionBatch:
        """Generate structured (MCQ) questions."""
        del correlation_id  # Used for tracing via telemetry
        return await self._generate(
            agent=self._structured_agent,
            context=context,
            count=count,
            difficulty_level=difficulty_level,
            question_type=QuestionType.STRUCTURED,
        )

    async def generate_non_structured(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> QuestionBatch:
        """Generate non-structured (open-ended) questions."""
        del correlation_id  # Used for tracing via telemetry
        return await self._generate(
            agent=self._non_structured_agent,
            context=context,
            count=count,
            difficulty_level=difficulty_level,
            question_type=QuestionType.NON_STRUCTURED,
        )

    async def _generate(
        self,
        *,
        agent: Agent,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        question_type: QuestionType,
    ) -> QuestionBatch:
        """Internal generation method."""
        prompt = build_user_prompt(
            context=context,
            count=count,
            difficulty_level=difficulty_level,
            question_type=question_type,
        )
        try:
            async with asyncio.timeout(self._timeout_seconds):
                result = await agent.invoke_async(
                    prompt,
                    structured_output_model=GeneratedQuestionBatchSchema,
                )
        except TimeoutError as error:
            raise LLMTransientError(
                "LLM request timed out",
                retry_after_seconds=10,
                model=self._model_id,
            ) from error
        except Exception as error:
            # Log full error details for debugging
            import traceback

            logger.error(
                "strands_generation_exception",
                error=str(error),
                error_type=type(error).__name__,
                traceback=traceback.format_exc()[-2000:],  # Last 2000 chars
                model=self._model_id,
                prompt_preview=prompt[:500]
                if isinstance(prompt, str)
                else str(prompt)[:500],
            )
            message = str(error).lower()
            if (
                "rate limit" in message
                or "timeout" in message
                or "temporarily unavailable" in message
            ):
                raise LLMTransientError(
                    "LLM request failed temporarily",
                    retry_after_seconds=30,
                    model=self._model_id,
                ) from error
            raise LLMPermanentError(
                "LLM request failed permanently",
                model=self._model_id,
                error=str(error),
            ) from error

        structured_output = result.structured_output

        # Debug: Log what we got back from Strands
        logger.info(
            "strands_generation_result",
            has_structured_output=structured_output is not None,
            structured_output_type=type(structured_output).__name__
            if structured_output
            else None,
            stop_reason=result.stop_reason if hasattr(result, "stop_reason") else None,
            message_preview=str(result.message)[:300]
            if hasattr(result, "message") and result.message
            else None,
        )

        if structured_output is None or not isinstance(
            structured_output, GeneratedQuestionBatchSchema
        ):
            raise LLMPermanentError(
                "Structured output was not returned by Strands",
                model=self._model_id,
                stop_reason=result.stop_reason
                if hasattr(result, "stop_reason")
                else None,
                raw_output_preview=str(result.raw_output)[:200]
                if hasattr(result, "raw_output") and result.raw_output
                else None,
            )

        questions = [
            QuestionDraft(
                question_text=item.question_text,
                answer_text=item.answer_text,
                question_type=question_type,
                difficulty_level=difficulty_level,
                explanation=item.explanation,
                references=item.references,
                topic_id=item.topic_id,
                metadata=item.metadata,
            )
            for item in structured_output.questions[:count]
        ]
        if len(questions) != count:
            raise LLMPermanentError(
                "Strands returned an unexpected number of questions",
                expected=count,
                actual=len(questions),
            )

        prompt_version = (
            "structured_v2"
            if question_type is QuestionType.STRUCTURED
            else "non_structured_v2"
        )
        return QuestionBatch(
            questions=questions,
            model_name=self._model_id,
            prompt_version=prompt_version,
        )

    async def health_check(self) -> bool:
        """Check connectivity to the LLM provider.

        Uses the shared AsyncOpenAI client to verify connectivity
        via the models.list() endpoint. This reuses the same client
        instance to avoid connection pool overhead.

        Returns:
            True if the provider is reachable and properly configured.
        """
        try:
            async with asyncio.timeout(5.0):
                await self._health_client.models.list()
            return True
        except AuthenticationError:
            # Authentication failure means misconfiguration - unhealthy
            logger.error("llm_health_check_failed", error="authentication_failed")
            return False
        except NotFoundError:
            # Model not found means invalid configuration - unhealthy
            logger.error("llm_health_check_failed", error="model_not_found")
            return False
        except (OpenAIError, OSError, TimeoutError) as error:
            logger.warning("llm_health_check_failed", error=str(error))
            return False

    async def generate_with_prompt(
        self,
        *,
        prompt: str,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
        question_type: str,
    ) -> QuestionBatch:
        """Generate questions using a pre-compiled prompt from Langfuse."""
        del correlation_id  # Used for tracing via telemetry

        # Select agent based on question type
        if question_type == "structured":
            agent = self._structured_agent
            qt = QuestionType.STRUCTURED
        else:
            agent = self._non_structured_agent
            qt = QuestionType.NON_STRUCTURED

        try:
            async with asyncio.timeout(self._timeout_seconds):
                result = await agent.invoke_async(
                    prompt,
                    structured_output_model=GeneratedQuestionBatchSchema,
                )
        except TimeoutError as error:
            raise LLMTransientError(
                "LLM request timed out",
                retry_after_seconds=10,
                model=self._model_id,
            ) from error
        except Exception as error:
            message = str(error).lower()
            if (
                "rate limit" in message
                or "timeout" in message
                or "temporarily unavailable" in message
            ):
                raise LLMTransientError(
                    "LLM request failed temporarily",
                    retry_after_seconds=30,
                    model=self._model_id,
                ) from error
            raise LLMPermanentError(
                "LLM request failed permanently",
                model=self._model_id,
                error=str(error),
            ) from error

        structured_output = result.structured_output
        if structured_output is None or not isinstance(
            structured_output, GeneratedQuestionBatchSchema
        ):
            raise LLMPermanentError(
                "Structured output was not returned by Strands",
                model=self._model_id,
            )

        questions = [
            QuestionDraft(
                question_text=item.question_text,
                answer_text=item.answer_text,
                question_type=qt,
                difficulty_level=difficulty_level,
                explanation=item.explanation,
                references=item.references,
                topic_id=item.topic_id,
                metadata=item.metadata,
            )
            for item in structured_output.questions[:count]
        ]
        if len(questions) != count:
            raise LLMPermanentError(
                "Strands returned an unexpected number of questions",
                expected=count,
                actual=len(questions),
            )

        return QuestionBatch(
            questions=questions,
            model_name=self._model_id,
            prompt_version="langfuse_managed",
        )
