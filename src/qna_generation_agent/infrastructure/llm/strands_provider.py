"""Strands-based LLM provider.

This module uses the Strands framework for structured LLM interactions.
Strands provides type-safe, schema-validated LLM calls which is intentional
for ensuring consistent question generation output.
"""

from __future__ import annotations

import asyncio
from typing import Any, cast

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    OpenAIError,
    PermissionDeniedError,
    RateLimitError,
    UnprocessableEntityError,
)
from pydantic import BaseModel, ValidationError
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
from qna_generation_agent.domain.enums import DifficultyLevel, QuestionType
from qna_generation_agent.infrastructure.llm.prompt_builder import (
    GeneratedQuestionBatchSchema,
    build_system_prompt,
    build_user_prompt,
)

logger = get_logger(__name__)


def _make_agent_key(model_id: str, system_prompt: str) -> str:
    """Create a cache key for agent lookup.

    Args:
        model_id: The model identifier.
        system_prompt: The system prompt content.

    Returns:
        A unique key for agent caching.
    """
    return f"{model_id}:{hash(system_prompt) & 0xFFFFFFFF:08x}"


def _parse_difficulty(value: str | None) -> DifficultyLevel | None:
    """Parse difficulty string to enum."""
    if value is None:
        return None
    try:
        return DifficultyLevel(value.lower())
    except ValueError:
        return None


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
        max_tokens: int = 4096,
        temperature: float = 0.2,
        cheap_model_id: str | None = None,
        expensive_model_id: str | None = None,
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

        # Determine model IDs with fallback to main model_id
        cheap_id = cheap_model_id if cheap_model_id else model_id
        expensive_id = expensive_model_id if expensive_model_id else model_id
        self._cheap_model_id = cheap_id
        self._expensive_model_id = expensive_id

        # Initialize Strands OpenAIModel for default (main) tier
        self._model = OpenAIModel(
            client_args={
                "api_key": api_key,
                "base_url": base_url,
            },
            model_id=model_id,
            params={
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        # Initialize tier-specific models
        self._cheap_model = OpenAIModel(
            client_args={
                "api_key": api_key,
                "base_url": base_url,
            },
            model_id=cheap_id,
            params={
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        self._expensive_model = OpenAIModel(
            client_args={
                "api_key": api_key,
                "base_url": base_url,
            },
            model_id=expensive_id,
            params={
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        self._structured_agent = Agent(
            model=self._model,
            system_prompt=build_system_prompt(QuestionType.STRUCTURED),
            callback_handler=None,
        )
        self._non_structured_agent = Agent(
            model=self._model,
            system_prompt=build_system_prompt(QuestionType.NON_STRUCTURED),
            callback_handler=None,
        )

        # Create shared AsyncOpenAI client for health checks
        # This reuses the same connection pool as Strands' internal client
        self._health_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        # Agent cache for invoke_with_system_and_user to avoid creating
        # new agents on every request (prevents resource exhaustion)
        self._agent_cache: dict[str, Agent] = {}

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

    def _get_cached_agent(self, model: OpenAIModel, model_id: str, system_prompt: str) -> Agent:
        """Get or create a cached Agent for the given model and system prompt.

        Args:
            model: The OpenAI model to use.
            model_id: The model identifier string.
            system_prompt: The system prompt content.

        Returns:
            A cached or newly created Agent instance.
        """
        cache_key = _make_agent_key(model_id, system_prompt)

        if cache_key not in self._agent_cache:
            logger.debug(
                "creating_new_agent",
                model_id=model_id,
                cache_key=cache_key,
            )
            self._agent_cache[cache_key] = Agent(
                model=model,
                system_prompt=system_prompt,
                callback_handler=None,
            )
        else:
            logger.debug(
                "reusing_cached_agent",
                model_id=model_id,
                cache_key=cache_key,
            )

        return self._agent_cache[cache_key]

    async def invoke_with_system_and_user[T: BaseModel](
        self,
        system_message: str,
        user_message: str,
        *,
        structured_output_model: type[T],
        model_tier: str = "expensive",
    ) -> T | None:
        """Invoke LLM with separate system and user messages.

        Args:
            system_message: Static system prompt from Langfuse.
            user_message: Dynamic user prompt built from template.
            structured_output_model: Pydantic model for response validation.
            model_tier: Which model to use ("expensive" or "cheap").

        Returns:
            Parsed structured output or None if failed.
        """
        start_time = asyncio.get_event_loop().time()

        # Select model based on tier
        if model_tier == "cheap":
            model = self._cheap_model
            tier_model_id = self._cheap_model_id
        else:
            model = self._expensive_model
            tier_model_id = self._expensive_model_id

        logger.debug(
            "invoke_with_system_and_user_started",
            model_tier=model_tier,
            model_id=tier_model_id,
            schema=structured_output_model.__name__,
        )

        try:
            # Use cached agent to avoid resource exhaustion from creating
            # new agents on every request
            agent = self._get_cached_agent(model, tier_model_id, system_message)

            async with asyncio.timeout(self._timeout_seconds):
                result = await agent.invoke_async(
                    user_message,
                    structured_output_model=structured_output_model,
                )

            structured_output = result.structured_output
            if structured_output is None:
                logger.warning(
                    "invoke_with_system_and_user_no_output",
                    model_tier=model_tier,
                    model_id=tier_model_id,
                    stop_reason=result.stop_reason
                    if hasattr(result, "stop_reason")
                    else None,
                )
                return None

            elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            logger.debug(
                "invoke_with_system_and_user_complete",
                model_tier=model_tier,
                model_id=tier_model_id,
                execution_time_ms=elapsed_ms,
                schema=structured_output_model.__name__,
            )

            return cast(T, structured_output)
        except TimeoutError as error:
            elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            logger.warning(
                "invoke_with_system_and_user_timeout",
                model_tier=model_tier,
                model_id=tier_model_id,
                timeout_seconds=self._timeout_seconds,
                execution_time_ms=elapsed_ms,
            )
            raise LLMTransientError(
                "LLM request timed out",
                retry_after_seconds=10,
                model=tier_model_id,
            ) from error
        except asyncio.CancelledError:
            elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            logger.warning(
                "invoke_with_system_and_user_cancelled",
                model_tier=model_tier,
                model_id=tier_model_id,
                execution_time_ms=elapsed_ms,
            )
            # Re-raise CancelledError to allow proper task cleanup
            raise
        except ValidationError as error:
            elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            logger.error(
                "invoke_with_system_and_user_validation_error",
                error_type=type(error).__name__,
                error=str(error),
                model_tier=model_tier,
                model_id=tier_model_id,
                schema=structured_output_model.__name__,
                execution_time_ms=elapsed_ms,
            )
            raise LLMPermanentError(
                "LLM returned output that failed schema validation",
                model=tier_model_id,
                error=str(error),
            ) from error
        except (AuthenticationError, NotFoundError) as error:
            elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            logger.error(
                "invoke_with_system_and_user_permanent_error",
                error_type=type(error).__name__,
                error=str(error),
                model_tier=model_tier,
                model_id=tier_model_id,
                execution_time_ms=elapsed_ms,
            )
            raise LLMPermanentError(
                "LLM request failed permanently",
                model=tier_model_id,
                error=str(error),
            ) from error
        except OpenAIError as error:
            elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            logger.warning(
                "invoke_with_system_and_user_openai_error",
                error_type=type(error).__name__,
                error=str(error),
                model_tier=model_tier,
                model_id=tier_model_id,
                execution_time_ms=elapsed_ms,
            )
            raise LLMTransientError(
                "LLM request failed temporarily",
                retry_after_seconds=30,
                model=tier_model_id,
            ) from error
        except Exception as error:
            elapsed_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            logger.exception(
                "invoke_with_system_and_user_unexpected_error",
                model_tier=model_tier,
                model_id=tier_model_id,
                execution_time_ms=elapsed_ms,
            )
            raise LLMPermanentError(
                "LLM request failed unexpectedly",
                model=tier_model_id,
                error=str(error),
            ) from error

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

    def _map_exception(self, error: Exception) -> LLMTransientError | LLMPermanentError:
        """Classify an exception from the LLM stack into a typed error.

        OpenAI SDK errors are classified by HTTP semantics:
        - Permanent (4xx client errors): auth, permissions, bad request, not found
        - Transient (5xx server / network errors): APIError, connection, timeout, rate limit
        - Unknown OpenAIError defaults to transient (safer to retry)
        - Non-OpenAI exceptions fall back to string heuristics.
        """
        # Permanent OpenAI client errors
        if isinstance(
            error,
            (
                AuthenticationError,
                PermissionDeniedError,
                BadRequestError,
                NotFoundError,
                UnprocessableEntityError,
            ),
        ):
            return LLMPermanentError(
                "LLM request failed permanently",
                model=self._model_id,
                error=str(error),
            )

        # Transient OpenAI server / network errors
        if isinstance(
            error,
            (APIError, APIConnectionError, APITimeoutError, RateLimitError, OpenAIError),
        ):
            return LLMTransientError(
                "LLM request failed temporarily",
                retry_after_seconds=30,
                model=self._model_id,
            )

        # Fallback for non-OpenAI exceptions (e.g. from Strands internals)
        message = str(error).lower()
        if (
            "rate limit" in message
            or "timeout" in message
            or "temporarily unavailable" in message
        ):
            return LLMTransientError(
                "LLM request failed temporarily",
                retry_after_seconds=30,
                model=self._model_id,
            )

        return LLMPermanentError(
            "LLM request failed permanently",
            model=self._model_id,
            error=str(error),
        )

    async def _invoke_agent(
        self,
        *,
        agent: Agent,
        prompt: str,
    ) -> Any:
        """Invoke Strands with timeout and mapped errors."""
        try:
            async with asyncio.timeout(self._timeout_seconds):
                return await agent.invoke_async(
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
            raise self._map_exception(error) from error

    def _build_question_batch(
        self,
        *,
        result: Any,
        count: int,
        difficulty_level: str | None,
        question_type: QuestionType,
        prompt_version: str,
    ) -> QuestionBatch:
        """Convert a Strands result into a question batch."""
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

        # Validate output doesn't contain garbage (JVM text, log output, etc.)
        raw_str = str(result.raw_output) if hasattr(result, "raw_output") else ""
        if "JVM" in raw_str or "Method overriding" in raw_str or "Tool #" in raw_str:
            raise LLMTransientError(
                "LLM returned garbage output (mixed content)",
                retry_after_seconds=5,
                model=self._model_id,
            )

        parsed_difficulty = _parse_difficulty(difficulty_level)
        questions: list[QuestionDraft] = []
        skipped_count = 0
        for item in structured_output.questions[:count]:
            # Skip questions without required fields
            if item.question_text is None:
                skipped_count += 1
                continue
            # Use placeholder if answer missing (legacy path) - 3-prompt workflow adds real answers
            answer = item.answer_text or "[Answer to be generated]"
            questions.append(
                QuestionDraft(
                    question_text=item.question_text,
                    answer_text=answer,
                    question_type=question_type,
                    difficulty_level=parsed_difficulty,
                    explanation=item.explanation,
                    references=item.references,
                    topic_id=item.topic_id,
                    metadata={k: str(v) for k, v in (item.metadata or {}).items()},
                )
            )

        if skipped_count > 0:
            logger.warning(
                "questions_skipped_missing_fields",
                skipped=skipped_count,
                expected=count,
                received=len(structured_output.questions),
            )

        if len(questions) == 0:
            raise LLMPermanentError(
                "No valid questions returned by LLM",
                expected=count,
                raw_preview=str(result.raw_output)[:500]
                if hasattr(result, "raw_output")
                else None,
            )

        return QuestionBatch(
            questions=questions,
            model_name=self._model_id,
            prompt_version=prompt_version,
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
        return self._build_question_batch(
            result=await self._invoke_agent(agent=agent, prompt=prompt),
            count=count,
            difficulty_level=difficulty_level,
            question_type=question_type,
            prompt_version=(
                "structured_v2"
                if question_type is QuestionType.STRUCTURED
                else "non_structured_v2"
            ),
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

    async def shutdown(self) -> None:
        """Close the health check client and release resources.

        This should be called during container shutdown to properly
        close the AsyncOpenAI client connection pool.
        """
        try:
            await self._health_client.close()
            logger.debug("strands_provider_shutdown_complete")
        except Exception as error:
            logger.warning(
                "strands_provider_shutdown_failed",
                error=str(error),
                error_type=type(error).__name__,
            )

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

        return self._build_question_batch(
            result=await self._invoke_agent(agent=agent, prompt=prompt),
            count=count,
            difficulty_level=difficulty_level,
            question_type=qt,
            prompt_version="langfuse_managed",
        )
