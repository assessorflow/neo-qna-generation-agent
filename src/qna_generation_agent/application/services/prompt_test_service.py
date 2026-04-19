"""Service for testing individual Langfuse prompts via LLM execution."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.application.ports.llm import LLMProvider
from qna_generation_agent.application.ports.prompt_provider import PromptProvider
from qna_generation_agent.infrastructure.llm.prompt_builder import (
    AssessmentGeneratorOutputSchema,
    MCQAnswerGeneratorOutputSchema,
    MCQExplanationOutputSchema,
    format_chunks_for_prompt,
)
from qna_generation_agent.infrastructure.llm.user_prompt_builder import (
    UserPromptBuilder,
)

logger = get_logger(__name__)


class PromptTestResult:
    """Result of a single prompt test execution."""

    def __init__(
        self,
        *,
        result: dict[str, Any] | None,
        prompt_version: str,
        execution_time_ms: int,
        raw_output: str | None = None,
        error: str | None = None,
    ) -> None:
        """Initialize a prompt test result."""
        self.result = result
        self.prompt_version = prompt_version
        self.execution_time_ms = execution_time_ms
        self.raw_output = raw_output
        self.error = error


class PromptTestService:
    """Test service for executing single prompts with separated system/user messages."""

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        prompt_provider: PromptProvider,
        timeout_seconds: int,
    ) -> None:
        """Initialize the prompt test service.

        Args:
            llm_provider: LLM provider for invoking with system/user separation.
            prompt_provider: Prompt provider for fetching system prompts.
            timeout_seconds: Timeout for LLM invocations.
        """
        self._llm_provider = llm_provider
        self._prompt_provider = prompt_provider
        self._timeout_seconds = timeout_seconds
        self._user_prompt_builder = UserPromptBuilder()

    async def test_assessment_generator(
        self,
        *,
        structured_count: int,
        non_structured_count: int,
        difficulty: str,
        topics: str,
        chunks: list[str],
    ) -> PromptTestResult:
        """Test the Assessment Generator prompt with structured output."""
        start_time = time.monotonic()
        prompt_version = "unknown"

        try:
            system_msg = await self._prompt_provider.get_system_prompt(
                "Assessment Generator",
                label=self._prompt_provider.default_label,
            )

            chunks_formatted = format_chunks_for_prompt(chunks)

            user_msg = self._user_prompt_builder.build_assessment_generator_prompt(
                structured_count=structured_count,
                non_structured_count=non_structured_count,
                difficulty=difficulty,
                topics=topics,
                chunks=chunks_formatted,
            )

            prompt_version = f"Assessment Generator@{self._prompt_provider.default_label}"

            async with asyncio.timeout(self._timeout_seconds):
                result = await self._llm_provider.invoke_with_system_and_user(
                    system_message=system_msg,
                    user_message=user_msg,
                    structured_output_model=AssessmentGeneratorOutputSchema,
                    model_tier="expensive",
                )

            execution_time_ms = int((time.monotonic() - start_time) * 1000)

            result_dict = (
                result.model_dump(mode="json")
                if result is not None
                else None
            )

            return PromptTestResult(
                result=result_dict,
                prompt_version=prompt_version,
                execution_time_ms=execution_time_ms,
                raw_output=None,
            )

        except TimeoutError:
            execution_time_ms = int((time.monotonic() - start_time) * 1000)
            logger.warning(
                "assessment_generator_test_timed_out",
                timeout_seconds=self._timeout_seconds,
                prompt_version=prompt_version,
            )
            return PromptTestResult(
                result=None,
                prompt_version=prompt_version,
                execution_time_ms=execution_time_ms,
                error=f"Prompt execution timed out after {self._timeout_seconds}s",
            )
        except Exception as error:
            execution_time_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "assessment_generator_test_failed",
                error=str(error),
                error_type=type(error).__name__,
            )
            return PromptTestResult(
                result=None,
                prompt_version=prompt_version,
                execution_time_ms=execution_time_ms,
                error=str(error),
            )

    async def test_mcq_answer_generator(
        self,
        *,
        question_text: str,
        topic: str,
        difficulty: str,
        chunk_content: str,
    ) -> PromptTestResult:
        """Test the MCQ Answer Generator prompt with structured output."""
        start_time = time.monotonic()
        prompt_version = "unknown"

        try:
            system_msg = await self._prompt_provider.get_system_prompt(
                "MCQ Answer Generator",
                label=self._prompt_provider.default_label,
            )

            user_msg = self._user_prompt_builder.build_mcq_answer_generator_prompt(
                question_text=question_text,
                topic=topic,
                difficulty=difficulty,
                chunk_content=chunk_content,
            )

            prompt_version = f"MCQ Answer Generator@{self._prompt_provider.default_label}"

            async with asyncio.timeout(self._timeout_seconds):
                result = await self._llm_provider.invoke_with_system_and_user(
                    system_message=system_msg,
                    user_message=user_msg,
                    structured_output_model=MCQAnswerGeneratorOutputSchema,
                    model_tier="cheap",
                )

            execution_time_ms = int((time.monotonic() - start_time) * 1000)

            result_dict = (
                result.model_dump(mode="json")
                if result is not None
                else None
            )

            return PromptTestResult(
                result=result_dict,
                prompt_version=prompt_version,
                execution_time_ms=execution_time_ms,
                raw_output=None,
            )

        except TimeoutError:
            execution_time_ms = int((time.monotonic() - start_time) * 1000)
            logger.warning(
                "mcq_answer_generator_test_timed_out",
                timeout_seconds=self._timeout_seconds,
                prompt_version=prompt_version,
            )
            return PromptTestResult(
                result=None,
                prompt_version=prompt_version,
                execution_time_ms=execution_time_ms,
                error=f"Prompt execution timed out after {self._timeout_seconds}s",
            )
        except Exception as error:
            execution_time_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "mcq_answer_generator_test_failed",
                error=str(error),
                error_type=type(error).__name__,
            )
            return PromptTestResult(
                result=None,
                prompt_version=prompt_version,
                execution_time_ms=execution_time_ms,
                error=str(error),
            )

    async def test_mcq_explanation_generator(
        self,
        *,
        question_text: str,
        topic: str,
        option_a: str,
        option_b: str,
        option_c: str,
        option_d: str,
        correct_answer: str,
        chunk_content: str,
    ) -> PromptTestResult:
        """Test the MCQ Explanation Generator prompt with structured output."""
        start_time = time.monotonic()
        prompt_version = "unknown"

        try:
            system_msg = await self._prompt_provider.get_system_prompt(
                "MCQ Explanation Generator",
                label=self._prompt_provider.default_label,
            )

            user_msg = self._user_prompt_builder.build_mcq_explanation_generator_prompt(
                question_text=question_text,
                topic=topic,
                correct_answer=correct_answer,
                option_a=option_a,
                option_b=option_b,
                option_c=option_c,
                option_d=option_d,
                chunk_content=chunk_content,
            )

            prompt_version = f"MCQ Explanation Generator@{self._prompt_provider.default_label}"

            async with asyncio.timeout(self._timeout_seconds):
                result = await self._llm_provider.invoke_with_system_and_user(
                    system_message=system_msg,
                    user_message=user_msg,
                    structured_output_model=MCQExplanationOutputSchema,
                    model_tier="cheap",
                )

            execution_time_ms = int((time.monotonic() - start_time) * 1000)

            result_dict = (
                result.model_dump(mode="json")
                if result is not None
                else None
            )

            return PromptTestResult(
                result=result_dict,
                prompt_version=prompt_version,
                execution_time_ms=execution_time_ms,
                raw_output=None,
            )

        except TimeoutError:
            execution_time_ms = int((time.monotonic() - start_time) * 1000)
            logger.warning(
                "mcq_explanation_generator_test_timed_out",
                timeout_seconds=self._timeout_seconds,
                prompt_version=prompt_version,
            )
            return PromptTestResult(
                result=None,
                prompt_version=prompt_version,
                execution_time_ms=execution_time_ms,
                error=f"Prompt execution timed out after {self._timeout_seconds}s",
            )
        except Exception as error:
            execution_time_ms = int((time.monotonic() - start_time) * 1000)
            logger.error(
                "mcq_explanation_generator_test_failed",
                error=str(error),
                error_type=type(error).__name__,
            )
            return PromptTestResult(
                result=None,
                prompt_version=prompt_version,
                execution_time_ms=execution_time_ms,
                error=str(error),
            )

    async def health_check(self) -> bool:
        """Check if the prompt test service is functional."""
        try:
            # Check prompt provider
            await self._prompt_provider.get_system_prompt("Assessment Generator")
            return True
        except Exception:
            return False
