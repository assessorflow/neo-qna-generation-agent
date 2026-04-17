"""Service for testing individual Langfuse prompts via Strands Agent execution."""

from __future__ import annotations

import time
from typing import Any

from strands import Agent
from strands.models.openai import OpenAIModel

from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.application.ports.prompt_provider import PromptProvider
from qna_generation_agent.infrastructure.llm.prompt_builder import (
    AssessmentGeneratorOutputSchema,
    MCQAnswerGeneratorOutputSchema,
    MCQExplanationOutputSchema,
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
        self.result = result
        self.prompt_version = prompt_version
        self.execution_time_ms = execution_time_ms
        self.raw_output = raw_output
        self.error = error


class PromptTestService:
    """Test service for executing single prompts via Strands Agent with structured output."""

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        base_url: str | None,
        timeout_seconds: int,
        prompt_provider: PromptProvider,
        cheap_model_id: str | None = None,
        expensive_model_id: str | None = None,
    ) -> None:
        """Initialize the prompt test service."""
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds
        self._prompt_provider = prompt_provider

        # Determine model IDs with fallback to main model_id
        cheap_id = cheap_model_id if cheap_model_id else model_id
        expensive_id = expensive_model_id if expensive_model_id else model_id

        # Initialize Strands OpenAIModel instances for different tiers
        self._cheap_model = OpenAIModel(
            client_args={
                "api_key": api_key,
                "base_url": base_url,
            },
            model_id=cheap_id,
            params={
                "max_tokens": 4000,
                "temperature": 0.7,
            },
        )

        self._expensive_model = OpenAIModel(
            client_args={
                "api_key": api_key,
                "base_url": base_url,
            },
            model_id=expensive_id,
            params={
                "max_tokens": 4000,
                "temperature": 0.7,
            },
        )

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

        try:
            prompt_obj = await self._prompt_provider.get_prompt(
                "Assessment Generator",
                label=self._prompt_provider.default_label,
            )

            chunks_formatted = "\n\n".join(
                f"[Chunk {i}] {chunk}" for i, chunk in enumerate(chunks, start=1)
            )

            compiled = prompt_obj.compile(
                structured_count=structured_count,
                non_structured_count=non_structured_count,
                difficulty=difficulty,
                topics=topics,
                chunks=chunks_formatted,
            )

            prompt_str = str(compiled)
            prompt_version = f"{prompt_obj.name}@v{prompt_obj.version}"

            agent = Agent(
                model=self._expensive_model,
                system_prompt="You are an expert assessment generator for ELP (English Language Proficiency) assessments targeting foreign students in Singapore.",
            )

            result = await agent.invoke_async(
                prompt_str,
                structured_output_model=AssessmentGeneratorOutputSchema,
            )

            execution_time_ms = int((time.monotonic() - start_time) * 1000)

            result_dict = (
                result.structured_output.model_dump(mode="json")
                if result and result.structured_output
                else None
            )

            return PromptTestResult(
                result=result_dict,
                prompt_version=prompt_version,
                execution_time_ms=execution_time_ms,
                raw_output=None,
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
                prompt_version="unknown",
                execution_time_ms=execution_time_ms,
                error=str(error),
            )

    async def test_mcq_answer_generator(
        self,
        *,
        question_stem: str,
        grammar_target: str,
        difficulty: str,
        l1_background: str,
    ) -> PromptTestResult:
        """Test the MCQ Answer Generator prompt with structured output."""
        start_time = time.monotonic()

        try:
            prompt_obj = await self._prompt_provider.get_prompt(
                "MCQ Answer Generator",
                label=self._prompt_provider.default_label,
            )

            compiled = prompt_obj.compile(
                question_text=question_stem,
                topic=grammar_target,
                difficulty=difficulty,
                chunk_content=l1_background,
            )

            prompt_str = str(compiled)
            prompt_version = f"{prompt_obj.name}@v{prompt_obj.version}"

            agent = Agent(
                model=self._cheap_model,
                system_prompt="You are an expert at creating MCQ answer options with L1-targeted distractors for English Language Proficiency assessments.",
            )

            result = await agent.invoke_async(
                prompt_str,
                structured_output_model=MCQAnswerGeneratorOutputSchema,
            )

            execution_time_ms = int((time.monotonic() - start_time) * 1000)

            result_dict = (
                result.structured_output.model_dump(mode="json")
                if result and result.structured_output
                else None
            )

            return PromptTestResult(
                result=result_dict,
                prompt_version=prompt_version,
                execution_time_ms=execution_time_ms,
                raw_output=None,
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
                prompt_version="unknown",
                execution_time_ms=execution_time_ms,
                error=str(error),
            )

    async def test_mcq_explanation_generator(
        self,
        *,
        question: str,
        options: dict[str, str],
        correct_answer: str,
        target_audience: str,
    ) -> PromptTestResult:
        """Test the MCQ Explanation Generator prompt with structured output."""
        start_time = time.monotonic()

        try:
            prompt_obj = await self._prompt_provider.get_prompt(
                "MCQ Explanation Generator",
                label=self._prompt_provider.default_label,
            )

            compiled = prompt_obj.compile(
                question_text=question,
                topic=target_audience,
                option_a=options.get("A", ""),
                option_b=options.get("B", ""),
                option_c=options.get("C", ""),
                option_d=options.get("D", ""),
                correct_answer=correct_answer,
                chunk_content=target_audience,
            )

            prompt_str = str(compiled)
            prompt_version = f"{prompt_obj.name}@v{prompt_obj.version}"

            agent = Agent(
                model=self._cheap_model,
                system_prompt="You are an expert at generating detailed explanations for MCQ distractors, with expertise in L1 interference patterns for English language learners.",
            )

            result = await agent.invoke_async(
                prompt_str,
                structured_output_model=MCQExplanationOutputSchema,
            )

            execution_time_ms = int((time.monotonic() - start_time) * 1000)

            result_dict = (
                result.structured_output.model_dump(mode="json")
                if result and result.structured_output
                else None
            )

            return PromptTestResult(
                result=result_dict,
                prompt_version=prompt_version,
                execution_time_ms=execution_time_ms,
                raw_output=None,
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
                prompt_version="unknown",
                execution_time_ms=execution_time_ms,
                error=str(error),
            )

    async def health_check(self) -> bool:
        """Check if the prompt test service is functional."""
        try:
            # Check prompt provider
            await self._prompt_provider.get_prompt("Assessment Generator")
            # Check both model instances are initialized
            if self._cheap_model is None or self._expensive_model is None:
                return False
            return True
        except Exception:
            return False
