"""LLM provider port."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from qna_generation_agent.application.dto import AssessmentContext, QuestionBatch

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Strategy pattern: structured and open-ended generation backends."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Return the model ID for this provider."""

    @abstractmethod
    async def generate_structured(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> QuestionBatch:
        """Generate structured questions."""

    @abstractmethod
    async def generate_non_structured(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> QuestionBatch:
        """Generate non-structured questions."""

    @abstractmethod
    async def generate_with_prompt(
        self,
        *,
        prompt: str,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
        question_type: str,
    ) -> QuestionBatch:
        """Generate questions using a pre-compiled prompt from Langfuse.

        Args:
            prompt: The compiled prompt text from Langfuse.
            count: Number of questions to generate.
            difficulty: Target difficulty level.
            correlation_id: Correlation ID for tracing.
            question_type: "structured" or "non_structured".

        Returns:
            Batch of generated question drafts.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check connectivity to the LLM provider.

        Returns True if the provider is reachable and responsive.
        """
        ...

    @abstractmethod
    async def invoke_with_schema(
        self,
        prompt: str,
        *,
        structured_output_model: type[T],
    ) -> T | None:
        """Invoke LLM with structured output schema.

        Args:
            prompt: The prompt to send to the LLM.
            structured_output_model: Pydantic model for structured output.

        Returns:
            Parsed structured output or None if failed.
        """
        ...
