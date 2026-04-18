"""HTTP response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health response."""

    model_config = ConfigDict(strict=True)

    status: str = "healthy"


class LiveResponse(BaseModel):
    """Liveness response."""

    model_config = ConfigDict(strict=True)

    alive: bool = True


class ReadyResponse(BaseModel):
    """Readiness response."""

    model_config = ConfigDict(strict=True)

    ready: bool
    checks: dict[str, bool] = Field(default_factory=dict)


class VersionResponse(BaseModel):
    """Version response."""

    model_config = ConfigDict(strict=True)

    version: str
    environment: str


class ErrorResponse(BaseModel):
    """Structured API error response."""

    model_config = ConfigDict(strict=True)

    error: str
    request_id: str


# =============================================================================
# Prompt Testing Schemas
# =============================================================================


class AssessmentGeneratorTestRequest(BaseModel):
    """Request to test the Assessment Generator prompt."""

    model_config = ConfigDict(strict=True)

    structured_count: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Number of MCQ questions to generate",
    )
    non_structured_count: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Number of open-ended questions to generate",
    )
    difficulty: str = Field(
        default="medium",
        pattern="^(easy|medium|hard)$",
        description="Target difficulty level",
    )
    topics: str = Field(
        default="Grammar, Vocabulary, Article Usage",
        min_length=1,
        description="Comma-separated list of topics",
    )
    chunks: list[str] = Field(
        default=[
            "Many foreign students struggle with English articles (a, an, the). Chinese and Vietnamese L1 speakers often omit articles because their native languages don't use them the same way. For example, they might say 'I bought book' instead of 'I bought a book'.",
            "Past perfect tense describes an action completed before another past action. Example: 'By the time I arrived, she had already left.' Common errors include using simple past instead: 'By the time I arrived, she already left.'",
        ],
        min_length=1,
        description="Document chunks to ground the questions",
    )


class MCQAnswerGeneratorTestRequest(BaseModel):
    """Request to test the MCQ Answer Generator prompt."""

    model_config = ConfigDict(strict=True)

    question_text: str = Field(
        default="The student ____ to school yesterday when it suddenly started raining.",
        min_length=1,
        description="The incomplete MCQ question stem",
    )
    topic: str = Field(
        default="past continuous tense",
        min_length=1,
        description="The grammar point or topic being tested",
    )
    difficulty: str = Field(
        default="medium",
        pattern="^(easy|medium|hard)$",
        description="Target difficulty level",
    )
    chunk_content: str = Field(
        default="Chinese",
        min_length=1,
        description="Source chunk content or target learner L1 background",
    )


class MCQExplanationGeneratorTestRequest(BaseModel):
    """Request to test the MCQ Explanation Generator prompt."""

    model_config = ConfigDict(strict=True)

    question_text: str = Field(
        default="Choose the correct article: I bought ____ book from the bookstore.",
        min_length=1,
        description="The MCQ question stem",
    )
    topic: str = Field(
        default="article usage",
        min_length=1,
        description="Topic or target audience description",
    )
    option_a: str = Field(
        default="a",
        min_length=1,
        description="Option A text",
    )
    option_b: str = Field(
        default="an",
        min_length=1,
        description="Option B text",
    )
    option_c: str = Field(
        default="the",
        min_length=1,
        description="Option C text",
    )
    option_d: str = Field(
        default="(no article)",
        min_length=1,
        description="Option D text",
    )
    correct_answer: str = Field(
        default="A",
        pattern="^[ABCD]$",
        description="The correct answer letter (A/B/C/D)",
    )
    chunk_content: str = Field(
        default="Chinese L1 students learning English in Singapore",
        min_length=1,
        description="Source chunk content or target audience info",
    )


class PromptTestResponse(BaseModel):
    """Response from a prompt test execution."""

    model_config = ConfigDict(strict=True)

    success: bool = Field(
        description="Whether the prompt executed successfully",
    )
    prompt_version: str = Field(
        description="Version of the Langfuse prompt used",
    )
    execution_time_ms: int = Field(
        description="Time taken to execute the prompt in milliseconds",
    )
    result: dict[str, Any] | None = Field(
        default=None,
        description="Parsed structured output from the LLM (if successful)",
    )
    raw_output: str | None = Field(
        default=None,
        description="Raw LLM output text (for debugging)",
    )
    error: str | None = Field(
        default=None,
        description="Error message if execution failed",
    )
