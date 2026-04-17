"""Prompt templates and structured-output schemas for Strands and Langfuse."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from qna_generation_agent.application.dto import AssessmentContext
from qna_generation_agent.domain.enums import QuestionType

# =============================================================================
# Legacy GeneratedQuestion Schemas (for backward compatibility)
# =============================================================================


class GeneratedQuestionSchema(BaseModel):
    """Structured output schema returned by the model.

    Accepts multiple field name variations for maximum LLM compatibility.
    """

    model_config = ConfigDict(strict=True)

    # Question field - many variations
    question_text: str | None = Field(default=None, min_length=1)
    content: str | None = Field(default=None, min_length=1)
    text: str | None = Field(default=None, min_length=1)
    prompt: str | None = Field(default=None, min_length=1)
    query: str | None = Field(default=None, min_length=1)

    # Answer field - many variations
    answer_text: str | None = Field(default=None, min_length=1)
    structured_answer: str | None = Field(default=None, min_length=1)
    answer: str | None = Field(default=None, min_length=1)
    response: str | None = Field(default=None, min_length=1)
    model_answer: str | None = Field(default=None, min_length=1)
    solution: str | None = Field(default=None, min_length=1)
    correct_answer: str | None = Field(default=None, min_length=1)

    explanation: str | None = None
    references: list[str] = Field(default_factory=list)
    topic_id: str | None = None
    # Flexible metadata - accepts any JSON-compatible values
    metadata: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_required_fields(self) -> GeneratedQuestionSchema:
        """Ensure at least one naming convention provides the question.

        Answer is optional - incomplete questions are filtered downstream.
        """
        # Get question from any field
        question = (
            self.question_text or self.content or self.text or self.prompt or self.query
        )
        # Get answer from any field (optional)
        answer = (
            self.answer_text
            or self.structured_answer
            or self.answer
            or self.response
            or self.model_answer
            or self.solution
            or self.correct_answer
        )

        if not question:
            raise ValueError(
                "No question field provided. Expected one of: "
                "question_text, content, text, prompt, query"
            )

        # Set the canonical field names
        self.question_text = question
        self.answer_text = answer  # May be None - filtered downstream
        return self


class GeneratedQuestionBatchSchema(BaseModel):
    """Structured output schema for one generation batch."""

    model_config = ConfigDict(strict=True)

    questions: list[GeneratedQuestionSchema]


# =============================================================================
# Langfuse Prompt: Assessment Generator
# Generates ELP assessment questions for foreign students in Singapore
# Variables: {structured_count}, {non_structured_count}, {difficulty}, {topics}, {chunks}
# =============================================================================


class MCQOptionsSchema(BaseModel):
    """MCQ options with A, B, C, D choices."""

    model_config = ConfigDict(strict=True)

    A: str = Field(min_length=1, description="Option A text")
    B: str = Field(min_length=1, description="Option B text")
    C: str = Field(min_length=1, description="Option C text")
    D: str = Field(min_length=1, description="Option D text")


class StructuredQuestionMetadataSchema(BaseModel):
    """Metadata for structured (MCQ) questions."""

    model_config = ConfigDict(strict=True)

    question_type: str = Field(
        default="structured",
        pattern="^structured$",
        description="Question type identifier",
    )
    options: MCQOptionsSchema = Field(
        description="Four MCQ options with keys A, B, C, D"
    )
    source_chunk_ids: list[str] = Field(
        min_length=1,
        description="Source chunk IDs that ground this question",
    )
    difficulty: str = Field(
        pattern="^(easy|medium|hard)$",
        description="Question difficulty level",
    )
    topic: str = Field(
        min_length=1,
        description="Topic name from the source material",
    )


class NonStructuredQuestionMetadataSchema(BaseModel):
    """Metadata for non-structured (open-ended) questions."""

    model_config = ConfigDict(strict=True)

    question_type: str = Field(
        default="non_structured",
        pattern="^non_structured$",
        description="Question type identifier",
    )
    source_chunk_ids: list[str] = Field(
        min_length=1,
        description="Source chunk IDs that ground this question",
    )
    difficulty: str = Field(
        pattern="^(easy|medium|hard)$",
        description="Question difficulty level",
    )
    topic: str = Field(
        min_length=1,
        description="Topic name from the source material",
    )
    rubric: str = Field(
        default="",
        description="Grading rubric for open-ended questions",
    )


# =============================================================================
# Langfuse Prompt: MCQ Explanation Generator
# Generates detailed explanations for why MCQ answers are correct/incorrect
# Variables: {question}, {options}, {correct_answer}, {target_audience}
# =============================================================================


class MCQDistractorExplanationSchema(BaseModel):
    """Explanation for a single MCQ distractor option."""

    model_config = ConfigDict(strict=True)

    option_letter: str = Field(
        pattern="^[ABCD]$",
        description="The option letter (A, B, C, or D)",
    )
    option_text: str = Field(
        min_length=1,
        description="The text of this option",
    )
    is_correct: bool = Field(
        default=False,
        description="Whether this is the correct answer",
    )
    explanation: str = Field(
        min_length=1,
        description="Detailed explanation of why this option is correct or incorrect",
    )
    l1_interference_note: str | None = Field(
        default=None,
        description="For distractors: the L1 interference error this represents",
    )


class MCQExplanationOutputSchema(BaseModel):
    """Structured output for the MCQ Explanation Generator prompt.

    Generates detailed explanations for each MCQ option, highlighting:
    - Why the correct answer is correct
    - Why each distractor is incorrect
    - Common L1 interference errors (e.g., Chinese article confusion)

    Prompt Variables:
        - question: The MCQ question stem
        - options: JSON object with A, B, C, D options
        - correct_answer: The correct option letter (A, B, C, D)
        - target_audience: Description of target learners (e.g., Chinese L1, Vietnamese L1)
    """

    model_config = ConfigDict(strict=True)

    question_analysis: str = Field(
        min_length=1,
        description="Brief analysis of what the question tests (grammar point, skill, etc.)",
    )
    option_explanations: list[MCQDistractorExplanationSchema] = Field(
        min_length=4,
        max_length=4,
        description="Explanations for all 4 options",
    )
    teaching_tip: str = Field(
        min_length=1,
        description="A teaching tip for instructors on how to address common errors",
    )
    cefr_level: str = Field(
        pattern="^(A1|A2|B1|B2|C1|C2)$",
        description="Estimated CEFR level of this question",
    )


# =============================================================================
# Langfuse Prompt: MCQ Answer Generator
# Generates model answers and distractors for MCQ questions
# Variables: {question_stem}, {grammar_target}, {difficulty}, {l1_background}
# =============================================================================


class MCQAnswerGeneratorOutputSchema(BaseModel):
    """Structured output for the MCQ Answer Generator prompt.

    Generates complete MCQ answer sets including:
    - The correct answer (grammatically correct)
    - Three distractors reflecting plausible L1 interference errors
    - Rationale for each distractor design

    Prompt Variables:
        - question_stem: The incomplete MCQ question stem
        - grammar_target: The grammar point being tested (e.g., "article usage")
        - difficulty: Target difficulty (easy, medium, hard)
        - l1_background: Target learner L1 (e.g., "Chinese", "Vietnamese", "Mixed")
    """

    model_config = ConfigDict(strict=True)

    question_stem: str = Field(
        min_length=1,
        description="The complete, self-contained question stem",
    )
    correct_answer: MCQDistractorExplanationSchema = Field(
        description="The correct answer with explanation",
    )
    distractors: list[MCQDistractorExplanationSchema] = Field(
        min_length=3,
        max_length=3,
        description="Three incorrect but plausible distractor options",
    )
    grammar_point_tested: str = Field(
        min_length=1,
        description="Specific grammar point this question assesses",
    )
    difficulty_justification: str = Field(
        min_length=1,
        description="Why this difficulty level is appropriate for this content",
    )
    l1_considerations: list[str] = Field(
        default_factory=list,
        description="Specific L1 interference errors targeted by distractors",
    )

    @field_validator("l1_considerations", mode="before")
    @classmethod
    def _normalize_l1_considerations(cls, value: object) -> list[str]:
        """Normalize l1_considerations to always be a list.

        Handles cases where LLM returns a single string, None, or other non-list values.
        """
        if value is None:
            return []
        if isinstance(value, str):
            # If string looks like a list representation, try to parse it
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(item) for item in parsed]
                except json.JSONDecodeError:
                    pass
            # Treat as single item or comma-separated
            if "," in value:
                return [item.strip() for item in value.split(",") if item.strip()]
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(item) for item in value]
        # For any other type, convert to string and wrap in list
        return [str(value)]


# =============================================================================
# Langfuse Prompt: Assessment Generator (v2)
# Generates ELP assessment questions with detailed metadata
# Variables: {structured_count}, {non_structured_count}, {difficulty}, {topics}, {chunks}
# =============================================================================


class AssessmentQuestionSchema(BaseModel):
    """Schema for a single assessment question with rich metadata.

    Accepts both old field names (content, structured_answer) and new field names
    (question_text, answer_text) for backward compatibility.
    """

    model_config = ConfigDict(strict=True)

    question_id: str = Field(
        min_length=1,
        pattern=r"^q-[0-9]+$",
        description="Unique identifier for this question (format: q-{number})",
    )
    question_type: str = Field(
        pattern="^(structured|non_structured)$",
        description="Type: 'structured' for MCQ, 'non_structured' for open-ended",
    )
    # Primary and alias field names - both optional during validation
    question_text: str | None = Field(
        default=None,
        min_length=1,
        description="The complete question text",
    )
    content: str | None = Field(
        default=None,
        min_length=1,
        description="Alias for question_text (backward compatibility)",
    )
    answer_text: str | None = Field(
        default=None,
        min_length=1,
        description="The answer (correct option for MCQ, model answer for open-ended)",
    )
    structured_answer: str | None = Field(
        default=None,
        min_length=1,
        description="Alias for answer_text (backward compatibility)",
    )
    non_structured_model_answer: str | None = Field(
        default=None,
        min_length=1,
        description="Model answer for non-structured questions",
    )
    explanation: str | None = Field(
        default=None,
        description="Explanation of the answer or marking scheme",
    )
    # Union type with metadata variants - validated at runtime
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Question metadata including source grounding and difficulty. Use StructuredQuestionMetadataSchema or NonStructuredQuestionMetadataSchema shape based on question_type.",
    )

    @model_validator(mode="after")
    def _ensure_required_fields(self) -> AssessmentQuestionSchema:
        """Ensure at least one naming convention provides the required fields."""
        # Get values from whichever field name was provided
        question = self.question_text or self.content
        # Answer can come from structured or non-structured field names
        answer = (
            self.answer_text
            or self.structured_answer
            or self.non_structured_model_answer
        )

        if not question:
            raise ValueError("Either 'question_text' or 'content' must be provided")
        if not answer:
            raise ValueError(
                "Either 'answer_text', 'structured_answer', or 'non_structured_model_answer' must be provided"
            )

        # Set the canonical field names
        self.question_text = question
        self.answer_text = answer
        return self


class AssessmentGeneratorOutputSchema(BaseModel):
    """Structured output for the Assessment Generator prompt.

    Generates ELP assessment questions targeting common English difficulties
    for foreign students in Singapore.

    Prompt Variables:
        - structured_count: Number of MCQ questions to generate
        - non_structured_count: Number of open-ended questions to generate
        - difficulty: Target difficulty (easy, medium, hard)
        - topics: Comma-separated list of topics to cover
        - chunks: Formatted knowledge chunks from source material
    """

    model_config = ConfigDict(strict=True)

    questions: list[AssessmentQuestionSchema] = Field(
        min_length=1,
        description="List of generated questions (both MCQ and open-ended)",
    )


# =============================================================================
# Prompt Input Schemas (for variable validation)
# =============================================================================


class AssessmentGeneratorInputSchema(BaseModel):
    """Input variables for the Assessment Generator prompt.

    Use this to validate inputs before compiling the prompt:
        prompt = await provider.get_assessment_generator_prompt()
        compiled = prompt.compile(**input_schema.model_dump())
    """

    model_config = ConfigDict(strict=True)

    structured_count: int = Field(
        ge=0,
        description="Number of MCQ questions to generate",
    )
    non_structured_count: int = Field(
        ge=0,
        description="Number of open-ended questions to generate",
    )
    difficulty: str = Field(
        pattern="^(easy|medium|hard)$",
        description="Target difficulty level",
    )
    topics: str = Field(
        min_length=1,
        description="Comma-separated list of topics to cover",
    )
    chunks: str = Field(
        min_length=1,
        description="Formatted knowledge chunks from source material",
    )

    @classmethod
    def from_context(
        cls,
        context: AssessmentContext,
        structured_count: int,
        non_structured_count: int,
        difficulty_level: str | None,
    ) -> AssessmentGeneratorInputSchema:
        """Build input schema from AssessmentContext."""
        chunks_formatted = "\n\n".join(
            f"[Chunk {i}] {chunk}" for i, chunk in enumerate(context.chunks, start=1)
        )
        return cls(
            structured_count=structured_count,
            non_structured_count=non_structured_count,
            difficulty=difficulty_level or "medium",
            topics=", ".join(context.topic_ids),
            chunks=chunks_formatted,
        )


class MCQExplanationInputSchema(BaseModel):
    """Input variables for the MCQ Explanation Generator prompt."""

    model_config = ConfigDict(strict=True)

    question: str = Field(
        min_length=1,
        description="The MCQ question stem",
    )
    options: str = Field(
        min_length=1,
        description="JSON string with A, B, C, D options",
    )
    correct_answer: str = Field(
        pattern="^[ABCD]$",
        description="The correct option letter",
    )
    target_audience: str = Field(
        min_length=1,
        description="Description of target learners (e.g., 'Chinese L1 students')",
    )


class MCQAnswerInputSchema(BaseModel):
    """Input variables for the MCQ Answer Generator prompt."""

    model_config = ConfigDict(strict=True)

    question_stem: str = Field(
        min_length=1,
        description="The incomplete MCQ question stem",
    )
    grammar_target: str = Field(
        min_length=1,
        description="Specific grammar point being tested",
    )
    difficulty: str = Field(
        pattern="^(easy|medium|hard)$",
        description="Target difficulty level",
    )
    l1_background: str = Field(
        min_length=1,
        description="Target learner L1 background (e.g., 'Chinese', 'Vietnamese')",
    )


# =============================================================================
# Legacy Prompt Builders (for backward compatibility)
# =============================================================================


def build_system_prompt(question_type: QuestionType) -> str:
    """Build a system prompt for the given question type.

    This is the legacy prompt builder. New code should use Langfuse prompts.

    Args:
        question_type: Type of question to generate.

    Returns:
        System prompt string.
    """
    if question_type is QuestionType.STRUCTURED:
        return (
            "You are an expert assessment creator for English language proficiency tests. "
            "Generate multiple-choice questions (MCQs) that test specific grammar points "
            "commonly challenging for foreign students in Singapore. "
            "Provide the question, the correct answer, and a brief explanation."
        )
    return (
        "You are an expert assessment creator for English language proficiency tests. "
        "Generate open-ended questions that test comprehension and application of concepts "
        "from the provided material. Provide the question and a model answer."
    )


def build_user_prompt(
    context: AssessmentContext,
    count: int,
    difficulty_level: str | None,
    question_type: QuestionType,
) -> str:
    """Build a user prompt for the given context and parameters.

    This is the legacy prompt builder. New code should use Langfuse prompts.

    Args:
        context: Assessment context with chunks and topic IDs.
        count: Number of questions to generate.
        difficulty_level: Target difficulty (easy, medium, hard).
        question_type: Type of question to generate.

    Returns:
        User prompt string.
    """
    difficulty = difficulty_level or "medium"
    type_label = (
        "multiple-choice" if question_type is QuestionType.STRUCTURED else "open-ended"
    )

    chunks_text = "\n\n".join(
        f"Chunk {i + 1}:\n{chunk}" for i, chunk in enumerate(context.chunks)
    )
    topics_text = ", ".join(context.topic_ids)

    schema_description = (
        "Return your response as a JSON object with this exact structure. "
        "Use ONLY these exact field names - do not use 'text', 'content', or other variations:\n"
        "{\n"
        '  "questions": [\n'
        "    {\n"
        '      "question_text": "The complete question text here",\n'
        '      "answer_text": "The complete answer text here",\n'
        '      "explanation": "Optional explanation here",\n'
        '      "references": ["optional"],\n'
        '      "topic_id": "optional",\n'
        '      "metadata": {}\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "CRITICAL: Both 'question_text' and 'answer_text' fields are REQUIRED for each question. "
        "Do not omit the answer. "
        f"Generate exactly {count} questions."
    )

    return (
        f"Generate {count} {difficulty} difficulty {type_label} questions\n"
        f"based on the following topics: {topics_text}\n\n"
        f"Source Material:\n{chunks_text}\n\n"
        f"{schema_description}"
    )
