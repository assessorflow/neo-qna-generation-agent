"""User prompt builder for the 3-prompt workflow.

Builds dynamic user prompts from templates with variable substitution.
System prompts are fetched separately from Langfuse as static text.
"""

from __future__ import annotations

from qna_generation_agent.application.errors import ValidationError

# =============================================================================
# Template Constants
# =============================================================================

_ASSESSMENT_GENERATOR_TEMPLATE: str = """Generate {structured_count} structured (MCQ) questions and {non_structured_count} open-ended questions at {difficulty} difficulty level.

Topics: {topics}

Source Material:
{chunks}

Return your response as a JSON object matching the AssessmentGeneratorOutputSchema structure with questions containing question_id, question_type, question_text, answer_text, explanation, and metadata fields."""

_MCQ_ANSWER_GENERATOR_TEMPLATE: str = """Complete the following MCQ question by generating the correct answer and three plausible distractors.

Question: {question_text}
Topic: {topic}
Difficulty: {difficulty}

Context: {chunk_content}

Return your response as a JSON object matching the MCQAnswerGeneratorOutputSchema structure with question_stem, correct_answer, distractors, grammar_point_tested, difficulty_justification, and l1_considerations fields."""

_MCQ_EXPLANATION_GENERATOR_TEMPLATE: str = """Generate detailed explanations for the following MCQ question and its options.

Question: {question_text}
Topic: {topic}
Correct Answer: {correct_answer}

Options:
A) {option_a}
B) {option_b}
C) {option_c}
D) {option_d}

Context: {chunk_content}

Return your response as a JSON object matching the MCQExplanationOutputSchema structure with question_analysis, option_explanations, teaching_tip, cefr_level, related_grammar, and common_errors fields."""


# =============================================================================
# UserPromptBuilder Class
# =============================================================================


class UserPromptBuilder:
    """Builds user prompts from templates for the 3-prompt workflow.

    User prompts are built in code (not fetched from Langfuse) and contain
    the dynamic, request-specific content for each LLM call.
    """

    @staticmethod
    def _validate_difficulty(difficulty: str) -> None:
        """Validate that difficulty is one of the allowed values.

        Args:
            difficulty: The difficulty level to validate.

        Raises:
            ValidationError: If difficulty is not "easy", "medium", or "hard".
        """
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValidationError(
                f"Invalid difficulty: {difficulty!r}. Must be one of: easy, medium, hard",
                field="difficulty",
                value=difficulty,
            )

    @staticmethod
    def _validate_non_negative_integer(value: int, name: str) -> None:
        """Validate that a value is a non-negative integer.

        Args:
            value: The integer value to validate.
            name: The name of the field for error messages.

        Raises:
            ValidationError: If value is negative or not an integer.
        """
        if not isinstance(value, int) or value < 0:
            raise ValidationError(
                f"{name} must be a non-negative integer, got {value!r}",
                field=name,
                value=value,
            )

    def build_assessment_generator_prompt(
        self,
        *,
        structured_count: int,
        non_structured_count: int,
        difficulty: str,
        topics: str,
        chunks: str,
    ) -> str:
        """Build the user prompt for the Assessment Generator.

        Args:
            structured_count: Number of MCQ questions to generate.
            non_structured_count: Number of open-ended questions to generate.
            difficulty: Target difficulty ("easy", "medium", or "hard").
            topics: Comma-separated list of topics to cover.
            chunks: Formatted knowledge chunks from source material.

        Returns:
            The compiled user prompt string.

        Raises:
            ValidationError: If inputs are invalid.
        """
        self._validate_non_negative_integer(structured_count, "structured_count")
        self._validate_non_negative_integer(non_structured_count, "non_structured_count")
        self._validate_difficulty(difficulty)

        return _ASSESSMENT_GENERATOR_TEMPLATE.format(
            structured_count=structured_count,
            non_structured_count=non_structured_count,
            difficulty=difficulty,
            topics=topics,
            chunks=chunks,
        )

    def build_mcq_answer_generator_prompt(
        self,
        *,
        question_text: str,
        topic: str,
        difficulty: str,
        chunk_content: str,
    ) -> str:
        """Build the user prompt for the MCQ Answer Generator.

        Args:
            question_text: The incomplete MCQ question stem.
            topic: The grammar point or topic being tested.
            difficulty: Target difficulty ("easy", "medium", or "hard").
            chunk_content: Source chunk content or target learner L1 background.

        Returns:
            The compiled user prompt string.

        Raises:
            ValidationError: If inputs are invalid.
        """
        self._validate_difficulty(difficulty)

        return _MCQ_ANSWER_GENERATOR_TEMPLATE.format(
            question_text=question_text,
            topic=topic,
            difficulty=difficulty,
            chunk_content=chunk_content,
        )

    def build_mcq_explanation_generator_prompt(
        self,
        *,
        question_text: str,
        topic: str,
        correct_answer: str,
        option_a: str,
        option_b: str,
        option_c: str,
        option_d: str,
        chunk_content: str,
    ) -> str:
        """Build the user prompt for the MCQ Explanation Generator.

        Args:
            question_text: The MCQ question stem.
            topic: Topic or target audience description.
            correct_answer: The correct option letter (A, B, C, or D).
            option_a: Text for option A.
            option_b: Text for option B.
            option_c: Text for option C.
            option_d: Text for option D.
            chunk_content: Source context or audience info.

        Returns:
            The compiled user prompt string.

        Raises:
            ValidationError: If inputs are invalid.
        """
        if correct_answer not in {"A", "B", "C", "D"}:
            raise ValidationError(
                f"Invalid correct_answer: {correct_answer!r}. Must be one of: A, B, C, D",
                field="correct_answer",
                value=correct_answer,
            )

        return _MCQ_EXPLANATION_GENERATOR_TEMPLATE.format(
            question_text=question_text,
            topic=topic,
            correct_answer=correct_answer,
            option_a=option_a,
            option_b=option_b,
            option_c=option_c,
            option_d=option_d,
            chunk_content=chunk_content,
        )
