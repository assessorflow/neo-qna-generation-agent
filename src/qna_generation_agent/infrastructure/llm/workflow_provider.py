"""Strands workflow-based LLM provider for parallel question generation.

This module implements a three-level parallel workflow architecture using
Langfuse-managed prompts for all generation stages.

Level 1: Assessment Generation (Parallel by Subtopic)
- Fetch "Assessment Generator" prompt from Langfuse
- Spawn parallel agents for each subtopic
- Aggregate question stems

Level 2: MCQ Answer Generation (Parallel by Question)
- Fetch "MCQ Answer Generator" prompt from Langfuse
- Spawn parallel agents for each question
- Aggregate answers with distractors

Level 3: MCQ Explanation Generation (Parallel by Question)
- Fetch "MCQ Explanation Generator" prompt from Langfuse
- Spawn parallel agents for each question
- Aggregate detailed explanations

Parallel Track: Non-Structured Generation
- Fetch "Assessment Generator" prompt (with non_structured_count > 0)
- Spawn parallel agents for each subtopic
- Aggregate open-ended questions
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel
from strands import Agent
from strands.models.openai import OpenAIModel
from strands_tools import workflow

from qna_generation_agent.app.json import dumps, loads
from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.application.dto import (
    AssessmentContext,
    QuestionBatch,
    QuestionDraft,
)
from qna_generation_agent.application.errors import LLMPermanentError, LLMTransientError
from qna_generation_agent.application.ports.llm import LLMProvider
from qna_generation_agent.domain.enums import QuestionType

if TYPE_CHECKING:
    from qna_generation_agent.application.ports.prompt_provider import PromptProvider

logger = get_logger(__name__)


class QuestionStem(BaseModel):
    """Intermediate representation of a generated question stem."""

    question_id: str
    content: str
    topic: str
    difficulty: str
    question_type: str


class MCQAnswer(BaseModel):
    """MCQ answer with correct answer and distractors."""

    question_id: str
    question_stem: str
    correct_letter: str
    correct_text: str
    correct_explanation: str
    distractors: list[dict[str, Any]]
    grammar_point: str
    difficulty_justification: str
    l1_considerations: list[str]


class MCQExplanation(BaseModel):
    """Detailed explanation for an MCQ question."""

    question_id: str
    question_analysis: str
    cefr_level: str
    teaching_tip: str
    option_explanations: list[dict[str, Any]]


class FinalQuestion(BaseModel):
    """Fully generated question with all components."""

    question_id: str
    text: str
    question_type: QuestionType
    difficulty_level: str | None
    topic_id: str | None
    metadata: dict[str, str]
    answer_text: str
    explanation: str


class WorkflowLLMProvider(LLMProvider):
    """Workflow-based LLM provider with parallel generation stages.

    Requires a PromptProvider (typically LangfusePromptProvider) to fetch
    versioned prompts for all generation stages.
    """

    def __init__(
        self,
        *,
        model_provider: str,
        model_id: str,
        api_key: str,
        base_url: str | None,
        timeout_seconds: int,
        prompt_provider: PromptProvider | None = None,
    ) -> None:
        if model_provider != "openai":
            raise LLMPermanentError(
                "Only the OpenAI Strands backend is configured",
                provider=model_provider,
            )

        self._model_id = model_id
        self._timeout_seconds = timeout_seconds
        self._prompt_provider = prompt_provider

        # Workflow mode requires prompt provider for Langfuse integration
        if self._prompt_provider is None:
            raise LLMPermanentError(
                "WorkflowLLMProvider requires a PromptProvider. "
                "Ensure LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set."
            )

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

        # Create orchestrator agent with workflow tool
        self._orchestrator = Agent(
            model=self._model,
            system_prompt="You are a workflow orchestrator for assessment generation.",
            tools=[workflow],
        )

    @property
    def model_id(self) -> str:
        """Return the model ID."""
        return self._model_id

    @property
    def timeout_seconds(self) -> int:
        """Return the timeout setting."""
        return self._timeout_seconds

    # =======================================================================
    # Langfuse Prompt Fetching
    # =======================================================================

    async def _get_assessment_generator_prompt(
        self,
        subtopic: str,
        chunks: list[str],
        structured_count: int,
        non_structured_count: int,
        difficulty: str | None,
    ) -> str:
        """Fetch and compile Assessment Generator prompt from Langfuse.

        Args:
            subtopic: Topic name for this generation batch
            chunks: Reference document chunks
            structured_count: Number of structured (MCQ) questions
            non_structured_count: Number of open-ended questions
            difficulty: Target difficulty level

        Returns:
            Compiled prompt string ready for workflow task

        Raises:
            LLMPermanentError: If prompt provider is not available
            LLMTransientError: If prompt fetch fails
        """
        if self._prompt_provider is None:
            raise LLMPermanentError("Prompt provider required for workflow generation")

        try:
            prompt_obj = await self._prompt_provider.get_prompt(
                "Assessment Generator",
                label="latest",
            )

            chunks_formatted = "\n\n".join(
                f"[Chunk {i}] {chunk}" for i, chunk in enumerate(chunks[:5], start=1)
            )

            compiled = prompt_obj.compile(
                structured_count=structured_count,
                non_structured_count=non_structured_count,
                difficulty=difficulty or "medium",
                topics=subtopic,
                chunks=chunks_formatted,
            )

            logger.debug(
                "langfuse_prompt_fetched",
                prompt_name="Assessment Generator",
                prompt_version=f"{prompt_obj.name}@v{prompt_obj.version}",
                subtopic=subtopic,
            )

            return str(compiled)

        except Exception as e:
            logger.error(
                "langfuse_prompt_fetch_failed",
                prompt_name="Assessment Generator",
                error=str(e),
            )
            raise LLMTransientError(
                f"Failed to fetch 'Assessment Generator' prompt from Langfuse: {e}",
                retry_after_seconds=5,
            ) from e

    async def _get_mcq_answer_generator_prompt(
        self,
        question_stem: str,
        difficulty: str,
    ) -> str:
        """Fetch and compile MCQ Answer Generator prompt from Langfuse.

        Args:
            question_stem: The MCQ question text
            difficulty: Target difficulty level

        Returns:
            Compiled prompt string ready for workflow task
        """
        if self._prompt_provider is None:
            raise LLMPermanentError("Prompt provider required for workflow generation")

        try:
            prompt_obj = await self._prompt_provider.get_prompt(
                "MCQ Answer Generator",
                label="latest",
            )

            # Extract grammar target from question stem (simple heuristic)
            grammar_target = self._extract_grammar_target(question_stem)

            compiled = prompt_obj.compile(
                question_stem=question_stem,
                grammar_target=grammar_target,
                difficulty=difficulty,
                l1_background="Mixed",  # TODO: Fetch from assessment config
            )

            logger.debug(
                "langfuse_prompt_fetched",
                prompt_name="MCQ Answer Generator",
                prompt_version=f"{prompt_obj.name}@v{prompt_obj.version}",
            )

            return str(compiled)

        except Exception as e:
            logger.error(
                "langfuse_prompt_fetch_failed",
                prompt_name="MCQ Answer Generator",
                error=str(e),
            )
            raise LLMTransientError(
                f"Failed to fetch 'MCQ Answer Generator' prompt: {e}",
                retry_after_seconds=5,
            ) from e

    async def _get_mcq_explanation_generator_prompt(
        self,
        question_stem: str,
        options: dict[str, str],
        correct_letter: str,
    ) -> str:
        """Fetch and compile MCQ Explanation Generator prompt from Langfuse.

        Args:
            question_stem: The MCQ question text
            options: Dict mapping option letters (A,B,C,D) to option text
            correct_letter: The correct answer letter

        Returns:
            Compiled prompt string ready for workflow task
        """
        if self._prompt_provider is None:
            raise LLMPermanentError("Prompt provider required for workflow generation")

        try:
            prompt_obj = await self._prompt_provider.get_prompt(
                "MCQ Explanation Generator",
                label="latest",
            )

            compiled = prompt_obj.compile(
                question=question_stem,
                options=dumps(options).decode("utf-8"),
                correct_answer=correct_letter,
                target_audience="Foreign students in Singapore (Chinese/Vietnamese/Malay L1)",
            )

            logger.debug(
                "langfuse_prompt_fetched",
                prompt_name="MCQ Explanation Generator",
                prompt_version=f"{prompt_obj.name}@v{prompt_obj.version}",
            )

            return str(compiled)

        except Exception as e:
            logger.error(
                "langfuse_prompt_fetch_failed",
                prompt_name="MCQ Explanation Generator",
                error=str(e),
            )
            raise LLMTransientError(
                f"Failed to fetch 'MCQ Explanation Generator' prompt: {e}",
                retry_after_seconds=5,
            ) from e

    async def _get_non_structured_generator_prompt(
        self,
        subtopic: str,
        chunks: list[str],
        count: int,
        difficulty: str | None,
    ) -> str:
        """Fetch and compile prompt for non-structured question generation.

        Uses the same "Assessment Generator" prompt but with non_structured_count > 0.

        Args:
            subtopic: Topic name for this generation batch
            chunks: Reference document chunks
            count: Number of open-ended questions to generate
            difficulty: Target difficulty level

        Returns:
            Compiled prompt string ready for workflow task
        """
        # Use Assessment Generator prompt configured for non-structured only
        return await self._get_assessment_generator_prompt(
            subtopic=subtopic,
            chunks=chunks,
            structured_count=0,
            non_structured_count=count,
            difficulty=difficulty,
        )

    # =======================================================================
    # Generation Methods
    # =======================================================================

    async def generate_structured(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> QuestionBatch:
        """Generate structured questions using three-level parallel workflow."""
        del correlation_id  # Used for tracing via telemetry

        # Distribute count across subtopics (topics from context)
        subtopics = self._distribute_subtopics(context, count)

        # ======================================================================
        # Level 1: Parallel Assessment Generation per Subtopic
        # ======================================================================
        logger.info(
            "workflow_level_1_start",
            subtopic_count=len(subtopics),
            total_questions=count,
        )

        # Pre-fetch all Level 1 prompts from Langfuse
        try:
            level1_prompts: list[str] = await self._gather_with_error_handling(
                [
                    self._get_assessment_generator_prompt(
                        subtopic=st["name"],
                        chunks=context.chunks,
                        structured_count=st["question_count"],
                        non_structured_count=0,
                        difficulty=difficulty_level,
                    )
                    for st in subtopics
                ]
            )
        except LLMTransientError:
            raise
        except Exception as e:
            raise LLMPermanentError(f"Failed to fetch Level 1 prompts: {e}") from e

        level1_tasks = [
            {
                "task_id": f"assess_{st['topic_id']}",
                "description": prompt,
                "priority": 5,
                "timeout": self._timeout_seconds,
            }
            for st, prompt in zip(subtopics, level1_prompts, strict=False)
        ]

        workflow_id = f"assessment_{uuid.uuid4().hex[:8]}"

        # Create and start workflow
        create_result = self._orchestrator.tool.workflow(
            action="create",
            workflow_id=workflow_id,
            tasks=level1_tasks,
        )

        if create_result.get("status") != "success":
            raise LLMPermanentError("Failed to create Level 1 workflow")

        self._orchestrator.tool.workflow(action="start", workflow_id=workflow_id)

        # Parse Level 1 results into QuestionStems
        question_stems = self._parse_level1_results(workflow_id)

        logger.info(
            "workflow_level_1_complete",
            questions_generated=len(question_stems),
        )

        if not question_stems:
            raise LLMPermanentError("Level 1 produced no question stems")

        # ======================================================================
        # Level 2: Parallel MCQ Answer Generation per Question
        # ======================================================================
        logger.info(
            "workflow_level_2_start",
            question_count=len(question_stems),
        )

        # Fetch Level 2 prompts (depends on Level 1 output)
        try:
            level2_prompts: list[str] = await self._gather_with_error_handling(
                [
                    self._get_mcq_answer_generator_prompt(
                        question_stem=qs.content,
                        difficulty=difficulty_level or "medium",
                    )
                    for qs in question_stems
                ]
            )
        except LLMTransientError:
            raise
        except Exception as e:
            raise LLMPermanentError(f"Failed to fetch Level 2 prompts: {e}") from e

        level2_tasks = [
            {
                "task_id": f"mcq_{qs.question_id}",
                "description": prompt,
                "priority": 5,
                "timeout": self._timeout_seconds,
            }
            for qs, prompt in zip(question_stems, level2_prompts, strict=False)
        ]

        workflow_id_l2 = f"mcq_{uuid.uuid4().hex[:8]}"

        create_result_l2 = self._orchestrator.tool.workflow(
            action="create",
            workflow_id=workflow_id_l2,
            tasks=level2_tasks,
        )

        if create_result_l2.get("status") != "success":
            raise LLMPermanentError("Failed to create Level 2 workflow")

        self._orchestrator.tool.workflow(action="start", workflow_id=workflow_id_l2)

        mcq_answers = self._parse_level2_results(workflow_id_l2)

        logger.info(
            "workflow_level_2_complete",
            answers_generated=len(mcq_answers),
        )

        if not mcq_answers:
            raise LLMPermanentError("Level 2 produced no MCQ answers")

        # ======================================================================
        # Level 3: Parallel MCQ Explanation Generation per Question
        # ======================================================================
        logger.info(
            "workflow_level_3_start",
            question_count=len(mcq_answers),
        )

        # Fetch Level 3 prompts (depends on Level 2 output)
        try:
            level3_prompts: list[str] = await self._gather_with_error_handling(
                [
                    self._get_mcq_explanation_generator_prompt(
                        question_stem=ans.question_stem,
                        options=self._build_options_dict(ans),
                        correct_letter=ans.correct_letter,
                    )
                    for ans in mcq_answers
                ]
            )
        except LLMTransientError:
            raise
        except Exception as e:
            raise LLMPermanentError(f"Failed to fetch Level 3 prompts: {e}") from e

        level3_tasks = [
            {
                "task_id": f"exp_{ans.question_id}",
                "description": prompt,
                "priority": 4,
                "timeout": self._timeout_seconds,
            }
            for ans, prompt in zip(mcq_answers, level3_prompts, strict=False)
        ]

        workflow_id_l3 = f"exp_{uuid.uuid4().hex[:8]}"

        create_result_l3 = self._orchestrator.tool.workflow(
            action="create",
            workflow_id=workflow_id_l3,
            tasks=level3_tasks,
        )

        if create_result_l3.get("status") != "success":
            raise LLMPermanentError("Failed to create Level 3 workflow")

        self._orchestrator.tool.workflow(action="start", workflow_id=workflow_id_l3)

        explanations = self._parse_level3_results(workflow_id_l3)

        logger.info(
            "workflow_level_3_complete",
            explanations_generated=len(explanations),
        )

        # ======================================================================
        # Final Aggregation
        # ======================================================================
        final_questions = self._aggregate_final_questions(
            question_stems, mcq_answers, explanations, difficulty_level
        )

        questions = [
            QuestionDraft(
                question_text=fq.text,
                answer_text=fq.answer_text,
                question_type=QuestionType.STRUCTURED,
                difficulty_level=fq.difficulty_level,
                explanation=fq.explanation,
                references=[],
                topic_id=fq.topic_id,
                metadata=fq.metadata,
            )
            for fq in final_questions
        ]

        return QuestionBatch(
            questions=questions,
            model_name=self._model_id,
            prompt_version="langfuse_workflow_v1",
        )

    async def generate_non_structured(
        self,
        *,
        context: AssessmentContext,
        count: int,
        difficulty_level: str | None,
        correlation_id: str,
    ) -> QuestionBatch:
        """Generate non-structured questions using parallel workflow per subtopic."""
        del correlation_id

        subtopics = self._distribute_subtopics(context, count)

        logger.info(
            "workflow_non_structured_start",
            subtopic_count=len(subtopics),
            total_questions=count,
        )

        # Fetch prompts from Langfuse
        try:
            prompts: list[str] = await self._gather_with_error_handling(
                [
                    self._get_non_structured_generator_prompt(
                        subtopic=st["name"],
                        chunks=context.chunks,
                        count=st["question_count"],
                        difficulty=difficulty_level,
                    )
                    for st in subtopics
                ]
            )
        except LLMTransientError:
            raise
        except Exception as e:
            raise LLMPermanentError(
                f"Failed to fetch non-structured prompts: {e}"
            ) from e

        tasks = [
            {
                "task_id": f"ns_{st['topic_id']}",
                "description": prompt,
                "priority": 5,
                "timeout": self._timeout_seconds,
            }
            for st, prompt in zip(subtopics, prompts, strict=False)
        ]

        workflow_id = f"ns_{uuid.uuid4().hex[:8]}"

        create_result = self._orchestrator.tool.workflow(
            action="create",
            workflow_id=workflow_id,
            tasks=tasks,
        )

        if create_result.get("status") != "success":
            raise LLMPermanentError("Failed to create non-structured workflow")

        self._orchestrator.tool.workflow(action="start", workflow_id=workflow_id)

        questions = self._parse_non_structured_results(workflow_id)

        logger.info(
            "workflow_non_structured_complete",
            questions_generated=len(questions),
        )

        return QuestionBatch(
            questions=questions,
            model_name=self._model_id,
            prompt_version="langfuse_workflow_v1",
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
        """Generate questions using a pre-compiled prompt (fallback)."""
        del correlation_id

        # For workflow provider, use single task workflow with provided prompt
        task = {
            "task_id": "single_generation",
            "description": prompt,
            "priority": 5,
            "timeout": self._timeout_seconds,
        }

        workflow_id = f"single_{uuid.uuid4().hex[:8]}"

        create_result = self._orchestrator.tool.workflow(
            action="create",
            workflow_id=workflow_id,
            tasks=[task],
        )

        if create_result.get("status") != "success":
            raise LLMPermanentError("Failed to create single workflow")

        self._orchestrator.tool.workflow(action="start", workflow_id=workflow_id)

        # Parse generic results
        status = self._orchestrator.tool.workflow(
            action="status", workflow_id=workflow_id
        )

        if status.get("status") != "success":
            raise LLMPermanentError("Failed to get workflow status")

        # Extract results and create QuestionDrafts
        qt = (
            QuestionType.STRUCTURED
            if question_type == "structured"
            else QuestionType.NON_STRUCTURED
        )

        questions = self._extract_questions_from_status(status, qt, difficulty_level)

        return QuestionBatch(
            questions=questions[:count],
            model_name=self._model_id,
            prompt_version="langfuse_workflow_single_v1",
        )

    async def invoke_with_schema[T: BaseModel](
        self,
        prompt: str,
        *,
        structured_output_model: type[T],
    ) -> T | None:
        """Invoke LLM with structured output schema using the orchestrator agent.

        This method uses the underlying Strands agent for schema-validated calls,
        bypassing the workflow tool for single-shot structured generation.

        Args:
            prompt: The prompt to send to the LLM.
            structured_output_model: The Pydantic model for the response.

        Returns:
            The structured output or None if parsing failed.
        """
        try:
            result = await self._orchestrator.invoke_async(
                prompt,
                structured_output_model=structured_output_model,
            )
            if result and result.structured_output:
                return cast(T, result.structured_output)
            return None
        except Exception as error:
            logger.warning(
                "invoke_with_schema_failed",
                error=str(error),
                error_type=type(error).__name__,
            )
            return None

    async def health_check(self) -> bool:
        """Check if the workflow system is functional."""
        try:
            # Test with minimal workflow
            test_result = self._orchestrator.tool.workflow(
                action="create",
                workflow_id=f"health_{uuid.uuid4().hex[:8]}",
                tasks=[
                    {
                        "task_id": "health_check",
                        "description": "Respond with 'healthy' if you can read this.",
                        "priority": 1,
                        "timeout": 10,
                    }
                ],
            )
            return bool(test_result.get("status") == "success")
        except Exception:
            return False

    # =======================================================================
    # Helper Methods
    # =======================================================================

    async def _gather_with_error_handling[T](self, coroutines: list[Any]) -> list[T]:
        """Gather async results with proper error handling.

        Raises LLMTransientError on any failure to allow retry.
        """
        import asyncio

        results = await asyncio.gather(*coroutines, return_exceptions=True)

        for result in results:
            if isinstance(result, BaseException):
                if isinstance(result, LLMTransientError):
                    raise result
                raise LLMTransientError(
                    f"Prompt fetch failed: {result}",
                    retry_after_seconds=5,
                ) from result

        return results  # type: ignore[return-value]

    def _distribute_subtopics(
        self, context: AssessmentContext, total_count: int
    ) -> list[dict[str, Any]]:
        """Distribute question count across available subtopics."""
        topic_ids = context.topic_ids if context.topic_ids else ["general"]

        base_count = total_count // len(topic_ids)
        remainder = total_count % len(topic_ids)

        subtopics = []
        for i, topic_id in enumerate(topic_ids):
            count = base_count + (1 if i < remainder else 0)
            if count > 0:
                subtopics.append(
                    {
                        "topic_id": topic_id,
                        "name": topic_id.replace("_", " ").title(),
                        "question_count": count,
                    }
                )

        return subtopics

    def _extract_grammar_target(self, question_stem: str) -> str:
        """Extract grammar target from question stem (simple heuristic)."""
        # TODO: Make this smarter or fetch from assessment config
        lower_stem = question_stem.lower()
        if any(w in lower_stem for w in ["had ", "past", "perfect"]):
            return "past perfect tense"
        if any(w in lower_stem for w in ["article", "a ", "an ", "the "]):
            return "article usage"
        if any(w in lower_stem for w in ["preposition", "in ", "on ", "at "]):
            return "preposition usage"
        if any(w in lower_stem for w in ["tense", "verb form"]):
            return "tense consistency"
        return "grammar application"

    def _build_options_dict(self, mcq_answer: MCQAnswer) -> dict[str, str]:
        """Build options dictionary from MCQ answer."""
        options = {mcq_answer.correct_letter: mcq_answer.correct_text}
        for d in mcq_answer.distractors:
            options[d.get("letter", "?")] = d.get("text", "")
        return options

    # =======================================================================
    # Result Parsing Methods
    # =======================================================================

    def _extract_questions_from_status(
        self,
        status: dict[str, Any],
        question_type: QuestionType,
        difficulty_level: str | None,
    ) -> list[QuestionDraft]:
        """Extract QuestionDrafts from workflow status."""
        questions: list[QuestionDraft] = []

        content = status.get("content", [])
        if content and isinstance(content[0], dict):
            text = content[0].get("text", "")
            try:
                workflow_data = loads(text.split("\n")[-1].encode()) if text else {}
                task_results = workflow_data.get("task_results", {})

                for _task_id, result in task_results.items():
                    if result.get("status") == "completed":
                        result_content = result.get("result", [])
                        if result_content:
                            for item in result_content:
                                if isinstance(item, dict) and "text" in item:
                                    try:
                                        data = loads(item["text"].encode())
                                        if isinstance(data, list):
                                            for q in data:
                                                questions.append(
                                                    QuestionDraft(
                                                        question_text=q.get(
                                                            "question", ""
                                                        ),
                                                        answer_text=q.get("answer", ""),
                                                        question_type=question_type,
                                                        difficulty_level=difficulty_level,
                                                        explanation=q.get(
                                                            "explanation", ""
                                                        ),
                                                        references=[],
                                                        topic_id=q.get("topic"),
                                                        metadata=q.get("metadata", {}),
                                                    )
                                                )
                                    except ValueError:
                                        continue
            except (ValueError, IndexError):
                pass

        return questions

    def _parse_level1_results(self, workflow_id: str) -> list[QuestionStem]:
        """Parse Level 1 workflow results into QuestionStems."""
        status = self._orchestrator.tool.workflow(
            action="status", workflow_id=workflow_id
        )

        question_stems: list[QuestionStem] = []

        content = status.get("content", [])
        if content and isinstance(content[0], dict):
            text = content[0].get("text", "")
            try:
                workflow_data = loads(text.split("\n")[-1].encode()) if text else {}
                task_results = workflow_data.get("task_results", {})

                for _task_id, result in task_results.items():
                    if result.get("status") == "completed":
                        result_content = result.get("result", [])
                        if result_content:
                            for item in result_content:
                                if isinstance(item, dict) and "text" in item:
                                    try:
                                        data = loads(item["text"].encode())
                                        if isinstance(data, list):
                                            for q in data:
                                                question_stems.append(
                                                    QuestionStem(
                                                        question_id=f"q_{uuid.uuid4().hex[:8]}",
                                                        content=q.get("content", ""),
                                                        topic=q.get("topic", ""),
                                                        difficulty=q.get(
                                                            "difficulty", "medium"
                                                        ),
                                                        question_type=q.get(
                                                            "type", "structured"
                                                        ),
                                                    )
                                                )
                                    except ValueError:
                                        continue
            except (ValueError, IndexError):
                pass

        return question_stems

    def _parse_level2_results(self, workflow_id: str) -> list[MCQAnswer]:
        """Parse Level 2 workflow results into MCQAnswers."""
        status = self._orchestrator.tool.workflow(
            action="status", workflow_id=workflow_id
        )

        mcq_answers: list[MCQAnswer] = []

        content = status.get("content", [])
        if content and isinstance(content[0], dict):
            text = content[0].get("text", "")
            try:
                workflow_data = loads(text.split("\n")[-1].encode()) if text else {}
                task_results = workflow_data.get("task_results", {})

                for _task_id, result in task_results.items():
                    if result.get("status") == "completed":
                        result_content = result.get("result", [])
                        if result_content:
                            for item in result_content:
                                if isinstance(item, dict) and "text" in item:
                                    try:
                                        data = loads(item["text"].encode())
                                        correct = data.get("correct_answer", {})
                                        mcq_answers.append(
                                            MCQAnswer(
                                                question_id=f"q_{uuid.uuid4().hex[:8]}",
                                                question_stem=data.get(
                                                    "question_stem", ""
                                                ),
                                                correct_letter=correct.get(
                                                    "option_letter", "A"
                                                ),
                                                correct_text=correct.get(
                                                    "option_text", ""
                                                ),
                                                correct_explanation=correct.get(
                                                    "explanation", ""
                                                ),
                                                distractors=data.get("distractors", []),
                                                grammar_point=data.get(
                                                    "grammar_point_tested", ""
                                                ),
                                                difficulty_justification=data.get(
                                                    "difficulty_justification", ""
                                                ),
                                                l1_considerations=data.get(
                                                    "l1_considerations", []
                                                ),
                                            )
                                        )
                                    except ValueError:
                                        continue
            except (ValueError, IndexError):
                pass

        return mcq_answers

    def _parse_level3_results(self, workflow_id: str) -> list[MCQExplanation]:
        """Parse Level 3 workflow results into MCQExplanations."""
        status = self._orchestrator.tool.workflow(
            action="status", workflow_id=workflow_id
        )

        explanations: list[MCQExplanation] = []

        content = status.get("content", [])
        if content and isinstance(content[0], dict):
            text = content[0].get("text", "")
            try:
                workflow_data = loads(text.split("\n")[-1].encode()) if text else {}
                task_results = workflow_data.get("task_results", {})

                for _task_id, result in task_results.items():
                    if result.get("status") == "completed":
                        result_content = result.get("result", [])
                        if result_content:
                            for item in result_content:
                                if isinstance(item, dict) and "text" in item:
                                    try:
                                        data = loads(item["text"].encode())
                                        explanations.append(
                                            MCQExplanation(
                                                question_id=f"q_{uuid.uuid4().hex[:8]}",
                                                question_analysis=data.get(
                                                    "question_analysis", ""
                                                ),
                                                cefr_level=data.get("cefr_level", "B1"),
                                                teaching_tip=data.get(
                                                    "teaching_tip", ""
                                                ),
                                                option_explanations=data.get(
                                                    "option_explanations", []
                                                ),
                                            )
                                        )
                                    except ValueError:
                                        continue
            except (ValueError, IndexError):
                pass

        return explanations

    def _parse_non_structured_results(self, workflow_id: str) -> list[QuestionDraft]:
        """Parse non-structured workflow results into QuestionDrafts."""
        status = self._orchestrator.tool.workflow(
            action="status", workflow_id=workflow_id
        )

        questions: list[QuestionDraft] = []

        content = status.get("content", [])
        if content and isinstance(content[0], dict):
            text = content[0].get("text", "")
            try:
                workflow_data = loads(text.split("\n")[-1].encode()) if text else {}
                task_results = workflow_data.get("task_results", {})

                for _task_id, result in task_results.items():
                    if result.get("status") == "completed":
                        result_content = result.get("result", [])
                        if result_content:
                            for item in result_content:
                                if isinstance(item, dict) and "text" in item:
                                    try:
                                        data = loads(item["text"].encode())
                                        if isinstance(data, list):
                                            for q in data:
                                                questions.append(
                                                    QuestionDraft(
                                                        question_text=q.get(
                                                            "question", ""
                                                        ),
                                                        answer_text=q.get(
                                                            "model_answer", ""
                                                        ),
                                                        question_type=QuestionType.NON_STRUCTURED,
                                                        difficulty_level=q.get(
                                                            "difficulty"
                                                        ),
                                                        explanation=q.get(
                                                            "explanation", ""
                                                        ),
                                                        references=[],
                                                        topic_id=q.get("topic"),
                                                        metadata=q.get("metadata", {}),
                                                    )
                                                )
                                    except ValueError:
                                        continue
            except (ValueError, IndexError):
                pass

        return questions

    def _aggregate_final_questions(
        self,
        stems: list[QuestionStem],
        answers: list[MCQAnswer],
        explanations: list[MCQExplanation],
        difficulty_level: str | None,
    ) -> list[FinalQuestion]:
        """Aggregate all three levels into final questions."""
        answer_by_stem: dict[str, MCQAnswer] = {}
        for ans in answers:
            answer_by_stem[ans.question_stem] = ans

        explanation_by_id: dict[str, MCQExplanation] = {}
        for exp in explanations:
            explanation_by_id[exp.question_id] = exp

        final_questions = []

        for stem in stems:
            answer: MCQAnswer | None = answer_by_stem.get(stem.content)
            if answer is None:
                continue

            explanation: MCQExplanation | None = explanation_by_id.get(stem.question_id)

            # Build formatted answer text
            options_dict = self._build_options_dict(answer)
            answer_parts = [
                f"Question: {answer.question_stem}",
                "",
                "Options:",
            ]
            for letter, text in options_dict.items():
                marker = " ✓ CORRECT" if letter == answer.correct_letter else ""
                answer_parts.append(f"  {letter}) {text}{marker}")
            answer_parts.append("")
            answer_parts.append(f"Correct Answer: {answer.correct_letter}")

            answer_text = "\n".join(answer_parts)

            # Build explanation
            explanation_parts = []
            if explanation:
                explanation_parts.extend(
                    [
                        f"Analysis: {explanation.question_analysis}",
                        f"CEFR Level: {explanation.cefr_level}",
                        f"Teaching Tip: {explanation.teaching_tip}",
                    ]
                )
                for opt_exp in explanation.option_explanations:
                    explanation_parts.append(
                        f"{opt_exp.get('option_letter', '?')}: {opt_exp.get('explanation', '')}"
                    )
            else:
                explanation_parts.append(answer.correct_explanation)
                for d in answer.distractors:
                    explanation_parts.append(
                        f"{d.get('option_letter', '?')}: {d.get('explanation', '')}"
                    )

            explanation_text = "\n".join(explanation_parts)

            final_questions.append(
                FinalQuestion(
                    question_id=stem.question_id,
                    text=answer.question_stem,
                    question_type=QuestionType.STRUCTURED,
                    difficulty_level=difficulty_level,
                    topic_id=stem.topic,
                    metadata={
                        "grammar_point": answer.grammar_point,
                        "l1_considerations": ", ".join(answer.l1_considerations),
                        "difficulty_justification": answer.difficulty_justification,
                    },
                    answer_text=answer_text,
                    explanation=explanation_text,
                )
            )

        return final_questions
