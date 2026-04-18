"""Strands Agent Tools for gRPC Knowledge and Submission Services.

This module provides tool classes that wrap Knowledge Service and Submission Service
gRPC operations as Strands Agent tools. These tools enable agents to:

- Retrieve topics and perform similarity search on knowledge bases
- Manage question sets and write generated questions to submission services
- Coordinate multi-service operations through the unified QnAGenerationToolkit

Example:
    >>> toolkit = QnAGenerationToolkit(
    ...     knowledge_target="knowledge-service:50051",
    ...     submission_target="submission-service:50051",
    ... )
    >>> context = await toolkit.get_generation_context(
    ...     workflow_id="wf-123",
    ...     assessment_id="assess-456",
    ...     query="machine learning fundamentals",
    ... )
    >>> await toolkit.close()
"""

from __future__ import annotations

import asyncio
from typing import Any

from strands import tool

from qna_generation_agent.application.ports.knowledge_client import (
    Chunk,
    GetChunksByIdsCommand,
    GetTopicsCommand,
    SimilaritySearchCommand,
    Topic,
)
from qna_generation_agent.application.ports.submission_client import (
    CreateQuestionSetCommand,
    GetAssessmentConfigCommand,
    IncrementIterationCommand,
    WriteGeneratedQuestionsCommand,
)
from qna_generation_agent.application.ports.submission_client import (
    Question as QuestionDataclass,
)
from qna_generation_agent.infrastructure.grpc.knowledge_client import (
    GrpcKnowledgeClient,
)
from qna_generation_agent.infrastructure.grpc.submission_client import (
    GrpcSubmissionClient,
)


def _serialize_topics(topics: list[Topic]) -> list[dict[str, Any]]:
    """Recursively convert Topic dataclasses to dictionaries.

    Args:
        topics: List of Topic dataclasses to serialize.

    Returns:
        List of dictionaries representing the topics with nested subtopics.
    """
    result = []
    for topic in topics:
        topic_dict: dict[str, Any] = {
            "topic_id": topic.topic_id,
            "name": topic.name,
        }
        if topic.subtopics:
            topic_dict["subtopics"] = _serialize_topics(topic.subtopics)
        else:
            topic_dict["subtopics"] = []
        result.append(topic_dict)
    return result


def _serialize_chunks(chunks: list[Chunk]) -> list[dict[str, Any]]:
    """Convert Chunk dataclasses to dictionaries.

    Args:
        chunks: List of Chunk dataclasses to serialize.

    Returns:
        List of dictionaries representing the chunks.
    """
    return [
        {
            "chunk_id": chunk.chunk_id,
            "workflow_id": chunk.workflow_id,
            "content": chunk.content,
            "source_type": chunk.source_type,
            "metadata": dict(chunk.metadata),
            "score": chunk.score,
        }
        for chunk in chunks
    ]


def _serialize_questions(questions: list[QuestionDataclass]) -> list[dict[str, Any]]:
    """Convert Question dataclasses to dictionaries.

    Args:
        questions: List of Question dataclasses to serialize.

    Returns:
        List of dictionaries representing the questions.
    """
    return [
        {
            "question_id": q.question_id,
            "question_type": q.question_type,
            "content": q.content,
            "structured_answer": q.structured_answer,
            "non_structured_model_answer": q.non_structured_model_answer,
            "metadata_json": q.metadata_json,
            "topic_id": q.topic_id,
            "iteration": q.iteration,
            "sort_order": q.sort_order,
        }
        for q in questions
    ]


class KnowledgeServiceTools:
    """Strands tools for Knowledge Service gRPC operations.

    This class wraps Knowledge Service operations as Strands Agent tools,
    enabling agents to query topics and perform similarity search on
    knowledge bases.

    Args:
        target: The gRPC service target (host:port).
        timeout_seconds: Timeout for gRPC calls in seconds. Defaults to 30.0.
        tls_enabled: Whether to use TLS for the connection. Defaults to False.
        tls_cert_path: Path to TLS certificate file. Defaults to None.

    Example:
        >>> tools = KnowledgeServiceTools("knowledge-service:50051")
        >>> topics = await tools.get_topics("workflow-123")
        >>> await tools.close()
    """

    def __init__(
        self,
        target: str,
        timeout_seconds: float = 30.0,
        tls_enabled: bool = False,
        tls_cert_path: str | None = None,
    ) -> None:
        self._client = GrpcKnowledgeClient(
            target=target,
            timeout_seconds=timeout_seconds,
            tls_enabled=tls_enabled,
            tls_cert_path=tls_cert_path,
        )

    @tool
    async def get_topics(self, workflow_id: str) -> dict[str, Any]:
        """Get topics for a workflow from the Knowledge Service.

        Retrieves the hierarchical topic structure for the specified workflow,
        including all nested subtopics.

        Args:
            workflow_id: The unique identifier of the workflow.

        Returns:
            A dictionary containing the list of topics:
            {"topics": [{"topic_id": str, "name": str, "subtopics": [...]}]}

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """
        command = GetTopicsCommand(workflow_id=workflow_id)
        topics = await self._client.get_topics(command)
        return {"topics": _serialize_topics(topics)}

    @tool
    async def similarity_search(
        self,
        workflow_id: str,
        query: str,
        kb_type: str = "document",
        top_k: int = 5,
        assessment_id: str = "",
    ) -> dict[str, Any]:
        """Search for similar chunks in the Knowledge Service.

        Performs a similarity search against the knowledge base using the
        provided query, returning the most relevant chunks.

        Args:
            workflow_id: The unique identifier of the workflow.
            query: The search query string.
            kb_type: Type of knowledge base ("document", "enriched", "policy").
                Defaults to "document".
            top_k: Maximum number of results to return. Defaults to 5.
            assessment_id: Optional assessment ID for context-aware search.
                Defaults to empty string.

        Returns:
            A dictionary containing the matching chunks:
            {"chunks": [{"chunk_id": str, "content": str, "score": float, ...}]}

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """
        command = SimilaritySearchCommand(
            query=query,
            workflow_id=workflow_id,
            kb_type=kb_type,
            top_k=top_k,
            assessment_id=assessment_id,
        )
        chunks = await self._client.similarity_search(command)
        return {"chunks": _serialize_chunks(chunks)}

    @tool
    async def get_chunks_by_ids(self, chunk_ids: list[str]) -> dict[str, Any]:
        """Get specific chunks by their IDs from the Knowledge Service.

        Retrieves the full content and metadata for the specified chunk IDs.

        Args:
            chunk_ids: List of chunk identifiers to retrieve.

        Returns:
            A dictionary containing the requested chunks:
            {"chunks": [{"chunk_id": str, "content": str, ...}]}

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """
        command = GetChunksByIdsCommand(chunk_ids=chunk_ids)
        chunks = await self._client.get_chunks_by_ids(command)
        return {"chunks": _serialize_chunks(chunks)}

    async def close(self) -> None:
        """Close the gRPC connection to the Knowledge Service.

        This method should be called when the tools are no longer needed
        to release resources properly.
        """
        await self._client.close()


class SubmissionServiceTools:
    """Strands tools for Assessment Submission Service gRPC operations.

    This class wraps Submission Service operations as Strands Agent tools,
    enabling agents to manage question sets and persist generated questions.

    Args:
        target: The gRPC service target (host:port).
        timeout_seconds: Timeout for gRPC calls in seconds. Defaults to 30.0.
        tls_enabled: Whether to use TLS for the connection. Defaults to False.
        tls_cert_path: Path to TLS certificate file. Defaults to None.

    Example:
        >>> tools = SubmissionServiceTools("submission-service:50051")
        >>> record = await tools.create_question_set("workflow-123")
        >>> await tools.close()
    """

    def __init__(
        self,
        target: str,
        timeout_seconds: float = 30.0,
        tls_enabled: bool = False,
        tls_cert_path: str | None = None,
    ) -> None:
        self._client = GrpcSubmissionClient(
            target=target,
            timeout_seconds=timeout_seconds,
            tls_enabled=tls_enabled,
            tls_cert_path=tls_cert_path,
        )

    @tool
    async def get_assessment_config(
        self,
        assessment_id: str,
        workflow_id: str,
    ) -> dict[str, Any]:
        """Get assessment configuration from the Submission Service.

        Retrieves the authoritative assessment configuration including
        generation parameters for regeneration scenarios.

        Args:
            assessment_id: The unique identifier of the assessment.
            workflow_id: The unique identifier of the workflow.

        Returns:
            A dictionary containing the assessment configuration:
            {
                "assessment_id": str,
                "workflow_id": str,
                "assessor_id": str,
                "assessment_title": str,
                "purpose": str | None,
                "duration_minutes": int,
                "difficulty_level": str | None,
                "structured_question_count": int,
                "non_structured_question_count": int,
                "web_research_mode": str,
                "status": str,
                "deadline": str | None,
            }

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """
        command = GetAssessmentConfigCommand(
            assessment_id=assessment_id,
            workflow_id=workflow_id,
        )
        config = await self._client.get_assessment_config(command)
        return {
            "assessment_id": config.assessment_id,
            "workflow_id": config.workflow_id,
            "assessor_id": config.assessor_id,
            "assessment_title": config.assessment_title,
            "purpose": config.purpose,
            "duration_minutes": config.duration_minutes,
            "difficulty_level": config.difficulty_level,
            "structured_question_count": config.structured_question_count,
            "non_structured_question_count": config.non_structured_question_count,
            "web_research_mode": config.web_research_mode,
            "status": config.status,
            "deadline": config.deadline,
        }

    @tool
    async def create_question_set(self, workflow_id: str) -> dict[str, Any]:
        """Create a new question set in the Submission Service.

        Initializes a new question set record for the specified workflow.

        Args:
            workflow_id: The unique identifier of the workflow.

        Returns:
            A dictionary containing the created question set record:
            {
                "id": str,
                "workflow_id": str,
                "iteration_count": int,
                "status": str,
                "created_at": str | None,
            }

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """
        command = CreateQuestionSetCommand(workflow_id=workflow_id)
        record = await self._client.create_question_set(command)
        return {
            "id": record.id,
            "workflow_id": record.workflow_id,
            "iteration_count": record.iteration_count,
            "status": record.status,
            "created_at": record.created_at.isoformat() if record.created_at else None,
        }

    @tool
    async def increment_iteration(self, question_set_id: str) -> dict[str, Any]:
        """Increment the iteration count for a question set.

        Advances the iteration counter for the specified question set,
        typically used when regenerating questions.

        Args:
            question_set_id: The unique identifier of the question set.

        Returns:
            A dictionary containing the updated iteration information:
            {
                "question_set_id": str,
                "iteration_count": int,
                "status": str,
            }

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """
        command = IncrementIterationCommand(question_set_id=question_set_id)
        result = await self._client.increment_iteration(command)
        return {
            "question_set_id": result.question_set_id,
            "iteration_count": result.iteration_count,
            "status": result.status,
        }

    @tool
    async def write_generated_questions(
        self,
        question_set_id: str,
        questions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Write generated questions to the Submission Service.

        Persists a batch of generated questions to the specified question set.

        Args:
            question_set_id: The unique identifier of the question set.
            questions: List of question dictionaries, each containing:
                - question_id: str
                - question_type: str
                - content: str
                - structured_answer: str (optional)
                - non_structured_model_answer: str (optional)
                - metadata_json: str (optional)
                - topic_id: str (optional)
                - iteration: int (optional)
                - sort_order: int (optional)

        Returns:
            A dictionary containing the write operation result:
            {
                "questions_written": int,
                "status": str,
            }

        Raises:
            StorageTransientError: If the request fails temporarily (retryable).
            StoragePermanentError: If the request fails permanently.
        """
        question_dataclasses = [
            QuestionDataclass(
                question_id=q["question_id"],
                question_type=q["question_type"],
                content=q["content"],
                structured_answer=q.get("structured_answer", ""),
                non_structured_model_answer=q.get("non_structured_model_answer", ""),
                metadata_json=q.get("metadata_json", ""),
                topic_id=q.get("topic_id", ""),
                iteration=q.get("iteration", 1),
                sort_order=q.get("sort_order", 1),
            )
            for q in questions
        ]
        command = WriteGeneratedQuestionsCommand(
            question_set_id=question_set_id,
            questions=question_dataclasses,
        )
        result = await self._client.write_generated_questions(command)
        return {
            "questions_written": result.questions_written,
            "status": result.status,
        }

    async def close(self) -> None:
        """Close the gRPC connection to the Submission Service.

        This method should be called when the tools are no longer needed
        to release resources properly.
        """
        await self._client.close()


class QnAGenerationToolkit:
    """Unified toolkit combining Knowledge and Submission Service tools.

    This class provides a high-level interface that coordinates operations
    across both Knowledge and Submission Services, including parallelized
    multi-service calls for efficient context gathering.

    Args:
        knowledge_target: The Knowledge Service gRPC target (host:port).
        submission_target: The Submission Service gRPC target (host:port).
        timeout_seconds: Timeout for gRPC calls in seconds. Defaults to 30.0.
        tls_enabled: Whether to use TLS for connections. Defaults to False.
        tls_cert_path: Path to TLS certificate file. Defaults to None.

    Example:
        >>> toolkit = QnAGenerationToolkit(
        ...     knowledge_target="knowledge:50051",
        ...     submission_target="submission:50051",
        ... )
        >>> context = await toolkit.get_generation_context(
        ...     workflow_id="wf-123",
        ...     assessment_id="assess-456",
        ...     query="python async programming",
        ... )
        >>> await toolkit.close()
    """

    def __init__(
        self,
        knowledge_target: str,
        submission_target: str,
        timeout_seconds: float = 30.0,
        tls_enabled: bool = False,
        tls_cert_path: str | None = None,
    ) -> None:
        self._knowledge_tools = KnowledgeServiceTools(
            target=knowledge_target,
            timeout_seconds=timeout_seconds,
            tls_enabled=tls_enabled,
            tls_cert_path=tls_cert_path,
        )
        self._submission_tools = SubmissionServiceTools(
            target=submission_target,
            timeout_seconds=timeout_seconds,
            tls_enabled=tls_enabled,
            tls_cert_path=tls_cert_path,
        )

    @tool
    async def get_generation_context(
        self,
        workflow_id: str,
        assessment_id: str,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Get unified generation context from both services.

        Parallelizes calls to get assessment configuration, topics, and perform
        similarity search (if query provided) using asyncio.gather().

        This method efficiently gathers all context needed for question generation
        in a single call, reducing latency through parallel execution.

        Args:
            workflow_id: The unique identifier of the workflow.
            assessment_id: The unique identifier of the assessment.
            query: Optional search query for similarity search. If None, no
                similarity search is performed. Defaults to None.

        Returns:
            A dictionary containing unified context:
            {
                "assessment_config": {...},
                "topics": [...],
                "chunks": [...],  # Only present if query was provided
            }

        Raises:
            StorageTransientError: If any request fails temporarily (retryable).
            StoragePermanentError: If any request fails permanently.
        """
        # Prepare coroutines
        config_coro = self._submission_tools.get_assessment_config(  # type: ignore[call-arg]
            assessment_id=assessment_id,
            workflow_id=workflow_id,
        )
        topics_coro = self._knowledge_tools.get_topics(workflow_id=workflow_id)  # type: ignore[call-arg]

        if query:
            search_coro = self._knowledge_tools.similarity_search(  # type: ignore[call-arg]
                workflow_id=workflow_id,
                query=query,
                assessment_id=assessment_id,
            )
            # Execute all three calls in parallel
            config_result, topics_result, search_result = await asyncio.gather(
                config_coro,
                topics_coro,
                search_coro,
            )
            return {
                "assessment_config": config_result,
                "topics": topics_result.get("topics", []),
                "chunks": search_result.get("chunks", []),
            }
        else:
            # Execute config and topics calls in parallel
            config_result, topics_result = await asyncio.gather(
                config_coro,
                topics_coro,
            )
            return {
                "assessment_config": config_result,
                "topics": topics_result.get("topics", []),
            }

    async def close(self) -> None:
        """Close both gRPC connections.

        This method should be called when the toolkit is no longer needed
        to release resources properly.
        """
        await self._knowledge_tools.close()
        await self._submission_tools.close()
