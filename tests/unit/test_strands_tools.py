"""Unit tests for Strands Agents tools wrapping gRPC clients."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from qna_generation_agent.application.errors import (
    StoragePermanentError,
    StorageTransientError,
)
from qna_generation_agent.application.ports.knowledge_client import Chunk, Topic
from qna_generation_agent.application.ports.submission_client import (
    AssessmentConfig,
    IncrementIterationResult,
    QuestionSetRecord,
    WriteGeneratedQuestionsResult,
)
from qna_generation_agent.application.ports.submission_client import (
    Question as QuestionDataclass,
)
from qna_generation_agent.infrastructure.grpc.strands_tools import (
    KnowledgeServiceTools,
    QnAGenerationToolkit,
    SubmissionServiceTools,
    _serialize_chunks,
    _serialize_questions,
    _serialize_topics,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_knowledge_client():
    """Return a mocked GrpcKnowledgeClient."""
    client = MagicMock()
    client.get_topics = AsyncMock()
    client.similarity_search = AsyncMock()
    client.get_chunks_by_ids = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def mock_submission_client():
    """Return a mocked GrpcSubmissionClient."""
    client = MagicMock()
    client.get_assessment_config = AsyncMock()
    client.create_question_set = AsyncMock()
    client.increment_iteration = AsyncMock()
    client.write_generated_questions = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def sample_topic():
    """Return a sample Topic dataclass."""
    return Topic(
        topic_id="topic_123",
        name="Grammar",
        subtopics=[
            Topic(topic_id="sub_1", name="Articles", subtopics=[]),
            Topic(topic_id="sub_2", name="Tenses", subtopics=[]),
        ],
    )


@pytest.fixture
def sample_chunk():
    """Return a sample Chunk dataclass."""
    return Chunk(
        chunk_id="chunk_123",
        workflow_id="wf_123",
        content="Sample content",
        source_type="document",
        metadata={"key": "value"},
        score=0.95,
    )


@pytest.fixture
def sample_question():
    """Return a sample Question dataclass."""
    return QuestionDataclass(
        question_id="q_123",
        question_type="STRUCTURED",
        content="What is 2+2?",
        structured_answer="4",
        non_structured_model_answer="",
        metadata_json='{"difficulty": "easy"}',
        topic_id="topic_123",
        iteration=1,
        sort_order=1,
    )


# ============================================================================
# Serialization Helper Tests
# ============================================================================


@pytest.mark.unit
class TestSerializeTopics:
    """Tests for _serialize_topics function."""

    def test_serialize_topics(self, sample_topic):
        """Test recursive conversion of Topic dataclasses with nested subtopics."""
        result = _serialize_topics([sample_topic])

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["topic_id"] == "topic_123"
        assert result[0]["name"] == "Grammar"
        assert len(result[0]["subtopics"]) == 2
        assert result[0]["subtopics"][0]["topic_id"] == "sub_1"
        assert result[0]["subtopics"][0]["name"] == "Articles"
        assert result[0]["subtopics"][1]["topic_id"] == "sub_2"
        assert result[0]["subtopics"][1]["name"] == "Tenses"

    def test_serialize_topics_empty_list(self):
        """Test empty topics list returns empty list."""
        result = _serialize_topics([])
        assert result == []
        assert isinstance(result, list)

    def test_serialize_topics_deeply_nested(self):
        """Test serialization of deeply nested topic structure."""
        deep_topic = Topic(
            topic_id="level_1",
            name="Level 1",
            subtopics=[
                Topic(
                    topic_id="level_2",
                    name="Level 2",
                    subtopics=[
                        Topic(
                            topic_id="level_3",
                            name="Level 3",
                            subtopics=[],
                        )
                    ],
                )
            ],
        )
        result = _serialize_topics([deep_topic])

        assert result[0]["topic_id"] == "level_1"
        assert result[0]["subtopics"][0]["topic_id"] == "level_2"
        assert result[0]["subtopics"][0]["subtopics"][0]["topic_id"] == "level_3"


@pytest.mark.unit
class TestSerializeChunks:
    """Tests for _serialize_chunks function."""

    def test_serialize_chunks(self, sample_chunk):
        """Test conversion of Chunk dataclasses."""
        result = _serialize_chunks([sample_chunk])

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "chunk_123"
        assert result[0]["workflow_id"] == "wf_123"
        assert result[0]["content"] == "Sample content"
        assert result[0]["source_type"] == "document"
        assert result[0]["metadata"] == {"key": "value"}
        assert result[0]["score"] == 0.95

    def test_serialize_chunks_empty_list(self):
        """Test empty chunks list returns empty list."""
        result = _serialize_chunks([])
        assert result == []

    def test_serialize_chunks_multiple(self, sample_chunk):
        """Test serialization of multiple chunks."""
        chunk2 = Chunk(
            chunk_id="chunk_456",
            workflow_id="wf_456",
            content="Another content",
            source_type="video",
            metadata={},
            score=0.85,
        )
        result = _serialize_chunks([sample_chunk, chunk2])

        assert len(result) == 2
        assert result[0]["chunk_id"] == "chunk_123"
        assert result[1]["chunk_id"] == "chunk_456"


@pytest.mark.unit
class TestSerializeQuestions:
    """Tests for _serialize_questions function."""

    def test_serialize_questions(self, sample_question):
        """Test conversion of Question dataclasses."""
        result = _serialize_questions([sample_question])

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["question_id"] == "q_123"
        assert result[0]["question_type"] == "STRUCTURED"
        assert result[0]["content"] == "What is 2+2?"
        assert result[0]["structured_answer"] == "4"
        assert result[0]["non_structured_model_answer"] == ""
        assert result[0]["metadata_json"] == '{"difficulty": "easy"}'
        assert result[0]["topic_id"] == "topic_123"
        assert result[0]["iteration"] == 1
        assert result[0]["sort_order"] == 1

    def test_serialize_questions_empty_list(self):
        """Test empty questions list returns empty list."""
        result = _serialize_questions([])
        assert result == []

    def test_serialize_questions_non_structured(self):
        """Test serialization of non-structured questions."""
        question = QuestionDataclass(
            question_id="q_456",
            question_type="NON_STRUCTURED",
            content="Explain machine learning",
            structured_answer="",
            non_structured_model_answer="Machine learning is...",
            metadata_json="{}",
            topic_id="topic_456",
            iteration=2,
            sort_order=5,
        )
        result = _serialize_questions([question])

        assert result[0]["question_type"] == "NON_STRUCTURED"
        assert result[0]["non_structured_model_answer"] == "Machine learning is..."
        assert result[0]["iteration"] == 2


# ============================================================================
# KnowledgeServiceTools Tests
# ============================================================================


@pytest.mark.unit
class TestKnowledgeServiceTools:
    """Tests for KnowledgeServiceTools class."""

    @pytest.mark.asyncio
    async def test_knowledge_tools_init(self, mock_knowledge_client):
        """Test initialization with correct parameters."""
        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
            return_value=mock_knowledge_client,
        ):
            tools = KnowledgeServiceTools(
                target="grpc://localhost:50052",
                timeout_seconds=30.0,
            )

            assert tools._client is mock_knowledge_client

    @pytest.mark.asyncio
    async def test_get_topics_tool(self, mock_knowledge_client, sample_topic):
        """Test get_topics tool with mocked client returning list of Topics."""
        mock_knowledge_client.get_topics.return_value = [sample_topic]

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
            return_value=mock_knowledge_client,
        ):
            tools = KnowledgeServiceTools(target="grpc://localhost:50052")
            result = await tools.get_topics(workflow_id="wf_123")

            assert isinstance(result, dict)
            assert "topics" in result
            assert len(result["topics"]) == 1
            assert result["topics"][0]["topic_id"] == "topic_123"
            assert result["topics"][0]["name"] == "Grammar"
            mock_knowledge_client.get_topics.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_similarity_search_tool(self, mock_knowledge_client, sample_chunk):
        """Test similarity_search tool with mocked client returning list of Chunks."""
        mock_knowledge_client.similarity_search.return_value = [sample_chunk]

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
            return_value=mock_knowledge_client,
        ):
            tools = KnowledgeServiceTools(target="grpc://localhost:50052")
            result = await tools.similarity_search(
                query="test query",
                workflow_id="wf_123",
                kb_type="document",
                top_k=5,
            )

            assert isinstance(result, dict)
            assert "chunks" in result
            assert len(result["chunks"]) == 1
            assert result["chunks"][0]["chunk_id"] == "chunk_123"
            assert result["chunks"][0]["score"] == 0.95
            mock_knowledge_client.similarity_search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_chunks_by_ids_tool(self, mock_knowledge_client, sample_chunk):
        """Test get_chunks_by_ids tool with mocked client."""
        mock_knowledge_client.get_chunks_by_ids.return_value = [sample_chunk]

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
            return_value=mock_knowledge_client,
        ):
            tools = KnowledgeServiceTools(target="grpc://localhost:50052")
            result = await tools.get_chunks_by_ids(chunk_ids=["chunk_123", "chunk_456"])

            assert isinstance(result, dict)
            assert "chunks" in result
            assert len(result["chunks"]) == 1
            assert result["chunks"][0]["chunk_id"] == "chunk_123"
            mock_knowledge_client.get_chunks_by_ids.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_knowledge_tools_close(self, mock_knowledge_client):
        """Test close method calls client.close()."""
        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
            return_value=mock_knowledge_client,
        ):
            tools = KnowledgeServiceTools(target="grpc://localhost:50052")
            await tools.close()

            mock_knowledge_client.close.assert_awaited_once()


# ============================================================================
# SubmissionServiceTools Tests
# ============================================================================


@pytest.mark.unit
class TestSubmissionServiceTools:
    """Tests for SubmissionServiceTools class."""

    @pytest.mark.asyncio
    async def test_submission_tools_init(self, mock_submission_client):
        """Test initialization with correct parameters."""
        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
            return_value=mock_submission_client,
        ):
            tools = SubmissionServiceTools(
                target="grpc://localhost:50051",
                timeout_seconds=30.0,
            )

            assert tools._client is mock_submission_client

    @pytest.mark.asyncio
    async def test_get_assessment_config_tool(self, mock_submission_client):
        """Test with mocked AssessmentConfig response."""
        config = AssessmentConfig(
            assessment_id="assess_123",
            workflow_id="wf_123",
            assessor_id="assessor_456",
            assessment_title="Test Assessment",
            purpose="assessment",
            duration_minutes=60,
            difficulty_level="medium",
            structured_question_count=5,
            non_structured_question_count=3,
            web_research_mode="disabled",
            status="active",
            deadline=None,
        )
        mock_submission_client.get_assessment_config.return_value = config

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
            return_value=mock_submission_client,
        ):
            tools = SubmissionServiceTools(target="grpc://localhost:50051")
            result = await tools.get_assessment_config(
                assessment_id="assess_123",
                workflow_id="wf_123",
            )

            assert isinstance(result, dict)
            assert result["assessment_id"] == "assess_123"
            assert result["workflow_id"] == "wf_123"
            assert result["assessor_id"] == "assessor_456"
            assert result["assessment_title"] == "Test Assessment"
            assert result["duration_minutes"] == 60
            assert result["structured_question_count"] == 5
            assert result["non_structured_question_count"] == 3
            mock_submission_client.get_assessment_config.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_create_question_set_tool(self, mock_submission_client):
        """Test with mocked QuestionSetRecord response."""
        record = QuestionSetRecord(
            id="qs_123",
            workflow_id="wf_123",
            iteration_count=1,
            status="created",
            created_at=datetime(2024, 1, 1, 12, 0, 0),
        )
        mock_submission_client.create_question_set.return_value = record

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
            return_value=mock_submission_client,
        ):
            tools = SubmissionServiceTools(target="grpc://localhost:50051")
            result = await tools.create_question_set(workflow_id="wf_123")

            assert isinstance(result, dict)
            assert result["id"] == "qs_123"
            assert result["workflow_id"] == "wf_123"
            assert result["iteration_count"] == 1
            assert result["status"] == "created"
            mock_submission_client.create_question_set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_increment_iteration_tool(self, mock_submission_client):
        """Test with mocked IncrementIterationResult response."""
        result_data = IncrementIterationResult(
            question_set_id="qs_123",
            iteration_count=2,
            status="incremented",
        )
        mock_submission_client.increment_iteration.return_value = result_data

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
            return_value=mock_submission_client,
        ):
            tools = SubmissionServiceTools(target="grpc://localhost:50051")
            result = await tools.increment_iteration(question_set_id="qs_123")

            assert isinstance(result, dict)
            assert result["question_set_id"] == "qs_123"
            assert result["iteration_count"] == 2
            assert result["status"] == "incremented"
            mock_submission_client.increment_iteration.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_generated_questions_tool(self, mock_submission_client):
        """Test with mocked WriteGeneratedQuestionsResult response."""
        write_result = WriteGeneratedQuestionsResult(
            questions_written=10,
            status="written",
        )
        mock_submission_client.write_generated_questions.return_value = write_result

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
            return_value=mock_submission_client,
        ):
            tools = SubmissionServiceTools(target="grpc://localhost:50051")
            # Pass questions as dictionaries as expected by the implementation
            questions = [
                {
                    "question_id": "q_001",
                    "question_type": "STRUCTURED",
                    "content": "What is 2+2?",
                    "structured_answer": "4",
                    "non_structured_model_answer": "",
                    "metadata_json": "{}",
                    "topic_id": "topic_123",
                    "iteration": 1,
                    "sort_order": 1,
                }
            ]
            result = await tools.write_generated_questions(
                question_set_id="qs_123",
                questions=questions,
            )

            assert isinstance(result, dict)
            assert result["questions_written"] == 10
            assert result["status"] == "written"
            mock_submission_client.write_generated_questions.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_submission_tools_close(self, mock_submission_client):
        """Test close method calls client.close()."""
        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
            return_value=mock_submission_client,
        ):
            tools = SubmissionServiceTools(target="grpc://localhost:50051")
            await tools.close()

            mock_submission_client.close.assert_awaited_once()


# ============================================================================
# QnAGenerationToolkit Tests
# ============================================================================


@pytest.mark.unit
class TestQnAGenerationToolkit:
    """Tests for QnAGenerationToolkit class."""

    @pytest.mark.asyncio
    async def test_toolkit_init(self, mock_knowledge_client, mock_submission_client):
        """Test initialization creates both tool instances."""
        with (
            patch(
                "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
                return_value=mock_knowledge_client,
            ),
            patch(
                "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
                return_value=mock_submission_client,
            ),
        ):
            toolkit = QnAGenerationToolkit(
                knowledge_target="grpc://localhost:50052",
                submission_target="grpc://localhost:50051",
            )

            assert toolkit._knowledge_tools is not None
            assert toolkit._submission_tools is not None
            assert toolkit._knowledge_tools._client is mock_knowledge_client
            assert toolkit._submission_tools._client is mock_submission_client

    @pytest.mark.asyncio
    async def test_get_generation_context(
        self, mock_knowledge_client, mock_submission_client, sample_topic
    ):
        """Test unified context tool with mocked parallel responses."""
        mock_knowledge_client.get_topics.return_value = [sample_topic]

        config = AssessmentConfig(
            assessment_id="assess_123",
            workflow_id="wf_123",
            assessor_id="assessor_456",
            assessment_title="Test Assessment",
            purpose="assessment",
            duration_minutes=60,
            difficulty_level="medium",
            structured_question_count=5,
            non_structured_question_count=3,
            web_research_mode="disabled",
            status="active",
            deadline=None,
        )
        mock_submission_client.get_assessment_config.return_value = config

        with (
            patch(
                "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
                return_value=mock_knowledge_client,
            ),
            patch(
                "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
                return_value=mock_submission_client,
            ),
        ):
            toolkit = QnAGenerationToolkit(
                knowledge_target="grpc://localhost:50052",
                submission_target="grpc://localhost:50051",
            )
            result = await toolkit.get_generation_context(
                assessment_id="assess_123",
                workflow_id="wf_123",
            )

            assert isinstance(result, dict)
            assert "assessment_config" in result
            assert "topics" in result
            assert result["assessment_config"]["assessment_id"] == "assess_123"
            assert result["assessment_config"]["duration_minutes"] == 60
            assert len(result["topics"]) == 1
            assert result["topics"][0]["topic_id"] == "topic_123"

            mock_knowledge_client.get_topics.assert_awaited_once()
            mock_submission_client.get_assessment_config.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_toolkit_close(self, mock_knowledge_client, mock_submission_client):
        """Test close method closes both knowledge and submission tools."""
        with (
            patch(
                "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
                return_value=mock_knowledge_client,
            ),
            patch(
                "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
                return_value=mock_submission_client,
            ),
        ):
            toolkit = QnAGenerationToolkit(
                knowledge_target="grpc://localhost:50052",
                submission_target="grpc://localhost:50051",
            )
            await toolkit.close()

            mock_knowledge_client.close.assert_awaited_once()
            mock_submission_client.close.assert_awaited_once()


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.unit
class TestErrorHandling:
    """Tests for error propagation through tools."""

    @pytest.mark.asyncio
    async def test_knowledge_tool_error_propagation(self, mock_knowledge_client):
        """Test that gRPC errors propagate correctly from knowledge tools."""
        mock_knowledge_client.get_topics.side_effect = StorageTransientError(
            "Service unavailable",
            retry_after_seconds=5,
        )

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
            return_value=mock_knowledge_client,
        ):
            tools = KnowledgeServiceTools(target="grpc://localhost:50052")

            with pytest.raises(StorageTransientError) as exc_info:
                await tools.get_topics(workflow_id="wf_123")

            assert "Service unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_knowledge_tool_permanent_error(self, mock_knowledge_client):
        """Test that permanent gRPC errors propagate correctly."""
        mock_knowledge_client.similarity_search.side_effect = StoragePermanentError(
            "Method not implemented",
            code="UNIMPLEMENTED",
        )

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
            return_value=mock_knowledge_client,
        ):
            tools = KnowledgeServiceTools(target="grpc://localhost:50052")

            with pytest.raises(StoragePermanentError) as exc_info:
                await tools.similarity_search(
                    query="test",
                    workflow_id="wf_123",
                    kb_type="document",
                )

            assert "Method not implemented" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_submission_tool_error_propagation(self, mock_submission_client):
        """Test that gRPC errors propagate correctly from submission tools."""
        mock_submission_client.create_question_set.side_effect = StorageTransientError(
            "Submission service unavailable",
            retry_after_seconds=10,
        )

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
            return_value=mock_submission_client,
        ):
            tools = SubmissionServiceTools(target="grpc://localhost:50051")

            with pytest.raises(StorageTransientError) as exc_info:
                await tools.create_question_set(workflow_id="wf_123")

            assert "Submission service unavailable" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_submission_tool_permanent_error(self, mock_submission_client):
        """Test that permanent gRPC errors propagate correctly."""
        mock_submission_client.write_generated_questions.side_effect = (
            StoragePermanentError(
                "Permission denied",
                code="PERMISSION_DENIED",
            )
        )

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
            return_value=mock_submission_client,
        ):
            tools = SubmissionServiceTools(target="grpc://localhost:50051")

            with pytest.raises(StoragePermanentError) as exc_info:
                await tools.write_generated_questions(
                    question_set_id="qs_123",
                    questions=[],
                )

            assert "Permission denied" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_toolkit_error_propagation(
        self, mock_knowledge_client, mock_submission_client
    ):
        """Test that errors in toolkit context gathering propagate correctly."""
        mock_submission_client.get_assessment_config.side_effect = (
            StorageTransientError(
                "Assessment config unavailable",
                retry_after_seconds=5,
            )
        )

        with (
            patch(
                "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
                return_value=mock_knowledge_client,
            ),
            patch(
                "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
                return_value=mock_submission_client,
            ),
        ):
            toolkit = QnAGenerationToolkit(
                knowledge_target="grpc://localhost:50052",
                submission_target="grpc://localhost:50051",
            )

            with pytest.raises(StorageTransientError) as exc_info:
                await toolkit.get_generation_context(
                    assessment_id="assess_123",
                    workflow_id="wf_123",
                )

            assert "Assessment config unavailable" in str(exc_info.value)


# ============================================================================
# Additional Edge Case Tests
# ============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_get_chunks_by_ids_empty_list(self, mock_knowledge_client):
        """Test get_chunks_by_ids with empty chunk IDs list."""
        mock_knowledge_client.get_chunks_by_ids.return_value = []

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
            return_value=mock_knowledge_client,
        ):
            tools = KnowledgeServiceTools(target="grpc://localhost:50052")
            result = await tools.get_chunks_by_ids(chunk_ids=[])

            assert isinstance(result, dict)
            assert "chunks" in result
            assert result["chunks"] == []

    @pytest.mark.asyncio
    async def test_similarity_search_default_parameters(self, mock_knowledge_client):
        """Test similarity_search uses default parameters correctly."""
        mock_knowledge_client.similarity_search.return_value = []

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
            return_value=mock_knowledge_client,
        ):
            tools = KnowledgeServiceTools(target="grpc://localhost:50052")
            await tools.similarity_search(
                query="test",
                workflow_id="wf_123",
            )

            call_args = mock_knowledge_client.similarity_search.call_args
            command = call_args[0][0]
            assert command.query == "test"
            assert command.workflow_id == "wf_123"
            assert command.kb_type == "document"  # default
            assert command.top_k == 5  # default

    @pytest.mark.asyncio
    async def test_write_generated_questions_empty_list(self, mock_submission_client):
        """Test write_generated_questions with empty questions list."""
        write_result = WriteGeneratedQuestionsResult(
            questions_written=0,
            status="written",
        )
        mock_submission_client.write_generated_questions.return_value = write_result

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
            return_value=mock_submission_client,
        ):
            tools = SubmissionServiceTools(target="grpc://localhost:50051")
            result = await tools.write_generated_questions(
                question_set_id="qs_123",
                questions=[],  # Empty list of question dicts
            )

            assert result["questions_written"] == 0
            assert result["status"] == "written"

    @pytest.mark.asyncio
    async def test_assessment_config_with_deadline(self, mock_submission_client):
        """Test assessment config serialization with deadline."""
        config = AssessmentConfig(
            assessment_id="assess_123",
            workflow_id="wf_123",
            assessor_id="assessor_456",
            assessment_title="Test Assessment",
            purpose="assessment",
            duration_minutes=60,
            difficulty_level="medium",
            structured_question_count=5,
            non_structured_question_count=3,
            web_research_mode="disabled",
            status="active",
            deadline="2024-12-31T23:59:59Z",
        )
        mock_submission_client.get_assessment_config.return_value = config

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcSubmissionClient",
            return_value=mock_submission_client,
        ):
            tools = SubmissionServiceTools(target="grpc://localhost:50051")
            result = await tools.get_assessment_config(
                assessment_id="assess_123",
                workflow_id="wf_123",
            )

            assert result["deadline"] == "2024-12-31T23:59:59Z"

    @pytest.mark.asyncio
    async def test_topic_with_empty_subtopics(self, mock_knowledge_client):
        """Test topic serialization when subtopics list is empty."""
        topic = Topic(
            topic_id="topic_simple",
            name="Simple Topic",
            subtopics=[],
        )
        mock_knowledge_client.get_topics.return_value = [topic]

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
            return_value=mock_knowledge_client,
        ):
            tools = KnowledgeServiceTools(target="grpc://localhost:50052")
            result = await tools.get_topics(workflow_id="wf_123")

            assert len(result["topics"]) == 1
            assert result["topics"][0]["subtopics"] == []

    @pytest.mark.asyncio
    async def test_chunk_with_empty_metadata(self, mock_knowledge_client):
        """Test chunk serialization with empty metadata dict."""
        chunk = Chunk(
            chunk_id="chunk_123",
            workflow_id="wf_123",
            content="Content",
            source_type="document",
            metadata={},
            score=0.9,
        )
        mock_knowledge_client.similarity_search.return_value = [chunk]

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
            return_value=mock_knowledge_client,
        ):
            tools = KnowledgeServiceTools(target="grpc://localhost:50052")
            result = await tools.similarity_search(
                query="test",
                workflow_id="wf_123",
            )

            assert result["chunks"][0]["metadata"] == {}

    @pytest.mark.asyncio
    async def test_multiple_chunks_returned(self, mock_knowledge_client):
        """Test handling multiple chunks returned from search."""
        chunks = [
            Chunk(
                chunk_id=f"chunk_{i}",
                workflow_id="wf_123",
                content=f"Content {i}",
                source_type="document",
                metadata={"index": str(i)},
                score=0.9 - (i * 0.1),
            )
            for i in range(5)
        ]
        mock_knowledge_client.similarity_search.return_value = chunks

        with patch(
            "qna_generation_agent.infrastructure.grpc.strands_tools.GrpcKnowledgeClient",
            return_value=mock_knowledge_client,
        ):
            tools = KnowledgeServiceTools(target="grpc://localhost:50052")
            result = await tools.similarity_search(
                query="test",
                workflow_id="wf_123",
                top_k=5,
            )

            assert len(result["chunks"]) == 5
            assert result["chunks"][0]["chunk_id"] == "chunk_0"
            assert result["chunks"][4]["chunk_id"] == "chunk_4"
