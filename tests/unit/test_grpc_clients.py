"""Unit tests for gRPC client error handling."""

# ruff: noqa: ASYNC109

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import grpc
import pytest

from qna_generation_agent.application.errors import (
    StoragePermanentError,
    StorageTransientError,
)
from qna_generation_agent.application.ports.submission_client import (
    CreateQuestionSetCommand,
    IncrementIterationCommand,
    Question,
    WriteGeneratedQuestionsCommand,
)
from qna_generation_agent.infrastructure.grpc.knowledge_client import (
    GetTopicsCommand,
    GrpcKnowledgeClient,
    SimilaritySearchCommand,
)
from qna_generation_agent.infrastructure.grpc.submission_client import (
    GrpcSubmissionClient,
)


class FakeGrpcChannel:
    """Fake gRPC channel for testing."""

    def __init__(
        self, should_fail: bool = False, error_code: grpc.StatusCode | None = None
    ) -> None:
        self.should_fail = should_fail
        self.error_code = error_code or grpc.StatusCode.UNAVAILABLE
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self


class FakeStub:
    """Fake gRPC stub for testing."""

    def __init__(self, channel: FakeGrpcChannel) -> None:
        self.channel = channel
        self.calls: list[Any] = []
        self.last_metadata: Any = None

    async def CreateQuestionSet(
        self, request: Any, timeout: float, metadata: Any = None
    ) -> Any:
        self.calls.append(("CreateQuestionSet", request, timeout))
        self.last_metadata = metadata
        if self.channel.should_fail:
            raise grpc.aio.AioRpcError(
                self.channel.error_code,
                None,
                None,
                "Service unavailable",
            )
        response = MagicMock()
        response.question_set_id = "qs_test_123"
        response.status = "created"
        # Simulate server returning fields when available
        response.workflow_id = "wf_123"
        response.iteration_count = 1
        return response

    async def WriteGeneratedQuestions(
        self, request: Any, timeout: float, metadata: Any = None
    ) -> Any:
        self.calls.append(("WriteGeneratedQuestions", request, timeout))
        self.last_metadata = metadata
        if self.channel.should_fail:
            raise grpc.aio.AioRpcError(
                self.channel.error_code,
                None,
                None,
                "Service unavailable",
            )
        response = MagicMock()
        response.questions_written = len(request.questions)
        response.status = "written"
        return response

    async def IncrementQuestionSetIteration(
        self, request: Any, timeout: float, metadata: Any = None
    ) -> Any:
        self.calls.append(("IncrementQuestionSetIteration", request, timeout))
        self.last_metadata = metadata
        if self.channel.should_fail:
            raise grpc.aio.AioRpcError(
                self.channel.error_code,
                None,
                None,
                "Service unavailable",
            )
        response = MagicMock()
        response.question_set_id = request.question_set_id
        response.iteration_count = 2
        response.status = "incremented"
        return response

    async def GetAssessmentConfig(
        self, request: Any, timeout: float, metadata: Any = None
    ) -> Any:
        self.calls.append(("GetAssessmentConfig", request, timeout))
        self.last_metadata = metadata
        if self.channel.should_fail:
            raise grpc.aio.AioRpcError(
                self.channel.error_code,
                None,
                None,
                "Service unavailable",
            )
        response = MagicMock()
        response.assessment_id = request.assessment_id
        response.workflow_id = request.workflow_id
        response.assessor_id = "assessor_123"
        response.assessment_title = "Test Assessment"
        response.purpose = "assessment"
        response.duration_minutes = 60
        response.difficulty_level = "medium"
        response.structured_question_count = 5
        response.non_structured_question_count = 3
        response.web_research_mode = "disabled"
        response.status = "active"
        response.deadline = None
        return response

    async def SimilaritySearch(
        self, request: Any, timeout: float, metadata: Any = None
    ) -> Any:
        self.calls.append(("SimilaritySearch", request, timeout))
        self.last_metadata = metadata
        if self.channel.should_fail:
            raise grpc.aio.AioRpcError(
                self.channel.error_code,
                None,
                None,
                "Service unavailable",
            )
        response = MagicMock()
        response.chunks = []
        return response

    async def GetChunksByIds(
        self, request: Any, timeout: float, metadata: Any = None
    ) -> Any:
        self.calls.append(("GetChunksByIds", request, timeout))
        self.last_metadata = metadata
        if self.channel.should_fail:
            raise grpc.aio.AioRpcError(
                self.channel.error_code,
                None,
                None,
                "Service unavailable",
            )
        response = MagicMock()
        response.chunks = []
        return response

    async def GetTopics(
        self, request: Any, timeout: float, metadata: Any = None
    ) -> Any:
        self.calls.append(("GetTopics", request, timeout))
        self.last_metadata = metadata
        if self.channel.should_fail:
            raise grpc.aio.AioRpcError(
                self.channel.error_code,
                None,
                None,
                "Service unavailable",
            )
        response = MagicMock()
        response.topics = []
        return response


@pytest.mark.unit
async def test_submission_client_create_question_set_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcSubmissionClient(target="grpc://localhost:50051")
    result = await client.create_question_set(
        CreateQuestionSetCommand(workflow_id="wf_123")
    )

    assert result.id == "qs_test_123"
    assert result.status == "created"
    assert result.workflow_id == "wf_123"

    await client.close()
    assert channel.closed


@pytest.mark.unit
async def test_submission_client_create_question_set_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = FakeGrpcChannel(should_fail=True, error_code=grpc.StatusCode.UNAVAILABLE)
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcSubmissionClient(target="grpc://localhost:50051")

    with pytest.raises(StorageTransientError) as _exc_info:
        await client.create_question_set(CreateQuestionSetCommand(workflow_id="wf_123"))

    assert "temporarily unavailable" in str(_exc_info.value).lower()
    await client.close()


@pytest.mark.unit
async def test_submission_client_create_question_set_permanent_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = FakeGrpcChannel(
        should_fail=True, error_code=grpc.StatusCode.PERMISSION_DENIED
    )
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcSubmissionClient(target="grpc://localhost:50051")

    with pytest.raises(StoragePermanentError):
        await client.create_question_set(CreateQuestionSetCommand(workflow_id="wf_123"))

    await client.close()


@pytest.mark.unit
async def test_submission_client_write_generated_questions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcSubmissionClient(target="grpc://localhost:50051")
    result = await client.write_generated_questions(
        WriteGeneratedQuestionsCommand(
            question_set_id="qs_123",
            questions=[
                Question(
                    question_id="q_001",
                    question_type="STRUCTURED",
                    content="What is 2+2?",
                    structured_answer="4",
                    non_structured_model_answer="",
                    metadata_json="{}",
                    topic_id="math",
                    iteration=1,
                    sort_order=1,
                )
            ],
        )
    )

    assert result.questions_written == 1
    assert result.status == "written"

    await client.close()


@pytest.mark.unit
async def test_submission_client_increment_iteration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcSubmissionClient(target="grpc://localhost:50051")
    result = await client.increment_iteration(
        IncrementIterationCommand(question_set_id="qs_123")
    )

    assert result.question_set_id == "qs_123"
    assert result.iteration_count == 2
    assert result.status == "incremented"

    await client.close()


@pytest.mark.unit
async def test_knowledge_client_similarity_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    result = await client.similarity_search(
        SimilaritySearchCommand(
            workflow_id="wf_123",
            query="test query",
            kb_type="document",
            top_k=5,
        )
    )

    assert result == []

    await client.close()


@pytest.mark.unit
async def test_knowledge_client_get_chunks_by_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    from qna_generation_agent.application.ports.knowledge_client import (
        GetChunksByIdsCommand,
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    result = await client.get_chunks_by_ids(
        GetChunksByIdsCommand(chunk_ids=["chunk_001", "chunk_002"])
    )

    assert result == []

    await client.close()


@pytest.mark.unit
async def test_knowledge_client_get_topics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    result = await client.get_topics(GetTopicsCommand(workflow_id="wf_123"))

    assert result == []

    await client.close()


@pytest.mark.unit
async def test_knowledge_client_closed_stub_raises_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    await client.close()

    with pytest.raises(StorageTransientError) as _exc_info:
        await client.similarity_search(
            SimilaritySearchCommand(
                workflow_id="wf_123",
                query="test",
                kb_type="document",
                top_k=5,
            )
        )

    assert "closed" in str(_exc_info.value).lower()


# ============================================================================
# Metadata Propagation Tests
# ============================================================================


@pytest.mark.unit
async def test_knowledge_client_similarity_search_propagates_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that similarity_search propagates trace metadata."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    # Set context variables
    import structlog

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        trace_id="trace_abc123",
        correlation_id="corr_xyz789",
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    await client.similarity_search(
        SimilaritySearchCommand(
            workflow_id="wf_123",
            query="test query",
            kb_type="document",
            top_k=5,
        )
    )

    # Verify metadata was passed
    assert fake_stub.last_metadata is not None
    metadata_dict = dict(fake_stub.last_metadata)
    assert metadata_dict.get("x-trace-id") == "trace_abc123"
    assert metadata_dict.get("x-correlation-id") == "corr_xyz789"

    await client.close()
    structlog.contextvars.clear_contextvars()


@pytest.mark.unit
async def test_knowledge_client_get_chunks_by_ids_propagates_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that get_chunks_by_ids propagates trace metadata."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    import structlog

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id="trace_test_123")

    from qna_generation_agent.application.ports.knowledge_client import (
        GetChunksByIdsCommand,
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    await client.get_chunks_by_ids(GetChunksByIdsCommand(chunk_ids=["chunk_001"]))

    assert fake_stub.last_metadata is not None
    metadata_dict = dict(fake_stub.last_metadata)
    assert metadata_dict.get("x-trace-id") == "trace_test_123"

    await client.close()
    structlog.contextvars.clear_contextvars()


@pytest.mark.unit
async def test_knowledge_client_get_topics_propagates_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that get_topics propagates trace metadata."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    import structlog

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id="corr_test_456")

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    await client.get_topics(GetTopicsCommand(workflow_id="wf_test"))

    assert fake_stub.last_metadata is not None
    metadata_dict = dict(fake_stub.last_metadata)
    assert metadata_dict.get("x-correlation-id") == "corr_test_456"

    await client.close()
    structlog.contextvars.clear_contextvars()


@pytest.mark.unit
async def test_submission_client_create_question_set_propagates_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that create_question_set propagates trace metadata."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    import structlog

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        trace_id="trace_sub_123",
        correlation_id="corr_sub_456",
    )

    client = GrpcSubmissionClient(target="grpc://localhost:50051")
    await client.create_question_set(CreateQuestionSetCommand(workflow_id="wf_123"))

    assert fake_stub.last_metadata is not None
    metadata_dict = dict(fake_stub.last_metadata)
    assert metadata_dict.get("x-trace-id") == "trace_sub_123"
    assert metadata_dict.get("x-correlation-id") == "corr_sub_456"

    await client.close()
    structlog.contextvars.clear_contextvars()


@pytest.mark.unit
async def test_submission_client_write_generated_questions_propagates_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that write_generated_questions propagates trace metadata."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    import structlog

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id="trace_write_789")

    client = GrpcSubmissionClient(target="grpc://localhost:50051")
    await client.write_generated_questions(
        WriteGeneratedQuestionsCommand(
            question_set_id="qs_123",
            questions=[
                Question(
                    question_id="q_001",
                    question_type="structured",
                    content="Test?",
                    structured_answer="Answer",
                )
            ],
        )
    )

    assert fake_stub.last_metadata is not None
    metadata_dict = dict(fake_stub.last_metadata)
    assert metadata_dict.get("x-trace-id") == "trace_write_789"

    await client.close()
    structlog.contextvars.clear_contextvars()


@pytest.mark.unit
async def test_submission_client_increment_iteration_propagates_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that increment_iteration propagates trace metadata."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    import structlog

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id="corr_inc_999")

    client = GrpcSubmissionClient(target="grpc://localhost:50051")
    await client.increment_iteration(
        IncrementIterationCommand(question_set_id="qs_123")
    )

    assert fake_stub.last_metadata is not None
    metadata_dict = dict(fake_stub.last_metadata)
    assert metadata_dict.get("x-correlation-id") == "corr_inc_999"

    await client.close()
    structlog.contextvars.clear_contextvars()


@pytest.mark.unit
async def test_grpc_metadata_no_context_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that when no context is set, metadata is None."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    # Clear any existing context
    import structlog

    structlog.contextvars.clear_contextvars()

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    await client.similarity_search(
        SimilaritySearchCommand(
            workflow_id="wf_123",
            query="test",
            kb_type="document",
            top_k=5,
        )
    )

    # When no context is set, metadata should be None
    assert fake_stub.last_metadata is None

    await client.close()


# ============================================================================
# Health Check Semantics Tests
# ============================================================================


@pytest.mark.unit
async def test_knowledge_client_health_check_unavailable_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that health_check returns False when service is UNAVAILABLE."""
    channel = FakeGrpcChannel(should_fail=True, error_code=grpc.StatusCode.UNAVAILABLE)
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    result = await client.health_check()

    assert result is False
    await client.close()


@pytest.mark.unit
async def test_knowledge_client_health_check_auth_error_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that health_check returns False on auth errors (misconfiguration)."""
    channel = FakeGrpcChannel(
        should_fail=True, error_code=grpc.StatusCode.PERMISSION_DENIED
    )
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    result = await client.health_check()

    assert result is False
    await client.close()


@pytest.mark.unit
async def test_knowledge_client_health_check_unimplemented_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that health_check returns False on UNIMPLEMENTED (contract drift)."""
    channel = FakeGrpcChannel(
        should_fail=True, error_code=grpc.StatusCode.UNIMPLEMENTED
    )
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    result = await client.health_check()

    assert result is False
    await client.close()


@pytest.mark.unit
async def test_knowledge_client_health_check_invalid_argument_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that health_check returns False on INVALID_ARGUMENT (contract drift)."""
    channel = FakeGrpcChannel(
        should_fail=True, error_code=grpc.StatusCode.INVALID_ARGUMENT
    )
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    result = await client.health_check()

    assert result is False
    await client.close()


@pytest.mark.unit
async def test_knowledge_client_health_check_internal_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that health_check returns False on INTERNAL error."""
    channel = FakeGrpcChannel(should_fail=True, error_code=grpc.StatusCode.INTERNAL)
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    result = await client.health_check()

    assert result is False
    await client.close()


@pytest.mark.unit
async def test_knowledge_client_health_check_success_returns_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that health_check returns True when service responds."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    result = await client.health_check()

    assert result is True
    await client.close()


@pytest.mark.unit
async def test_knowledge_client_health_check_closed_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that health_check returns False when client is closed."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.knowledge_client.KnowledgeServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    await client.close()

    result = await client.health_check()
    assert result is False


# ============================================================================
# GetAssessmentConfig Implementation Tests
# ============================================================================


@pytest.mark.unit
async def test_submission_client_get_assessment_config_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that get_assessment_config returns proper AssessmentConfig."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    import structlog

    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id="trace_config_123")

    from qna_generation_agent.application.ports.submission_client import (
        GetAssessmentConfigCommand,
    )

    client = GrpcSubmissionClient(target="grpc://localhost:50051")
    result = await client.get_assessment_config(
        GetAssessmentConfigCommand(
            assessment_id="assess_123",
            workflow_id="wf_123",
        )
    )

    assert result.assessment_id == "assess_123"
    assert result.workflow_id == "wf_123"
    assert result.assessor_id == "assessor_123"
    assert result.assessment_title == "Test Assessment"
    assert result.duration_minutes == 60
    assert result.structured_question_count == 5
    assert result.non_structured_question_count == 3

    # Verify metadata was propagated
    assert fake_stub.last_metadata is not None
    metadata_dict = dict(fake_stub.last_metadata)
    assert metadata_dict.get("x-trace-id") == "trace_config_123"

    await client.close()
    structlog.contextvars.clear_contextvars()


@pytest.mark.unit
async def test_submission_client_get_assessment_config_unimplemented_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that UNIMPLEMENTED error raises StoragePermanentError."""
    channel = FakeGrpcChannel(
        should_fail=True, error_code=grpc.StatusCode.UNIMPLEMENTED
    )
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    from qna_generation_agent.application.ports.submission_client import (
        GetAssessmentConfigCommand,
    )

    client = GrpcSubmissionClient(target="grpc://localhost:50051")

    with pytest.raises(StoragePermanentError) as exc_info:
        await client.get_assessment_config(
            GetAssessmentConfigCommand(
                assessment_id="assess_123",
                workflow_id="wf_123",
            )
        )

    assert "not yet implemented" in str(exc_info.value).lower()
    await client.close()


@pytest.mark.unit
async def test_submission_client_get_assessment_config_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that retryable errors raise StorageTransientError."""
    channel = FakeGrpcChannel(should_fail=True, error_code=grpc.StatusCode.UNAVAILABLE)
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    from qna_generation_agent.application.ports.submission_client import (
        GetAssessmentConfigCommand,
    )

    client = GrpcSubmissionClient(target="grpc://localhost:50051")

    with pytest.raises(StorageTransientError) as exc_info:
        await client.get_assessment_config(
            GetAssessmentConfigCommand(
                assessment_id="assess_123",
                workflow_id="wf_123",
            )
        )

    assert "temporarily unavailable" in str(exc_info.value).lower()
    await client.close()


# ============================================================================
# Closed Client Behavior Tests
# ============================================================================


@pytest.mark.unit
async def test_submission_client_closed_raises_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that closed client raises StorageTransientError."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcSubmissionClient(target="grpc://localhost:50051")
    await client.close()

    with pytest.raises(StorageTransientError) as exc_info:
        await client.create_question_set(CreateQuestionSetCommand(workflow_id="wf_123"))

    assert "closed" in str(exc_info.value).lower()


@pytest.mark.unit
async def test_submission_client_write_questions_closed_raises_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that write_generated_questions raises error when closed."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcSubmissionClient(target="grpc://localhost:50051")
    await client.close()

    with pytest.raises(StorageTransientError) as exc_info:
        await client.write_generated_questions(
            WriteGeneratedQuestionsCommand(
                question_set_id="qs_123",
                questions=[],
            )
        )

    assert "closed" in str(exc_info.value).lower()


# ============================================================================
# create_question_set Server State Tests
# ============================================================================


@pytest.mark.unit
async def test_submission_client_create_question_set_uses_server_fields_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that create_question_set uses server-returned fields when available."""
    channel = FakeGrpcChannel()
    fake_stub = FakeStub(channel)

    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.create_channel",
        lambda *args, **kwargs: channel,
    )
    monkeypatch.setattr(
        "qna_generation_agent.infrastructure.grpc.submission_client.SubmissionServiceStub",
        lambda channel: fake_stub,
    )

    client = GrpcSubmissionClient(target="grpc://localhost:50051")
    result = await client.create_question_set(
        CreateQuestionSetCommand(workflow_id="wf_123")
    )

    # Server returns these values in FakeStub
    assert result.workflow_id == "wf_123"  # From server response
    assert result.iteration_count == 1  # From server response
    assert result.created_at is None  # Not fabricated when server doesn't return it

    await client.close()
