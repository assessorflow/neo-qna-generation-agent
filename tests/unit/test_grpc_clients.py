"""Unit tests for gRPC client error handling."""

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

    async def CreateQuestionSet(self, request: Any, timeout: float) -> Any:  # noqa: ASYNC109
        self.calls.append(("CreateQuestionSet", request, timeout))
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
        return response

    async def WriteGeneratedQuestions(self, request: Any, timeout: float) -> Any:  # noqa: ASYNC109
        self.calls.append(("WriteGeneratedQuestions", request, timeout))
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

    async def IncrementQuestionSetIteration(self, request: Any, timeout: float) -> Any:  # noqa: ASYNC109
        self.calls.append(("IncrementQuestionSetIteration", request, timeout))
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

    async def SimilaritySearch(self, request: Any, timeout: float) -> Any:  # noqa: ASYNC109
        self.calls.append(("SimilaritySearch", request, timeout))
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

    async def GetChunksByIds(self, request: Any, timeout: float) -> Any:  # noqa: ASYNC109
        self.calls.append(("GetChunksByIds", request, timeout))
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

    async def GetTopics(self, request: Any, timeout: float) -> Any:  # noqa: ASYNC109
        self.calls.append(("GetTopics", request, timeout))
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

    client = GrpcKnowledgeClient(target="grpc://localhost:50052")
    result = await client.get_chunks_by_ids(["chunk_001", "chunk_002"])

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
