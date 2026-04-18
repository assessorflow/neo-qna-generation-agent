"""gRPC client for Knowledge Service."""

from __future__ import annotations

import asyncio
from typing import Any

import grpc

from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.application.errors import StorageTransientError
from qna_generation_agent.application.ports.knowledge_client import (
    Chunk,
    GetChunksByIdsCommand,
    GetTopicsCommand,
    KnowledgeClient,
    SimilaritySearchCommand,
    Topic,
)
from qna_generation_agent.infrastructure.grpc.channel import (
    create_channel,
    get_grpc_metadata,
    handle_grpc_error,
)
from qna_generation_agent.infrastructure.grpc.stubs.assessorflow.knowledge.v1.knowledge_pb2 import (  # type: ignore[attr-defined]
    GetChunksByIdsRequest,
    GetTopicsRequest,
    SimilaritySearchRequest,
)
from qna_generation_agent.infrastructure.grpc.stubs.assessorflow.knowledge.v1.knowledge_pb2_grpc import (
    KnowledgeServiceStub,
)

logger = get_logger(__name__)


class GrpcKnowledgeClient(KnowledgeClient):
    """gRPC implementation of Knowledge Service client."""

    def __init__(
        self,
        target: str,
        *,
        timeout_seconds: float = 30.0,
        tls_enabled: bool = False,
        tls_cert_path: str | None = None,
    ) -> None:
        self._target = target
        self._timeout_seconds = timeout_seconds
        self._channel = create_channel(
            target,
            tls_enabled=tls_enabled,
            tls_cert_path=tls_cert_path,
        )
        self._stub: KnowledgeServiceStub | None = KnowledgeServiceStub(self._channel)  # type: ignore[no-untyped-call]
        self._close_lock = asyncio.Lock()
        logger.info(
            "knowledge_client_initialized",
            target=target,
            tls_enabled=tls_enabled,
        )

    async def similarity_search(
        self,
        command: SimilaritySearchCommand,
    ) -> list[Chunk]:
        """Search for similar chunks in the Knowledge Service."""
        if self._stub is None:
            raise StorageTransientError(
                "Knowledge Service client is closed",
                retry_after_seconds=5,
            )
        request = SimilaritySearchRequest(
            query=command.query,
            workflow_id=command.workflow_id,
            kb_type=command.kb_type,
            top_k=command.top_k,
            assessment_id=command.assessment_id,
        )

        metadata = get_grpc_metadata()
        try:
            response = await self._stub.SimilaritySearch(
                request,
                timeout=self._timeout_seconds,
                metadata=metadata,
            )
        except grpc.aio.AioRpcError as error:
            raise handle_grpc_error(error, "Knowledge Service") from error

        return [
            Chunk(
                chunk_id=c.chunk_id,
                workflow_id=c.workflow_id,
                content=c.content,
                source_type=c.source_type,
                metadata=dict(c.metadata),
                score=c.score,
            )
            for c in response.chunks
        ]

    async def get_chunks_by_ids(
        self,
        command: GetChunksByIdsCommand,
    ) -> list[Chunk]:
        """Get chunks by their IDs from the Knowledge Service."""
        if self._stub is None:
            raise StorageTransientError(
                "Knowledge Service client is closed",
                retry_after_seconds=5,
            )
        request = GetChunksByIdsRequest(chunk_ids=command.chunk_ids)

        metadata = get_grpc_metadata()
        try:
            response = await self._stub.GetChunksByIds(
                request,
                timeout=self._timeout_seconds,
                metadata=metadata,
            )
        except grpc.aio.AioRpcError as error:
            raise handle_grpc_error(error, "Knowledge Service") from error

        return [
            Chunk(
                chunk_id=c.chunk_id,
                workflow_id=c.workflow_id,
                content=c.content,
                source_type=c.source_type,
                metadata=dict(c.metadata),
                score=c.score,
            )
            for c in response.chunks
        ]

    async def get_topics(
        self,
        command: GetTopicsCommand,
    ) -> list[Topic]:
        """Get topics for a workflow from the Knowledge Service."""
        if self._stub is None:
            raise StorageTransientError(
                "Knowledge Service client is closed",
                retry_after_seconds=5,
            )
        request = GetTopicsRequest(workflow_id=command.workflow_id)

        metadata = get_grpc_metadata()
        try:
            response = await self._stub.GetTopics(
                request,
                timeout=self._timeout_seconds,
                metadata=metadata,
            )
        except grpc.aio.AioRpcError as error:
            raise handle_grpc_error(error, "Knowledge Service") from error

        def _convert_topic(proto_topic: Any) -> Topic:
            """Recursively convert proto topic to dataclass."""
            subtopics = [_convert_topic(st) for st in proto_topic.subtopics]
            return Topic(
                topic_id=proto_topic.topic_id,
                name=proto_topic.name,
                subtopics=subtopics,
            )

        return [_convert_topic(t) for t in response.topics]

    async def close(self) -> None:
        """Close the client connection."""
        async with self._close_lock:
            channel = self._channel
            if channel is None:
                return
            self._channel = None
            self._stub = None
            await channel.close()
            logger.info("knowledge_client_closed")

    async def health_check(self) -> bool:
        """Check connectivity to the Knowledge Service.

        Attempts a lightweight GetChunksByIds call with empty chunk IDs.
        This is a non-destructive operation that verifies the service is responsive.
        Treats auth/permission errors and contract drift as unhealthy.
        """
        if self._channel is None or self._stub is None:
            return False
        try:
            async with asyncio.timeout(5.0):
                request = GetChunksByIdsRequest(chunk_ids=[])
                metadata = get_grpc_metadata()
                await self._stub.GetChunksByIds(request, timeout=5.0, metadata=metadata)
            return True
        except TimeoutError:
            logger.warning("knowledge_health_check_timeout")
            return False
        except grpc.aio.AioRpcError as error:
            # UNAVAILABLE means service is not reachable
            if error.code() == grpc.StatusCode.UNAVAILABLE:
                logger.warning("knowledge_service_unavailable", error=str(error))
                return False
            # Auth/permission errors indicate misconfiguration - unhealthy
            if error.code() in (
                grpc.StatusCode.UNAUTHENTICATED,
                grpc.StatusCode.PERMISSION_DENIED,
            ):
                logger.error("knowledge_service_auth_failed", error=str(error))
                return False
            # Contract drift / method errors indicate incompatibility - unhealthy
            if error.code() in (
                grpc.StatusCode.UNIMPLEMENTED,
                grpc.StatusCode.INVALID_ARGUMENT,
                grpc.StatusCode.INTERNAL,
                grpc.StatusCode.UNKNOWN,
            ):
                logger.error(
                    "knowledge_service_contract_drift",
                    error=str(error),
                    code=str(error.code()),
                )
                return False
            # Any other error means the service is reachable but rejected the request
            return True
        except Exception as error:
            logger.warning("knowledge_health_check_failed", error=str(error))
            return False
