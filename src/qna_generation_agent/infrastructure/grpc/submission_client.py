"""gRPC client for Assessment Submission Service."""

from __future__ import annotations

import asyncio
from typing import Any

import grpc

from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.application.errors import (
    StoragePermanentError,
    StorageTransientError,
)
from qna_generation_agent.application.ports.submission_client import (
    CreateQuestionSetCommand,
    IncrementIterationCommand,
    IncrementIterationResult,
    QuestionSetRecord,
    SubmissionClient,
    WriteGeneratedQuestionsCommand,
    WriteGeneratedQuestionsResult,
)
from qna_generation_agent.infrastructure.grpc.channel import (
    create_channel,
    get_grpc_metadata,
    handle_grpc_error,
)
from qna_generation_agent.infrastructure.grpc.stubs.assessorflow.submission.v1.submission_pb2 import (  # type: ignore[attr-defined]
    CreateQuestionSetRequest,
    IncrementIterationRequest,
    WriteGeneratedQuestionsRequest,
)
from qna_generation_agent.infrastructure.grpc.stubs.assessorflow.submission.v1.submission_pb2 import (  # type: ignore[attr-defined]
    Question as QuestionProto,
)
from qna_generation_agent.infrastructure.grpc.stubs.assessorflow.submission.v1.submission_pb2_grpc import (
    SubmissionServiceStub,
)

logger = get_logger(__name__)


class GrpcSubmissionClient(SubmissionClient):
    """gRPC implementation of Submission Service client."""

    def __init__(
        self,
        target: str,
        *,
        timeout_seconds: float = 30.0,
        tls_enabled: bool = False,
        tls_cert_path: str | None = None,
    ) -> None:
        """Initialize the gRPC Submission Service client.

        Args:
            target: gRPC target URL.
            timeout_seconds: Default timeout for RPC calls.
            tls_enabled: Whether to use TLS.
            tls_cert_path: Optional path to a custom CA certificate.
        """
        self._target = target
        self._timeout_seconds = timeout_seconds
        self._channel = create_channel(
            target,
            tls_enabled=tls_enabled,
            tls_cert_path=tls_cert_path,
        )
        self._stub: SubmissionServiceStub | None = SubmissionServiceStub(self._channel)  # type: ignore[no-untyped-call]
        self._close_lock = asyncio.Lock()
        logger.info(
            "submission_client_initialized",
            target=target,
            tls_enabled=tls_enabled,
        )

    async def get_assessment_config(
        self,
        command: Any,  # GetAssessmentConfigCommand
    ) -> Any:  # AssessmentConfig
        """Get assessment configuration from Submission Service.

        Retrieves the authoritative assessment config including generation
        parameters for regeneration scenarios.
        """
        from qna_generation_agent.application.ports.submission_client import (
            AssessmentConfig,
        )

        if self._stub is None:
            raise StorageTransientError(
                "Submission Service client is closed",
                retry_after_seconds=5,
            )

        # Import proto types dynamically to avoid issues if proto not yet updated
        try:
            from qna_generation_agent.infrastructure.grpc.stubs.assessorflow.submission.v1.submission_pb2 import (  # type: ignore[attr-defined]
                GetAssessmentConfigRequest,
            )
        except ImportError as e:
            # Proto not yet updated - use fallback response
            logger.warning("get_assessment_config_proto_not_available", error=str(e))
            raise StoragePermanentError(
                "GetAssessmentConfig proto not yet available",
                details=str(e),
            ) from e

        request = GetAssessmentConfigRequest(
            assessment_id=command.assessment_id,
            workflow_id=command.workflow_id,
        )

        metadata = get_grpc_metadata()
        try:
            response = await self._stub.GetAssessmentConfig(
                request,
                timeout=self._timeout_seconds,
                metadata=metadata,
            )
        except grpc.aio.AioRpcError as error:
            if error.code() == grpc.StatusCode.UNIMPLEMENTED:
                raise StoragePermanentError(
                    "GetAssessmentConfig not yet implemented in Submission Service",
                    code=str(error.code()),
                    details=error.details(),
                ) from error
            raise handle_grpc_error(error, "Submission Service") from error

        return AssessmentConfig(
            assessment_id=response.assessment_id,
            workflow_id=response.workflow_id,
            assessor_id=response.assessor_id,
            assessment_title=response.assessment_title,
            purpose=response.purpose if hasattr(response, "purpose") else None,
            duration_minutes=response.duration_minutes,
            difficulty_level=response.difficulty_level
            if hasattr(response, "difficulty_level")
            else None,
            structured_question_count=response.structured_question_count
            if hasattr(response, "structured_question_count")
            else 0,
            non_structured_question_count=response.non_structured_question_count
            if hasattr(response, "non_structured_question_count")
            else 0,
            web_research_mode=response.web_research_mode,
            status=response.status,
            deadline=response.deadline if hasattr(response, "deadline") else None,
        )

    async def create_question_set(
        self,
        command: CreateQuestionSetCommand,
    ) -> QuestionSetRecord:
        """Create a question set record in the Submission Service."""
        if self._stub is None:
            raise StorageTransientError(
                "Submission Service client is closed",
                retry_after_seconds=5,
            )
        request = CreateQuestionSetRequest(
            workflow_id=command.workflow_id,
        )

        metadata = get_grpc_metadata()
        try:
            response = await self._stub.CreateQuestionSet(
                request,
                timeout=self._timeout_seconds,
                metadata=metadata,
            )
        except grpc.aio.AioRpcError as error:
            raise handle_grpc_error(error, "Submission Service") from error

        # Only populate fields returned by the server
        # Server may not return: created_at, workflow_id, iteration_count
        return QuestionSetRecord(
            id=response.question_set_id,
            workflow_id=getattr(response, "workflow_id", command.workflow_id),
            iteration_count=getattr(response, "iteration_count", 0),
            status=response.status,
            created_at=None,
        )

    async def write_generated_questions(
        self,
        command: WriteGeneratedQuestionsCommand,
    ) -> WriteGeneratedQuestionsResult:
        """Write generated questions to the Submission Service."""
        if self._stub is None:
            raise StorageTransientError(
                "Submission Service client is closed",
                retry_after_seconds=5,
            )
        proto_questions = [
            QuestionProto(
                question_id=q.question_id,
                question_type=q.question_type,
                content=q.content,
                structured_answer=q.structured_answer,
                non_structured_model_answer=q.non_structured_model_answer,
                metadata_json=q.metadata_json,
                topic_id=q.topic_id,
                iteration=q.iteration,
                sort_order=q.sort_order,
            )
            for q in command.questions
        ]
        request = WriteGeneratedQuestionsRequest(
            question_set_id=command.question_set_id,
            questions=proto_questions,
        )

        metadata = get_grpc_metadata()
        try:
            response = await self._stub.WriteGeneratedQuestions(
                request,
                timeout=self._timeout_seconds,
                metadata=metadata,
            )
        except grpc.aio.AioRpcError as error:
            raise handle_grpc_error(error, "Submission Service") from error

        return WriteGeneratedQuestionsResult(
            questions_written=response.questions_written,
            status=response.status,
        )

    async def increment_iteration(
        self,
        command: IncrementIterationCommand,
    ) -> IncrementIterationResult:
        """Increment the iteration count for a question set."""
        if self._stub is None:
            raise StorageTransientError(
                "Submission Service client is closed",
                retry_after_seconds=5,
            )
        request = IncrementIterationRequest(
            question_set_id=command.question_set_id,
        )

        metadata = get_grpc_metadata()
        try:
            response = await self._stub.IncrementQuestionSetIteration(
                request,
                timeout=self._timeout_seconds,
                metadata=metadata,
            )
        except grpc.aio.AioRpcError as error:
            raise handle_grpc_error(error, "Submission Service") from error

        return IncrementIterationResult(
            question_set_id=response.question_set_id,
            iteration_count=response.iteration_count,
            status=response.status,
        )

    async def close(self) -> None:
        """Close the client connection."""
        async with self._close_lock:
            channel = self._channel
            if channel is None:
                return
            self._channel = None
            self._stub = None
            await channel.close()
            logger.info("submission_client_closed")

    async def health_check(self) -> bool:
        """Check connectivity to the Submission Service.

        Uses gRPC channel connectivity state to verify the service is reachable
        without performing any mutating operations. This avoids creating
        test records in the submission system.
        """
        if self._channel is None or self._stub is None:
            return False

        try:
            # Check if channel is in a ready state
            channel_state = self._channel.get_state(try_to_connect=True)
            if channel_state == grpc.ChannelConnectivity.READY:
                return True

            # Wait briefly for connection attempt
            async with asyncio.timeout(5.0):
                # Try to connect and wait for READY state
                await self._channel.wait_for_state_change(channel_state)
                new_state: grpc.ChannelConnectivity = self._channel.get_state()
                is_ready: bool = new_state == grpc.ChannelConnectivity.READY
                return is_ready

        except TimeoutError:
            logger.warning("submission_health_check_timeout")
            return False
        except Exception as error:
            logger.warning("submission_health_check_failed", error=str(error))
            return False
