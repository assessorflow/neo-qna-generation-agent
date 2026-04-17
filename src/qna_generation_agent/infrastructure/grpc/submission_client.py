"""gRPC client for Assessment Submission Service."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
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
from qna_generation_agent.infrastructure.grpc.channel import create_channel
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

# gRPC status codes that indicate transient (retryable) failures
_RETRYABLE_CODES: frozenset[grpc.StatusCode] = frozenset(
    {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
    }
)


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
        self._target = target
        self._timeout_seconds = timeout_seconds
        self._channel = create_channel(
            target,
            tls_enabled=tls_enabled,
            tls_cert_path=tls_cert_path,
        )
        self._stub: SubmissionServiceStub | None = SubmissionServiceStub(self._channel)  # type: ignore[no-untyped-call]
        logger.info(
            "submission_client_initialized",
            target=target,
            tls_enabled=tls_enabled,
        )

    async def get_assessment_config(
        self,
        command: Any,  # GetAssessmentConfigCommand
    ) -> Any:  # AssessmentConfig
        """Stub: GetAssessmentConfig not yet implemented in proto.

        This method is defined in the spec but not yet available
        in the gRPC service. Use trigger payload data instead.
        """
        raise StoragePermanentError(
            "GetAssessmentConfig not yet implemented in Submission Service proto"
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

        try:
            response = await self._stub.CreateQuestionSet(
                request,
                timeout=self._timeout_seconds,
            )
        except grpc.aio.AioRpcError as error:
            if error.code() in _RETRYABLE_CODES:
                raise StorageTransientError(
                    "Submission Service temporarily unavailable",
                    retry_after_seconds=5,
                    code=str(error.code()),
                    details=error.details(),
                ) from error
            raise StoragePermanentError(
                "Submission Service call failed",
                code=str(error.code()),
                details=error.details(),
            ) from error

        # Note: The proto doesn't return created_at, workflow_id, iteration_count
        # Using defaults for backwards compatibility
        return QuestionSetRecord(
            id=response.question_set_id,
            workflow_id=command.workflow_id,
            iteration_count=0,
            status=response.status,
            created_at=datetime.now(UTC),
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

        try:
            response = await self._stub.WriteGeneratedQuestions(
                request,
                timeout=self._timeout_seconds,
            )
        except grpc.aio.AioRpcError as error:
            if error.code() in _RETRYABLE_CODES:
                raise StorageTransientError(
                    "Submission Service temporarily unavailable",
                    retry_after_seconds=5,
                    code=str(error.code()),
                    details=error.details(),
                ) from error
            raise StoragePermanentError(
                "Submission Service call failed",
                code=str(error.code()),
                details=error.details(),
            ) from error

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

        try:
            response = await self._stub.IncrementQuestionSetIteration(
                request,
                timeout=self._timeout_seconds,
            )
        except grpc.aio.AioRpcError as error:
            if error.code() in _RETRYABLE_CODES:
                raise StorageTransientError(
                    "Submission Service temporarily unavailable",
                    retry_after_seconds=5,
                    code=str(error.code()),
                    details=error.details(),
                ) from error
            raise StoragePermanentError(
                "Submission Service call failed",
                code=str(error.code()),
                details=error.details(),
            ) from error

        return IncrementIterationResult(
            question_set_id=response.question_set_id,
            iteration_count=response.iteration_count,
            status=response.status,
        )

    async def close(self) -> None:
        """Close the client connection."""
        channel = self._channel
        if channel is None:
            return
        self._channel = None
        self._stub = None
        await channel.close()
        logger.info("submission_client_closed")

    async def health_check(self) -> bool:
        """Check connectivity to the Submission Service.

        Attempts a lightweight CreateQuestionSet call with minimal data.
        This verifies the service is responsive (even if the call fails validation).
        Treats auth/permission errors as unhealthy (misconfiguration).
        """
        if self._channel is None or self._stub is None:
            return False
        try:
            async with asyncio.timeout(5.0):
                request = CreateQuestionSetRequest(workflow_id="health-check")
                await self._stub.CreateQuestionSet(request, timeout=5.0)
            return True
        except grpc.aio.AioRpcError as error:
            # UNAVAILABLE means service is not reachable
            if error.code() == grpc.StatusCode.UNAVAILABLE:
                logger.warning("submission_service_unavailable", error=str(error))
                return False
            # Auth/permission errors indicate misconfiguration - unhealthy
            if error.code() in (
                grpc.StatusCode.UNAUTHENTICATED,
                grpc.StatusCode.PERMISSION_DENIED,
            ):
                logger.error("submission_service_auth_failed", error=str(error))
                return False
            # Any other error means the service is reachable but rejected the request
            return True
        except Exception as error:
            logger.warning("submission_health_check_failed", error=str(error))
            return False
