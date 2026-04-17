"""Shared gRPC channel creation utilities."""

from __future__ import annotations

from typing import Any

import grpc
import structlog
from grpc.aio import Channel

from qna_generation_agent.app.logging import get_logger
from qna_generation_agent.application.errors import (
    StoragePermanentError,
    StorageTransientError,
)

logger = get_logger(__name__)

# gRPC metadata key names for context propagation
_TRACE_ID_KEY = "x-trace-id"
_CORRELATION_ID_KEY = "x-correlation-id"

# gRPC status codes that indicate transient (retryable) failures
RETRYABLE_CODES: frozenset[grpc.StatusCode] = frozenset(
    {
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.RESOURCE_EXHAUSTED,
    }
)


def handle_grpc_error(
    error: grpc.aio.AioRpcError,
    service_name: str,
    retryable_codes: frozenset[grpc.StatusCode] | None = None,
) -> StorageTransientError | StoragePermanentError:
    """Convert gRPC error to appropriate storage error.

    Args:
        error: The gRPC AioRpcError to handle.
        service_name: Name of the service for error messages.
        retryable_codes: Optional set of retryable status codes. Defaults to RETRYABLE_CODES.

    Returns:
        Either a StorageTransientError or StoragePermanentError.
    """
    codes = retryable_codes or RETRYABLE_CODES
    code_str = str(error.code())
    details = error.details()

    if error.code() in codes:
        return StorageTransientError(
            f"{service_name} temporarily unavailable",
            retry_after_seconds=5,
            code=code_str,
            details=details,
        )
    return StoragePermanentError(
        f"{service_name} call failed",
        code=code_str,
        details=details,
    )


def create_channel(
    target: str,
    *,
    tls_enabled: bool = False,
    tls_cert_path: str | None = None,
) -> Channel:
    """Create gRPC channel with optional TLS encryption.

    Security: Uses TLS when tls_enabled=True. If tls_cert_path is provided,
    uses custom CA certificate; otherwise uses system default roots.

    Args:
        target: The gRPC target (host:port).
        tls_enabled: Whether to enable TLS encryption.
        tls_cert_path: Optional path to custom CA certificate file.

    Returns:
        A grpc.aio Channel instance.
    """
    if not tls_enabled:
        logger.debug("grpc_insecure_channel", target=target)
        return grpc.aio.insecure_channel(target)

    if tls_cert_path:
        with open(tls_cert_path, "rb") as f:
            credentials = grpc.ssl_channel_credentials(f.read())
        logger.info(
            "grpc_tls_channel_with_custom_ca", target=target, cert_path=tls_cert_path
        )
    else:
        credentials = grpc.ssl_channel_credentials()
        logger.info("grpc_tls_channel_with_system_roots", target=target)

    return grpc.aio.secure_channel(target, credentials)


def get_grpc_metadata(
    trace_id: str | None = None,
    correlation_id: str | None = None,
    additional: dict[str, str] | None = None,
) -> tuple[tuple[str, str], ...] | None:
    """Build gRPC metadata tuple from context variables or explicit values.

    Extracts trace_id and correlation_id from structlog contextvars if not
    explicitly provided. Returns None if no metadata values are available.

    Args:
        trace_id: Optional explicit trace ID. If None, attempts to read
            from structlog contextvars.
        correlation_id: Optional explicit correlation ID. If None, attempts
            to read from structlog contextvars.
        additional: Optional additional metadata key-value pairs.

    Returns:
        A tuple of (key, value) pairs suitable for gRPC metadata,
        or None if no metadata is available.

    Example:
        metadata = get_grpc_metadata()
        response = await stub.SomeMethod(request, metadata=metadata)
    """
    # Get contextvars if explicit values not provided
    ctx: dict[str, Any] = structlog.contextvars.get_contextvars()

    effective_trace_id = trace_id if trace_id is not None else ctx.get("trace_id")
    effective_correlation_id = (
        correlation_id if correlation_id is not None else ctx.get("correlation_id")
    )

    metadata: list[tuple[str, str]] = []

    if effective_trace_id:
        metadata.append((_TRACE_ID_KEY, str(effective_trace_id)))
    if effective_correlation_id:
        metadata.append((_CORRELATION_ID_KEY, str(effective_correlation_id)))

    if additional:
        for key, value in additional.items():
            if value is not None:
                metadata.append((key, str(value)))

    return tuple(metadata) if metadata else None
