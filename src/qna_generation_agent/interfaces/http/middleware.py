"""BlackSheep middleware functions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from blacksheep import Request, Response
from blacksheep.exceptions import BadRequestFormat
from blacksheep.server.responses import json

from qna_generation_agent.app.logging import bind_context, clear_context, get_logger
from qna_generation_agent.application.errors import (
    IdempotencyConflict,
    TransientError,
    ValidationError,
)
from qna_generation_agent.errors import AppError
from qna_generation_agent.interfaces.http.schemas import ErrorResponse

logger = get_logger(__name__)

Handler = Callable[[Request], Awaitable[Response]]


def _get_cors_origin_header(
    request_origin: bytes | None,
    allowed_origins: list[str],
    allow_credentials: bool,
) -> tuple[bytes, bytes]:
    """Determine CORS origin header value based on configured origins.

    Security: Returns no origin header if request origin is not in allowed list.
    Empty allowed_origins means no CORS restrictions (development mode).
    """
    if not allowed_origins:
        return (b"access-control-allow-origin", b"*")

    if request_origin is None:
        return (b"access-control-allow-origin", b"")

    decoded_origin = request_origin.decode("utf-8", errors="replace")

    if "*" in allowed_origins and not allow_credentials:
        return (b"access-control-allow-origin", b"*")

    if decoded_origin in allowed_origins:
        return (b"access-control-allow-origin", request_origin)

    return (b"access-control-allow-origin", b"")


def _resolve_request_id(request: Request) -> str:
    raw_request_id = request.get_first_header(
        b"x-request-id"
    ) or request.get_first_header(b"x-correlation-id")
    if raw_request_id is None:
        return str(uuid4())
    return raw_request_id.decode("utf-8")


def _resolve_trace_id(request: Request) -> str | None:
    """Extract trace ID from request headers if present."""
    for header in [b"x-trace-id", b"trace-id", b"x-b3-traceid"]:
        raw = request.get_first_header(header)
        if raw is not None:
            return raw.decode("utf-8", errors="replace")
    return None


def _get_request_id_from_context() -> str | None:
    """Get request_id from structlog context if available."""
    import structlog

    ctx = structlog.contextvars.get_contextvars()
    return ctx.get("request_id")


async def correlation_middleware(request: Request, handler: Handler) -> Response:
    """Bind request id / trace id context for the request lifecycle."""
    clear_context()
    request_id = _resolve_request_id(request)
    trace_id = _resolve_trace_id(request)
    bind_context(request_id=request_id, trace_id=trace_id)
    try:
        response = await handler(request)
        response.add_header(b"x-request-id", request_id.encode("utf-8"))
        if trace_id:
            response.add_header(b"x-trace-id", trace_id.encode("utf-8"))
        return response
    finally:
        clear_context()


async def request_logging_middleware(request: Request, handler: Handler) -> Response:
    """Log request start and completion."""
    started_at = perf_counter()
    logger.info("request_started", method=request.method, path=request.path)
    response = await handler(request)
    duration_ms = round((perf_counter() - started_at) * 1000, 2)
    logger.info(
        "request_completed",
        method=request.method,
        path=request.path,
        status=response.status,
        duration_ms=duration_ms,
    )
    return response


def _map_error_to_status(error: AppError) -> int:
    """Map typed errors to appropriate HTTP status codes.

    - ValidationError -> 400 Bad Request
    - IdempotencyConflict -> 409 Conflict
    - TransientError (and subclasses) -> 503 Service Unavailable
    - All other AppError subclasses -> 500 Internal Server Error
    """
    if isinstance(error, ValidationError):
        return 400
    if isinstance(error, IdempotencyConflict):
        return 409
    if isinstance(error, TransientError):
        return 503
    return 500


async def error_middleware(request: Request, handler: Handler) -> Response:
    """Convert uncaught errors into structured JSON responses.

    Maps typed AppError subclasses to appropriate HTTP status codes:
    - ValidationError -> 400 Bad Request
    - IdempotencyConflict -> 409 Conflict
    - TransientError (and subclasses like StorageTransientError, LLMTransientError) -> 503 Service Unavailable
    - PermanentError and other AppError subclasses -> 500 Internal Server Error
    - BadRequestFormat (malformed JSON, etc.) -> 400 Bad Request
    """
    try:
        return await handler(request)
    except BadRequestFormat as error:
        request_id = _get_request_id_from_context() or _resolve_request_id(request)
        logger.warning(
            "bad_request_format",
            method=request.method,
            path=request.path,
            error=str(error),
            request_id=request_id,
        )
        return json(
            ErrorResponse(
                error="bad_request",
                request_id=request_id,
            ).model_dump(mode="json"),
            status=400,
        )
    except AppError as error:
        request_id = _get_request_id_from_context() or _resolve_request_id(request)
        status = _map_error_to_status(error)
        logger.error(
            "handled_http_error",
            method=request.method,
            path=request.path,
            error=str(error),
            error_type=error.__class__.__name__,
            status=status,
        )
        response = json(
            ErrorResponse(
                error=error.__class__.__name__, request_id=request_id
            ).model_dump(mode="json"),
            status=status,
        )
        if isinstance(error, TransientError) and error.retry_after_seconds:
            response.add_header(
                b"retry-after", str(error.retry_after_seconds).encode("utf-8")
            )
        return response
    except Exception as error:
        request_id = _get_request_id_from_context() or _resolve_request_id(request)
        logger.exception(
            "unhandled_http_error",
            method=request.method,
            path=request.path,
            error=str(error),
        )
        return json(
            ErrorResponse(
                error="internal_server_error", request_id=request_id
            ).model_dump(mode="json"),
            status=500,
        )
