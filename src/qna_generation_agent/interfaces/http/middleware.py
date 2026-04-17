"""BlackSheep middleware functions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from time import perf_counter
from uuid import uuid4

from blacksheep import Request, Response
from blacksheep.server.responses import json

from qna_generation_agent.app.logging import bind_context, clear_context, get_logger
from qna_generation_agent.app.settings import Settings
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
    # No restrictions configured - development mode, allow any origin
    if not allowed_origins:
        return (b"access-control-allow-origin", b"*")

    # Specific origins configured - validate request origin
    if request_origin is None:
        return (b"access-control-allow-origin", b"")

    decoded_origin = request_origin.decode("utf-8", errors="replace")

    # Check for wildcard (not recommended for production with credentials)
    if "*" in allowed_origins and not allow_credentials:
        return (b"access-control-allow-origin", b"*")

    # Exact match required when credentials are used or specific origins listed
    if decoded_origin in allowed_origins:
        return (b"access-control-allow-origin", request_origin)

    # Origin not allowed - return empty (browser will block)
    return (b"access-control-allow-origin", b"")


async def cors_middleware(
    request: Request,
    handler: Handler,
    settings: Settings,
) -> Response:
    """Add CORS headers for browser access with configurable origins.

    Security:
        - Validates Origin header against configured allowlist
        - Blocks credentials when wildcard is used
        - Returns empty origin header for disallowed origins
    """
    if request.method == "OPTIONS":
        response = Response(204)
    else:
        response = await handler(request)

    request_origin = request.get_first_header(b"origin")
    origin_header_name, origin_header_value = _get_cors_origin_header(
        request_origin,
        settings.cors_origins_list,
        settings.cors_allow_credentials,
    )

    response.add_header(origin_header_name, origin_header_value)
    response.add_header(b"access-control-allow-methods", b"GET, POST, OPTIONS")
    response.add_header(
        b"access-control-allow-headers",
        b"content-type, accept, authorization, x-request-id, x-correlation-id",
    )

    if settings.cors_allow_credentials:
        response.add_header(b"access-control-allow-credentials", b"true")

    return response


def _resolve_request_id(request: Request) -> str:
    raw_request_id = request.get_first_header(
        b"x-request-id"
    ) or request.get_first_header(b"x-correlation-id")
    if raw_request_id is None:
        return str(uuid4())
    return raw_request_id.decode("utf-8")


def _get_request_id_from_context() -> str | None:
    """Get request_id from structlog context if available."""
    # Import locally to avoid circular imports at module level
    import structlog

    ctx = structlog.contextvars.get_contextvars()
    return ctx.get("request_id")


async def correlation_middleware(request: Request, handler: Handler) -> Response:
    """Bind request id / trace id context for the request lifecycle."""
    clear_context()
    request_id = _resolve_request_id(request)
    bind_context(request_id=request_id, trace_id=None)
    try:
        response = await handler(request)
        response.add_header(b"x-request-id", request_id.encode("utf-8"))
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


async def error_middleware(request: Request, handler: Handler) -> Response:
    """Convert uncaught errors into structured JSON responses."""
    try:
        return await handler(request)
    except AppError as error:
        # Use request_id from context if available (set by correlation_middleware),
        # otherwise resolve from headers. This ensures consistent request_id even
        # when the original request had no x-request-id header.
        request_id = _get_request_id_from_context() or _resolve_request_id(request)
        logger.error(
            "handled_http_error",
            method=request.method,
            path=request.path,
            error=str(error),
        )
        return json(
            ErrorResponse(
                error=error.__class__.__name__, request_id=request_id
            ).model_dump(mode="json"),
            status=500,
        )
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
