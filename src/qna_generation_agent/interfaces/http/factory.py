"""BlackSheep application factory."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, cast

from blacksheep import Application, Request, Response
from blacksheep.server.openapi.v3 import OpenAPIHandler
from blacksheep.settings.json import json_settings
from openapidocs.v3 import Info

from qna_generation_agent import __version__
from qna_generation_agent.app.bootstrap import ApplicationContainer, bootstrap_serve
from qna_generation_agent.app.json import dumps, loads
from qna_generation_agent.app.logging import configure_logging, get_logger
from qna_generation_agent.app.settings import Settings, load_settings
from qna_generation_agent.interfaces.http.middleware import (
    _get_cors_origin_header,
    correlation_middleware,
    error_middleware,
    request_logging_middleware,
)
from qna_generation_agent.interfaces.http.routes_health import register_health_routes
from qna_generation_agent.interfaces.http.routes_prompts import (
    register_prompt_test_routes,
)

logger = get_logger(__name__)


def _configure_json() -> None:
    cast(Any, json_settings).loads = loads
    cast(Any, json_settings).dumps = lambda value: dumps(value).decode("utf-8")


def _configure_openapi(app: Application) -> None:
    """Configure OpenAPI spec and Swagger UI."""
    info = Info(
        title="QnA Generation Agent",
        version=__version__,
        description="API for health checks and operational endpoints",
    )
    docs = OpenAPIHandler(
        info=info,
        ui_path="/docs",
        json_spec_path="/openapi.json",
    )
    docs.bind_app(app)

    # Silence favicon 404 noise from Swagger UI
    @app.router.get("/docs/favicon.png")
    async def _favicon() -> Response:
        return Response(204)


Handler = Callable[[Request], Awaitable[Response]]


def _make_cors_middleware(
    settings: Settings,
) -> Callable[[Request, Handler], Awaitable[Response]]:
    """Create CORS middleware with injected settings."""

    async def cors_middleware(request: Request, handler: Handler) -> Response:
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

    return cors_middleware


def create_blacksheep_app(
    container: ApplicationContainer | None = None,
) -> Application:
    """Factory pattern: build the BlackSheep application boundary."""
    _configure_json()
    if container is None:
        settings = load_settings()
        configure_logging(settings.log_level.value)
        app_container = bootstrap_serve(settings)
    else:
        app_container = container
    application = Application(show_error_details=app_container.settings.is_development)
    cast(Any, application.services).add_instance(app_container, ApplicationContainer)

    # Middlewares are applied in order: first appended = outermost
    # Order: CORS (outer layer) → correlation (set context) → logging (log with context) → errors (catch with context)
    # CORS must be outermost to ensure error responses include CORS headers for browser clients
    cors_middleware = _make_cors_middleware(app_container.settings)
    application.middlewares.append(cors_middleware)
    application.middlewares.append(correlation_middleware)
    application.middlewares.append(request_logging_middleware)
    application.middlewares.append(error_middleware)

    # Startup/shutdown handlers - used by both TestClient (app.start/stop) and ASGI lifespan
    # ASGI lifespan (in serve/app.py) provides graceful shutdown; these handlers ensure
    # TestClient mode also works correctly.
    @application.on_start
    async def on_start() -> None:
        await app_container.startup()
        logger.info("http_application_started")

    @application.on_stop
    async def on_stop() -> None:
        await app_container.shutdown()
        logger.info("http_application_stopped")

    register_health_routes(application)

    # Register prompt test routes in development or when explicitly enabled
    # These routes require Langfuse configuration
    if app_container.settings.is_development or getattr(
        app_container.settings, "enable_test_routes", False
    ):
        register_prompt_test_routes(application)
        logger.info(
            "prompt_test_routes_registered",
            is_development=app_container.settings.is_development,
        )

    _configure_openapi(application)
    return application


__all__ = ["create_blacksheep_app"]
