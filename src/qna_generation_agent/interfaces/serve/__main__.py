"""Unified serve process entrypoint."""

from __future__ import annotations

from granian import Granian
from granian.constants import Interfaces

from qna_generation_agent.app.logging import configure_logging, get_logger
from qna_generation_agent.app.settings import load_settings

logger = get_logger(__name__)


def main() -> None:
    """Run Granian in ASGI mode with HTTP and Pub/Sub subscriber."""
    # Load settings once at startup - this is the single source of truth
    settings = load_settings()
    settings.validate_settings()
    configure_logging(settings.log_level.value)

    logger.info(
        "starting_serve_server",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        interface="asgi",
        subscriber=settings.pubsub_subscription_trigger,
    )

    # Configure Granian with proper shutdown behavior
    # respawn_failed_workers=False prevents auto-restart on clean exit
    # This allows graceful shutdown to work properly via ASGI lifespan events
    server = Granian(
        target="qna_generation_agent.interfaces.serve.app:app",
        interface=Interfaces.ASGI,
        address=settings.host,
        port=settings.port,
        workers=settings.workers,
        respawn_failed_workers=False,
    )

    server.serve()


if __name__ == "__main__":
    main()
