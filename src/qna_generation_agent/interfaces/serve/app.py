"""Granian target module for the unified serve runtime."""

from __future__ import annotations

from qna_generation_agent.app.bootstrap import bootstrap_serve
from qna_generation_agent.app.logging import configure_logging
from qna_generation_agent.app.settings import load_settings
from qna_generation_agent.interfaces.http.factory import create_blacksheep_app

# Load settings and configure logging once at import time
# This ensures consistent initialization when running via Granian
settings = load_settings()
configure_logging(settings.log_level.value)
container = bootstrap_serve(settings)

app = create_blacksheep_app(container)

__all__ = ["app"]
