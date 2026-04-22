"""HTTP response schemas."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = ConfigDict(strict=True)

    status: str = "healthy"


class LiveResponse(BaseModel):
    """Liveness probe response."""

    model_config = ConfigDict(strict=True)

    alive: bool = True


class ReadyResponse(BaseModel):
    """Readiness probe response."""

    model_config = ConfigDict(strict=True)

    ready: bool
    checks: dict[str, bool] = Field(default_factory=dict)


class VersionResponse(BaseModel):
    """Version information response."""

    model_config = ConfigDict(strict=True)

    version: str
    environment: str


class ErrorResponse(BaseModel):
    """Standard error response."""

    model_config = ConfigDict(strict=True)

    error: str
    request_id: str
