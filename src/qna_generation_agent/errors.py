"""Shared application error hierarchy.

This module defines the root exception taxonomy used across layers.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for typed application errors."""

    def __init__(self, message: str, **context: Any) -> None:
        """Initialize with a message and optional structured context."""
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        """Return a human-readable representation with context details."""
        if not self.context:
            return self.message

        details = ", ".join(f"{key}={value!r}" for key, value in self.context.items())
        return f"{self.message} ({details})"


class ConfigurationError(AppError):
    """Raised when process configuration is invalid."""
