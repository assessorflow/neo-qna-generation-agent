"""Immutable value objects for the domain."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self
from uuid import uuid4

from qna_generation_agent.domain.errors import ValidationError

__all__ = [
    "AnswerId",
    "ContentHash",
    "QuestionId",
]


@dataclass(frozen=True)
class QuestionId:
    """Typed identifier for questions."""

    value: str

    def __post_init__(self) -> None:
        """Validate the identifier value is non-empty."""
        if not self.value or not self.value.strip():
            raise ValidationError("QuestionId cannot be empty")

    @classmethod
    def generate(cls, prefix: str = "q") -> Self:
        """Generate a new QuestionId with random suffix."""
        return cls(f"{prefix}_{uuid4().hex[:8]}")


@dataclass(frozen=True)
class AnswerId:
    """Typed identifier for answers."""

    value: str

    def __post_init__(self) -> None:
        """Validate the identifier value is non-empty."""
        if not self.value or not self.value.strip():
            raise ValidationError("AnswerId cannot be empty")

    @classmethod
    def generate(cls, prefix: str = "a") -> Self:
        """Generate a new AnswerId with random suffix."""
        return cls(f"{prefix}_{uuid4().hex[:8]}")


@dataclass(frozen=True)
class ContentHash:
    """Hash of document content for deduplication and versioning."""

    algorithm: str
    value: str

    def __post_init__(self) -> None:
        """Validate hash value and algorithm are valid."""
        if not self.value or not self.value.strip():
            raise ValidationError("ContentHash value cannot be empty")
        if self.algorithm not in ("sha256", "md5", "blake2b"):
            raise ValidationError(f"Unsupported hash algorithm: {self.algorithm}")

    @classmethod
    def from_content(cls, content: str, algorithm: str = "sha256") -> Self:
        """Create a ContentHash from string content."""
        import hashlib
        from typing import Any

        hasher: Any
        if algorithm == "sha256":
            hasher = hashlib.sha256()
        elif algorithm == "md5":
            hasher = hashlib.md5(usedforsecurity=False)
        elif algorithm == "blake2b":
            hasher = hashlib.blake2b()
        else:
            raise ValidationError(f"Unsupported algorithm: {algorithm}")

        hasher.update(content.encode("utf-8"))
        return cls(algorithm=algorithm, value=hasher.hexdigest())

    def __eq__(self, other: object) -> bool:
        """Compare two ContentHash instances by algorithm and value."""
        if not isinstance(other, ContentHash):
            return NotImplemented
        return self.algorithm == other.algorithm and self.value == other.value

    def __hash__(self) -> int:
        """Return a hash based on algorithm and value."""
        return hash((self.algorithm, self.value))
