"""Unit tests for security features (CORS, TLS)."""

from __future__ import annotations

import pytest
from pytest import MonkeyPatch

from qna_generation_agent.app.settings import Settings
from qna_generation_agent.interfaces.http.middleware import _get_cors_origin_header


@pytest.mark.unit
class TestCorsOriginHeader:
    """Tests for CORS origin header validation."""

    def test_no_restrictions_returns_wildcard(self) -> None:
        """When no origins configured, return wildcard (dev mode)."""
        header_name, header_value = _get_cors_origin_header(
            b"http://example.com",
            [],
            False,
        )
        assert header_name == b"access-control-allow-origin"
        assert header_value == b"*"

    def test_specific_origin_allowed(self) -> None:
        """Request origin in allowlist should be returned."""
        _header_name, header_value = _get_cors_origin_header(
            b"https://app.example.com",
            ["https://app.example.com"],
            False,
        )
        assert header_value == b"https://app.example.com"

    def test_origin_not_in_allowlist_blocked(self) -> None:
        """Request origin not in allowlist should be blocked."""
        _header_name, header_value = _get_cors_origin_header(
            b"https://evil.com",
            ["https://app.example.com"],
            False,
        )
        assert header_value == b""

    def test_wildcard_blocked_with_credentials(self) -> None:
        """Wildcard should be blocked when credentials are enabled."""
        _header_name, header_value = _get_cors_origin_header(
            b"http://example.com",
            ["*"],
            True,
        )
        # When credentials are enabled, wildcard is blocked
        assert header_value == b""

    def test_wildcard_allowed_without_credentials(self) -> None:
        """Wildcard should be allowed when credentials are disabled."""
        _header_name, header_value = _get_cors_origin_header(
            b"http://example.com",
            ["*"],
            False,
        )
        assert header_value == b"*"

    def test_missing_origin_header(self) -> None:
        """When no origin header provided, return empty."""
        _header_name, header_value = _get_cors_origin_header(
            None,
            ["https://app.example.com"],
            False,
        )
        assert header_value == b""

    def test_origin_with_whitespace_trimmed(self) -> None:
        """Origins with whitespace should be handled."""
        _header_name, header_value = _get_cors_origin_header(
            b"https://app.example.com ",  # Trailing space
            ["https://app.example.com"],
            False,
        )
        # The decoding happens in the function, spaces are preserved in match
        # This test documents current behavior
        assert header_value == b""  # Won't match due to space


@pytest.mark.unit
class TestCorsSettings:
    """Tests for CORS settings integration."""

    def test_cors_origins_list_parsing(self, monkeypatch: MonkeyPatch) -> None:
        """Settings should parse comma-separated origins."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
        monkeypatch.setenv("SUBMISSION_SERVICE_URL", "grpc://localhost:50051")
        monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "grpc://localhost:50052")
        monkeypatch.setenv("IDENTITY_ACCESS_SERVICE_URL", "grpc://localhost:50053")
        monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://a.com, https://b.com")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.cors_origins_list == ["https://a.com", "https://b.com"]

    def test_cors_origins_list_empty(self, monkeypatch: MonkeyPatch) -> None:
        """Empty CORS origins should return empty list."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
        monkeypatch.setenv("SUBMISSION_SERVICE_URL", "grpc://localhost:50051")
        monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "grpc://localhost:50052")
        monkeypatch.setenv("IDENTITY_ACCESS_SERVICE_URL", "grpc://localhost:50053")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.cors_origins_list == []

    def test_cors_allow_credentials_default(self, monkeypatch: MonkeyPatch) -> None:
        """CORS credentials should default to False."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
        monkeypatch.setenv("SUBMISSION_SERVICE_URL", "grpc://localhost:50051")
        monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "grpc://localhost:50052")
        monkeypatch.setenv("IDENTITY_ACCESS_SERVICE_URL", "grpc://localhost:50053")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.cors_allow_credentials is False


@pytest.mark.unit
class TestGrpcTlsSettings:
    """Tests for gRPC TLS settings."""

    def test_grpc_tls_disabled_by_default(self, monkeypatch: MonkeyPatch) -> None:
        """gRPC TLS should be disabled by default."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
        monkeypatch.setenv("SUBMISSION_SERVICE_URL", "grpc://localhost:50051")
        monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "grpc://localhost:50052")
        monkeypatch.setenv("IDENTITY_ACCESS_SERVICE_URL", "grpc://localhost:50053")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.grpc_tls_enabled is False

    def test_grpc_tls_enabled_via_env(self, monkeypatch: MonkeyPatch) -> None:
        """gRPC TLS can be enabled via environment."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
        monkeypatch.setenv("SUBMISSION_SERVICE_URL", "grpc://localhost:50051")
        monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "grpc://localhost:50052")
        monkeypatch.setenv("IDENTITY_ACCESS_SERVICE_URL", "grpc://localhost:50053")
        monkeypatch.setenv("GRPC_TLS_ENABLED", "true")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.grpc_tls_enabled is True

    def test_grpc_tls_cert_path_optional(self, monkeypatch: MonkeyPatch) -> None:
        """gRPC TLS cert path should be optional."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4")
        monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com")
        monkeypatch.setenv("SUBMISSION_SERVICE_URL", "grpc://localhost:50051")
        monkeypatch.setenv("KNOWLEDGE_SERVICE_URL", "grpc://localhost:50052")
        monkeypatch.setenv("IDENTITY_ACCESS_SERVICE_URL", "grpc://localhost:50053")

        settings = Settings()  # type: ignore[call-arg]
        assert settings.grpc_tls_cert_path is None
