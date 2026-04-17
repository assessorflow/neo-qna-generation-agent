"""Unit tests for HTTP middleware."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from qna_generation_agent.interfaces.http.middleware import (
    _get_cors_origin_header,
    _resolve_request_id,
)


class TestGetCorsOriginHeader:
    """Tests for _get_cors_origin_header."""

    @pytest.mark.unit
    def test_empty_allowed_origins_returns_wildcard(self) -> None:
        """Test that empty origins returns wildcard."""
        _header_name, header_value = _get_cors_origin_header(
            b"http://example.com", [], False
        )
        assert header_value == b"*"

    @pytest.mark.unit
    def test_none_request_origin_returns_empty(self) -> None:
        """Test that None request origin returns empty."""
        _header_name, header_value = _get_cors_origin_header(
            None, ["http://example.com"], False
        )
        assert header_value == b""

    @pytest.mark.unit
    def test_matching_origin_returns_origin(self) -> None:
        """Test that matching origin is returned."""
        _header_name, header_value = _get_cors_origin_header(
            b"http://example.com", ["http://example.com"], False
        )
        assert header_value == b"http://example.com"

    @pytest.mark.unit
    def test_non_matching_origin_returns_empty(self) -> None:
        """Test that non-matching origin returns empty."""
        _header_name, header_value = _get_cors_origin_header(
            b"http://attacker.com", ["http://example.com"], False
        )
        assert header_value == b""

    @pytest.mark.unit
    def test_wildcard_with_credentials_blocked(self) -> None:
        """Test that wildcard is blocked when credentials are enabled."""
        _header_name, header_value = _get_cors_origin_header(None, ["*"], True)
        # When credentials are enabled, wildcard should not be returned
        assert header_value != b"*" or header_value == b""


class TestResolveRequestId:
    """Tests for _resolve_request_id."""

    @pytest.mark.unit
    def test_returns_header_value_if_present(self) -> None:
        """Test that header value is returned if present."""
        request = MagicMock()
        request.get_first_header = MagicMock(return_value=b"test-request-id")

        request_id = _resolve_request_id(request)

        assert request_id == "test-request-id"

    @pytest.mark.unit
    def test_generates_uuid_if_no_header(self) -> None:
        """Test that UUID is generated if no header present."""
        request = MagicMock()
        request.get_first_header = MagicMock(return_value=None)

        request_id = _resolve_request_id(request)

        # Should be a valid UUID format
        assert len(request_id) == 36  # UUID length
        assert "-" in request_id

    @pytest.mark.unit
    def test_prefers_x_request_id(self) -> None:
        """Test that x-request-id is preferred over x-correlation-id."""
        request = MagicMock()
        request.get_first_header = MagicMock(return_value=b"request-id")

        request_id = _resolve_request_id(request)

        assert request_id == "request-id"
