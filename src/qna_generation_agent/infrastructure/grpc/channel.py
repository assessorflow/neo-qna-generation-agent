"""Shared gRPC channel creation utilities."""

from __future__ import annotations

import grpc
from grpc.aio import Channel

from qna_generation_agent.app.logging import get_logger

logger = get_logger(__name__)


def create_channel(
    target: str,
    *,
    tls_enabled: bool = False,
    tls_cert_path: str | None = None,
) -> Channel:
    """Create gRPC channel with optional TLS encryption.

    Security: Uses TLS when tls_enabled=True. If tls_cert_path is provided,
    uses custom CA certificate; otherwise uses system default roots.

    Args:
        target: The gRPC target (host:port).
        tls_enabled: Whether to enable TLS encryption.
        tls_cert_path: Optional path to custom CA certificate file.

    Returns:
        A grpc.aio Channel instance.
    """
    if not tls_enabled:
        logger.debug("grpc_insecure_channel", target=target)
        return grpc.aio.insecure_channel(target)

    if tls_cert_path:
        with open(tls_cert_path, "rb") as f:
            credentials = grpc.ssl_channel_credentials(f.read())
        logger.info(
            "grpc_tls_channel_with_custom_ca", target=target, cert_path=tls_cert_path
        )
    else:
        credentials = grpc.ssl_channel_credentials()
        logger.info("grpc_tls_channel_with_system_roots", target=target)

    return grpc.aio.secure_channel(target, credentials)
