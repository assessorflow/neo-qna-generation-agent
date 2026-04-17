"""Application-wide constants for timeouts, limits, and configuration values.

This module centralizes magic numbers to improve maintainability and make
tuning parameters discoverable.
"""

from __future__ import annotations

# gRPC client settings
GRPC_DEFAULT_TIMEOUT_SECONDS = 30.0
GRPC_RETRYABLE_STATUS_CODES = frozenset(
    {
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "RESOURCE_EXHAUSTED",
    }
)
GRPC_RETRY_AFTER_SECONDS = 5

# LLM provider settings
LLM_DEFAULT_TIMEOUT_SECONDS = 120
LLM_RETRY_AFTER_TRANSIENT_SECONDS = 10
LLM_RETRY_AFTER_RATE_LIMIT_SECONDS = 30

# HTTP server settings
HTTP_DEFAULT_HOST = "0.0.0.0"
HTTP_DEFAULT_PORT = 8000
HTTP_DEFAULT_WORKERS = 1

# Pub/Sub settings
PUBSUB_DEFAULT_MAX_WORKERS = 10
PUBSUB_SHUTDOWN_TIMEOUT_SECONDS = 5.0
PUBSUB_FLOW_CONTROL_MAX_MESSAGES = 10

# CORS settings
CORS_DEFAULT_METHODS = "GET, POST, OPTIONS"
CORS_DEFAULT_HEADERS = (
    "content-type, accept, authorization, x-request-id, x-correlation-id"
)

# Domain constraints
MAX_QUESTION_TEXT_LENGTH = 10000
MAX_ANSWER_TEXT_LENGTH = 10000
MAX_EXPLANATION_LENGTH = 5000
MAX_REFERENCES_COUNT = 20
MAX_METADATA_ITEMS = 50
MAX_METADATA_KEY_LENGTH = 100
MAX_METADATA_VALUE_LENGTH = 1000

# Retry and circuit breaker settings
MAX_RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0
RETRY_MAX_DELAY_SECONDS = 60.0
RETRY_EXPONENTIAL_BASE = 2.0

# Telemetry settings
TELEMETRY_DEFAULT_BATCH_SIZE = 100
TELEMETRY_FLUSH_INTERVAL_SECONDS = 5.0
TELEMETRY_MAX_QUEUE_SIZE = 1000

# Service identity
SERVICE_NAME = "qna-generation-agent"
SERVICE_VERSION_HEADER = "x-service-version"
