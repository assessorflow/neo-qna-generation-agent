# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Skills and Tools

When working with this codebase, use these resources for authoritative guidance:

- **Strands MCP** — Use the local Strands MCP server for Strands Agents framework questions, SDK usage, `Agent`/`OpenAIModel` patterns, structured output schemas, or workflow tool documentation
- **`/langfuse`** — Use for Langfuse prompt management, telemetry, `@observe` decorator usage, tracing patterns, or prompt fetching via the Langfuse SDK

## Project Overview

This is the **Question and Answer Generation Agent** — one component of the larger AssessorFlow platform, a multi-agent AI system for automating knowledge assessment lifecycles. This specific microservice generates questions and model answers based on classified topics and retrieved document chunks.

**Current State:** This repository contains:
- HTTP health server using Granian (ASGI interface)
- Pub/Sub worker entry point with command handler wiring
- Domain models and application service structure
- Full QnA generation logic (LLM integration, gRPC calls)
- In-memory persistence for question sets and idempotency
- **Langfuse prompt management** — centralized prompts with version control
- **Langfuse telemetry** — distributed tracing and observability
- **Structured output schemas** — type-safe LLM responses
- **Sequential 3-prompt workflow** — Assessment Generator → MCQ Answer Generator → MCQ Explanation Generator
- **Server-side iteration increment** via gRPC for regeneration flow
- **Graceful shutdown** — SIGTERM/SIGINT handling with 10s timeout
- **Runtime state management** — lifespan-based health tracking

## Architecture

### Tech Stack
- **Python 3.13+** with strict typing (mypy enforces `disallow_untyped_defs`)
- **Granian>=2.7** — ASGI server with `respawn_failed_workers=False` for graceful shutdown
- **BlackSheep>=2.6** — ASGI framework with lifespan hooks
- **orjson>=3.11** — JSON serialization (returns bytes, no decode needed)
- **structlog>=25.5** — Structured JSON logging with `get_logger(__name__)`
- **pydantic>=2.13 + pydantic-settings>=2.13** — Environment configuration with `ConfigDict(strict=True)`
- **Langfuse>=4.2** — Prompt management (`@observe` decorator) and telemetry. Use `/langfuse` skill for SDK questions
- **Strands>=1.35** — LLM agent framework (Agent initialized once and reused). Consult Strands MCP for framework questions
- **google-cloud-pubsub>=2.37** — Async Pub/Sub with `SubscriberClient` and `StreamingPullFuture`
- **grpcio>=1.80** — Async gRPC with `grpc.aio` channels (shared, created at startup)

### Project Structure
```
src/qna_generation_agent/
├── app/                    # Settings, bootstrap, JSON, logging, lifespan
│   ├── bootstrap.py        # DI container initialization
│   ├── lifespan.py         # Runtime state, health checks, lifespan manager
│   ├── logging.py          # structlog configuration
│   ├── json.py             # orjson wrappers
│   ├── settings.py         # pydantic-settings
│   └── constants.py        # Service constants
├── application/            # Use cases, DTOs, ports, services
│   ├── commands.py         # Command handlers
│   ├── dto.py              # Data transfer objects
│   ├── errors.py           # Application errors
│   ├── services/           # Application services
│   │   ├── generate_qna.py       # Main generation workflow
│   │   └── prompt_test_service.py # Prompt testing endpoints
│   └── ports/              # Repository interfaces
│       ├── llm.py              # LLM provider port
│       ├── prompt_provider.py  # Prompt provider port
│       ├── telemetry.py        # Telemetry port
│       ├── idempotency.py    # Idempotency store
│       ├── repository.py     # Question set repository
│       ├── publisher.py      # Event publisher
│       ├── knowledge_client.py # Knowledge service client
│       └── submission_client.py # Submission service client
├── domain/                 # Entities, value objects, enums, events
│   ├── entities.py         # Question, Answer, QuestionSet, GenerationRequest
│   ├── enums.py            # QuestionType, DifficultyLevel, Purpose, GenerationStatus
│   ├── events.py           # QnAGenerationTriggered, QnAGenerationCompleted
│   ├── errors.py           # Domain errors
│   └── value_objects.py    # Type-safe IDs
├── infrastructure/         # External adapters
│   ├── llm/                # LLM provider + Prompt provider
│   │   ├── strands_provider.py          # Strands LLM implementation
│   │   ├── langfuse_prompt_provider.py  # Langfuse prompt fetching
│   │   └── prompt_builder.py            # Input/output schemas
│   ├── messaging/          # Pub/Sub publisher/subscriber
│   ├── persistence/        # In-memory repositories
│   ├── grpc/               # gRPC clients
│   └── telemetry/          # Langfuse tracing
└── interfaces/
    ├── http/               # HTTP server
    │   ├── factory.py      # BlackSheep app factory with lifespan
    │   ├── routes_health.py # Health check endpoints
    │   ├── routes_prompts.py # Prompt testing endpoints (development)
    │   ├── middleware.py   # Error handling, logging, correlation, CORS
    │   └── schemas.py      # Pydantic request/response models
    └── serve/              # Unified HTTP + Pub/Sub process
        ├── __main__.py     # Serve entry point
        └── app.py          # Granian ASGI target
```

### Transport Layers
The service is designed with three interface types:

1. **HTTP** (`interfaces/http/`) — Health checks (`/healthz`, `/readyz`, `/livez`, `/version`) and prompt testing (`/test/prompt/*`)
2. **Pub/Sub** — Async event consumption from Google Cloud Pub/Sub (triggered by Orchestrator)
3. **gRPC** — Synchronous RPC for Knowledge Service and Assessment Submission Service calls

### Runtime Mode

The service runs as a unified HTTP + Pub/Sub process:

```bash
uv run qna-serve
```

This single process handles:
- **HTTP** — Health checks and prompt testing (when enabled)
- **Pub/Sub** — Async event consumption with graceful shutdown
- **gRPC** — Synchronous RPC for inter-service communication

**Graceful Shutdown Behavior:**
- SIGTERM/SIGINT triggers 10-second graceful shutdown
- Pub/Sub subscriber stops accepting new messages immediately
- In-flight messages have 10 seconds to complete before forced termination
- Granian `respawn_failed_workers=False` prevents automatic restart on exit

### Lifespan and Runtime State

The `lifespan.py` module manages the complete application lifecycle:

```python
@asynccontextmanager
async def lifespan(container: ApplicationContainer) -> AsyncIterator[RuntimeState]:
    # 1. Initialize runtime state
    # 2. Start container (subscriber + health checks)
    # 3. Yield RuntimeState for app runtime
    # 4. Graceful shutdown on exit
```

**RuntimeState** tracks:
- `lifecycle` — INITIALIZING, RUNNING, SHUTTING_DOWN, etc.
- `subscriber_running` — Whether Pub/Sub subscriber is active
- `llm_healthy`, `knowledge_service_healthy`, `submission_service_healthy`
- `telemetry_healthy`, `prompt_provider_healthy`

Health checks run at startup and inform `/readyz` responses.

### Dependency Injection
The `bootstrap.py` module implements a pure-DI container pattern:

```python
container = bootstrap_serve(settings)
# Container holds all adapters: gRPC clients, LLM provider, Prompt provider, etc.
```

Configuration determines which adapters are created:
- **In-memory repositories** — Always used for QuestionSet and idempotency
- **No Pub/Sub config** → Null event publisher
- **No Langfuse keys** → No-op telemetry and no prompt provider
- **Langfuse keys present** → LangfuseTelemetry + LangfusePromptProvider
- **ENABLE_TEST_ROUTES=true + Langfuse** → PromptTestService available

### Event Flow

1. Orchestrator publishes trigger event to Pub/Sub
2. Worker receives event → `HandleGenerationTrigger` command
3. `GenerateQnAService`:
   - Fetches prompt from Langfuse (e.g., "Assessment Generator")
   - Calls Knowledge Service (gRPC) to get topics and chunks
   - Executes **sequential 3-prompt workflow** for structured questions
   - Persists question set via repository
   - Publishes completion event

### Sequential 3-Prompt Workflow

**Consult Strands MCP for Strands Agents framework questions (Agent, OpenAIModel, structured output patterns).**

For structured (MCQ) questions, the service uses a sequential 3-prompt workflow:

```python
# Step 1: Assessment Generator → Question stems
prompt = await prompt_provider.get_prompt("Assessment Generator")
result = await llm.invoke_with_schema(prompt, AssessmentGeneratorOutputSchema)

# Step 2: MCQ Answer Generator → Answers with L1-targeted distractors
for question in questions:
    answer_prompt = await prompt_provider.get_prompt("MCQ Answer Generator")
    answer_result = await llm.invoke_with_schema(answer_prompt, MCQAnswerGeneratorOutputSchema)

# Step 3: MCQ Explanation Generator → Detailed explanations
    explanation_prompt = await prompt_provider.get_prompt("MCQ Explanation Generator")
    explanation_result = await llm.invoke_with_schema(explanation_prompt, MCQExplanationOutputSchema)
```

**Note:** The previous `WorkflowLLMProvider` was removed due to production-safety issues (timing-dependent parsing, broken ID aggregation, brittle exception handling). The sequential workflow is more predictable and debuggable.

### Cost-Optimized Model Routing

The 3-prompt workflow uses tiered model routing to optimize costs while maintaining quality:

| Step | Prompt | Model Tier | Purpose |
|------|--------|------------|---------|
| 1 | Assessment Generator | **EXPENSIVE** | Creative question stems requiring high-quality reasoning |
| 2 | MCQ Answer Generator | **CHEAP** | Structured distractor generation |
| 3 | MCQ Explanation Generator | **CHEAP** | Structured explanation generation |

Configure via environment variables:
- `EXPENSIVE_MODEL_ID` — High-quality model (defaults to `OPENAI_MODEL`)
- `CHEAP_MODEL_ID` — Cost-optimized model (defaults to `OPENAI_MODEL`)

When `CHEAP_MODEL_ID` is not configured, all stages use the expensive model for backwards compatibility.

### Prompt Management Architecture

**Use `/langfuse` skill for prompt management SDK questions.**

```python
# Three-tier prompt system:
# 1. Port (interface)       → application/ports/prompt_provider.py
# 2. Provider (adapter)     → infrastructure/llm/langfuse_prompt_provider.py
# 3. Input/Output schemas   → infrastructure/llm/prompt_builder.py

# Available prompts:
ASSESSMENT_GENERATOR = "Assessment Generator"
MCQ_EXPLANATION_GENERATOR = "MCQ Explanation Generator"
MCQ_ANSWER_GENERATOR = "MCQ Answer Generator"

# Usage:
prompt = await container.prompt_provider.get_assessment_generator_prompt(label="production")
compiled = prompt.compile(structured_count=5, non_structured_count=3, ...)
result = await agent.invoke_async(compiled, structured_output_model=AssessmentGeneratorOutputSchema)
```

**Prompt Caching:** Langfuse SDK caches prompts locally for 5 minutes. Updates in Langfuse UI take effect within 5 minutes without service restart.

## Configuration

### Environment Variables
Environment variables (loaded from `.env`):
- `HOST` — Server host (default: `0.0.0.0`)
- `PORT` — Server port (default: `8000`)
- `WORKERS` — Worker process count (default: `1`, **enforced to 1**)
- `LOG_LEVEL` — Logging level: `debug`, `info`, `warning`, `error` (default: `info`)
- `PUBSUB_PROJECT_ID`, `PUBSUB_SUBSCRIPTION_TRIGGER`, `PUBSUB_TOPIC_COMPLETE` — Pub/Sub settings
- `PUBSUB_TOPIC_DECISION_AUDIT` — Topic for decision audit events
- `PUBSUB_TOPIC_TOKEN_USAGE` — Topic for token usage events
- `PUBSUB_TOPIC_TRIGGER_DLQ` — Dead letter queue topic for failed triggers
- `PUBSUB_ENABLED` — Enable/disable Pub/Sub (default: `true`)
- `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` — Prompt management AND telemetry
- `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_BASE_URL` — LLM configuration
- `CHEAP_MODEL_ID` — Cost-optimized model for MCQ answer/explanation (defaults to `OPENAI_MODEL`)
- `EXPENSIVE_MODEL_ID` — High-quality model for assessment generation (defaults to `OPENAI_MODEL`)
- `OPENAI_TEMPERATURE` — LLM temperature (default: `0.2`)
- `OPENAI_MAX_OUTPUT_TOKENS` — Max tokens per request (default: `4096`)
- `SUBMISSION_SERVICE_URL` — gRPC URL for Assessment Submission Service
- `KNOWLEDGE_SERVICE_URL` — gRPC URL for Knowledge Service
- `GRPC_TIMEOUT_SECONDS` — gRPC call timeout (default: `30.0`)
- `GRPC_TLS_ENABLED` — Enable TLS for gRPC (default: `false`)
- `GRPC_TLS_CERT_PATH` — Path to custom CA certificate
- `RELEASE` — Release version for observability
- `ENABLE_TEST_ROUTES` — Enable `/test/*` prompt testing endpoints (default: `false`)
- `PROMPT_LABEL` — Default Langfuse prompt label (default: `production`)
- `QA_GEN_MAX_ITERATIONS` — Max regeneration iterations (default: `3`)
- `QA_GEN_TIMEOUT_MS` — Generation timeout in milliseconds (default: `30000`)
- `LLM_TIMEOUT_SECONDS` — LLM call timeout (default: `120`)

### Security Settings
- `CORS_ALLOWED_ORIGINS` — Comma-separated allowed origins (default: empty = no restrictions in dev)
- `CORS_ALLOW_CREDENTIALS` — Enable credentials in CORS (default: `false`)

## Security Guidelines

### CORS Security
The CORS middleware validates the `Origin` header against a configured allowlist:

```python
# Production: specific origins
CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
CORS_ALLOW_CREDENTIALS=true

# Development: no restrictions (default)
CORS_ALLOWED_ORIGINS=
```

**Security rules:**
- Wildcard (`*`) origins are blocked when `CORS_ALLOW_CREDENTIALS=true`
- Empty configuration allows all origins (development mode only)
- Unrecognized origins return empty `Access-Control-Allow-Origin` header

### gRPC TLS
Enable TLS for production gRPC connections to protect data in transit:

```python
# With system default CA roots
GRPC_TLS_ENABLED=true

# With custom CA certificate (for internal PKI)
GRPC_TLS_ENABLED=true
GRPC_TLS_CERT_PATH=/etc/ssl/certs/internal-ca.pem
```

### Secrets Management
- Never commit `.env` files containing secrets
- Use secret management systems in production (Google Secret Manager, AWS Secrets Manager)
- API keys are validated at startup; missing keys raise `ConfigurationError`

### Input Validation
All inputs are validated at boundaries:
- **Domain entities**: `__post_init__` validation for invariants
- **API schemas**: Pydantic validation for HTTP requests
- **Pub/Sub messages**: `TriggerEnvelope.model_validate()` for event validation
- **Prompt inputs**: Pydantic schemas in `prompt_builder.py` validate variables

### Error Handling
Sensitive error details are not exposed to clients:
- HTTP errors return generic messages with `request_id` for correlation
- Detailed error information is logged server-side only
- gRPC error codes are mapped to application error types without exposing internal details

## Development Commands

```bash
# Run unified server (HTTP + Pub/Sub)
uv run qna-serve

# Code quality
uv run ruff check .
uv run ruff check . --fix
uv run mypy .

# Tests
uv run pytest
uv run pytest -m unit
uv run pytest -m integration
uv run pytest --cov=qna_generation_agent --cov-report=html
```

## Code Quality Standards

- **Line length:** 88 characters
- **Ruff rules:** E, F, W, I, UP, B, C4, ASYNC, RUF
- **Typing:** Strict mypy — all functions must have typed arguments and returns
- **Logging:** Always use `get_logger(__name__)` from `app.logging`, never print
- **JSON:** Use `dumps()`/`loads()` from `app.json` (orjson-based)
- **Coverage:** Minimum 80% test coverage required

## HTTP Endpoints

### Operational Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /healthz` | Returns health status based on runtime state |
| `GET /readyz` | Kubernetes readiness probe — checks subscriber, LLM, gRPC services, telemetry |
| `GET /livez` | Kubernetes liveness probe — returns alive unless shutting down |
| `GET /version` | Returns version and environment |
| `GET /docs` | Swagger UI documentation (OpenAPI) |
| `GET /openapi.json` | OpenAPI 3.0 specification |

### Prompt Testing Endpoints (Development)

When `ENABLE_TEST_ROUTES=true` AND Langfuse is configured:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /test/prompt/assessment` | Test Assessment Generator prompt |
| `POST /test/prompt/mcq-answer` | Test MCQ Answer Generator prompt |
| `POST /test/prompt/mcq-explanation` | Test MCQ Explanation Generator prompt |

All prompt testing endpoints return `PromptTestResponse` with:
- `success` — Whether execution succeeded
- `prompt_version` — Langfuse prompt version used
- `execution_time_ms` — Duration of execution
- `result` — Parsed structured output (if successful)
- `error` — Error message (if failed)

## Domain Model

### Entities
- **Question**: text, type, difficulty, answer, topic_id, metadata
- **Answer**: text, explanation, references, confidence_score
- **QuestionSet**: collection of questions with status (pending/in_progress/completed/failed)
- **GenerationRequest**: normalized request with topic_ids, question counts, difficulty

### State Machine
QuestionSet status transitions:
```
PENDING → IN_PROGRESS → COMPLETED
   ↓           ↓
   └──────→ FAILED
```

### Enums
- **QuestionType**: STRUCTURED (MCQ), NON_STRUCTURED (open-ended)
- **DifficultyLevel**: EASY, MEDIUM, HARD
- **Purpose**: ASSESSMENT, PRACTICE, REVIEW
- **GenerationStatus**: PENDING, IN_PROGRESS, COMPLETED, FAILED

## Structured Output Schemas

Located in `infrastructure/llm/prompt_builder.py`:

| Schema | Purpose | Used With Prompt |
|--------|---------|------------------|
| `AssessmentGeneratorOutputSchema` | MCQ + open-ended questions with metadata | Assessment Generator |
| `MCQExplanationOutputSchema` | Distractor explanations with L1 interference notes | MCQ Explanation Generator |
| `MCQAnswerGeneratorOutputSchema` | MCQ answers with targeted distractors | MCQ Answer Generator |

**Input Schemas** for variable validation:
- `AssessmentGeneratorInputSchema` — with `from_context()` factory
- `MCQExplanationInputSchema`
- `MCQAnswerInputSchema`

## Error Handling Patterns

### Application Errors
All domain/infrastructure errors extend `AppError`:
```python
class AppError(Exception):
    """Base for all application errors."""
    message: str
    context: dict[str, Any]
```

### Error Categories
- **PermanentError**: Invalid input, bad configuration — don't retry
- **TransientError**: Network issues, rate limiting — retry with backoff
- **ValidationError**: Schema violations — return 400 to caller
- **StoragePermanentError**: Data not found, constraint violations
- **StorageTransientError**: Connection issues, timeouts — retry
- **LLMPermanentError**: Bad prompt, unsupported model
- **LLMTransientError**: Rate limit, timeout — retry
- **IdempotencyConflict**: Duplicate event detected, return cached result
- **WorkflowEscalationError**: Max regeneration iterations exceeded
- **GenerationError**: Question generation failed
- **RetrievalError**: Failed to retrieve data from external service

## Platform Context

This service operates within a larger event-driven architecture:
- **Orchestrator Agent** — Central router (stateful, LangGraph-based)
- **Knowledge Service** — RAG pipeline (document chunks, embeddings)
- **Classification Agent** — Topic extraction and material sufficiency checks
- **Validator Agent** — Content quality gating (OCR, content safety)
- **Evaluator Agent** — Question validation and participant scoring
- **Reporting Agent** — Feedback generation

This QnA Generation Agent receives `assessorflow.qa-generation.trigger` events from the Orchestrator via Pub/Sub, **fetches versioned prompts from Langfuse**, retrieves topics and chunks from the Knowledge Service via gRPC, generates questions using LLM with structured output schemas, and writes results back to the Assessment Submission Service.

### Regeneration Flow

When validation fails, the service supports regeneration with iteration tracking:

1. Inbound event has `question_set_id`, `validation_result=fail`, `iteration=N`
2. Service calls `IncrementIteration` gRPC to get authoritative iteration count
3. If `iteration > max_iterations`, raises `WorkflowEscalationError`
4. Otherwise, generates new questions with the incremented iteration

## Testing Strategy

### Test Categories
- **Unit tests** (`@pytest.mark.unit`): Fast, no external services, mock adapters
- **Integration tests** (`@pytest.mark.integration`): Exercise real adapters (gRPC)
- **Contract tests** (`@pytest.mark.contract`): Event schema validation
- **Schema tests** (`tests/unit/test_prompt_schemas.py`): Prompt input/output validation

### Test Fixtures
Key fixtures in `conftest.py`:
- `event_loop` — Async test support
- `fake_settings` — Test configuration

### Writing Tests
```python
@pytest.mark.unit
@pytest.mark.asyncio
async def test_service_generates_questions() -> None:
    service = GenerateQnAService(...)
    receipt = await service.execute(command)
    assert receipt.question_count > 0
```

### Testing Prompts
```python
@pytest.mark.unit
def test_prompt_compiles_correctly() -> None:
    prompt = Prompt(...)
    result = prompt.compile(name="Alice", score=100)
    assert "Alice" in result
```

## Performance Considerations

### Throughput
- Pub/Sub subscriber uses flow control (`max_messages=1`) for backpressure
- Single-worker architecture prevents concurrency issues
- LLM calls are the primary bottleneck

### Memory
- Large document chunks are streamed through without buffering
- Question sets are stored in memory (non-persistent)
- Structlog processors are cached on first use

### Latency
- gRPC clients maintain persistent connections
- Correlation ID propagation avoids re-fetching context
- Prompt compilation is local (no network call)

## Common Tasks

### Adding a New Endpoint
1. Add route in `interfaces/http/routes_*.py`
2. Define request/response schema in `schemas.py` if needed
3. Add tests in `tests/integration/test_http_app.py`

### Adding a New Repository
1. Define interface in `application/ports/repository.py`
2. Implement in `infrastructure/persistence/`
3. Add to bootstrap `_build_container()`
4. Add tests in `tests/unit/`

### Adding a New Event Type
1. Define in `domain/events.py` (dataclass)
2. Define envelope in `infrastructure/messaging/envelope_models.py`
3. Add contract tests in `tests/contract/`

### Adding a New Prompt
1. Create prompt in Langfuse UI with variables
2. Add prompt name constant in `LangfusePromptProvider`
3. Add input/output schema in `prompt_builder.py` (if new variable pattern)
4. Add convenience method in `LangfusePromptProvider` (optional)
5. Add unit tests in `tests/unit/test_prompt_schemas.py`

### Using a Prompt in Service
```python
# In GenerateQnAService:
if self._prompt_provider:
    prompt = await self._prompt_provider.get_assessment_generator_prompt(label="production")
    input_data = AssessmentGeneratorInputSchema.from_context(
        context=context,
        structured_count=command.structured_count,
        non_structured_count=command.non_structured_count,
        difficulty=command.difficulty,
    )
    compiled = prompt.compile(**input_data.model_dump())
    result = await self._llm_provider.invoke_with_schema(
        compiled, structured_output_model=AssessmentGeneratorOutputSchema
    )
```

## Troubleshooting

### Build Issues
- Ensure Python 3.13+ is installed
- Use `uv` for dependency management
- Run `uv sync` after pulling changes

### Type Errors
- Run `mypy --show-error-codes src/` for detailed diagnostics
- Check `pyproject.toml` mypy overrides for third-party stubs

### Test Failures
- Verify environment variables are set for bootstrap tests
- Use `pytest -s` to see log output during tests

### Prompt Provider Not Available
```
readyz check: prompt_provider_healthy: false
```
**Solutions:**
- Verify `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
- Check network connectivity to `LANGFUSE_BASE_URL`
- Ensure prompts exist in Langfuse project

### Graceful Shutdown Issues
```
Process hangs on SIGTERM / keeps restarting
```
**Solutions:**
- Verify `respawn_failed_workers=False` is set in Granian config
- Check logs for `subscriber_shutdown_timeout` (in-flight messages > 10s)
- Kubernetes: ensure `preStop` hook allows time for shutdown

## Audit Summary

### Current State (April 2026)

| Metric | Status |
|--------|--------|
| Test Coverage | ~79% (271 tests passing) |
| Type Safety | Strict mypy (no issues) |
| Linting | Ruff passing |
| Architecture | Hexagonal + Clean Architecture |
| Python Version | 3.13+ |

### Code Quality Highlights

- **All 271 tests pass** with coverage near 80% requirement
- **Strict mypy enabled** with `disallow_untyped_defs` - no type errors
- **Modern Python 3.13+** patterns: `StrEnum`, `T: BaseModel` syntax, improved asyncio
- **No legacy imports**: `import json` replaced with `orjson` throughout
- **Proper exception handling**: Specific exceptions with structured logging in boundaries
- **No print statements**: All logging via structlog

### Dependency Versions Verified

| Package | Version | Pattern |
|---------|---------|---------|
| blacksheep | >=2.6 | `Application` with lifespan |
| granian | >=2.7 | ASGI interface with `respawn_failed_workers=False` |
| google-cloud-pubsub | >=2.37 | `SubscriberClient` with `StreamingPullFuture` |
| grpcio | >=1.80 | `grpc.aio` for async channels |
| langfuse | >=4.2 | `@observe` decorator + context managers |
| orjson | >=3.11 | `dumps()` returns bytes (no `.decode()` needed) |
| strands-agents | >=1.35 | `Agent` initialized once and reused |
| structlog | >=25.5 | `get_logger()` with bound contextvars |
| pydantic | >=2.13 | `model_config = ConfigDict(strict=True)` |
| pydantic-settings | >=2.13 | `SettingsConfigDict(env_file=".env")` |

### Design Principles

1. **Hexagonal Architecture**: Domain at center, adapters at edges, ports define boundaries
2. **Dependency Inversion**: High-level modules don't depend on low-level details
3. **Fail Fast**: Validate at boundaries, assert invariants in domain
4. **Observability**: Structured logging, distributed tracing, health checks
5. **Security by Default**: TLS available, CORS configurable, secrets externalized
6. **Prompt as Code**: Versioned prompts in Langfuse, structured schemas in code
7. **Unified Process**: HTTP + Pub/Sub coexist in single process with proper lifespan
8. **Single-Worker Architecture**: Enforced single worker for predictable execution
