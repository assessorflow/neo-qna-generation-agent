# QnA Generation Agent

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)

A production-grade microservice for generating grounded questions and model answers as part of the AssessorFlow platform. This service integrates with Google Cloud Pub/Sub for event-driven workflows, gRPC for inter-service communication, and Langfuse for prompt management and telemetry. Persistence is handled via in-memory repositories (extensible to external stores).

## Architecture

### Overview

```mermaid
flowchart TB
    subgraph "External Systems"
        Orchestrator[Orchestrator Agent]
        Knowledge[Knowledge Service]
        Submission[Submission Service]
        Langfuse[Langfuse Prompt Management]
    end

    subgraph "QnA Generation Agent"
        HTTP[HTTP Interface Granian + BlackSheep]
        PubSub[Pub/Sub Subscriber]
        Core[Core Domain]

        subgraph "Application Layer"
            Commands[Command Handlers]
            Services[GenerateQnAService]
            Ports[Repository Ports]
        end

        subgraph "Infrastructure Layer"
            LLM[LLM Provider Strands/OpenAI]
            PromptProvider[Prompt Provider Langfuse]
            Telemetry[Langfuse Telemetry]
        end
    end

    Orchestrator -->|Trigger Event| PubSub
    PubSub -->|Process| Commands
    Commands -->|Generate| Services
    Services -->|Fetch Prompt| Langfuse
    Services -->|Fetch Context| Knowledge
    Services -->|Publish Results| Submission
    Services -->|Trace| Telemetry
    HTTP -->|Health Checks| Core
```

### Event Flow

```mermaid
sequenceDiagram
    participant O as Orchestrator
    participant PS as Pub/Sub
    participant QNA as QnA Agent
    participant LF as Langfuse
    participant KS as Knowledge Service
    participant SS as Submission Service

    O->>PS: Publish assessorflow.qa-generation.trigger
    PS->>QNA: Deliver message
    QNA->>LF: Fetch Assessment Generator prompt
    LF-->>QNA: Return prompt with variables
    QNA->>KS: gRPC: GetChunksByIds
    KS-->>QNA: Return document chunks
    QNA->>QNA: Generate questions via LLM
    QNA->>SS: gRPC: CreateQuestionSet
    QNA->>PS: Publish completion event
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Runtime | Python 3.13+ | Async-first Python with strict typing |
| HTTP Server | Granian + BlackSheep | High-performance ASGI server |
| Message Bus | Google Cloud Pub/Sub | Event-driven communication |
| Persistence | In-Memory | Question sets, idempotency (extensible to external stores) |
| LLM Integration | Strands Agents | Structured output generation |
| Prompt Management | Langfuse | Versioned prompt storage and retrieval |
| Telemetry | Langfuse | Distributed tracing and observability |
| gRPC | grpcio | Inter-service RPC |
| Serialization | orjson | Fast JSON handling |

## Project Structure

```
src/qna_generation_agent/
├── app/                    # Settings, bootstrap, JSON, logging
│   ├── bootstrap.py        # DI container initialization
│   ├── logging.py          # Structured JSON logging
│   ├── json.py             # orjson wrappers
│   └── settings.py         # Environment configuration
├── application/            # Use cases, DTOs, ports, services
│   ├── commands.py         # Command handlers
│   ├── dto.py              # Data transfer objects
│   ├── services/           # GenerateQnAService
│   └── ports/              # Repository interfaces (LLM, prompts, telemetry)
├── domain/                 # Entities, value objects, enums, events
│   ├── entities.py         # Question, Answer, QuestionSet
│   ├── enums.py            # QuestionType, DifficultyLevel
│   ├── events.py           # Domain events
│   ├── errors.py           # Domain errors
│   └── value_objects.py    # Type-safe IDs
├── infrastructure/         # External adapters
│   ├── llm/                # LLM provider (Strands) + Prompt provider (Langfuse)
│   ├── messaging/          # Pub/Sub publisher/subscriber
│   ├── persistence/        # In-memory repositories
│   ├── grpc/               # gRPC clients
│   └── telemetry/          # Langfuse tracing
└── interfaces/
    ├── http/               # HTTP server endpoints
    └── serve/              # Unified HTTP + Pub/Sub process
```

## Installation

### Prerequisites

- Python 3.13 or higher
- Google Cloud project with Pub/Sub enabled (for production)
- Langfuse account (optional, for prompt management and telemetry)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd neo-qna-generation-agent

# Install dependencies using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# Or using pip
pip install -e ".[dev,test,lint]"
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for LLM |
| `OPENAI_MODEL` | Yes | - | Model ID (e.g., `gpt-4o`) |
| `OPENAI_BASE_URL` | Yes | - | OpenAI API base URL |
| `SUBMISSION_SERVICE_URL` | Yes | - | gRPC URL for submission service |
| `KNOWLEDGE_SERVICE_URL` | Yes | - | gRPC URL for knowledge service |
| `HOST` | No | `0.0.0.0` | HTTP server host |
| `PORT` | No | `8000` | HTTP server port |
| `WORKERS` | No | `1` | **Hardcoded to 1** — HTTP and Pub/Sub run in a single unified process |
| `LOG_LEVEL` | No | `info` | Logging level |
| `PUBSUB_PROJECT_ID` | No | - | GCP project ID |
| `PUBSUB_SUBSCRIPTION_TRIGGER` | No | - | Pub/Sub subscription name |
| `PUBSUB_TOPIC_COMPLETE` | No | - | Pub/Sub topic for completion |
| `LANGFUSE_PUBLIC_KEY` | No | - | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | No | - | Langfuse secret key |
| `LANGFUSE_BASE_URL` | No | `https://cloud.langfuse.com` | Langfuse host |
| `CORS_ALLOWED_ORIGINS` | No | - | Comma-separated allowed origins |
| `GRPC_TLS_ENABLED` | No | `false` | Enable gRPC TLS |
| `GRPC_TLS_CERT_PATH` | No | - | Path to TLS CA certificate |
| `ENABLE_TEST_ROUTES` | No | `false` | Enable `/test/*` debug endpoints |

### Example .env File

```bash
# Required LLM configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com

# Service dependencies
SUBMISSION_SERVICE_URL=grpc://localhost:50051
KNOWLEDGE_SERVICE_URL=grpc://localhost:50052

# Optional: Pub/Sub for event-driven mode
PUBSUB_PROJECT_ID=my-gcp-project
PUBSUB_SUBSCRIPTION_TRIGGER=qna-trigger-sub
PUBSUB_TOPIC_COMPLETE=qna-complete-topic

# Optional: Langfuse for prompts and telemetry
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# Security settings
CORS_ALLOWED_ORIGINS=https://app.example.com
GRPC_TLS_ENABLED=true
```

## Usage

### Running the Server

The service runs as a unified HTTP + Pub/Sub process:

```bash
uv run qna-serve
```

This single process handles both HTTP health checks and Pub/Sub message consumption, ensuring proper lifecycle management and graceful shutdown.

> **Note:** Workers are hardcoded to 1 (enforced in `settings.py` via `ge=1, le=1`). This is intentional — the service uses a unified single-process model where HTTP and Pub/Sub share the same event loop. This prevents concurrency issues with Strands workflow state and in-memory repositories. For scaling, use horizontal pod autoscaling (multiple replicas) rather than increasing workers per pod.

**Graceful Shutdown:** The server responds to SIGTERM and SIGINT signals with a 10-second timeout for in-flight message processing.

### Health Check Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /healthz` | Basic health check |
| `GET /livez` | Kubernetes liveness probe |
| `GET /readyz` | Readiness probe with dependency checks (Pub/Sub, LLM, Prompt Provider) |
| `GET /version` | Application version and environment |
| `GET /docs` | Swagger UI documentation |

### API Examples

```bash
# Health check
curl http://localhost:8000/healthz

# Readiness check (includes Pub/Sub, LLM, Prompt Provider status)
curl http://localhost:8000/readyz

# Version info
curl http://localhost:8000/version
```

Response example:

```json
{
  "version": "0.1.0",
  "environment": "local"
}
```

### Prompt Testing Endpoints (Development)

When `ENABLE_TEST_ROUTES=true` or in non-production environments, three endpoints are available for testing individual Langfuse prompts via Strands workflow execution:

| Endpoint | Method | Description | Default Values |
|----------|--------|-------------|----------------|
| `/test/prompt/assessment` | `POST` | Test Assessment Generator | `structured_count=2`, `non_structured_count=1`, `difficulty=medium`, `topics="Grammar, Vocabulary, Article Usage"` |
| `/test/prompt/mcq-answer` | `POST` | Test MCQ Answer Generator | `question_stem="The student ____ to school yesterday..."`, `grammar_target="past continuous tense"`, `l1_background="Chinese"` |
| `/test/prompt/mcq-explanation` | `POST` | Test MCQ Explanation Generator | `question="Choose the correct article..."`, `options={A,B,C,D}`, `correct_answer="A"` |

These endpoints return:
- `success`: Whether execution succeeded
- `prompt_version`: Which Langfuse prompt version was used
- `execution_time_ms`: Duration of the workflow execution
- `result`: Parsed structured output from the LLM
- `raw_output`: Raw LLM response (for debugging)

**Example:**

```bash
# Test Assessment Generator with defaults
curl -X POST http://localhost:8000/test/prompt/assessment \
  -H "Content-Type: application/json" \
  -d '{}'

# Test with custom values
curl -X POST http://localhost:8000/test/prompt/assessment \
  -H "Content-Type: application/json" \
  -d '{
    "structured_count": 5,
    "non_structured_count": 2,
    "difficulty": "hard",
    "topics": "Academic Writing",
    "chunks": ["Your custom document chunk here..."]
  }'
```

**Requirements:**
- Langfuse must be configured (`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`)
- Returns 503 if Langfuse is not available
- Returns 500 if prompt execution fails

## Prompt Management

The service integrates with Langfuse for centralized prompt management. Prompts are fetched on-demand (with 5-minute SDK caching) ensuring updates take effect without service restart.

### Available Prompts

| Prompt Name | Purpose | Variables |
|-------------|---------|-----------|
| `Assessment Generator` | Generate ELP assessment questions for Singapore foreign students | `{structured_count}`, `{non_structured_count}`, `{difficulty}`, `{topics}`, `{chunks}` |
| `MCQ Explanation Generator` | Generate detailed explanations for MCQ distractors with L1 interference notes | `{question}`, `{options}`, `{correct_answer}`, `{target_audience}` |
| `MCQ Answer Generator` | Generate MCQ answers with L1-targeted distractors | `{question_stem}`, `{grammar_target}`, `{difficulty}`, `{l1_background}` |

## LLM Generation Workflow

The service supports two LLM provider modes, configurable via `LLM_WORKFLOW_MODE`:

### Mode 1: Sequential Provider (Default)
Direct sequential calls to Strands Agents for structured output generation.

### Mode 2: Strands Workflow Provider (`LLM_WORKFLOW_MODE=true`)
A **three-level parallel workflow** using [Strands Agents](https://strandsagents.com/) with the `workflow` tool. This enables efficient parallel generation optimized for ELP assessments.

```mermaid
flowchart TB
    subgraph "Workflow Architecture"
        O[Orchestrator Agent - tools: workflow]

        subgraph "Level 1: Assessment Generation (Parallel by Subtopic)"
            L1A[Task: Subtopic A Prompt: Assessment Generator]
            L1B[Task: Subtopic B Prompt: Assessment Generator]
            L1N[Task: Subtopic N Prompt: Assessment Generator]
        end

        subgraph "Level 2: MCQ Answer Generation (Parallel by Question)"
            L2A[Task: Question 1 Prompt: MCQ Answer Generator]
            L2B[Task: Question 2 Prompt: MCQ Answer Generator]
            L2N[Task: Question N Prompt: MCQ Answer Generator]
        end

        subgraph "Level 3: MCQ Explanation Generation (Parallel by Question)"
            L3A[Task: Question 1 Prompt: MCQ Explanation Generator]
            L3B[Task: Question 2 Prompt: MCQ Explanation Generator]
            L3N[Task: Question N Prompt: MCQ Explanation Generator]
        end

        subgraph "Parallel Track: Non-Structured (Parallel by Subtopic)"
            NSA[Task: Subtopic A Prompt: Assessment Generator]
            NSB[Task: Subtopic B Prompt: Assessment Generator]
        end
    end

    O -->|workflow.create() workflow.start()| L1A
    O --> L1B
    O --> L1N

    L1A -->|Output: QuestionStem| L2A
    L1B --> L2B
    L1N --> L2N

    L2A -->|Output: MCQAnswer| L3A
    L2B --> L3B
    L2N --> L3N

    L3A -->|Output: MCQExplanation| Agg[Final Aggregation]
    L3B --> Agg
    L3N --> Agg

    O -.-> NSA
    O -.-> NSB
    NSA -->|Output: Non-Structured Questions| Agg
    NSB --> Agg
```

#### Workflow Levels

| Level | Purpose | Parallelism | Output |
|-------|---------|-------------|--------|
| **Level 1** | Generate question stems from source chunks | By subtopic | `QuestionStem[]` |
| **Level 2** | Generate MCQ answers with L1-targeted distractors | By question | `MCQAnswer[]` |
| **Level 3** | Generate detailed explanations with CEFR analysis | By question | `MCQExplanation[]` |
| **Parallel Track** | Generate open-ended questions with rubrics | By subtopic | `QuestionDraft[]` |

Each level is a **separate workflow** created via:
```python
workflow(action="create", workflow_id="...", tasks=[...])
workflow(action="start", workflow_id="...")
workflow(action="status", workflow_id="...")  # → Aggregated results
```

### Using Prompts in Code

```python
from qna_generation_agent.infrastructure.llm.langfuse_prompt_provider import (
    LangfusePromptProvider,
)
from qna_generation_agent.infrastructure.llm.prompt_builder import (
    AssessmentGeneratorInputSchema,
    AssessmentGeneratorOutputSchema,
)

# Fetch prompt from Langfuse
provider = LangfusePromptProvider(...)
prompt = await provider.get_assessment_generator_prompt(label="production")

# Compile with validated inputs
input_data = AssessmentGeneratorInputSchema(
    structured_count=5,
    non_structured_count=3,
    difficulty="medium",
    topics="Grammar, Vocabulary",
    chunks="[Chunk 1] ...",
)
compiled_prompt = prompt.compile(**input_data.model_dump())

# Use with Strands for structured output
result = await agent.invoke_async(
    compiled_prompt,
    structured_output_model=AssessmentGeneratorOutputSchema,
)
```

### Creating Prompts in Langfuse

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Create Assessment Generator prompt
langfuse.create_prompt(
    name="Assessment Generator",
    type="text",
    prompt="""Generate EXACTLY {structured_count} MCQ questions...
    
Respond in JSON format:
{{
    "questions": [...]
}}""",
    labels=["production"],
)
```

## Development

### Code Quality

```bash
# Run all quality checks
uv run ruff check .
uv run ruff check . --fix
uv run mypy .

# Run tests
uv run pytest
uv run pytest -m unit
uv run pytest -m integration

# Run with coverage
uv run pytest --cov=qna_generation_agent --cov-report=html
```

### Code Standards

- **Line length:** 88 characters (Black-compatible)
- **Type checking:** Strict mypy with `disallow_untyped_defs`
- **Linting:** Ruff with E, F, W, I, UP, B, C4, ASYNC, RUF rules
- **Logging:** Structured JSON via structlog
- **Testing:** pytest with 80% coverage minimum

## Security Considerations

### CORS Configuration

By default, CORS is unrestricted in development. For production:

```bash
CORS_ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
CORS_ALLOW_CREDENTIALS=true
```

- Wildcard (`*`) origins are blocked when credentials are enabled
- Empty configuration allows all origins (development mode only)

### gRPC TLS

Enable TLS for production gRPC connections:

```bash
GRPC_TLS_ENABLED=true
GRPC_TLS_CERT_PATH=/path/to/ca-cert.pem  # Optional: custom CA
```

### Input Validation

- All domain entities validate inputs in `__post_init__`
- API schemas use Pydantic for runtime validation
- Pub/Sub messages are validated against `TriggerEnvelope` schema

### Secrets Management

- Never commit `.env` files
- Use secret management in production (Google Secret Manager, etc.)
- API keys are validated at startup; missing keys raise `ConfigurationError`

## Deployment

### Docker

```dockerfile
FROM python:3.13-slim

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync

COPY src/ ./src/
EXPOSE 8000

CMD ["uv", "run", "qna-serve"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neo-qna-generation-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neo-qna-generation-agent
  template:
    metadata:
      labels:
        app: neo-qna-generation-agent
    spec:
      containers:
        - name: app
          image: neo-qna-generation-agent:latest
          ports:
            - containerPort: 8000
          envFrom:
            - secretRef:
                name: qna-secrets
            - configMapRef:
                name: qna-config
          livenessProbe:
            httpGet:
              path: /livez
              port: 8000
          readinessProbe:
            httpGet:
              path: /readyz
              port: 8000
          lifecycle:
            preStop:
              exec:
                command: ["/bin/sh", "-c", "sleep 15"]
```

**Important:** The service is designed to run with a single worker per pod (`workers=1`). Do not override this — the unified process model requires single-threaded operation. Scale horizontally via replicas, not vertical workers.

**Note:** The `preStop` hook gives the service 15 seconds to gracefully shutdown before Kubernetes sends SIGKILL.

### Horizontal Pod Autoscaling

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neo-qna-generation-agent
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neo-qna-generation-agent
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

## Troubleshooting

### Common Issues

#### Pub/Sub Authentication Errors

```
Error: Failed to authenticate with Pub/Sub
```

**Solution:**
- Ensure GCP credentials are configured
- Check service account permissions
- Verify `PUBSUB_PROJECT_ID` is correct

#### LLM Rate Limiting

```
Error: Rate limit exceeded
```

**Solution:**
- Implement exponential backoff in caller
- Consider request batching
- Monitor rate limits in Langfuse traces

#### Prompt Provider Not Available

```
readyz check fails: prompt_provider_available: false
```

**Solution:**
- Verify `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
- Check Langfuse host connectivity
- Ensure prompts exist in Langfuse project

#### Service Won't Shut Down

```
Process hangs on SIGTERM
```

**Solution:**
- The service has `respawn_failed_workers=False` configured — this prevents Granian from restarting workers on exit
- Check for long-running message processing in logs
- Pub/Sub subscriber has a 10-second timeout for in-flight messages

### Logging

Structured JSON logs are emitted to stdout. Key log fields:

- `event`: Log event type
- `level`: Log level
- `timestamp`: ISO 8601 timestamp
- `request_id`: Correlation ID for tracing
- `error`: Error details (when applicable)

Example query in Cloud Logging:

```
jsonPayload.event="message_processed"
resource.labels.container_name="neo-qna-generation-agent"
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make changes with tests
4. Run quality checks: `uv run ruff check . && uv run mypy .`
5. Commit with clear messages
6. Push and create a PR

### Development Setup

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run all checks before committing
uv run ruff check . && uv run mypy . && uv run pytest
```

## Related Documentation

- [CLAUDE.md](CLAUDE.md) - Architecture and development guide
- [API Documentation](http://localhost:8000/docs) - Swagger UI (when running)
- [Langfuse Documentation](https://langfuse.com/docs) - Prompt management and telemetry
