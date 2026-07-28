# LLM Gateway

Production-grade API gateway for LLM providers with multi-tenant rate limiting, automatic fallback, and observability.

## Features

- **Multi-Provider Support** — OpenAI, Anthropic (extensible to others)
- **Per-Team Rate Limiting** — Redis-backed token bucket algorithm
- **Budget Enforcement** — Per-team monthly spending limits
- **Automatic Fallback** — Proactive (health checks) + reactive (on failure)
- **Circuit Breaker** — Prevent cascading failures (CLOSED → OPEN → HALF_OPEN)
- **Observability** — Structured logging + Prometheus metrics
- **Streaming** — SSE support for real-time responses

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         LLM Gateway                              │
├─────────────────────────────────────────────────────────────────┤
│  Request → Auth → Model Router → Budget → Rate Limit → Provider │
│                        ↓                                         │
│              Health Check (proactive fallback)                   │
│              Circuit Breaker (reactive fallback)                 │
├─────────────────────────────────────────────────────────────────┤
│  Redis: rate limits, budgets    │  Prometheus: /metrics          │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### With Docker (recommended)

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env with your API keys

# 2. Start everything
docker-compose up -d

# 3. Test
curl http://localhost:8000/health
```

### Local Development

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Start Redis
docker run -d -p 6379:6379 redis:7-alpine

# 3. Configure
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Run
uvicorn llm_gateway.main:app --reload
```

## API Usage

### Create Completion

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "X-API-Key: team-alpha-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Streaming

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "X-API-Key: team-alpha-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## Configuration

Teams, models, and providers are configured in `config/gateway.yaml`:

```yaml
teams:
  team-alpha:
    api_key: "team-alpha-key"
    rate_limit:
      requests_per_minute: 100
    budget:
      monthly_limit_usd: 500.0

models:
  gpt-4o:
    provider: openai
    model_id: gpt-4o-2024-08-06
    fallback: claude-sonnet
  claude-sonnet:
    provider: anthropic
    model_id: claude-3-sonnet-20240229
```

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/completions` | Create completion (streaming supported) |
| `GET /health` | Health check with provider status |
| `POST /health/check` | Trigger immediate health check |
| `GET /metrics` | Prometheus metrics |

## Observability

### Prometheus Metrics

- `llm_gateway_requests_total` — Request count by team/model/status
- `llm_gateway_request_latency_seconds` — Latency histogram
- `llm_gateway_tokens_input_total` — Input token count
- `llm_gateway_tokens_output_total` — Output token count
- `llm_gateway_cost_usd_total` — Cost tracking
- `llm_gateway_provider_healthy` — Provider health gauge
- `llm_gateway_circuit_state` — Circuit breaker state

### Grafana

Access at http://localhost:3000 (admin/admin) when using docker-compose.

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=llm_gateway
```

## Project Structure

```
llm-gateway/
├── src/llm_gateway/
│   ├── main.py                 # FastAPI app
│   ├── api/
│   │   └── completions.py      # /v1/completions endpoint
│   ├── core/
│   │   ├── auth.py             # API key authentication
│   │   ├── budget.py           # Spending tracking
│   │   ├── circuit_breaker.py  # Circuit breaker pattern
│   │   ├── config.py           # Configuration loading
│   │   ├── health.py           # Background health checks
│   │   ├── logging.py          # Structured logging
│   │   ├── metrics.py          # Prometheus metrics
│   │   ├── rate_limit.py       # Token bucket rate limiting
│   │   └── router.py           # Model → provider routing
│   ├── models/
│   │   └── llm.py              # Request/response models
│   └── providers/
│       ├── base.py             # Provider protocol
│       ├── openai.py           # OpenAI implementation
│       └── anthropic.py        # Anthropic implementation
├── config/
│   └── gateway.yaml            # Teams, models, providers
├── tests/
├── Dockerfile
└── docker-compose.yml
```

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Provider interface | Protocol (structural typing) | No inheritance, easy to extend |
| Rate limiting | Token bucket + Redis Lua | Atomic, allows bursts |
| Health checking | Background polling | No request latency impact |
| Circuit breaker | 3-state pattern | Graceful recovery testing |
| Config | YAML + env overrides | Readable, secrets via env |

## License

MIT
