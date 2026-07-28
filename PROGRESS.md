# LLM Gateway — Progress Tracker

## Current Status
**Phase:** 9 In Progress (Polish)
**Last completed:** Step 28 — Ollama provider (local models)
**Next up:** Step 29 — Integration tests
**Status:** Core gateway complete. Step 27 (Output Guardrails / PII) dropped — see note.
**Note:** Step 27 was built then reverted per decision; not included.

---

## Completed Steps

### Phase 0: Setup
- [x] Step 0: Created docs/PLAN.md and PROGRESS.md
- [x] Step 1: FastAPI app skeleton + project structure

### Phase 1: Core Proxy MVP
- [x] Step 2: Provider abstraction interface (Protocol class)
- [x] Step 3: OpenAI provider implementation (non-streaming)
- [x] Step 4: Basic routing — POST /v1/completions endpoint
- [x] Step 5: Streaming support with SSE

### Phase 2: Auth + Config + Rate Limiting
- [x] Step 6: YAML config schema + loader (Pydantic validation)
- [x] Step 7: Team auth middleware (API key → team context)
- [x] Step 8: Redis setup + connection management
- [x] Step 9: Token bucket rate limiting (Lua script)
- [x] Step 10: Budget tracking (spending accumulator)

### Phase 3: Multi-Provider + Routing
- [x] Step 11: Anthropic provider implementation
- [x] Step 12: Model-based provider routing
- [x] Step 13: Health checking (background monitoring)
- [x] Step 14: Fallback routing (reactive retry on failure)
- [x] Step 15: Circuit breaker (prevent hammering failed providers)

### Phase 4: Observability
- [x] Step 16: Request logging (structured JSON logs)
- [x] Step 17: Prometheus metrics (/metrics endpoint)

### Phase 5: Deployment & Testing
- [x] Step 18: Docker + docker-compose (gateway, Redis, Prometheus, Grafana)
- [x] Step 19: Unit tests (circuit breaker, health checker, router)
- [x] Step 20: README documentation

### Phase 6: Alerting
- [x] Step 21: Prometheus alert rules + Alertmanager config
- [ ] Step 22: Slack webhook integration (TODO)

### Phase 7: Cost Optimization
- [x] Step 24: Prompt Compression (pluggable Compressor Protocol + rule-based impl)
- [x] Step 25: Cost Analytics Dashboard (provisioned Grafana dashboard, 10 panels)

### Phase 8: Security Guardrails
- [x] Step 26: Input Guardrails (prompt injection detection, block/flag modes)
- [~] Step 27: Output Guardrails (PII) — built then dropped per decision

### Phase 9: Polish
- [x] Step 28: Ollama provider (local models, no API key, NDJSON streaming)
- [ ] Step 29: Integration tests
- [ ] Step 30: Load testing

---

## Key Decisions Log

| Decision | Choice | Why | Alternatives |
|----------|--------|-----|--------------|
| Planning docs | PLAN.md + PROGRESS.md | Separate static plan from living status | Single file |
| Python version | 3.9+ | Available on system | Require 3.11+ |
| Provider interface | Protocol | Structural typing, no inheritance | ABC |
| HTTP client | httpx | Async-native, consistent across providers | SDK per provider |
| Config format | YAML + env overrides | Comments, secrets via env | JSON, TOML |
| Rate limiting | Token bucket + Redis Lua | Atomic, allows bursts | Fixed window |
| Budget storage | Redis (in-memory) | Fast, simple | PostgreSQL (durable) |
| Model routing | Config-based mapping | Aliases, versioning, fallback chains | Pass-through |
| Health checking | Background polling (30s) | Proactive routing, no request latency | Check-on-request |
| Fallback routing | Proactive + reactive | Cover both pre-known and sudden failures | Reactive only |
| Circuit breaker | 3-state (closed/open/half-open) | Graceful recovery testing | Binary on/off |
| Prompt compression | Pluggable Protocol + rule-based | Seam for LLMLingua later; ship safe whitespace trim now | Model-based (LLMLingua) upfront |
| Input guardrails | Regex patterns + block/flag modes | Cheap first line; shadow-test before enforcing; Protocol seam for ML | ML classifier upfront, no shadow mode |

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Gateway                          │
├─────────────────────────────────────────────────────────────────┤
│  Request                                                         │
│     ↓                                                            │
│  Auth (API key → Team)                                          │
│     ↓                                                            │
│  Input Guardrails (prompt injection detection) ← TODO Phase 8    │
│     ↓                                                            │
│  Model Access Check                                              │
│     ↓                                                            │
│  Prompt Compression (token reduction) ← TODO Phase 7             │
│     ↓                                                            │
│  Budget Check (Redis)                                            │
│     ↓                                                            │
│  Rate Limit Check (Redis token bucket)                          │
│     ↓                                                            │
│  Provider (OpenAI or Anthropic)                                 │
│     ↓                                                            │
│  Output Guardrails (PII detection) ← TODO Phase 8                │
│     ↓                                                            │
│  Response → Track Spending → Return to Client                   │
├─────────────────────────────────────────────────────────────────┤
│  Redis: rate limits, budgets    │  Config: teams, models, limits │
├─────────────────────────────────────────────────────────────────┤
│  Background: Health Checker (pings providers every 30s)          │
│  Grafana: Cost Analytics Dashboard ← TODO Phase 7               │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure
```
llm-gateway/
├── src/llm_gateway/
│   ├── main.py
│   ├── api/
│   │   └── completions.py
│   ├── core/
│   │   ├── config.py, loader.py, schemas.py
│   │   ├── auth.py, teams.py
│   │   ├── providers.py, router.py, health.py, circuit_breaker.py
│   │   ├── logging.py, metrics.py
│   │   ├── redis.py, rate_limit.py, budget.py
│   ├── models/
│   │   └── llm.py
│   └── providers/
│       ├── base.py, exceptions.py
│       ├── openai.py
│       └── anthropic.py                  # NEW
├── config/gateway.yaml
├── docs/PLAN.md
└── pyproject.toml
```

---

## What Works Now

1. **Auth**: API key → team lookup, model access control
2. **Rate limiting**: Per-team RPM with token bucket algorithm
3. **Budget tracking**: Per-team monthly spending, 402 on exceed
4. **OpenAI provider**: Non-streaming and streaming
5. **Anthropic provider**: Non-streaming and streaming
6. **Model routing**: Route by model name to correct provider
7. **Config**: YAML with env var overrides, model→provider mapping
8. **Health checking**: Background task monitors provider availability every 30s
9. **Fallback routing**: Auto-switch when primary fails (proactive + reactive)
10. **Circuit breaker**: Skip failing providers (CLOSED→OPEN→HALF_OPEN→CLOSED)
11. **Request logging**: Structured JSON logs (team, model, latency, tokens, cost)
12. **Prometheus metrics**: /metrics endpoint with counters, histograms, gauges
13. **Docker**: Multi-stage build, docker-compose with Redis/Prometheus/Grafana
14. **Unit tests**: Circuit breaker, health checker, router tests
15. **Alert rules**: Prometheus alerts for errors, budget, latency, circuit breaker

## Next Phase: Cost Optimization

### Phase 7: Cost Optimization
- [ ] Step 24: Prompt Compression (reduce tokens before LLM call)
- [ ] Step 25: Cost Analytics Dashboard (Grafana panels)

### Phase 8: Security Guardrails
- [ ] Step 26: Input Guardrails (prompt injection detection)
- [ ] Step 27: Output Guardrails (PII detection/redaction)

### Phase 9: Polish
- [ ] Step 28: Ollama provider
- [ ] Step 29: Integration tests
- [ ] Step 30: Load testing

## Backlog (Deferred)

- Smart model routing (complexity-based tier selection)
- OpenTelemetry distributed tracing
- PostgreSQL request audit trail
- Admin API for runtime CRUD
- Priority queues / VIP routing
- Semantic caching (GPTCache-style)
- Hot reload config

## TODO

- [ ] **Slack webhook**: Set `SLACK_WEBHOOK_URL` env var to enable alert delivery

---

## Session Notes

### Session 1
- Discussed project spec
- Agreed on build plan and scope
- Deferred: priority queues, hot reload, admin API, Slack alerts

### Session 2
- Completed Steps 1-12
- Both providers working (OpenAI, Anthropic)
- Model routing by config
- Explained: token bucket, Lua scripts, SSE streaming, fallback logic
