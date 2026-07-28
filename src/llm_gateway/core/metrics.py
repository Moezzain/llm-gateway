"""Prometheus metrics for observability.

Exposes metrics at /metrics endpoint for Prometheus scraping.

Metric types:
- Counter: Cumulative count (requests, errors)
- Histogram: Distribution (latency, tokens)
- Gauge: Current value (active requests, circuit state)
"""

from prometheus_client import Counter, Gauge, Histogram

# Request metrics
REQUEST_COUNT = Counter(
    "llm_gateway_requests_total",
    "Total requests by team, model, provider, and status",
    ["team_id", "model", "provider", "status"],
)

REQUEST_LATENCY = Histogram(
    "llm_gateway_request_latency_seconds",
    "Request latency in seconds",
    ["team_id", "model", "provider"],
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# Token metrics
TOKENS_INPUT = Counter(
    "llm_gateway_tokens_input_total",
    "Total input tokens by team and model",
    ["team_id", "model"],
)

TOKENS_OUTPUT = Counter(
    "llm_gateway_tokens_output_total",
    "Total output tokens by team and model",
    ["team_id", "model"],
)

# Cost metrics
COST_USD = Counter(
    "llm_gateway_cost_usd_total",
    "Total cost in USD by team and model",
    ["team_id", "model"],
)

# Compression metrics
COMPRESSION_TOKENS_SAVED = Counter(
    "llm_gateway_compression_tokens_saved_total",
    "Estimated input tokens saved by prompt compression, by team and model",
    ["team_id", "model"],
)

# Guardrail metrics
GUARDRAIL_HITS = Counter(
    "llm_gateway_guardrail_hits_total",
    "Input guardrail rule matches by team, rule, and action (blocked/flagged)",
    ["team_id", "rule", "action"],
)

# Rate limiting metrics
RATE_LIMIT_HITS = Counter(
    "llm_gateway_rate_limit_hits_total",
    "Rate limit rejections by team",
    ["team_id"],
)

BUDGET_EXCEEDED = Counter(
    "llm_gateway_budget_exceeded_total",
    "Budget exceeded rejections by team",
    ["team_id"],
)

# Provider health metrics
PROVIDER_HEALTH = Gauge(
    "llm_gateway_provider_healthy",
    "Provider health status (1=healthy, 0=unhealthy)",
    ["provider"],
)

# Circuit breaker metrics
CIRCUIT_STATE = Gauge(
    "llm_gateway_circuit_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
    ["provider"],
)

CIRCUIT_STATE_MAP = {
    "closed": 0,
    "open": 1,
    "half_open": 2,
}

# Fallback metrics
FALLBACK_COUNT = Counter(
    "llm_gateway_fallback_total",
    "Fallback activations by model and reason",
    ["model", "reason"],  # reason: health_check, request_failed, circuit_open
)

# Active requests (for concurrency monitoring)
ACTIVE_REQUESTS = Gauge(
    "llm_gateway_active_requests",
    "Currently active requests by provider",
    ["provider"],
)


class MetricsRecorder:
    """Helper for recording metrics during request lifecycle.

    Usage:
        metrics = MetricsRecorder(team_id="alpha", model="gpt-4o")
        metrics.set_provider("openai")
        with metrics.track_latency():
            response = await provider.complete(request)
        metrics.record_success(input_tokens=100, output_tokens=50, cost=0.01)
    """

    def __init__(self, team_id: str, model: str) -> None:
        self.team_id = team_id
        self.model = model
        self.provider: str = "unknown"
        self._start_time: float = 0

    def set_provider(self, provider: str) -> None:
        """Set the provider being used."""
        self.provider = provider

    def start_request(self) -> None:
        """Mark request start for latency tracking."""
        import time
        self._start_time = time.time()
        ACTIVE_REQUESTS.labels(provider=self.provider).inc()

    def end_request(self) -> None:
        """Mark request end, record latency."""
        import time
        if self._start_time > 0:
            latency = time.time() - self._start_time
            REQUEST_LATENCY.labels(
                team_id=self.team_id,
                model=self.model,
                provider=self.provider,
            ).observe(latency)
        ACTIVE_REQUESTS.labels(provider=self.provider).dec()

    def record_success(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0,
    ) -> None:
        """Record successful request metrics."""
        self.end_request()

        REQUEST_COUNT.labels(
            team_id=self.team_id,
            model=self.model,
            provider=self.provider,
            status="success",
        ).inc()

        if input_tokens > 0:
            TOKENS_INPUT.labels(
                team_id=self.team_id,
                model=self.model,
            ).inc(input_tokens)

        if output_tokens > 0:
            TOKENS_OUTPUT.labels(
                team_id=self.team_id,
                model=self.model,
            ).inc(output_tokens)

        if cost_usd > 0:
            COST_USD.labels(
                team_id=self.team_id,
                model=self.model,
            ).inc(cost_usd)

    def record_error(self, error_type: str = "error") -> None:
        """Record failed request."""
        self.end_request()

        REQUEST_COUNT.labels(
            team_id=self.team_id,
            model=self.model,
            provider=self.provider,
            status=error_type,
        ).inc()

    def record_rate_limited(self) -> None:
        """Record rate limit hit."""
        REQUEST_COUNT.labels(
            team_id=self.team_id,
            model=self.model,
            provider=self.provider,
            status="rate_limited",
        ).inc()
        RATE_LIMIT_HITS.labels(team_id=self.team_id).inc()

    def record_budget_exceeded(self) -> None:
        """Record budget exceeded."""
        REQUEST_COUNT.labels(
            team_id=self.team_id,
            model=self.model,
            provider=self.provider,
            status="budget_exceeded",
        ).inc()
        BUDGET_EXCEEDED.labels(team_id=self.team_id).inc()

    def record_guardrail(self, rule: str, blocked: bool) -> None:
        """Record an input guardrail match (blocked or just flagged)."""
        GUARDRAIL_HITS.labels(
            team_id=self.team_id,
            rule=rule,
            action="blocked" if blocked else "flagged",
        ).inc()
        if blocked:
            REQUEST_COUNT.labels(
                team_id=self.team_id,
                model=self.model,
                provider=self.provider,
                status="guardrail_blocked",
            ).inc()

    def record_compression(self, tokens_saved: int) -> None:
        """Record estimated input tokens saved by prompt compression."""
        if tokens_saved > 0:
            COMPRESSION_TOKENS_SAVED.labels(
                team_id=self.team_id,
                model=self.model,
            ).inc(tokens_saved)

    def record_fallback(self, reason: str) -> None:
        """Record fallback activation."""
        FALLBACK_COUNT.labels(
            model=self.model,
            reason=reason,
        ).inc()


def update_provider_health(provider: str, healthy: bool) -> None:
    """Update provider health gauge."""
    PROVIDER_HEALTH.labels(provider=provider).set(1 if healthy else 0)


def update_circuit_state(provider: str, state: str) -> None:
    """Update circuit breaker state gauge."""
    CIRCUIT_STATE.labels(provider=provider).set(
        CIRCUIT_STATE_MAP.get(state, 0)
    )
