"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict

from fastapi import FastAPI, Response

from llm_gateway.api.completions import router as completions_router
from llm_gateway.core.circuit_breaker import circuit_breaker
from llm_gateway.core.health import health_checker
from llm_gateway.core.logging import setup_logging
from llm_gateway.core.providers import registry
from llm_gateway.core.redis import redis_manager
from llm_gateway.core.teams import team_registry

# Configure logging (JSON in production, human-readable in dev)
setup_logging(json_format=False, level="INFO")

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: startup and shutdown."""
    # Startup
    registry.initialize()  # Also registers providers with health_checker
    team_registry.initialize()

    # Redis is optional - rate limiting won't work without it
    try:
        await redis_manager.connect()
        logger.info("Redis connected")
    except ConnectionError as e:
        logger.warning(f"Redis unavailable: {e}. Rate limiting disabled.")

    # Start background health checks
    await health_checker.start()

    yield

    # Shutdown
    await health_checker.stop()
    await registry.close_all()
    await redis_manager.disconnect()


app = FastAPI(
    title="LLM Gateway",
    description="Production API gateway for LLM providers",
    version="0.1.0",
    lifespan=lifespan,
)

# Include routers
app.include_router(completions_router)


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint with provider status and circuit breaker state."""
    providers_status: Dict[str, Any] = {}

    for name in registry.list_providers():
        provider_info: Dict[str, Any] = {}

        # Health checker status
        health = health_checker.get_health(name)
        if health:
            provider_info["healthy"] = health.healthy
            provider_info["last_check_seconds_ago"] = round(health.seconds_since_check, 1)
            if health.last_error:
                provider_info["last_error"] = health.last_error

        # Circuit breaker status
        cb_stats = circuit_breaker.get_stats(name)
        provider_info["circuit"] = cb_stats["state"]
        provider_info["failure_count"] = cb_stats["failure_count"]

        providers_status[name] = provider_info

    return {
        "status": "healthy",
        "providers": providers_status,
    }


@app.post("/health/check")
async def trigger_health_check() -> Dict[str, Any]:
    """Trigger immediate health check for all providers.

    Useful for:
    - Admin operations (manually verify after incident)
    - Testing (don't wait for background interval)
    """
    results = await health_checker.check_all()
    return {
        "checked": list(results.keys()),
        "results": results,
    }


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint.

    Returns metrics in Prometheus text format for scraping.
    Configure Prometheus to scrape this endpoint.
    """
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
