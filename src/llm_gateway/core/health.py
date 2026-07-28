"""Provider health checking.

Background task monitors provider availability.
Used for proactive routing (skip unhealthy) vs reactive (retry after fail).
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

from llm_gateway.providers.base import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ProviderHealth:
    """Health status for a single provider."""

    name: str
    healthy: bool = True
    last_check: float = 0.0  # Unix timestamp
    consecutive_failures: int = 0
    last_error: Optional[str] = None

    @property
    def seconds_since_check(self) -> float:
        """Seconds since last health check."""
        if self.last_check == 0:
            return float("inf")
        return time.time() - self.last_check


@dataclass
class HealthCheckerConfig:
    """Configuration for health checker."""

    check_interval: float = 30.0  # Seconds between checks
    unhealthy_threshold: int = 3  # Consecutive failures to mark unhealthy
    healthy_threshold: int = 1  # Consecutive successes to mark healthy


class HealthChecker:
    """Monitors provider health in background.

    Periodically pings each provider's health_check() endpoint.
    Tracks consecutive failures/successes to determine health.

    Why background checks vs check-on-request?
    - Proactive: Know health before request arrives
    - No latency hit: Health check happens async
    - Smoother failover: Don't wait for timeout to discover outage
    """

    def __init__(self, config: Optional[HealthCheckerConfig] = None) -> None:
        self._config = config or HealthCheckerConfig()
        self._providers: Dict[str, LLMProvider] = {}
        self._health: Dict[str, ProviderHealth] = {}
        self._task: Optional[asyncio.Task] = None
        self._running = False

    def register(self, provider: LLMProvider) -> None:
        """Register a provider for health monitoring."""
        self._providers[provider.name] = provider
        self._health[provider.name] = ProviderHealth(name=provider.name)
        logger.info(f"Registered provider '{provider.name}' for health monitoring")

    def get_health(self, provider_name: str) -> Optional[ProviderHealth]:
        """Get current health status for a provider."""
        return self._health.get(provider_name)

    def is_healthy(self, provider_name: str) -> bool:
        """Check if provider is healthy.

        Returns True if:
        - Provider not registered (assume healthy, let request try)
        - Provider registered and marked healthy
        """
        health = self._health.get(provider_name)
        if health is None:
            return True  # Unknown = assume healthy
        return health.healthy

    def list_healthy(self) -> list:
        """List all healthy provider names."""
        return [name for name, health in self._health.items() if health.healthy]

    def list_unhealthy(self) -> list:
        """List all unhealthy provider names."""
        return [name for name, health in self._health.items() if not health.healthy]

    async def check_provider(self, provider_name: str) -> bool:
        """Run health check for a specific provider.

        Returns True if healthy, False otherwise.
        Updates internal health state.
        """
        provider = self._providers.get(provider_name)
        if not provider:
            return True

        health = self._health[provider_name]

        try:
            is_healthy = await provider.health_check()
            health.last_check = time.time()

            if is_healthy:
                health.consecutive_failures = 0
                health.last_error = None

                if not health.healthy:
                    # Was unhealthy, now recovered
                    health.healthy = True
                    logger.info(f"Provider '{provider_name}' recovered")
            else:
                health.consecutive_failures += 1
                health.last_error = "Health check returned false"
                self._maybe_mark_unhealthy(health)

            return is_healthy

        except Exception as e:
            health.last_check = time.time()
            health.consecutive_failures += 1
            health.last_error = str(e)
            self._maybe_mark_unhealthy(health)
            return False

    def _maybe_mark_unhealthy(self, health: ProviderHealth) -> None:
        """Mark provider unhealthy if threshold reached."""
        if (
            health.healthy
            and health.consecutive_failures >= self._config.unhealthy_threshold
        ):
            health.healthy = False
            logger.warning(
                f"Provider '{health.name}' marked unhealthy after "
                f"{health.consecutive_failures} consecutive failures. "
                f"Last error: {health.last_error}"
            )

    async def check_all(self) -> Dict[str, bool]:
        """Run health checks for all providers concurrently."""
        if not self._providers:
            return {}

        # Run all checks in parallel
        tasks = [
            self.check_provider(name)
            for name in self._providers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            name: (result if isinstance(result, bool) else False)
            for name, result in zip(self._providers.keys(), results)
        }

    async def start(self) -> None:
        """Start background health checking task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            f"Health checker started (interval: {self._config.check_interval}s)"
        )

    async def stop(self) -> None:
        """Stop background health checking task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Health checker stopped")

    async def _run_loop(self) -> None:
        """Background loop that periodically checks all providers."""
        # Initial check on startup
        await self.check_all()

        while self._running:
            try:
                await asyncio.sleep(self._config.check_interval)
                if self._running:
                    await self.check_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                # Don't crash the loop, continue checking


# Singleton instance
health_checker = HealthChecker()
