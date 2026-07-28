"""Tests for health checker."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from llm_gateway.core.health import (
    HealthChecker,
    HealthCheckerConfig,
    ProviderHealth,
)


@pytest.fixture
def health_checker() -> HealthChecker:
    """Health checker with low thresholds for testing."""
    config = HealthCheckerConfig(
        check_interval=0.1,
        unhealthy_threshold=2,
        healthy_threshold=1,
    )
    return HealthChecker(config)


@pytest.fixture
def healthy_provider() -> MagicMock:
    """Provider that always returns healthy."""
    provider = MagicMock()
    provider.name = "healthy_provider"
    provider.health_check = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def unhealthy_provider() -> MagicMock:
    """Provider that always returns unhealthy."""
    provider = MagicMock()
    provider.name = "unhealthy_provider"
    provider.health_check = AsyncMock(return_value=False)
    return provider


class TestHealthCheckerRegistration:
    """Test provider registration."""

    def test_register_provider(
        self, health_checker: HealthChecker, healthy_provider: MagicMock
    ) -> None:
        """Can register provider."""
        health_checker.register(healthy_provider)

        health = health_checker.get_health("healthy_provider")
        assert health is not None
        assert health.healthy is True

    def test_unknown_provider_is_healthy(self, health_checker: HealthChecker) -> None:
        """Unknown provider assumed healthy."""
        assert health_checker.is_healthy("unknown") is True


class TestHealthCheckerChecks:
    """Test health checking logic."""

    @pytest.mark.asyncio
    async def test_healthy_check(
        self, health_checker: HealthChecker, healthy_provider: MagicMock
    ) -> None:
        """Healthy provider stays healthy."""
        health_checker.register(healthy_provider)

        result = await health_checker.check_provider("healthy_provider")

        assert result is True
        assert health_checker.is_healthy("healthy_provider") is True

    @pytest.mark.asyncio
    async def test_unhealthy_after_threshold(
        self, health_checker: HealthChecker, unhealthy_provider: MagicMock
    ) -> None:
        """Provider marked unhealthy after threshold failures."""
        health_checker.register(unhealthy_provider)

        # First failure
        await health_checker.check_provider("unhealthy_provider")
        assert health_checker.is_healthy("unhealthy_provider") is True

        # Second failure (threshold = 2)
        await health_checker.check_provider("unhealthy_provider")
        assert health_checker.is_healthy("unhealthy_provider") is False

    @pytest.mark.asyncio
    async def test_recovery(
        self, health_checker: HealthChecker, unhealthy_provider: MagicMock
    ) -> None:
        """Provider recovers when check succeeds."""
        health_checker.register(unhealthy_provider)

        # Mark unhealthy
        await health_checker.check_provider("unhealthy_provider")
        await health_checker.check_provider("unhealthy_provider")
        assert health_checker.is_healthy("unhealthy_provider") is False

        # Recover
        unhealthy_provider.health_check = AsyncMock(return_value=True)
        await health_checker.check_provider("unhealthy_provider")

        assert health_checker.is_healthy("unhealthy_provider") is True

    @pytest.mark.asyncio
    async def test_check_all_concurrent(
        self, health_checker: HealthChecker, healthy_provider: MagicMock
    ) -> None:
        """check_all runs all checks concurrently."""
        provider2 = MagicMock()
        provider2.name = "provider2"
        provider2.health_check = AsyncMock(return_value=True)

        health_checker.register(healthy_provider)
        health_checker.register(provider2)

        results = await health_checker.check_all()

        assert results["healthy_provider"] is True
        assert results["provider2"] is True


class TestHealthCheckerLists:
    """Test list methods."""

    @pytest.mark.asyncio
    async def test_list_healthy_unhealthy(
        self,
        health_checker: HealthChecker,
        healthy_provider: MagicMock,
        unhealthy_provider: MagicMock,
    ) -> None:
        """Can list healthy and unhealthy providers."""
        health_checker.register(healthy_provider)
        health_checker.register(unhealthy_provider)

        # Make unhealthy_provider actually unhealthy
        await health_checker.check_provider("unhealthy_provider")
        await health_checker.check_provider("unhealthy_provider")

        assert "healthy_provider" in health_checker.list_healthy()
        assert "unhealthy_provider" in health_checker.list_unhealthy()
