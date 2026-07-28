"""Tests for model router."""

import pytest
from unittest.mock import MagicMock, patch

from llm_gateway.core.router import ModelRouter, RouteResult
from llm_gateway.core.schemas import ModelConfig


@pytest.fixture
def mock_config() -> MagicMock:
    """Mock config with test models."""
    config = MagicMock()
    config.models = {
        "gpt-4o": ModelConfig(
            provider="openai",
            model_id="gpt-4o-2024-01-01",
            fallback="claude-sonnet",
        ),
        "claude-sonnet": ModelConfig(
            provider="anthropic",
            model_id="claude-3-sonnet-20240229",
            fallback=None,
        ),
    }
    return config


@pytest.fixture
def mock_registry(mock_provider: MagicMock) -> MagicMock:
    """Mock provider registry."""
    registry = MagicMock()
    
    openai_provider = MagicMock()
    openai_provider.name = "openai"
    
    anthropic_provider = MagicMock()
    anthropic_provider.name = "anthropic"
    
    def get_provider(name: str) -> MagicMock:
        if name == "openai":
            return openai_provider
        elif name == "anthropic":
            return anthropic_provider
        return None
    
    registry.get = get_provider
    return registry


class TestModelRouter:
    """Test model routing."""

    def test_route_existing_model(
        self, mock_config: MagicMock, mock_registry: MagicMock
    ) -> None:
        """Routes to correct provider."""
        with patch("llm_gateway.core.router.config", mock_config), \
             patch("llm_gateway.core.router.registry", mock_registry), \
             patch("llm_gateway.core.router.health_checker") as mock_health:
            mock_health.is_healthy.return_value = True
            
            router = ModelRouter()
            result = router.route("gpt-4o")

            assert result is not None
            assert result.provider.name == "openai"
            assert result.model_id == "gpt-4o-2024-01-01"
            assert result.is_fallback is False

    def test_route_unknown_model(
        self, mock_config: MagicMock, mock_registry: MagicMock
    ) -> None:
        """Returns None for unknown model."""
        with patch("llm_gateway.core.router.config", mock_config), \
             patch("llm_gateway.core.router.registry", mock_registry):
            router = ModelRouter()
            result = router.route("unknown-model")

            assert result is None

    def test_route_uses_fallback_when_unhealthy(
        self, mock_config: MagicMock, mock_registry: MagicMock
    ) -> None:
        """Uses fallback when primary is unhealthy."""
        with patch("llm_gateway.core.router.config", mock_config), \
             patch("llm_gateway.core.router.registry", mock_registry), \
             patch("llm_gateway.core.router.health_checker") as mock_health:
            # Primary unhealthy, fallback healthy
            mock_health.is_healthy.side_effect = lambda name: name != "openai"
            
            router = ModelRouter()
            result = router.route("gpt-4o")

            assert result is not None
            assert result.provider.name == "anthropic"
            assert result.is_fallback is True

    def test_get_fallback(
        self, mock_config: MagicMock, mock_registry: MagicMock
    ) -> None:
        """get_fallback returns fallback route."""
        with patch("llm_gateway.core.router.config", mock_config), \
             patch("llm_gateway.core.router.registry", mock_registry), \
             patch("llm_gateway.core.router.health_checker") as mock_health:
            mock_health.is_healthy.return_value = True
            
            router = ModelRouter()
            result = router.get_fallback("gpt-4o")

            assert result is not None
            assert result.provider.name == "anthropic"
            assert result.is_fallback is True

    def test_no_fallback_configured(
        self, mock_config: MagicMock, mock_registry: MagicMock
    ) -> None:
        """Returns None when no fallback configured."""
        with patch("llm_gateway.core.router.config", mock_config), \
             patch("llm_gateway.core.router.registry", mock_registry):
            router = ModelRouter()
            result = router.get_fallback("claude-sonnet")

            assert result is None
