"""Model routing — maps model names to providers."""

from dataclasses import dataclass
from typing import Optional

from llm_gateway.core.config import config
from llm_gateway.core.health import health_checker
from llm_gateway.core.providers import registry
from llm_gateway.core.schemas import ModelConfig
from llm_gateway.providers.base import LLMProvider


@dataclass
class RouteResult:
    """Result of routing a model request."""

    provider: LLMProvider
    model_id: str  # Actual model ID to send to provider
    model_config: ModelConfig
    is_fallback: bool = False  # True if this is a fallback route


class ModelRouter:
    """Routes model requests to appropriate providers.

    Uses config to map model names (e.g., "gpt-4o") to:
    - Provider (openai, anthropic)
    - Actual model ID (might differ from request)

    Checks provider health and uses fallback if primary is unhealthy.
    """

    def route(self, model_name: str, check_health: bool = True) -> Optional[RouteResult]:
        """Route a model name to provider.

        Args:
            model_name: Requested model (e.g., "gpt-4o", "claude-sonnet")
            check_health: If True, skip unhealthy providers and try fallback

        Returns:
            RouteResult with provider and model details, or None if not found.
        """
        model_config = config.models.get(model_name)

        if not model_config:
            return None

        provider = registry.get(model_config.provider)

        if not provider:
            return None

        # Check if primary provider is healthy
        if check_health and not health_checker.is_healthy(model_config.provider):
            # Try fallback if configured
            fallback = self.get_fallback(model_name)
            if fallback:
                return fallback

        return RouteResult(
            provider=provider,
            model_id=model_config.model_id,
            model_config=model_config,
            is_fallback=False,
        )

    def get_fallback(self, model_name: str) -> Optional[RouteResult]:
        """Get fallback route for a model.

        Args:
            model_name: Original model that failed

        Returns:
            RouteResult for fallback model, or None if no fallback configured.
        """
        model_config = config.models.get(model_name)

        if not model_config or not model_config.fallback:
            return None

        # Route fallback without health check to avoid infinite recursion
        fallback_result = self.route(model_config.fallback, check_health=False)
        if fallback_result:
            fallback_result.is_fallback = True
        return fallback_result


# Singleton
model_router = ModelRouter()
