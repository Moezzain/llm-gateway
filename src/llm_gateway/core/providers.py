"""Provider registry and lifecycle management."""

from typing import Dict, List, Optional

from llm_gateway.core.config import config
from llm_gateway.core.health import health_checker
from llm_gateway.providers import (
    AnthropicProvider,
    OllamaProvider,
    OpenAIProvider,
)
from llm_gateway.providers.base import LLMProvider


class ProviderRegistry:
    """Manages provider instances.

    Creates providers on startup, provides access by name,
    handles cleanup on shutdown. Also registers providers
    with health checker for monitoring.
    """

    def __init__(self) -> None:
        self._providers: Dict[str, LLMProvider] = {}

    def initialize(self) -> None:
        """Initialize providers from config."""
        for name, provider_config in config.providers.items():
            if not provider_config.enabled:
                continue

            provider: Optional[LLMProvider] = None

            if name == "ollama":
                # Local provider — no API key, auth is network-level.
                provider = OllamaProvider(
                    timeout=provider_config.timeout,
                    base_url=provider_config.base_url,
                )
            elif not provider_config.api_key:
                # Cloud providers need a key; skip if not configured.
                continue
            elif name == "openai":
                provider = OpenAIProvider(
                    api_key=provider_config.api_key,
                    timeout=provider_config.timeout,
                    base_url=provider_config.base_url,
                )
            elif name == "anthropic":
                provider = AnthropicProvider(
                    api_key=provider_config.api_key,
                    timeout=provider_config.timeout,
                    base_url=provider_config.base_url,
                )

            if provider:
                self._providers[name] = provider
                health_checker.register(provider)

    def get(self, name: str) -> Optional[LLMProvider]:
        """Get provider by name."""
        return self._providers.get(name)

    def get_default(self) -> Optional[LLMProvider]:
        """Get first available provider."""
        if self._providers:
            return next(iter(self._providers.values()))
        return None

    def list_providers(self) -> List[str]:
        """List available provider names."""
        return list(self._providers.keys())

    async def close_all(self) -> None:
        """Close all provider HTTP clients."""
        for provider in self._providers.values():
            if hasattr(provider, "close"):
                await provider.close()


# Singleton instance
registry = ProviderRegistry()
