"""Provider abstraction interface.

Uses Protocol for structural typing — any class implementing these methods
works as a provider, no inheritance required.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from llm_gateway.models.llm import CompletionRequest, CompletionResponse, StreamChunk


@runtime_checkable
class LLMProvider(Protocol):
    """Interface that all LLM providers must implement.

    Protocol enables structural typing (duck typing with type checker support).
    A class doesn't need to inherit from this — it just needs matching methods.
    """

    @property
    def name(self) -> str:
        """Provider name for logging/metrics (e.g., 'openai', 'anthropic')."""
        ...

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Send a completion request and return the full response.

        Args:
            request: Unified completion request.

        Returns:
            Unified completion response.

        Raises:
            ProviderError: On provider-specific errors.
            RateLimitError: When provider rate limits the request.
            AuthenticationError: On invalid API key.
        """
        ...

    async def stream(self, request: CompletionRequest) -> AsyncIterator[StreamChunk]:
        """Send a completion request and stream the response.

        Args:
            request: Unified completion request (stream flag ignored).

        Yields:
            Stream chunks with delta content.

        Raises:
            ProviderError: On provider-specific errors.
        """
        ...

    async def health_check(self) -> bool:
        """Check if provider is healthy and responding.

        Returns:
            True if healthy, False otherwise.
        """
        ...
