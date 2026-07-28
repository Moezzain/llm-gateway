"""Provider-related exceptions.

Unified exceptions across all providers. The retry/fallback logic uses these
to decide whether to retry, fallback, or fail immediately.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base exception for all provider errors."""

    def __init__(self, message: str, provider: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class RateLimitError(ProviderError):
    """Provider rate limited the request. Retryable after delay."""

    def __init__(
        self, message: str, provider: str, retry_after: float | None = None
    ) -> None:
        super().__init__(message, provider, retryable=True)
        self.retry_after = retry_after  # Seconds to wait before retry


class AuthenticationError(ProviderError):
    """Invalid API key or authentication failure. Not retryable."""

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(message, provider, retryable=False)


class ModelNotFoundError(ProviderError):
    """Requested model doesn't exist. Not retryable."""

    def __init__(self, message: str, provider: str, model: str) -> None:
        super().__init__(message, provider, retryable=False)
        self.model = model


class ContentFilterError(ProviderError):
    """Content blocked by provider's safety filters. Not retryable."""

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(message, provider, retryable=False)


class TimeoutError(ProviderError):
    """Request timed out. Retryable."""

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(message, provider, retryable=True)


class ServiceUnavailableError(ProviderError):
    """Provider service is down. Retryable with fallback."""

    def __init__(self, message: str, provider: str) -> None:
        super().__init__(message, provider, retryable=True)
