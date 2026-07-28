"""Pytest fixtures for tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from llm_gateway.models.llm import (
    CompletionRequest,
    CompletionResponse,
    Message,
    Role,
    TokenUsage,
)


@pytest.fixture
def sample_request() -> CompletionRequest:
    """Sample completion request."""
    return CompletionRequest(
        model="gpt-4o",
        messages=[
            Message(role=Role.USER, content="Hello"),
        ],
        temperature=0.7,
    )


@pytest.fixture
def sample_response() -> CompletionResponse:
    """Sample completion response."""
    return CompletionResponse(
        content="Hello! How can I help you?",
        model="gpt-4o",
        usage=TokenUsage(input_tokens=10, output_tokens=20),
        finish_reason="stop",
    )


@pytest.fixture
def mock_provider(sample_response: CompletionResponse) -> MagicMock:
    """Mock LLM provider."""
    provider = MagicMock()
    provider.name = "mock_provider"
    provider.complete = AsyncMock(return_value=sample_response)
    provider.health_check = AsyncMock(return_value=True)
    return provider


@pytest.fixture
def mock_redis() -> MagicMock:
    """Mock Redis client."""
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.incrbyfloat = AsyncMock(return_value=1.0)
    redis.evalsha = AsyncMock(return_value=[1, 100, 60.0])
    return redis
