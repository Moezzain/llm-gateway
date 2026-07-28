"""Unified LLM request/response models.

These models are provider-agnostic. The gateway translates between these
and each provider's native format.
"""

from enum import Enum
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


# Type alias for finish reasons
FinishReason = Optional[Literal["stop", "length", "content_filter"]]


class Role(str, Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class Message(BaseModel):
    """Single message in a conversation."""

    role: Role
    content: str


class CompletionRequest(BaseModel):
    """Unified completion request format.

    Maps to OpenAI's chat completion, Anthropic's messages API, etc.
    """

    model: str = Field(description="Model identifier (e.g., 'gpt-4o', 'claude-sonnet')")
    messages: List[Message] = Field(description="Conversation history")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, description="Max tokens to generate")
    stream: bool = Field(default=False, description="Whether to stream response")


class TokenUsage(BaseModel):
    """Token usage statistics."""

    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class CompletionResponse(BaseModel):
    """Unified completion response format."""

    content: str = Field(description="Generated text")
    model: str = Field(description="Model that served the request")
    usage: TokenUsage
    finish_reason: FinishReason = None


class StreamChunk(BaseModel):
    """Single chunk in a streaming response."""

    content: str = Field(default="", description="Delta content")
    finish_reason: FinishReason = None
    usage: Optional[TokenUsage] = None  # Only present in final chunk (some providers)
