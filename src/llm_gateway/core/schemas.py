"""Configuration schemas.

Pydantic models for YAML config validation.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    api_key: Optional[str] = Field(default=None, description="Provider API key")
    base_url: Optional[str] = Field(default=None, description="Override base URL")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    enabled: bool = Field(default=True, description="Whether provider is enabled")


class RateLimitConfig(BaseModel):
    """Rate limit configuration for a team."""

    requests_per_minute: int = Field(default=60, ge=1)
    tokens_per_minute: int = Field(default=100000, ge=1)


class BudgetConfig(BaseModel):
    """Budget configuration for a team."""

    monthly_limit_usd: float = Field(default=100.0, ge=0)
    warning_threshold: float = Field(
        default=0.8, ge=0, le=1, description="Warn at this % of budget"
    )


class TeamConfig(BaseModel):
    """Configuration for a team using the gateway."""

    name: str = Field(description="Team display name")
    api_key: str = Field(description="Team's API key for gateway auth")
    allowed_models: List[str] = Field(
        default_factory=list, description="Models team can use (empty = all)"
    )
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    enabled: bool = Field(default=True)


class ModelConfig(BaseModel):
    """Configuration for model routing."""

    provider: str = Field(description="Provider name (openai, anthropic)")
    model_id: str = Field(description="Provider's model ID")
    fallback: Optional[str] = Field(
        default=None, description="Fallback model name if this fails"
    )
    input_cost_per_1k: float = Field(default=0.0, description="Cost per 1k input tokens")
    output_cost_per_1k: float = Field(default=0.0, description="Cost per 1k output tokens")


class CompressionConfig(BaseModel):
    """Prompt compression configuration.

    Disabled by default: compression is opt-in because even conservative
    whitespace normalization can affect whitespace-sensitive payloads.
    """

    enabled: bool = Field(default=False, description="Enable prompt compression")
    strategy: Literal["rule_based"] = Field(
        default="rule_based", description="Compression strategy to apply"
    )
    min_chars: int = Field(
        default=200,
        ge=0,
        description="Skip prompts smaller than this (not worth compressing)",
    )


class GuardrailsConfig(BaseModel):
    """Input guardrails (prompt injection detection) configuration.

    Enabled by default — this is a security gateway. `flag` mode lets you
    shadow-test detection (log only) before flipping to `block` enforcement.
    """

    enabled: bool = Field(default=True, description="Enable input guardrails")
    mode: Literal["block", "flag"] = Field(
        default="block",
        description="'block' rejects flagged requests; 'flag' logs but allows",
    )


class RedisConfig(BaseModel):
    """Redis connection configuration."""

    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    password: Optional[str] = Field(default=None)
    ssl: bool = Field(default=False)


class GatewayConfig(BaseModel):
    """Root configuration schema."""

    providers: Dict[str, ProviderConfig] = Field(
        default_factory=dict, description="Provider configurations"
    )
    teams: Dict[str, TeamConfig] = Field(
        default_factory=dict, description="Team configurations keyed by team ID"
    )
    models: Dict[str, ModelConfig] = Field(
        default_factory=dict, description="Model routing configurations"
    )

    # Prompt compression (cost optimization)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)

    # Input guardrails (security)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)

    # Redis for rate limiting
    redis: RedisConfig = Field(default_factory=RedisConfig)

    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    debug: bool = Field(default=False)
