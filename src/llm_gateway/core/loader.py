"""Configuration loader.

Loads and validates YAML config, with environment variable overrides.
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from llm_gateway.core.schemas import GatewayConfig


class ConfigError(Exception):
    """Configuration loading or validation error."""

    pass


def load_config(path: Optional[str] = None) -> GatewayConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to config file. Defaults to CONFIG_PATH env var or config/gateway.yaml

    Returns:
        Validated GatewayConfig

    Raises:
        ConfigError: If file not found or validation fails
    """
    config_path = _resolve_path(path)

    if not config_path.exists():
        # Return default config if no file exists
        return GatewayConfig()

    try:
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {config_path}: {e}")

    # Apply environment variable overrides
    raw = _apply_env_overrides(raw)

    try:
        return GatewayConfig(**raw)
    except ValidationError as e:
        raise ConfigError(f"Config validation failed: {e}")


def _resolve_path(path: Optional[str]) -> Path:
    """Resolve config file path."""
    if path:
        return Path(path)

    env_path = os.environ.get("CONFIG_PATH")
    if env_path:
        return Path(env_path)

    return Path("config/gateway.yaml")


def _apply_env_overrides(raw: dict) -> dict:
    """Override config values from environment variables.

    Supports:
    - OPENAI_API_KEY -> providers.openai.api_key
    - ANTHROPIC_API_KEY -> providers.anthropic.api_key
    """
    if "providers" not in raw:
        raw["providers"] = {}

    # OpenAI
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        if "openai" not in raw["providers"]:
            raw["providers"]["openai"] = {}
        raw["providers"]["openai"]["api_key"] = openai_key

    # Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if anthropic_key:
        if "anthropic" not in raw["providers"]:
            raw["providers"]["anthropic"] = {}
        raw["providers"]["anthropic"]["api_key"] = anthropic_key

    return raw
