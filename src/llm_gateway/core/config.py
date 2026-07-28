"""Application configuration.

Loads from YAML file with environment variable overrides.
"""

from llm_gateway.core.loader import load_config
from llm_gateway.core.schemas import GatewayConfig

# Load config on module import
# YAML file + env var overrides
config: GatewayConfig = load_config()
