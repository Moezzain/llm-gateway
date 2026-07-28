"""Redis connection management.

Provides async Redis client with connection pooling.
"""

from typing import Optional

import redis.asyncio as redis

from llm_gateway.core.config import config


class RedisManager:
    """Manages Redis connection lifecycle.

    Creates connection pool on startup, provides client access,
    cleans up on shutdown.
    """

    def __init__(self) -> None:
        self._client: Optional[redis.Redis] = None

    async def connect(self) -> None:
        """Initialize Redis connection pool."""
        redis_config = config.redis

        self._client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db,
            password=redis_config.password,
            ssl=redis_config.ssl,
            decode_responses=True,  # Return strings, not bytes
        )

        # Test connection
        try:
            await self._client.ping()
        except redis.ConnectionError as e:
            self._client = None
            raise ConnectionError(f"Failed to connect to Redis: {e}")

    async def disconnect(self) -> None:
        """Close Redis connection pool."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> redis.Redis:
        """Get Redis client.

        Raises:
            RuntimeError: If not connected.
        """
        if self._client is None:
            raise RuntimeError("Redis not connected. Call connect() first.")
        return self._client

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._client is not None


# Singleton
redis_manager = RedisManager()
