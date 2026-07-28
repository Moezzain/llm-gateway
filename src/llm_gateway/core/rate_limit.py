"""Token bucket rate limiting with Redis.

Uses Lua script for atomic check-and-consume operations.
"""

import time
from dataclasses import dataclass
from typing import Optional

from llm_gateway.core.redis import redis_manager


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int  # Tokens remaining
    reset_at: float  # Unix timestamp when bucket refills
    retry_after: Optional[float] = None  # Seconds to wait if rejected


# Lua script for atomic token bucket operation
# KEYS[1] = bucket key
# ARGV[1] = capacity (max tokens)
# ARGV[2] = refill_rate (tokens per second)
# ARGV[3] = current timestamp
# ARGV[4] = tokens to consume (usually 1)
#
# Returns: [allowed (0/1), remaining, reset_at]
TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])

-- Get current bucket state
local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
local tokens = tonumber(bucket[1])
local last_refill = tonumber(bucket[2])

-- Initialize if bucket doesn't exist
if tokens == nil then
    tokens = capacity
    last_refill = now
end

-- Calculate tokens to add since last refill
local elapsed = now - last_refill
local tokens_to_add = elapsed * refill_rate
tokens = math.min(capacity, tokens + tokens_to_add)
last_refill = now

-- Try to consume tokens
local allowed = 0
if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
end

-- Save bucket state
redis.call('HMSET', key, 'tokens', tokens, 'last_refill', last_refill)
-- Expire bucket after 2x the time to fully refill (cleanup stale buckets)
local ttl = math.ceil((capacity / refill_rate) * 2)
redis.call('EXPIRE', key, ttl)

-- Calculate when bucket will be full again
local reset_at = now + ((capacity - tokens) / refill_rate)

return {allowed, math.floor(tokens), reset_at}
"""


class RateLimiter:
    """Token bucket rate limiter using Redis."""

    def __init__(self) -> None:
        self._script_sha: Optional[str] = None

    async def _ensure_script(self) -> str:
        """Load Lua script into Redis if not already loaded."""
        if self._script_sha is None:
            self._script_sha = await redis_manager.client.script_load(
                TOKEN_BUCKET_SCRIPT
            )
        return self._script_sha

    async def check(
        self,
        key: str,
        capacity: int,
        refill_rate: float,
        tokens: int = 1,
    ) -> RateLimitResult:
        """Check rate limit and consume tokens if allowed.

        Args:
            key: Unique identifier (e.g., "ratelimit:team-alpha:rpm")
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
            tokens: Tokens to consume (default 1)

        Returns:
            RateLimitResult with allowed status and metadata
        """
        if not redis_manager.is_connected:
            # No Redis = no rate limiting (allow all)
            return RateLimitResult(
                allowed=True,
                remaining=capacity,
                reset_at=time.time(),
            )

        script_sha = await self._ensure_script()
        now = time.time()

        result = await redis_manager.client.evalsha(
            script_sha,
            1,  # Number of keys
            key,
            str(capacity),
            str(refill_rate),
            str(now),
            str(tokens),
        )

        allowed = bool(result[0])
        remaining = int(result[1])
        reset_at = float(result[2])

        retry_after = None
        if not allowed:
            # Calculate how long until enough tokens are available
            retry_after = (tokens - remaining) / refill_rate

        return RateLimitResult(
            allowed=allowed,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=retry_after,
        )

    async def check_requests_per_minute(
        self, team_id: str, requests_per_minute: int
    ) -> RateLimitResult:
        """Check requests-per-minute limit for a team."""
        key = f"ratelimit:{team_id}:rpm"
        # Convert RPM to tokens per second
        refill_rate = requests_per_minute / 60.0
        return await self.check(key, requests_per_minute, refill_rate)

    async def check_tokens_per_minute(
        self, team_id: str, tokens_per_minute: int, token_count: int
    ) -> RateLimitResult:
        """Check tokens-per-minute limit for a team.

        Called after response to track token usage.
        """
        key = f"ratelimit:{team_id}:tpm"
        # Convert TPM to tokens per second
        refill_rate = tokens_per_minute / 60.0
        return await self.check(key, tokens_per_minute, refill_rate, token_count)


# Singleton
rate_limiter = RateLimiter()
