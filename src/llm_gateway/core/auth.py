"""Authentication dependencies."""

from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException

from llm_gateway.core.schemas import TeamConfig
from llm_gateway.core.teams import team_registry


@dataclass
class TeamContext:
    """Authenticated team context for a request."""

    team_id: str
    config: TeamConfig


async def require_team(
    authorization: Optional[str] = Header(None, alias="Authorization"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> TeamContext:
    """Dependency that requires valid team authentication.

    Accepts API key via:
    - Authorization: Bearer <key>
    - X-API-Key: <key>

    Returns:
        TeamContext with team ID and config.

    Raises:
        HTTPException 401: Missing or invalid API key.
        HTTPException 403: Team disabled.
    """
    api_key = _extract_api_key(authorization, x_api_key)

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Provide via Authorization: Bearer <key> or X-API-Key header.",
        )

    result = team_registry.get_by_api_key(api_key)

    if result is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    team_id, team_config = result

    if not team_config.enabled:
        raise HTTPException(
            status_code=403,
            detail="Team is disabled.",
        )

    return TeamContext(team_id=team_id, config=team_config)


def _extract_api_key(
    authorization: Optional[str], x_api_key: Optional[str]
) -> Optional[str]:
    """Extract API key from headers."""
    # Prefer X-API-Key
    if x_api_key:
        return x_api_key

    # Try Bearer token
    if authorization:
        parts = authorization.split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1]

    return None
