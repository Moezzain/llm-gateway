"""Team management and lookup."""

from typing import Dict, Optional

from llm_gateway.core.config import config
from llm_gateway.core.schemas import TeamConfig


class TeamRegistry:
    """Manages team lookups.

    Builds reverse index from API key -> team for fast auth.
    """

    def __init__(self) -> None:
        self._teams_by_id: Dict[str, TeamConfig] = {}
        self._teams_by_key: Dict[str, tuple[str, TeamConfig]] = {}

    def initialize(self) -> None:
        """Build team indexes from config."""
        self._teams_by_id = {}
        self._teams_by_key = {}

        for team_id, team_config in config.teams.items():
            if not team_config.enabled:
                continue

            self._teams_by_id[team_id] = team_config
            self._teams_by_key[team_config.api_key] = (team_id, team_config)

    def get_by_id(self, team_id: str) -> Optional[TeamConfig]:
        """Get team by ID."""
        return self._teams_by_id.get(team_id)

    def get_by_api_key(self, api_key: str) -> Optional[tuple[str, TeamConfig]]:
        """Get team ID and config by API key.

        Returns:
            Tuple of (team_id, TeamConfig) or None if not found.
        """
        return self._teams_by_key.get(api_key)

    def list_teams(self) -> list:
        """List all team IDs."""
        return list(self._teams_by_id.keys())


# Singleton
team_registry = TeamRegistry()
