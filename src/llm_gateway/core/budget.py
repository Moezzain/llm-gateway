"""Budget tracking and enforcement.

Tracks spending per team, enforces budget caps.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from llm_gateway.core.config import config
from llm_gateway.core.redis import redis_manager
from llm_gateway.core.schemas import BudgetConfig


@dataclass
class BudgetStatus:
    """Current budget status for a team."""

    spent_usd: float
    limit_usd: float
    remaining_usd: float
    percentage_used: float
    is_warning: bool  # Above warning threshold
    is_exceeded: bool  # At or above limit


@dataclass
class CostCalculation:
    """Cost breakdown for a request."""

    input_tokens: int
    output_tokens: int
    input_cost_usd: float
    output_cost_usd: float
    total_cost_usd: float
    model: str


class BudgetTracker:
    """Tracks and enforces team budgets."""

    def __init__(self) -> None:
        pass

    def _get_month_key(self, team_id: str) -> str:
        """Get Redis key for current month's spending."""
        now = datetime.now(timezone.utc)
        return f"budget:{team_id}:{now.year}:{now.month:02d}"

    async def get_status(
        self, team_id: str, budget_config: BudgetConfig
    ) -> BudgetStatus:
        """Get current budget status for a team."""
        spent = await self._get_spent(team_id)
        limit = budget_config.monthly_limit_usd
        remaining = max(0, limit - spent)
        percentage = spent / limit if limit > 0 else 0

        return BudgetStatus(
            spent_usd=spent,
            limit_usd=limit,
            remaining_usd=remaining,
            percentage_used=percentage,
            is_warning=percentage >= budget_config.warning_threshold,
            is_exceeded=spent >= limit,
        )

    async def _get_spent(self, team_id: str) -> float:
        """Get amount spent this month."""
        if not redis_manager.is_connected:
            return 0.0

        key = self._get_month_key(team_id)
        value = await redis_manager.client.get(key)
        return float(value) if value else 0.0

    async def record_spending(
        self, team_id: str, cost_usd: float
    ) -> float:
        """Record spending and return new total.

        Args:
            team_id: Team identifier
            cost_usd: Cost to add

        Returns:
            New total spent this month
        """
        if not redis_manager.is_connected:
            return 0.0

        key = self._get_month_key(team_id)

        # Atomic increment (INCRBYFLOAT)
        # Cost is in dollars, store as cents for precision
        cost_cents = int(cost_usd * 100)
        new_total_cents = await redis_manager.client.incrbyfloat(key, cost_cents)

        # Set expiry to ~40 days (outlive the month)
        await redis_manager.client.expire(key, 60 * 60 * 24 * 40)

        return new_total_cents / 100

    def calculate_cost(
        self, model_name: str, input_tokens: int, output_tokens: int
    ) -> Optional[CostCalculation]:
        """Calculate cost for a request.

        Args:
            model_name: Model used (e.g., "gpt-4o")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            CostCalculation or None if model not found in config
        """
        model_config = config.models.get(model_name)
        if not model_config:
            return None

        input_cost = (input_tokens / 1000) * model_config.input_cost_per_1k
        output_cost = (output_tokens / 1000) * model_config.output_cost_per_1k

        return CostCalculation(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost_usd=input_cost,
            output_cost_usd=output_cost,
            total_cost_usd=input_cost + output_cost,
            model=model_name,
        )


# Singleton
budget_tracker = BudgetTracker()
