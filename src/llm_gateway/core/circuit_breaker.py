"""Circuit breaker pattern for provider resilience.

Prevents hammering failed providers by tracking failures and
"opening" the circuit after threshold is reached.

States:
- CLOSED: Normal operation, requests flow through
- OPEN: Provider failing, reject requests immediately
- HALF_OPEN: Testing if provider recovered
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal, requests allowed
    OPEN = "open"  # Failing, requests blocked
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures to open circuit
    success_threshold: int = 2  # Successes in half-open to close
    timeout: float = 30.0  # Seconds before trying half-open
    half_open_max_calls: int = 3  # Max concurrent calls in half-open


@dataclass
class CircuitStats:
    """Statistics for a single circuit."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0  # Only tracked in half-open
    last_failure_time: float = 0.0
    half_open_calls: int = 0  # Current calls in half-open

    def reset(self) -> None:
        """Reset to closed state."""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0


class CircuitBreaker:
    """Circuit breaker for provider calls.

    Usage:
        if circuit_breaker.allow_request("openai"):
            try:
                result = await provider.complete(request)
                circuit_breaker.record_success("openai")
            except ProviderError as e:
                if e.retryable:
                    circuit_breaker.record_failure("openai")
                raise
        else:
            # Circuit open, use fallback or fail fast
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._circuits: Dict[str, CircuitStats] = {}

    def _get_circuit(self, provider_name: str) -> CircuitStats:
        """Get or create circuit for provider."""
        if provider_name not in self._circuits:
            self._circuits[provider_name] = CircuitStats()
        return self._circuits[provider_name]

    def get_state(self, provider_name: str) -> CircuitState:
        """Get current circuit state for provider."""
        circuit = self._get_circuit(provider_name)
        self._maybe_transition_to_half_open(circuit)
        return circuit.state

    def allow_request(self, provider_name: str) -> bool:
        """Check if request should be allowed through.

        Returns True if request can proceed, False if circuit is open.
        """
        circuit = self._get_circuit(provider_name)

        # Check if we should transition from open to half-open
        self._maybe_transition_to_half_open(circuit)

        if circuit.state == CircuitState.CLOSED:
            return True

        if circuit.state == CircuitState.OPEN:
            return False

        # Half-open: allow limited requests to test recovery
        if circuit.half_open_calls < self._config.half_open_max_calls:
            circuit.half_open_calls += 1
            return True

        return False

    def record_success(self, provider_name: str) -> None:
        """Record successful request."""
        circuit = self._get_circuit(provider_name)

        if circuit.state == CircuitState.HALF_OPEN:
            circuit.success_count += 1
            circuit.half_open_calls = max(0, circuit.half_open_calls - 1)

            # Enough successes to close circuit
            if circuit.success_count >= self._config.success_threshold:
                logger.info(f"Circuit closed for '{provider_name}' (recovered)")
                circuit.reset()

        elif circuit.state == CircuitState.CLOSED:
            # Reset failure count on success
            circuit.failure_count = 0

    def record_failure(self, provider_name: str) -> None:
        """Record failed request (only for retryable failures)."""
        circuit = self._get_circuit(provider_name)
        circuit.failure_count += 1
        circuit.last_failure_time = time.time()

        if circuit.state == CircuitState.HALF_OPEN:
            circuit.half_open_calls = max(0, circuit.half_open_calls - 1)
            # Any failure in half-open reopens circuit
            circuit.state = CircuitState.OPEN
            circuit.success_count = 0
            logger.warning(
                f"Circuit reopened for '{provider_name}' (failed in half-open)"
            )

        elif circuit.state == CircuitState.CLOSED:
            if circuit.failure_count >= self._config.failure_threshold:
                circuit.state = CircuitState.OPEN
                logger.warning(
                    f"Circuit opened for '{provider_name}' "
                    f"after {circuit.failure_count} failures"
                )

    def _maybe_transition_to_half_open(self, circuit: CircuitStats) -> None:
        """Transition from open to half-open if timeout elapsed."""
        if circuit.state != CircuitState.OPEN:
            return

        elapsed = time.time() - circuit.last_failure_time
        if elapsed >= self._config.timeout:
            circuit.state = CircuitState.HALF_OPEN
            circuit.success_count = 0
            circuit.half_open_calls = 0
            logger.info(
                f"Circuit half-open for provider "
                f"(testing recovery after {elapsed:.1f}s)"
            )

    def force_open(self, provider_name: str) -> None:
        """Manually open circuit (e.g., from health checker)."""
        circuit = self._get_circuit(provider_name)
        if circuit.state != CircuitState.OPEN:
            circuit.state = CircuitState.OPEN
            circuit.last_failure_time = time.time()
            logger.info(f"Circuit force-opened for '{provider_name}'")

    def force_close(self, provider_name: str) -> None:
        """Manually close circuit (e.g., admin override)."""
        circuit = self._get_circuit(provider_name)
        circuit.reset()
        logger.info(f"Circuit force-closed for '{provider_name}'")

    def get_stats(self, provider_name: str) -> dict:
        """Get circuit stats for monitoring."""
        circuit = self._get_circuit(provider_name)
        self._maybe_transition_to_half_open(circuit)

        return {
            "state": circuit.state.value,
            "failure_count": circuit.failure_count,
            "success_count": circuit.success_count,
            "seconds_since_last_failure": (
                time.time() - circuit.last_failure_time
                if circuit.last_failure_time > 0
                else None
            ),
        }


# Singleton instance
circuit_breaker = CircuitBreaker()
