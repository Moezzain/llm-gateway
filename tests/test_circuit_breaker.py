"""Tests for circuit breaker."""

import pytest
from llm_gateway.core.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
)


@pytest.fixture
def circuit_breaker() -> CircuitBreaker:
    """Circuit breaker with low thresholds for testing."""
    config = CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=0.1,  # 100ms for fast tests
        half_open_max_calls=2,
    )
    return CircuitBreaker(config)


class TestCircuitBreakerStates:
    """Test circuit breaker state transitions."""

    def test_starts_closed(self, circuit_breaker: CircuitBreaker) -> None:
        """Circuit starts in closed state."""
        assert circuit_breaker.get_state("test") == CircuitState.CLOSED
        assert circuit_breaker.allow_request("test") is True

    def test_opens_after_failures(self, circuit_breaker: CircuitBreaker) -> None:
        """Circuit opens after threshold failures."""
        # Record failures
        for _ in range(3):
            circuit_breaker.record_failure("test")

        assert circuit_breaker.get_state("test") == CircuitState.OPEN
        assert circuit_breaker.allow_request("test") is False

    def test_success_resets_failure_count(self, circuit_breaker: CircuitBreaker) -> None:
        """Success in closed state resets failure count."""
        circuit_breaker.record_failure("test")
        circuit_breaker.record_failure("test")
        circuit_breaker.record_success("test")
        circuit_breaker.record_failure("test")
        circuit_breaker.record_failure("test")

        # Should still be closed (count reset after success)
        assert circuit_breaker.get_state("test") == CircuitState.CLOSED

    def test_transitions_to_half_open(self, circuit_breaker: CircuitBreaker) -> None:
        """Circuit transitions to half-open after timeout."""
        import time

        # Open the circuit
        for _ in range(3):
            circuit_breaker.record_failure("test")

        assert circuit_breaker.get_state("test") == CircuitState.OPEN

        # Wait for timeout
        time.sleep(0.15)

        # Should be half-open now
        assert circuit_breaker.get_state("test") == CircuitState.HALF_OPEN
        assert circuit_breaker.allow_request("test") is True

    def test_half_open_closes_on_success(self, circuit_breaker: CircuitBreaker) -> None:
        """Circuit closes after successes in half-open."""
        import time

        # Open and wait
        for _ in range(3):
            circuit_breaker.record_failure("test")
        time.sleep(0.15)

        # Should be half-open
        assert circuit_breaker.get_state("test") == CircuitState.HALF_OPEN

        # Record successes
        circuit_breaker.record_success("test")
        circuit_breaker.record_success("test")

        # Should be closed now
        assert circuit_breaker.get_state("test") == CircuitState.CLOSED

    def test_half_open_reopens_on_failure(self, circuit_breaker: CircuitBreaker) -> None:
        """Circuit reopens on failure in half-open."""
        import time

        # Open and wait
        for _ in range(3):
            circuit_breaker.record_failure("test")
        time.sleep(0.15)

        # Should be half-open
        assert circuit_breaker.get_state("test") == CircuitState.HALF_OPEN

        # Single failure reopens
        circuit_breaker.record_failure("test")

        assert circuit_breaker.get_state("test") == CircuitState.OPEN


class TestCircuitBreakerIsolation:
    """Test that circuits are isolated per provider."""

    def test_independent_providers(self, circuit_breaker: CircuitBreaker) -> None:
        """Each provider has independent circuit."""
        # Break provider A
        for _ in range(3):
            circuit_breaker.record_failure("provider_a")

        # Provider A should be open
        assert circuit_breaker.get_state("provider_a") == CircuitState.OPEN
        assert circuit_breaker.allow_request("provider_a") is False

        # Provider B should still be closed
        assert circuit_breaker.get_state("provider_b") == CircuitState.CLOSED
        assert circuit_breaker.allow_request("provider_b") is True


class TestCircuitBreakerStats:
    """Test circuit breaker stats."""

    def test_get_stats(self, circuit_breaker: CircuitBreaker) -> None:
        """Stats include state and failure count."""
        circuit_breaker.record_failure("test")
        circuit_breaker.record_failure("test")

        stats = circuit_breaker.get_stats("test")

        assert stats["state"] == "closed"
        assert stats["failure_count"] == 2
        assert stats["success_count"] == 0

    def test_force_open(self, circuit_breaker: CircuitBreaker) -> None:
        """Force open works."""
        circuit_breaker.force_open("test")
        assert circuit_breaker.get_state("test") == CircuitState.OPEN

    def test_force_close(self, circuit_breaker: CircuitBreaker) -> None:
        """Force close works."""
        for _ in range(3):
            circuit_breaker.record_failure("test")
        
        circuit_breaker.force_close("test")
        assert circuit_breaker.get_state("test") == CircuitState.CLOSED
