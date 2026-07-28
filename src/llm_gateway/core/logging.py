"""Structured logging configuration.

Provides JSON-formatted logs for production observability.
Includes request context (team, model, latency, tokens).
"""

import json
import logging
import sys
import time
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

# Context variable for request-scoped data
request_context: ContextVar[Dict[str, Any]] = ContextVar("request_context", default={})


@dataclass
class RequestLog:
    """Structured log entry for a completion request."""

    timestamp: float = field(default_factory=time.time)
    team_id: Optional[str] = None
    model: Optional[str] = None
    provider: Optional[str] = None
    is_fallback: bool = False
    is_streaming: bool = False

    # Timing
    latency_ms: Optional[float] = None

    # Tokens (non-streaming only)
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    # Cost
    cost_usd: Optional[float] = None

    # Status
    status: str = "success"  # success, error, rate_limited, budget_exceeded
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Circuit breaker
    circuit_state: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


class JsonFormatter(logging.Formatter):
    """JSON log formatter for structured logging.

    Output format:
    {"level": "INFO", "message": "...", "timestamp": "...", ...context}
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record),
        }

        # Add request context if available
        ctx = request_context.get()
        if ctx:
            log_data.update(ctx)

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        for key in ["team_id", "model", "provider", "latency_ms", "status"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)

        return json.dumps(log_data)


class RequestLogger:
    """Helper for logging request lifecycle.

    Usage:
        req_log = RequestLogger(team_id="alpha", model="gpt-4o")
        req_log.set_provider("openai")
        try:
            response = await provider.complete(request)
            req_log.set_tokens(response.usage.input_tokens, response.usage.output_tokens)
            req_log.success()
        except Exception as e:
            req_log.error(e)
    """

    def __init__(
        self,
        team_id: Optional[str] = None,
        model: Optional[str] = None,
        is_streaming: bool = False,
    ) -> None:
        self._log = RequestLog(
            team_id=team_id,
            model=model,
            is_streaming=is_streaming,
        )
        self._start_time = time.time()
        self._logger = logging.getLogger("llm_gateway.requests")

    def set_provider(self, provider: str, is_fallback: bool = False) -> None:
        """Set provider info."""
        self._log.provider = provider
        self._log.is_fallback = is_fallback

    def set_tokens(self, input_tokens: int, output_tokens: int) -> None:
        """Set token counts."""
        self._log.input_tokens = input_tokens
        self._log.output_tokens = output_tokens

    def set_cost(self, cost_usd: float) -> None:
        """Set cost."""
        self._log.cost_usd = cost_usd

    def set_circuit_state(self, state: str) -> None:
        """Set circuit breaker state."""
        self._log.circuit_state = state

    def success(self) -> None:
        """Log successful request."""
        self._finalize("success")
        self._logger.info(
            f"Completion: {self._log.model} via {self._log.provider}",
            extra=self._log.to_dict(),
        )

    def error(self, exc: Exception, error_type: Optional[str] = None) -> None:
        """Log failed request."""
        self._log.error = str(exc)
        self._log.error_type = error_type or type(exc).__name__
        self._finalize("error")
        self._logger.warning(
            f"Completion failed: {self._log.model} - {self._log.error}",
            extra=self._log.to_dict(),
        )

    def rate_limited(self) -> None:
        """Log rate-limited request."""
        self._finalize("rate_limited")
        self._logger.info(
            f"Rate limited: {self._log.team_id}",
            extra=self._log.to_dict(),
        )

    def budget_exceeded(self) -> None:
        """Log budget-exceeded request."""
        self._finalize("budget_exceeded")
        self._logger.info(
            f"Budget exceeded: {self._log.team_id}",
            extra=self._log.to_dict(),
        )

    def _finalize(self, status: str) -> None:
        """Calculate latency and set status."""
        self._log.latency_ms = (time.time() - self._start_time) * 1000
        self._log.status = status


def setup_logging(json_format: bool = True, level: str = "INFO") -> None:
    """Configure application logging.

    Args:
        json_format: If True, use JSON formatter. False for human-readable.
        level: Log level (DEBUG, INFO, WARNING, ERROR).
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, level.upper()))

    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )

    root_logger.addHandler(handler)

    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
