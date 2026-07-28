"""Prompt compression.

Reduces the size of prompts *before* they reach the provider, so every team
gets consistent token savings without changing their client code. This is the
gateway's job, not the client's — see interview angle #7 in docs/PLAN.md.

Design: the actual compression is behind a `Compressor` Protocol so the
strategy is pluggable. We ship a conservative rule-based implementation now;
a model-based one (LLMLingua-style) can drop in later by implementing the same
Protocol and registering it in `_build_compressor`.
"""

import re
from dataclasses import dataclass
from typing import List, Protocol

from llm_gateway.core.schemas import CompressionConfig
from llm_gateway.models.llm import Message

# Rough char→token heuristic. Real tokenization needs the model's tokenizer
# (e.g. tiktoken for OpenAI), but that's a per-provider dependency and we only
# need an estimate for metrics/logging, not for billing. Billing uses the
# provider's reported usage. ~4 chars/token holds for English prose.
CHARS_PER_TOKEN = 4


@dataclass(frozen=True)
class CompressionResult:
    """Outcome of a compression pass.

    Immutable: holds the (possibly) new message list plus before/after sizes
    so callers can log/meter savings without recomputing.
    """

    messages: List[Message]
    original_chars: int
    compressed_chars: int
    was_compressed: bool

    @property
    def chars_saved(self) -> int:
        return self.original_chars - self.compressed_chars

    @property
    def est_tokens_saved(self) -> int:
        return self.chars_saved // CHARS_PER_TOKEN

    @property
    def ratio(self) -> float:
        """Fraction of original size removed (0.0 = nothing, 1.0 = everything)."""
        if self.original_chars == 0:
            return 0.0
        return self.chars_saved / self.original_chars


class Compressor(Protocol):
    """A pluggable compression strategy.

    Takes a list of messages, returns a new list with compressed content.
    Must NOT mutate the input (immutability — see coding-style rules).
    """

    name: str

    def compress(self, messages: List[Message]) -> List[Message]: ...


# Trailing whitespace on each line, and runs of 3+ newlines.
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_EXTRA_BLANK_LINES = re.compile(r"\n{3,}")


class RuleBasedCompressor:
    """Conservative, deterministic whitespace normalization.

    Strips trailing whitespace per line, collapses 3+ blank lines into one
    blank line, and trims the whole message. We deliberately do NOT collapse
    internal single spaces or reflow text — that would corrupt code blocks and
    whitespace-sensitive content (Markdown, YAML, diffs). The savings are
    modest but the risk is near zero, which is the right trade for a gateway
    applying this to every team's traffic blind.

    Tradeoff: low compression ratio. A model-based compressor (LLMLingua)
    achieves 2-5x but adds latency, a model dependency, and quality risk.
    """

    name = "rule_based"

    def compress(self, messages: List[Message]) -> List[Message]:
        return [
            m.model_copy(update={"content": self._clean(m.content)})
            for m in messages
        ]

    def _clean(self, text: str) -> str:
        text = _TRAILING_WS.sub("", text)
        text = _EXTRA_BLANK_LINES.sub("\n\n", text)
        return text.strip()


def _build_compressor(strategy: str) -> Compressor:
    """Map a strategy name to its implementation.

    Add new strategies (e.g. "llmlingua") here once implemented.
    """
    if strategy == "rule_based":
        return RuleBasedCompressor()
    raise ValueError(f"Unknown compression strategy: {strategy!r}")


def _total_chars(messages: List[Message]) -> int:
    return sum(len(m.content) for m in messages)


class PromptCompressor:
    """Orchestrates compression for the request pipeline.

    Decides *whether* to compress (config + min-size gate), delegates the
    *how* to the configured Compressor, and reports the savings. A no-op when
    disabled, so it's safe to always call from the hot path.
    """

    def __init__(self, config: CompressionConfig) -> None:
        self._config = config
        self._impl: Compressor = _build_compressor(config.strategy)

    def compress(self, messages: List[Message]) -> CompressionResult:
        original = _total_chars(messages)

        # Skip when disabled or the prompt is too small to be worth touching.
        # Tiny prompts can't save meaningful tokens and aren't worth the risk.
        if not self._config.enabled or original < self._config.min_chars:
            return CompressionResult(
                messages=messages,
                original_chars=original,
                compressed_chars=original,
                was_compressed=False,
            )

        compressed_messages = self._impl.compress(messages)
        compressed = _total_chars(compressed_messages)

        return CompressionResult(
            messages=compressed_messages,
            original_chars=original,
            compressed_chars=compressed,
            was_compressed=compressed < original,
        )


# Singleton wired from config, same pattern as rate_limiter / circuit_breaker.
from llm_gateway.core.config import config  # noqa: E402

prompt_compressor = PromptCompressor(config.compression)
