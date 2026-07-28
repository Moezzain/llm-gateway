"""Tests for prompt compression."""

import pytest

from llm_gateway.core.compression import (
    PromptCompressor,
    RuleBasedCompressor,
)
from llm_gateway.core.schemas import CompressionConfig
from llm_gateway.models.llm import Message, Role


def _msg(content: str, role: Role = Role.USER) -> Message:
    return Message(role=role, content=content)


@pytest.mark.unit
class TestRuleBasedCompressor:
    def test_strips_trailing_whitespace_per_line(self) -> None:
        c = RuleBasedCompressor()
        result = c.compress([_msg("line one   \nline two\t\n")])
        assert result[0].content == "line one\nline two"

    def test_collapses_extra_blank_lines(self) -> None:
        c = RuleBasedCompressor()
        result = c.compress([_msg("a\n\n\n\n\nb")])
        assert result[0].content == "a\n\nb"

    def test_trims_surrounding_whitespace(self) -> None:
        c = RuleBasedCompressor()
        result = c.compress([_msg("   hello   ")])
        assert result[0].content == "hello"

    def test_preserves_internal_single_spaces(self) -> None:
        # Must NOT reflow text — would corrupt code/markdown.
        c = RuleBasedCompressor()
        text = "def f(x):\n    return x + 1"
        result = c.compress([_msg(text)])
        assert result[0].content == text

    def test_does_not_mutate_input(self) -> None:
        c = RuleBasedCompressor()
        original = _msg("padded   ")
        c.compress([original])
        assert original.content == "padded   "


@pytest.mark.unit
class TestPromptCompressor:
    def test_disabled_is_passthrough(self) -> None:
        comp = PromptCompressor(CompressionConfig(enabled=False))
        msgs = [_msg("lots of   trailing space   " * 20)]
        result = comp.compress(msgs)
        assert result.was_compressed is False
        assert result.messages is msgs

    def test_skips_prompts_below_min_chars(self) -> None:
        comp = PromptCompressor(CompressionConfig(enabled=True, min_chars=100))
        result = comp.compress([_msg("short   ")])  # under 100 chars
        assert result.was_compressed is False

    def test_compresses_when_enabled_and_large_enough(self) -> None:
        comp = PromptCompressor(CompressionConfig(enabled=True, min_chars=10))
        msgs = [_msg("padding line   \n\n\n\n" * 10)]
        result = comp.compress(msgs)
        assert result.was_compressed is True
        assert result.compressed_chars < result.original_chars
        assert result.chars_saved > 0

    def test_no_savings_reported_as_not_compressed(self) -> None:
        comp = PromptCompressor(CompressionConfig(enabled=True, min_chars=0))
        # Already-clean content: nothing to strip.
        clean = "already clean content with no extra whitespace at all here"
        result = comp.compress([_msg(clean)])
        assert result.was_compressed is False
        assert result.chars_saved == 0

    def test_ratio_and_token_estimate(self) -> None:
        comp = PromptCompressor(CompressionConfig(enabled=True, min_chars=0))
        result = comp.compress([_msg("x" + " " * 40 + "\n")])  # trailing ws stripped
        assert 0.0 < result.ratio <= 1.0
        assert result.est_tokens_saved == result.chars_saved // 4

    def test_empty_messages_safe(self) -> None:
        comp = PromptCompressor(CompressionConfig(enabled=True, min_chars=0))
        result = comp.compress([])
        assert result.was_compressed is False
        assert result.ratio == 0.0
