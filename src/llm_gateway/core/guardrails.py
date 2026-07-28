"""Input guardrails — prompt injection detection.

Runs at the gateway, before any provider call, so org-wide input policy is
enforced in one place instead of trusting every client. See interview angle
#8/#9 in docs/PLAN.md: defense in depth, block injection before it reaches an
expensive (and potentially jailbreakable) model.

Threat model: a client (or an end-user whose text a client forwards) tries to
override instructions, exfiltrate a system prompt, or jailbreak the model.
We pattern-match known attack shapes. This is a first line, NOT a complete
defense — pattern matching catches the obvious attempts and raises the bar.
Sophisticated/obfuscated attacks need an ML classifier (noted as the upgrade
path), which is why detection sits behind a Protocol seam.

Two modes (config):
- "block": reject the request (400). Enforcing.
- "flag":  allow but log + meter. Shadow mode — run it in prod to measure
           false positives before you flip to blocking.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Protocol

from llm_gateway.core.schemas import GuardrailsConfig
from llm_gateway.models.llm import Message


class GuardrailMode(str, Enum):
    BLOCK = "block"
    FLAG = "flag"


@dataclass(frozen=True)
class Rule:
    """A single named detection pattern."""

    name: str
    pattern: "re.Pattern[str]"
    description: str


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of scanning one request.

    Immutable. `triggered` is False when nothing matched.
    """

    triggered: bool
    rule_name: Optional[str] = None
    reason: Optional[str] = None
    matched_text: Optional[str] = None


def _rule(name: str, pattern: str, description: str) -> Rule:
    return Rule(name=name, pattern=re.compile(pattern, re.IGNORECASE), description=description)


# Curated prompt-injection signatures. Each is deliberately specific to keep
# false positives low — a guardrail that blocks legitimate traffic is worse
# than useless because teams route around it.
DEFAULT_RULES: List[Rule] = [
    _rule(
        "ignore_previous",
        r"ignore\s+(?:all\s+|the\s+|your\s+|any\s+|these\s+)*(?:previous|prior|above|preceding|earlier)\s+(?:instructions?|prompts?|messages?|rules?|directions?)",
        "Attempt to override earlier instructions",
    ),
    _rule(
        "disregard_instructions",
        r"disregard\s+(?:all\s+|the\s+|your\s+|any\s+)*(?:previous|prior|above)\s+(?:instructions?|prompts?|rules?)",
        "Attempt to discard prior instructions",
    ),
    _rule(
        "forget_instructions",
        r"forget\s+(?:everything|all\s+(?:previous|prior)|your\s+(?:instructions?|rules?|training))",
        "Attempt to reset model context",
    ),
    _rule(
        "reveal_system_prompt",
        r"(?:reveal|print|repeat|show|output|display|tell\s+me)\s+(?:your|the)\s+(?:system\s+)?(?:prompt|instructions?|rules?)",
        "Attempt to exfiltrate the system prompt",
    ),
    _rule(
        "repeat_above",
        r"repeat\s+(?:the\s+)?(?:words?|text|everything)\s+above",
        "Attempt to dump preceding context",
    ),
    _rule(
        "do_anything_now",
        r"\bdo\s+anything\s+now\b|\bDAN\s+mode\b",
        "Classic DAN jailbreak",
    ),
    _rule(
        "developer_mode",
        r"\b(?:enable\s+)?developer\s+mode\b",
        "Fake 'developer mode' jailbreak",
    ),
    _rule(
        "pretend_unrestricted",
        r"(?:pretend|act\s+as)\s+(?:if\s+)?(?:you\s+are\s+|to\s+be\s+)?(?:a\s+|an\s+)?(?:jailbroken|unrestricted|uncensored|unfiltered|amoral)",
        "Attempt to assume an unrestricted persona",
    ),
    _rule(
        "fake_role_delimiter",
        r"<\|?(?:im_start|im_end|system)\|?>|(?:^|\n)\s*#{1,4}\s*system\b",
        "Injected chat/role delimiter to spoof a system turn",
    ),
]


class InjectionScanner(Protocol):
    """Pluggable input-scanning strategy.

    Implement this with an ML classifier later; the engine doesn't care how
    detection works, only that it returns a GuardrailResult.
    """

    def scan(self, messages: List[Message]) -> GuardrailResult: ...


class PatternScanner:
    """Regex-based prompt-injection detector.

    Scans every message's content (all roles are client-supplied here, so all
    are untrusted) and returns the first rule that fires.
    """

    name = "pattern"

    def __init__(self, rules: Optional[List[Rule]] = None) -> None:
        self._rules = rules if rules is not None else DEFAULT_RULES

    def scan(self, messages: List[Message]) -> GuardrailResult:
        for message in messages:
            for rule in self._rules:
                match = rule.pattern.search(message.content)
                if match:
                    return GuardrailResult(
                        triggered=True,
                        rule_name=rule.name,
                        reason=rule.description,
                        matched_text=match.group(0)[:120],
                    )
        return GuardrailResult(triggered=False)


class GuardrailEngine:
    """Applies input guardrails per config.

    No-op when disabled, so it's safe to always call from the hot path.
    `check()` returns a result; the caller decides what to do with it based
    on `should_block`.
    """

    def __init__(self, config: GuardrailsConfig) -> None:
        self._config = config
        self._scanner: InjectionScanner = PatternScanner()

    @property
    def mode(self) -> GuardrailMode:
        return GuardrailMode(self._config.mode)

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def check(self, messages: List[Message]) -> GuardrailResult:
        if not self._config.enabled:
            return GuardrailResult(triggered=False)
        return self._scanner.scan(messages)

    def should_block(self, result: GuardrailResult) -> bool:
        """True only when a rule fired AND we're enforcing (block mode)."""
        return result.triggered and self.mode is GuardrailMode.BLOCK


# Singleton wired from config, same pattern as the other core services.
from llm_gateway.core.config import config  # noqa: E402

input_guardrail = GuardrailEngine(config.guardrails)
