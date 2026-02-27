"""
Input Guardrails
================
Pre-query validation and sanitization layer.

Validates user inputs before they reach the retrieval pipeline.
All checks are fail-open: any internal error returns passed=True so
the query continues rather than being silently dropped.

Checks performed (all configurable):
  1. Length check   — min 3 chars, max 1000 chars
  2. Sanitization   — strip whitespace, normalize spaces
  3. Empty rejection — blank-only queries are rejected
  4. Off-topic warn  — loose BakkesMod domain keyword check (warn only, never block)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("bakkesmod_rag.guardrails")

# Loose keyword set for BakkesMod domain relevance check (warn only, never block)
_DOMAIN_KEYWORDS: list[str] = [
    "bakkesmod", "bakkes", "plugin", "rocket league", "rl",
    "hook", "event", "gamewrapper", "carwrapper", "boost",
    "notifier", "sdk", "cpp", "c++", "code", "api",
    "function", "method", "class", "variable", "cvargWrapper",
    "ball", "car", "player", "score", "goal", "speed",
    "how", "what", "why", "when", "where", "which",
    "write", "create", "generate", "make", "implement",
]


@dataclass
class GuardrailResult:
    """Result of a guardrail check.

    Attributes:
        passed: Whether the query passed all hard checks.
        reason: Human-readable explanation of any rejection (or warning).
        sanitized_query: The query after whitespace cleanup.
    """
    passed: bool
    reason: str
    sanitized_query: str


class InputGuardrail:
    """Pre-query validation and sanitization.

    All checks fail-open: any internal error returns passed=True.

    Usage::

        guardrail = InputGuardrail(config=cfg.guardrails)
        result = guardrail.check(user_input)
        if not result.passed:
            return f"Error: {result.reason}"
        query = result.sanitized_query  # use the sanitized version
    """

    def __init__(self, config=None):
        """Initialise guardrails.

        Args:
            config: GuardrailConfig (uses defaults if None).
        """
        if config is None:
            from bakkesmod_rag.config import GuardrailConfig
            config = GuardrailConfig()
        self._config = config

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def check(self, query: str) -> GuardrailResult:
        """Run all guardrail checks on the input query.

        Always fails open: returns passed=True on any internal error.

        Args:
            query: Raw user input string.

        Returns:
            GuardrailResult with passed flag, reason, and sanitized query.
        """
        if not self._config.enabled:
            sanitized = self._sanitize(query)
            return GuardrailResult(
                passed=True,
                reason="Guardrails disabled",
                sanitized_query=sanitized,
            )

        try:
            return self._check_safe(query)
        except Exception as e:
            logger.warning("Guardrail check error (failing open): %s", e)
            # Fail open: do not block the query on internal error
            return GuardrailResult(
                passed=True,
                reason=f"Guardrail check error (passed through): {e}",
                sanitized_query=str(query),
            )

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _sanitize(self, query: str) -> str:
        """Strip leading/trailing whitespace and normalize multiple spaces."""
        return re.sub(r" {2,}", " ", str(query).strip())

    def _check_safe(self, query: str) -> GuardrailResult:
        """Run all checks without exception handling."""
        sanitized = self._sanitize(query)

        # 1. Empty / whitespace-only check
        if not sanitized:
            return GuardrailResult(
                passed=False,
                reason="Query is empty or whitespace-only",
                sanitized_query=sanitized,
            )

        # 2. Minimum length check
        if len(sanitized) < self._config.min_length:
            return GuardrailResult(
                passed=False,
                reason=(
                    f"Query too short: {len(sanitized)} chars "
                    f"(minimum {self._config.min_length})"
                ),
                sanitized_query=sanitized,
            )

        # 3. Maximum length check
        if len(sanitized) > self._config.max_length:
            return GuardrailResult(
                passed=False,
                reason=(
                    f"Query too long: {len(sanitized)} chars "
                    f"(maximum {self._config.max_length})"
                ),
                sanitized_query=sanitized,
            )

        # 4. Off-topic domain relevance check (warn only, never block)
        norm = sanitized.lower()
        on_topic = any(kw in norm for kw in _DOMAIN_KEYWORDS)
        if not on_topic:
            # Warn but still pass — the user might be asking about something
            # adjacent to BakkesMod or using non-standard terminology
            logger.info(
                "Query may be off-topic for BakkesMod domain: %s...",
                sanitized[:60],
            )
            return GuardrailResult(
                passed=True,
                reason="Query passed (off-topic warning: may not be BakkesMod-related)",
                sanitized_query=sanitized,
            )

        return GuardrailResult(
            passed=True,
            reason="All guardrail checks passed",
            sanitized_query=sanitized,
        )
