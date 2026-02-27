"""
Intent Router
=============
Classifies user queries into intent categories to route them to the
correct handler automatically.

Rule-based detection runs at zero API cost and handles the vast majority
of queries.  Optional LLM confirmation activates only for borderline cases
above a configurable confidence threshold.

Intent hierarchy:
  QUESTION         -- pure Q&A (default)
  CODE_GENERATION  -- user wants plugin code written
  HYBRID           -- question + code example wanted
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

logger = logging.getLogger("bakkesmod_rag.intent_router")


class Intent(Enum):
    """Possible routing destinations for a user query."""
    QUESTION = "question"
    CODE_GENERATION = "code_generation"
    HYBRID = "hybrid"


@dataclass
class IntentResult:
    """Outcome of intent classification.

    Attributes:
        intent: The detected intent category.
        confidence: Confidence score in [0, 1].
        routing_reason: Human-readable explanation of why this intent was chosen.
    """
    intent: Intent
    confidence: float
    routing_reason: str


# ---------------------------------------------------------------------------
# Keyword sets for rule-based detection
# ---------------------------------------------------------------------------

_CODE_GEN_PHRASES: list[str] = [
    "write a plugin",
    "generate code",
    "create a plugin",
    "implement a plugin",
    "make me a plugin",
    "scaffold",
    "boilerplate",
    "write me a plugin",
    "generate a plugin",
    "create plugin",
    "make a plugin",
    "write plugin",
    "code a plugin",
    "build a plugin",
    "build me a plugin",
]

_HYBRID_HOW_WORDS: list[str] = [
    "how do i",
    "how to",
    "show me",
]

_HYBRID_EXAMPLE_WORDS: list[str] = [
    "example",
    "with code",
    "sample code",
    "code snippet",
    "snippet",
    "demonstrate",
]


def _normalise(query: str) -> str:
    """Lowercase and collapse whitespace for pattern matching."""
    return re.sub(r"\s+", " ", query.strip().lower())


def _matches_any(text: str, phrases: list[str]) -> bool:
    """Check if any phrase is found as a substring."""
    return any(p in text for p in phrases)


class IntentRouter:
    """Rule-based + optional LLM intent classifier.

    Usage::

        router = IntentRouter(llm=my_llm, config=cfg.intent_router)
        result = router.classify("Write me a boost meter plugin")
        # IntentResult(intent=Intent.CODE_GENERATION, confidence=0.95, ...)
    """

    def __init__(self, llm=None, config=None):
        """Initialise the router.

        Args:
            llm: Optional LLM for low-confidence confirmation calls.
            config: IntentRouterConfig (uses defaults if None).
        """
        self._llm = llm
        if config is None:
            from bakkesmod_rag.config import IntentRouterConfig
            config = IntentRouterConfig()
        self._config = config

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def classify(self, query: str) -> IntentResult:
        """Classify a user query into an intent category.

        Always fails open: if any error occurs, returns QUESTION intent.

        Args:
            query: Raw user query string.

        Returns:
            IntentResult with intent, confidence, and routing_reason.
        """
        try:
            return self._classify_safe(query)
        except Exception as e:
            logger.warning("Intent router error (failing open to QUESTION): %s", e)
            return IntentResult(
                intent=Intent.QUESTION,
                confidence=0.5,
                routing_reason=f"Router error — defaulting to QUESTION: {e}",
            )

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _classify_safe(self, query: str) -> IntentResult:
        """Run classification without exception handling."""
        norm = _normalise(query)

        # 1. Check for CODE_GENERATION triggers (highest priority)
        if _matches_any(norm, _CODE_GEN_PHRASES):
            return IntentResult(
                intent=Intent.CODE_GENERATION,
                confidence=0.95,
                routing_reason="Matched code generation keyword phrase",
            )

        # 2. Check for HYBRID (how-do-I + example)
        has_how = _matches_any(norm, _HYBRID_HOW_WORDS)
        has_example = _matches_any(norm, _HYBRID_EXAMPLE_WORDS)
        if has_how and has_example:
            return IntentResult(
                intent=Intent.HYBRID,
                confidence=0.85,
                routing_reason="Matched how-to + example pattern (HYBRID)",
            )

        # 3. Borderline LLM confirmation
        if self._config.enabled and self._llm is not None:
            llm_result = self._llm_confirm(query, norm)
            if llm_result is not None:
                return llm_result

        # 4. Default: QUESTION
        return IntentResult(
            intent=Intent.QUESTION,
            confidence=0.80,
            routing_reason="No code generation triggers found — routing to Q&A",
        )

    def _llm_confirm(self, query: str, norm: str) -> Optional[IntentResult]:
        """Ask the LLM for clarification on borderline queries.

        Returns None if confidence is high enough to skip, or on any error.
        """
        # Only call LLM for queries that are ambiguous (low rule-based confidence)
        threshold = self._config.llm_confirmation_threshold

        # Heuristic: "code" word without explicit generation phrase
        has_code_word = "code" in norm or "plugin" in norm or "implement" in norm
        if not has_code_word:
            return None  # Clearly a question — skip LLM call

        try:
            prompt = (
                "Classify this query into one of: QUESTION, CODE_GENERATION, HYBRID.\n"
                "QUESTION = user wants an explanation or fact.\n"
                "CODE_GENERATION = user wants code written for them.\n"
                "HYBRID = user wants an explanation with a code example.\n\n"
                f'Query: "{query}"\n\n'
                "Reply with exactly one word: QUESTION, CODE_GENERATION, or HYBRID."
            )
            response = self._llm.complete(prompt)
            raw = response.text.strip().upper()

            if "CODE_GENERATION" in raw or "CODE" in raw:
                return IntentResult(
                    intent=Intent.CODE_GENERATION,
                    confidence=threshold,
                    routing_reason="LLM confirmed CODE_GENERATION intent",
                )
            elif "HYBRID" in raw:
                return IntentResult(
                    intent=Intent.HYBRID,
                    confidence=threshold,
                    routing_reason="LLM confirmed HYBRID intent",
                )
            elif "QUESTION" in raw:
                return IntentResult(
                    intent=Intent.QUESTION,
                    confidence=threshold,
                    routing_reason="LLM confirmed QUESTION intent",
                )
        except Exception as e:
            logger.warning("LLM intent confirmation failed: %s", e)

        return None
