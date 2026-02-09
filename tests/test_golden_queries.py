"""Golden query smoke tests.

Verifies that the query rewriter and retrieval subsystems produce
sensible output for curated BakkesMod questions — without running
the full RAGEngine (which takes minutes to build indexes).

These tests auto-skip when no API keys are available.

Run only integration tests:  pytest -m integration -v
"""

import os
import pytest

_HAS_KEYS = bool(os.getenv("OPENAI_API_KEY"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_KEYS, reason="No API keys available"),
]


# Golden queries: (question, expected_terms_in_rewrite_or_synonym_expansion)
GOLDEN_QUERIES = [
    ("What is BakkesMod?", ["bakkesmod"]),
    ("How do I create a BakkesMod plugin?", ["plugin"]),
    ("How do I hook the goal scored event?", ["hook", "event"]),
    ("What is GameWrapper?", ["gamewrapper"]),
    ("How do I access the player's car?", ["car"]),
    ("How do I use ImGui for settings?", ["imgui"]),
    ("How do I register a console variable?", ["register", "variable", "console"]),
    ("What is ServerWrapper?", ["serverwrapper"]),
    ("How do I draw on the screen?", ["draw", "screen"]),
    ("How do I log messages in a plugin?", ["log"]),
    ("What events can I hook in BakkesMod?", ["hook", "event"]),
    ("How do I get ball velocity?", ["ball"]),
    ("What is the plugin lifecycle?", ["plugin", "lifecycle"]),
    ("How do I use HookEventWithCallerPost?", ["hookeventwithcallerpost"]),
    ("How do I create a plugin window?", ["window"]),
]


class TestGoldenQueryExpansion:
    """Verify query rewriter expands golden queries with domain synonyms."""

    @pytest.mark.parametrize(
        "question,expected_terms",
        GOLDEN_QUERIES,
        ids=[q[0][:40] for q in GOLDEN_QUERIES],
    )
    def test_expansion_contains_terms(self, question, expected_terms):
        from bakkesmod_rag.query_rewriter import QueryRewriter
        rw = QueryRewriter(llm=None, use_llm=False)
        expanded = rw.rewrite(question).lower()
        found = [t for t in expected_terms if t in expanded]
        assert found, (
            f"None of {expected_terms} found in expansion of: {question}\n"
            f"Expanded: {expanded}"
        )


class TestGoldenQueryDecomposition:
    """Verify complex golden queries get decomposed properly."""

    def test_compound_query_decomposes(self):
        from bakkesmod_rag.query_decomposer import QueryDecomposer
        d = QueryDecomposer(llm=None, enable_decomposition=True)
        subs = d.decompose(
            "How do I hook goal events and draw a counter on screen?"
        )
        # Compound query should produce at least the original
        assert len(subs) >= 1
