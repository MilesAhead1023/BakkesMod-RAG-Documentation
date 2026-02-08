"""Golden query regression tests.

A curated set of BakkesMod development questions with expected terms
in the answer. These serve as a quality gate: if retrieval or generation
quality degrades, these tests will catch it.

Run with: pytest tests/test_golden_queries.py -v -m integration
"""

import os
import pytest

pytestmark = pytest.mark.integration


def _has_api_keys() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture(scope="module")
def engine():
    if not _has_api_keys():
        pytest.skip("No API keys available for integration tests")
    from bakkesmod_rag import RAGEngine
    return RAGEngine()


# Each golden query: (question, list_of_expected_terms, min_confidence)
GOLDEN_QUERIES = [
    (
        "What is BakkesMod?",
        ["Rocket League", "plugin", "mod"],
        0.40,
    ),
    (
        "How do I create a BakkesMod plugin?",
        ["onLoad", "BakkesModPlugin"],
        0.40,
    ),
    (
        "How do I hook the goal scored event?",
        ["HookEvent", "goal"],
        0.30,
    ),
    (
        "What is GameWrapper?",
        ["gameWrapper", "game"],
        0.30,
    ),
    (
        "How do I access the player's car?",
        ["CarWrapper", "car"],
        0.30,
    ),
    (
        "How do I use ImGui for settings?",
        ["ImGui"],
        0.30,
    ),
    (
        "How do I register a console variable?",
        ["registerCvar", "cvar"],
        0.30,
    ),
    (
        "What is ServerWrapper?",
        ["ServerWrapper", "server"],
        0.30,
    ),
    (
        "How do I draw on the screen?",
        ["canvas", "draw"],
        0.30,
    ),
    (
        "How do I log messages in a plugin?",
        ["LOG", "log"],
        0.30,
    ),
    (
        "What events can I hook in BakkesMod?",
        ["HookEvent", "event"],
        0.30,
    ),
    (
        "How do I get ball velocity?",
        ["BallWrapper", "velocity"],
        0.30,
    ),
    (
        "What is the plugin lifecycle?",
        ["onLoad", "onUnload"],
        0.30,
    ),
    (
        "How do I use HookEventWithCallerPost?",
        ["HookEventWithCallerPost", "caller"],
        0.25,
    ),
    (
        "How do I create a plugin window?",
        ["PluginWindow", "window"],
        0.25,
    ),
]


class TestGoldenQueries:
    @pytest.mark.parametrize(
        "question,expected_terms,min_confidence",
        GOLDEN_QUERIES,
        ids=[q[0][:40] for q in GOLDEN_QUERIES],
    )
    def test_golden_query(self, engine, question, expected_terms, min_confidence):
        """Each golden query must contain expected terms and meet confidence threshold."""
        result = engine.query(question, use_cache=False)

        answer_lower = result.answer.lower()

        # Check at least one expected term appears
        found = [
            term for term in expected_terms
            if term.lower() in answer_lower
        ]
        assert found, (
            f"None of {expected_terms} found in answer for: {question}\n"
            f"Answer (first 200 chars): {result.answer[:200]}"
        )

        # Check confidence meets minimum
        assert result.confidence >= min_confidence, (
            f"Confidence {result.confidence:.2f} < {min_confidence} for: {question}"
        )

        # Check we got sources
        assert len(result.sources) > 0, f"No sources for: {question}"
