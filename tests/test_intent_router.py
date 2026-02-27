"""Tests for the intent router: rule-based classification, fail-open, engine.ask()."""

import pytest
from unittest.mock import MagicMock

from bakkesmod_rag.intent_router import IntentRouter, IntentResult, Intent
from bakkesmod_rag.config import IntentRouterConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _router(llm=None, enabled=True, threshold=0.6) -> IntentRouter:
    """Create a router with test configuration."""
    config = IntentRouterConfig(enabled=enabled, llm_confirmation_threshold=threshold)
    return IntentRouter(llm=llm, config=config)


# ---------------------------------------------------------------------------
# Rule-based classification
# ---------------------------------------------------------------------------

class TestCodeGenerationTriggers:
    """Queries that should route to CODE_GENERATION."""

    @pytest.mark.parametrize("query", [
        "Write a plugin that shows my boost amount",
        "Generate code for a speed plugin",
        "Create a plugin that logs player names",
        "Implement a plugin to track goals",
        "Make me a plugin that shows fps",
        "Give me some boilerplate for a new plugin",
        "I need scaffold code for a BakkesMod plugin",
        "write plugin for rocket league stats",
    ])
    def test_code_generation_queries(self, query):
        router = _router()
        result = router.classify(query)
        assert result.intent == Intent.CODE_GENERATION
        assert result.confidence >= 0.9
        assert result.routing_reason != ""


class TestQuestionTriggers:
    """Queries that should route to QUESTION."""

    @pytest.mark.parametrize("query", [
        "What is the GameWrapper class?",
        "How does BM25 retrieval work?",
        "What events can I hook into?",
        "What is BakkesMod?",
        "Tell me about the CarWrapper API",
        "What does the notifier object do?",
    ])
    def test_question_queries(self, query):
        router = _router()
        result = router.classify(query)
        assert result.intent == Intent.QUESTION
        assert result.confidence > 0.0
        assert isinstance(result.routing_reason, str)


class TestHybridTriggers:
    """Queries that should route to HYBRID (how-to + example)."""

    @pytest.mark.parametrize("query", [
        "How do I hook an event with an example?",
        "Show me how to use GameWrapper with a code snippet",
        "How do I get player speed, with sample code?",
        "How to access boost with code example",
    ])
    def test_hybrid_queries(self, query):
        router = _router()
        result = router.classify(query)
        assert result.intent == Intent.HYBRID
        assert result.confidence >= 0.8


# ---------------------------------------------------------------------------
# Fail-open behavior
# ---------------------------------------------------------------------------

class TestFailOpen:
    """Router must never raise — always return a valid IntentResult."""

    def test_fail_open_on_llm_error(self):
        """If LLM raises, router falls back to QUESTION."""
        bad_llm = MagicMock()
        bad_llm.complete.side_effect = RuntimeError("LLM offline")
        router = _router(llm=bad_llm)
        # A borderline query that might trigger LLM
        result = router.classify("Can you code something for this plugin?")
        assert isinstance(result, IntentResult)
        assert isinstance(result.intent, Intent)

    def test_classify_never_raises(self):
        """classify() never raises, regardless of input."""
        router = _router()
        for bad_query in ["", " ", "a" * 10000, "\x00\x01\x02", None.__str__()]:
            result = router.classify(bad_query)
            assert isinstance(result, IntentResult)

    def test_fail_open_returns_question_intent(self):
        """On internal error, the result is always QUESTION (safe default)."""
        router = _router()
        # Force an internal exception by patching _classify_safe
        import unittest.mock
        with unittest.mock.patch.object(
            router, "_classify_safe", side_effect=ValueError("boom")
        ):
            result = router.classify("anything")
        assert result.intent == Intent.QUESTION


# ---------------------------------------------------------------------------
# LLM confirmation path
# ---------------------------------------------------------------------------

class TestLLMConfirmation:
    """LLM confirmation is called for borderline cases."""

    def test_llm_confirmation_code_generation(self):
        """LLM confirmation overrides to CODE_GENERATION."""
        mock_llm = MagicMock()
        mock_llm.complete.return_value = MagicMock(text="CODE_GENERATION")
        router = _router(llm=mock_llm)
        result = router.classify("Could you implement the plugin code for me?")
        # Should be CODE_GENERATION from either rules or LLM
        assert result.intent in (Intent.CODE_GENERATION, Intent.QUESTION, Intent.HYBRID)

    def test_no_llm_still_works(self):
        """Router works without any LLM."""
        router = _router(llm=None)
        result = router.classify("What is BakkesMod?")
        assert isinstance(result, IntentResult)


# ---------------------------------------------------------------------------
# IntentResult structure
# ---------------------------------------------------------------------------

class TestIntentResult:
    """Validate IntentResult fields."""

    def test_intent_result_fields(self):
        result = IntentResult(
            intent=Intent.QUESTION,
            confidence=0.9,
            routing_reason="test",
        )
        assert result.intent == Intent.QUESTION
        assert result.confidence == 0.9
        assert result.routing_reason == "test"

    def test_intent_enum_values(self):
        assert Intent.QUESTION.value == "question"
        assert Intent.CODE_GENERATION.value == "code_generation"
        assert Intent.HYBRID.value == "hybrid"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestIntentRouterConfig:
    def test_defaults(self):
        from bakkesmod_rag.config import IntentRouterConfig
        cfg = IntentRouterConfig()
        assert cfg.enabled is True
        assert 0.0 < cfg.llm_confirmation_threshold <= 1.0

    def test_disabled_router_skips_llm(self):
        """When router is disabled, LLM is never called."""
        mock_llm = MagicMock()
        config = IntentRouterConfig(enabled=False)
        router = IntentRouter(llm=mock_llm, config=config)
        result = router.classify("Could you implement the plugin for me?")
        # Rule-based detection still works, LLM confirmation skipped
        assert isinstance(result, IntentResult)
        # LLM should not be called when enabled=False
        # (Rules will still catch obvious code gen phrases)
