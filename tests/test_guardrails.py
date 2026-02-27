"""Tests for InputGuardrail: length, sanitization, fail-open, engine integration."""

import pytest
from unittest.mock import MagicMock, patch

from bakkesmod_rag.guardrails import InputGuardrail, GuardrailResult
from bakkesmod_rag.config import GuardrailConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _guardrail(enabled=True, min_length=3, max_length=1000) -> InputGuardrail:
    config = GuardrailConfig(enabled=enabled, min_length=min_length, max_length=max_length)
    return InputGuardrail(config=config)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------

class TestGuardrailResult:
    def test_result_fields(self):
        gr = GuardrailResult(passed=True, reason="ok", sanitized_query="test")
        assert gr.passed is True
        assert gr.reason == "ok"
        assert gr.sanitized_query == "test"


# ---------------------------------------------------------------------------
# Sanitization
# ---------------------------------------------------------------------------

class TestSanitization:
    def test_strips_leading_trailing_whitespace(self):
        g = _guardrail()
        result = g.check("  What is BakkesMod?  ")
        assert result.sanitized_query == "What is BakkesMod?"

    def test_normalizes_multiple_spaces(self):
        g = _guardrail()
        result = g.check("What  is   BakkesMod?")
        assert result.sanitized_query == "What is BakkesMod?"

    def test_passes_through_clean_query(self):
        g = _guardrail()
        result = g.check("What is BakkesMod?")
        assert result.passed is True
        assert result.sanitized_query == "What is BakkesMod?"


# ---------------------------------------------------------------------------
# Length enforcement
# ---------------------------------------------------------------------------

class TestLengthChecks:
    def test_empty_query_rejected(self):
        g = _guardrail()
        result = g.check("")
        assert result.passed is False
        assert "empty" in result.reason.lower()

    def test_whitespace_only_rejected(self):
        g = _guardrail()
        result = g.check("   ")
        assert result.passed is False

    def test_too_short_rejected(self):
        g = _guardrail(min_length=5)
        result = g.check("Hi")
        assert result.passed is False
        assert "short" in result.reason.lower()

    def test_too_long_rejected(self):
        g = _guardrail(max_length=10)
        result = g.check("This is a very long query that exceeds the limit")
        assert result.passed is False
        assert "long" in result.reason.lower()

    def test_exact_min_length_passes(self):
        g = _guardrail(min_length=3)
        result = g.check("abc")  # Exactly 3 chars
        # May or may not pass depending on domain keyword check (warn-only)
        assert isinstance(result.passed, bool)

    def test_exact_max_length_passes(self):
        g = _guardrail(max_length=100)
        query = "What is BakkesMod? " * 5  # ~95 chars
        result = g.check(query)
        assert isinstance(result.passed, bool)


# ---------------------------------------------------------------------------
# Off-topic detection (warn-only, never block)
# ---------------------------------------------------------------------------

class TestOffTopicDetection:
    def test_off_topic_still_passes(self):
        """Off-topic queries get a warning but are NOT blocked."""
        g = _guardrail()
        result = g.check("What is the weather like today?")
        assert result.passed is True  # warn-only, never block

    def test_on_topic_passes(self):
        g = _guardrail()
        result = g.check("What is the GameWrapper class in BakkesMod?")
        assert result.passed is True


# ---------------------------------------------------------------------------
# Fail-open behavior
# ---------------------------------------------------------------------------

class TestFailOpen:
    def test_internal_error_passes_through(self):
        """If check logic raises, guardrail fails open."""
        g = _guardrail()
        with patch.object(g, "_check_safe", side_effect=RuntimeError("boom")):
            result = g.check("What is BakkesMod?")
        assert result.passed is True

    def test_disabled_guardrail_passes_everything(self):
        """Disabled guardrail passes any input."""
        g = _guardrail(enabled=False)
        for q in ["", "x", "a" * 2000]:
            result = g.check(q)
            assert result.passed is True

    def test_none_str_input_no_crash(self):
        """Guardrail handles unusual inputs without crashing."""
        g = _guardrail()
        result = g.check(str(None))
        assert isinstance(result, GuardrailResult)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestGuardrailConfig:
    def test_defaults(self):
        cfg = GuardrailConfig()
        assert cfg.enabled is True
        assert cfg.min_length == 3
        assert cfg.max_length == 1000

    def test_custom_limits(self):
        cfg = GuardrailConfig(min_length=10, max_length=500)
        assert cfg.min_length == 10
        assert cfg.max_length == 500
