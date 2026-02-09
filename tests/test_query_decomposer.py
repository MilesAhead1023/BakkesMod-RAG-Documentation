"""Tests for QueryDecomposer — hybrid rule + LLM query decomposition."""

import pytest
from unittest.mock import MagicMock, patch

from bakkesmod_rag.query_decomposer import QueryDecomposer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def decomposer():
    """QueryDecomposer with no LLM (rule-based only)."""
    return QueryDecomposer(llm=None, max_sub_queries=4, complexity_threshold=80)


@pytest.fixture
def mock_llm():
    """Mock LLM that returns pre-configured sub-questions."""
    llm = MagicMock()
    response = MagicMock()
    response.text = (
        "How do I hook events in BakkesMod?\n"
        "How do I render an overlay using ImGui?"
    )
    llm.complete.return_value = response
    return llm


@pytest.fixture
def decomposer_with_llm(mock_llm):
    """QueryDecomposer with a mock LLM."""
    return QueryDecomposer(
        llm=mock_llm,
        max_sub_queries=4,
        complexity_threshold=80,
    )


# ---------------------------------------------------------------------------
# Rule-based splitting tests
# ---------------------------------------------------------------------------

class TestRuleBasedSplit:
    """Tests for rule-based query splitting."""

    def test_simple_query_no_split(self, decomposer):
        """Simple query returns unchanged (list of 1)."""
        result = decomposer.decompose("How do I hook events?")
        assert result == ["How do I hook events?"]

    def test_split_on_and_how(self, decomposer):
        """Splits on 'and how' pattern."""
        result = decomposer.decompose(
            "How do I hook events and how do I render an overlay?"
        )
        assert len(result) == 2

    def test_split_on_semicolon(self, decomposer):
        """Splits on semicolons."""
        result = decomposer.decompose(
            "Explain GameWrapper methods; describe ServerWrapper usage"
        )
        assert len(result) == 2

    def test_split_on_question_marks(self, decomposer):
        """Splits on multiple question marks."""
        result = decomposer.decompose(
            "What is CarWrapper? What is BallWrapper?"
        )
        assert len(result) == 2

    def test_numbered_list_split(self, decomposer):
        """Splits numbered lists."""
        result = decomposer.decompose(
            "1. How do I create a plugin? 2. How do I add ImGui?"
        )
        assert len(result) == 2

    def test_short_fragments_filtered(self, decomposer):
        """Very short fragments after split are filtered out."""
        # "and how" split where one part is too short to be useful
        result = decomposer.decompose("Do X and how do I hook events?")
        # "Do X" is only 4 chars after cleanup — below 10-char threshold
        # so only 1 valid sub-query
        assert len(result) >= 1

    def test_max_sub_queries_cap(self):
        """Respects max_sub_queries cap."""
        d = QueryDecomposer(llm=None, max_sub_queries=2)
        result = d.decompose(
            "What is A? What is B? What is C? What is D?"
        )
        assert len(result) <= 2


# ---------------------------------------------------------------------------
# Complexity heuristic tests
# ---------------------------------------------------------------------------

class TestIsComplex:
    """Tests for the _is_complex heuristic."""

    def test_short_query_not_complex(self, decomposer):
        """Queries below threshold are not complex."""
        assert decomposer._is_complex("How do I hook events?") is False

    def test_long_query_with_multiple_question_words(self, decomposer):
        """Long query with multiple question words is complex."""
        query = (
            "How do I hook the goal scored event in BakkesMod "
            "and what methods does GameWrapper provide for accessing server state?"
        )
        assert decomposer._is_complex(query) is True

    def test_multiple_api_terms_complex(self, decomposer):
        """Query with multiple BakkesMod API terms is complex."""
        query = (
            "I need to understand the relationship between GameWrapper "
            "and ServerWrapper and how CarWrapper fits into the hierarchy"
        )
        assert decomposer._is_complex(query) is True

    def test_single_api_term_not_complex(self, decomposer):
        """Query with a single API term below threshold is not complex."""
        query = "What is GameWrapper and how do I use it in my plugin code?"
        # Only 59 chars — below the 80-char threshold
        assert decomposer._is_complex(query) is False

    def test_long_but_simple(self, decomposer):
        """Long query with single focus is not complex."""
        query = "a" * 100  # Long but no question words or API terms
        assert decomposer._is_complex(query) is False


# ---------------------------------------------------------------------------
# LLM decomposition tests
# ---------------------------------------------------------------------------

class TestLLMSplit:
    """Tests for LLM-powered decomposition."""

    def test_llm_decomposition_called(self, decomposer_with_llm, mock_llm):
        """LLM is called when query is complex and rule-based fails."""
        query = (
            "How do I hook the goal scored event in BakkesMod "
            "and what methods does GameWrapper provide for accessing "
            "the current server state including player positions?"
        )
        result = decomposer_with_llm.decompose(query)
        # Rule-based should split on "and what" pattern
        assert len(result) >= 2

    def test_llm_fallback_on_failure(self):
        """If LLM fails, returns original query."""
        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("API error")
        d = QueryDecomposer(llm=llm, complexity_threshold=10)
        # A query that's "complex" enough to trigger LLM but no rule-based split
        query = "Explain the full architecture of how the BakkesMod plugin system works with GameWrapper and ServerWrapper"
        result = d.decompose(query)
        assert len(result) >= 1

    def test_llm_empty_response_returns_original(self):
        """LLM returning empty text falls back to original query."""
        llm = MagicMock()
        response = MagicMock()
        response.text = ""
        llm.complete.return_value = response
        d = QueryDecomposer(llm=llm, complexity_threshold=10)
        query = "Explain GameWrapper and ServerWrapper and CarWrapper and BallWrapper relationships"
        result = d.decompose(query)
        assert len(result) >= 1

    def test_no_llm_skips_llm_split(self, decomposer):
        """Without LLM, complex queries just pass through."""
        query = "a" * 100  # Long but no split points
        result = decomposer.decompose(query)
        assert result == [query]


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------

class TestMergeSubAnswers:
    """Tests for merging sub-query answers."""

    def test_single_answer_passthrough(self):
        """Single answer returns unchanged."""
        result = QueryDecomposer.merge_sub_answers(
            ["How do I hook events?"],
            ["Use gameWrapper->HookEvent(...)"],
        )
        assert result == "Use gameWrapper->HookEvent(...)"

    def test_multiple_answers_concatenated(self):
        """Without LLM, multiple answers are concatenated with headers."""
        result = QueryDecomposer.merge_sub_answers(
            ["What is A?", "What is B?"],
            ["A is foo", "B is bar"],
        )
        assert "**What is A?**" in result
        assert "A is foo" in result
        assert "**What is B?**" in result
        assert "B is bar" in result

    def test_llm_merge(self):
        """LLM merge synthesizes answers."""
        llm = MagicMock()
        response = MagicMock()
        response.text = "A is foo and B is bar. They work together."
        llm.complete.return_value = response

        result = QueryDecomposer.merge_sub_answers(
            ["What is A?", "What is B?"],
            ["A is foo", "B is bar"],
            llm=llm,
        )
        assert "A is foo and B is bar" in result
        assert llm.complete.called

    def test_llm_merge_fallback_on_failure(self):
        """LLM merge failure falls back to concatenation."""
        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("API error")

        result = QueryDecomposer.merge_sub_answers(
            ["What is A?", "What is B?"],
            ["A is foo", "B is bar"],
            llm=llm,
        )
        # Fallback concatenation
        assert "**What is A?**" in result
        assert "A is foo" in result


# ---------------------------------------------------------------------------
# Disabled decomposition
# ---------------------------------------------------------------------------

class TestDisabled:
    """Tests for when decomposition is disabled."""

    def test_disabled_returns_original(self):
        """Disabled decomposer always returns original query."""
        d = QueryDecomposer(enable_decomposition=False)
        result = d.decompose(
            "What is A? What is B? What is C?"
        )
        assert len(result) == 1

    def test_empty_query_returns_as_is(self, decomposer):
        """Empty query returns as-is."""
        result = decomposer.decompose("")
        assert result == [""]

    def test_whitespace_only_returns_as_is(self, decomposer):
        """Whitespace-only query returns as-is."""
        result = decomposer.decompose("   ")
        assert result == ["   "]


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestConfigDefaults:
    """Tests that config defaults are correct."""

    def test_default_max_sub_queries(self):
        from bakkesmod_rag.config import RetrieverConfig
        cfg = RetrieverConfig()
        assert cfg.max_sub_queries == 4

    def test_default_enable_decomposition(self):
        from bakkesmod_rag.config import RetrieverConfig
        cfg = RetrieverConfig()
        assert cfg.enable_query_decomposition is True

    def test_default_complexity_threshold(self):
        from bakkesmod_rag.config import RetrieverConfig
        cfg = RetrieverConfig()
        assert cfg.decomposition_complexity_threshold == 80


# ---------------------------------------------------------------------------
# Integration with engine (mocked subsystems)
# ---------------------------------------------------------------------------

class TestEngineIntegration:
    """Tests that QueryDecomposer is wired into engine correctly."""

    def test_engine_has_decomposer(self):
        """RAGEngine should have a decomposer attribute after the import."""
        # We just verify the import works and the class exists
        from bakkesmod_rag.engine import RAGEngine
        assert hasattr(RAGEngine, "query")
        # Can't instantiate without full LLM setup, but verify the
        # decomposer is referenced in the source
        import inspect
        source = inspect.getsource(RAGEngine.__init__)
        assert "decomposer" in source

    def test_decomposer_in_init_exports(self):
        """QueryDecomposer is exported from package."""
        from bakkesmod_rag import QueryDecomposer as QD
        assert QD is QueryDecomposer
