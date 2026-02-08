"""Tests for QueryRewriter: domain synonym expansion."""

import pytest
from bakkesmod_rag.query_rewriter import QueryRewriter


@pytest.fixture
def rewriter():
    return QueryRewriter(llm=None, use_llm=False)


class TestSynonymExpansion:
    def test_expands_plugin_keyword(self, rewriter):
        result = rewriter.expand_with_synonyms("How do I create a plugin?")
        assert "plugin" in result
        assert "mod" in result or "extension" in result

    def test_expands_hook_keyword(self, rewriter):
        result = rewriter.expand_with_synonyms("How to hook an event?")
        assert "hook" in result
        assert "event" in result

    def test_expands_car_keyword(self, rewriter):
        result = rewriter.expand_with_synonyms("How to get the car velocity?")
        assert "CarWrapper" in result

    def test_expands_ball_keyword(self, rewriter):
        result = rewriter.expand_with_synonyms("Where is the ball?")
        assert "BallWrapper" in result

    def test_expands_gui_keyword(self, rewriter):
        result = rewriter.expand_with_synonyms("How to create a GUI?")
        assert "ImGui" in result

    def test_expands_settings_keyword(self, rewriter):
        result = rewriter.expand_with_synonyms("How to add settings?")
        assert "config" in result or "preferences" in result

    def test_no_expansion_for_unknown_terms(self, rewriter):
        query = "What is the meaning of life?"
        result = rewriter.expand_with_synonyms(query)
        assert result == query  # No synonyms matched

    def test_multiple_expansions(self, rewriter):
        result = rewriter.expand_with_synonyms("How to hook a goal event and render car info?")
        # Should expand hook, event, render, and car
        assert len(result) > len("How to hook a goal event and render car info?")

    def test_case_insensitive(self, rewriter):
        result = rewriter.expand_with_synonyms("How to use GAMEWRAPPER?")
        assert "gameWrapper" in result or "GameWrapper" in result

    def test_rewrite_without_llm_uses_synonyms(self, rewriter):
        result = rewriter.rewrite("How to create a plugin?")
        assert "mod" in result or "extension" in result

    def test_rewrite_with_llm_flag_false(self):
        rewriter = QueryRewriter(llm=None, use_llm=False)
        assert rewriter.use_llm is False
