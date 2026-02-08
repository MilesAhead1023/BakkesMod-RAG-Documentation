"""Tests for CostTracker: token counting, budgets, persistence."""

import json
import pytest
from pathlib import Path
from bakkesmod_rag.cost_tracker import CostTracker, get_tracker


@pytest.fixture
def tracker(tmp_path):
    """Create a CostTracker with a temporary persistence path."""
    path = tmp_path / "costs.json"
    return CostTracker(config=None, persistence_path=path)


class TestTokenCounting:
    def test_count_tokens_fallback(self, tracker):
        """Without tiktoken, falls back to len/4."""
        count = tracker.count_tokens("hello world this is a test")
        assert count > 0

    def test_count_tokens_empty_string(self, tracker):
        count = tracker.count_tokens("")
        assert count == 0


class TestCostTracking:
    def test_track_embedding_records_cost(self, tracker):
        tracker.track_embedding(1_000_000, provider="openai")
        assert tracker.costs["total"] > 0
        assert "openai" in tracker.costs["by_provider"]

    def test_track_llm_call_openai(self, tracker):
        tracker.track_llm_call(
            input_tokens=1000,
            output_tokens=500,
            provider="openai",
            model="gpt-4o-mini",
        )
        assert tracker.costs["total"] > 0

    def test_track_llm_call_free_provider(self, tracker):
        """OpenRouter/DeepSeek is free tier -- no cost recorded."""
        tracker.track_llm_call(
            input_tokens=1000,
            output_tokens=500,
            provider="openrouter",
            model="deepseek-v3",
        )
        assert tracker.costs["total"] == 0.0

    def test_track_llm_call_anthropic(self, tracker):
        tracker.track_llm_call(
            input_tokens=1000,
            output_tokens=500,
            provider="anthropic",
            model="claude-sonnet",
        )
        assert tracker.costs["total"] > 0

    def test_track_llm_call_gemini(self, tracker):
        tracker.track_llm_call(
            input_tokens=1000,
            output_tokens=500,
            provider="gemini",
            model="gemini-flash",
        )
        assert tracker.costs["total"] > 0


class TestPersistence:
    def test_costs_persist_to_disk(self, tracker, tmp_path):
        tracker.track_embedding(100_000, provider="openai")
        # Load from disk
        with open(tracker.persistence_path) as f:
            data = json.load(f)
        assert data["total"] > 0

    def test_reload_from_disk(self, tmp_path):
        path = tmp_path / "costs.json"
        # Create first tracker and record cost
        t1 = CostTracker(config=None, persistence_path=path)
        t1.track_embedding(1_000_000, provider="openai")
        saved_total = t1.costs["total"]

        # Create second tracker loading from same file
        t2 = CostTracker(config=None, persistence_path=path)
        assert t2.costs["total"] == pytest.approx(saved_total, abs=0.001)


class TestBudget:
    def test_budget_exceeded_raises(self, tmp_path):
        from bakkesmod_rag.config import RAGConfig
        config = RAGConfig(
            openai_api_key="test",
            cost={"daily_budget_usd": 0.001, "alert_threshold_pct": 50.0},
        )
        path = tmp_path / "costs.json"
        tracker = CostTracker(config=config, persistence_path=path)

        with pytest.raises(RuntimeError, match="budget exceeded"):
            tracker.track_llm_call(
                input_tokens=1_000_000,
                output_tokens=1_000_000,
                provider="anthropic",
                model="claude-sonnet",
            )


class TestDailyCost:
    def test_get_daily_cost_zero_initially(self, tracker):
        assert tracker.get_daily_cost() == 0.0

    def test_reset_daily(self, tracker):
        tracker.track_embedding(1_000_000, provider="openai")
        assert tracker.get_daily_cost() > 0
        tracker.reset_daily()
        assert tracker.get_daily_cost() == 0.0


class TestReport:
    def test_report_format(self, tracker):
        tracker.track_embedding(100_000, provider="openai")
        report = tracker.get_report(days=3)
        assert "Cost Report" in report
        assert "Total All-Time" in report
        assert "By Provider" in report


class TestSingleton:
    def test_get_tracker_returns_same_instance(self):
        # Reset global
        import bakkesmod_rag.cost_tracker as ct
        ct._tracker_instance = None

        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2

        ct._tracker_instance = None  # cleanup
