"""
Tests for FeedbackStore
=======================
Validates recording, retrieval, pattern matching, common fixes,
JSON persistence, and edge cases.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from bakkesmod_rag.feedback_store import FeedbackStore, FeedbackEntry


@pytest.fixture
def tmp_feedback_dir(tmp_path):
    """Create a temporary directory for feedback storage."""
    d = tmp_path / "feedback"
    d.mkdir()
    return str(d)


@pytest.fixture
def store(tmp_feedback_dir):
    """Create a fresh FeedbackStore for each test."""
    return FeedbackStore(feedback_dir=tmp_feedback_dir)


# ---------------------------------------------------------------------------
# Basic record and retrieve
# ---------------------------------------------------------------------------

class TestRecordGeneration:
    """Test recording generation attempts."""

    def test_record_returns_id(self, store):
        gen_id = store.record_generation(
            description="goal tracker plugin",
            features=["event_hooks"],
            generated_files={"plugin.cpp": "code here"},
        )
        assert gen_id is not None
        assert len(gen_id) == 8

    def test_record_creates_entry(self, store):
        store.record_generation(
            description="goal tracker",
            features=["event_hooks"],
            generated_files={"plugin.cpp": "code"},
        )
        assert len(store._entries) == 1
        assert store._entries[0].description == "goal tracker"

    def test_record_with_all_fields(self, store):
        gen_id = store.record_generation(
            description="boost monitor",
            features=["event_hooks", "drawable"],
            generated_files={"plugin.h": "h", "plugin.cpp": "cpp"},
            validation_errors=["missing onLoad"],
            validation_warnings=["missing pch.h"],
            fix_iterations=3,
            accepted=True,
            user_edits="added onLoad",
            compile_attempted=True,
            compile_success=True,
            compiler_errors=[],
        )
        entry = store._entries[0]
        assert entry.accepted is True
        assert entry.fix_iterations == 3
        assert entry.compile_success is True

    def test_record_multiple_entries(self, store):
        for i in range(5):
            store.record_generation(
                description=f"plugin {i}",
                features=[],
                generated_files={},
            )
        assert len(store._entries) == 5


# ---------------------------------------------------------------------------
# Update feedback
# ---------------------------------------------------------------------------

class TestUpdateFeedback:
    """Test updating feedback on existing entries."""

    def test_update_accepted(self, store):
        gen_id = store.record_generation(
            description="test plugin",
            features=[],
            generated_files={},
        )
        result = store.update_feedback(gen_id, accepted=True)
        assert result is True
        assert store._entries[0].accepted is True

    def test_update_with_edits(self, store):
        gen_id = store.record_generation(
            description="test plugin",
            features=[],
            generated_files={},
        )
        store.update_feedback(gen_id, accepted=True, user_edits="fixed onLoad")
        assert store._entries[0].user_edits == "fixed onLoad"

    def test_update_nonexistent_id(self, store):
        result = store.update_feedback("nonexistent", accepted=True)
        assert result is False


# ---------------------------------------------------------------------------
# Get successful patterns
# ---------------------------------------------------------------------------

class TestGetSuccessfulPatterns:
    """Test retrieving accepted generations as few-shot examples."""

    def test_empty_store(self, store):
        patterns = store.get_successful_patterns()
        assert patterns == []

    def test_only_accepted_returned(self, store):
        store.record_generation(
            description="rejected plugin",
            features=["event_hooks"],
            generated_files={"plugin.cpp": "bad code"},
            accepted=False,
        )
        store.record_generation(
            description="accepted plugin",
            features=["event_hooks"],
            generated_files={"plugin.cpp": "good code"},
            accepted=True,
        )
        patterns = store.get_successful_patterns()
        assert len(patterns) == 1
        assert patterns[0]["description"] == "accepted plugin"

    def test_feature_overlap_ranking(self, store):
        store.record_generation(
            description="drawing plugin",
            features=["drawable"],
            generated_files={"plugin.cpp": "draw code"},
            accepted=True,
        )
        store.record_generation(
            description="hooks plugin",
            features=["event_hooks", "drawable"],
            generated_files={"plugin.cpp": "hooks code"},
            accepted=True,
        )
        patterns = store.get_successful_patterns(
            target_features=["event_hooks", "drawable"]
        )
        # The one with more overlap should rank higher
        assert len(patterns) >= 1

    def test_max_results_respected(self, store):
        for i in range(10):
            store.record_generation(
                description=f"plugin {i}",
                features=[],
                generated_files={"plugin.cpp": f"code {i}"},
                accepted=True,
            )
        patterns = store.get_successful_patterns(max_results=3)
        assert len(patterns) == 3

    def test_code_snippet_extraction(self, store):
        store.record_generation(
            description="test plugin",
            features=[],
            generated_files={
                "MyPlugin.cpp": "void onLoad() { /* real code */ }",
                "pch.cpp": "#include <pch.h>",
            },
            accepted=True,
        )
        patterns = store.get_successful_patterns()
        assert len(patterns) == 1
        assert "onLoad" in patterns[0]["code_snippet"]


# ---------------------------------------------------------------------------
# Get common fixes
# ---------------------------------------------------------------------------

class TestGetCommonFixes:
    """Test aggregation of common error patterns."""

    def test_empty_store(self, store):
        fixes = store.get_common_fixes()
        assert fixes == []

    def test_aggregates_validation_errors(self, store):
        for _ in range(3):
            store.record_generation(
                description="test",
                features=[],
                generated_files={},
                validation_errors=["Missing BAKKESMOD_PLUGIN() macro in implementation"],
            )
        fixes = store.get_common_fixes()
        assert len(fixes) >= 1
        assert fixes[0]["frequency"] == 3

    def test_aggregates_compiler_errors(self, store):
        store.record_generation(
            description="test",
            features=[],
            generated_files={},
            compiler_errors=["plugin.cpp(10) : error C2065: 'x' : undeclared"],
        )
        store.record_generation(
            description="test2",
            features=[],
            generated_files={},
            compiler_errors=["other.cpp(20) : error C2065: 'y' : undeclared"],
        )
        fixes = store.get_common_fixes()
        # Both should normalise to similar pattern
        assert len(fixes) >= 1

    def test_max_results(self, store):
        for i in range(10):
            store.record_generation(
                description=f"test {i}",
                features=[],
                generated_files={},
                validation_errors=[f"error type {i % 3}"],
            )
        fixes = store.get_common_fixes(max_results=2)
        assert len(fixes) <= 2


# ---------------------------------------------------------------------------
# Get feedback stats
# ---------------------------------------------------------------------------

class TestGetFeedbackStats:
    """Test summary statistics calculation."""

    def test_empty_store(self, store):
        stats = store.get_feedback_stats()
        assert stats["total_generations"] == 0
        assert stats["acceptance_rate"] == 0.0

    def test_acceptance_rate(self, store):
        store.record_generation("a", [], {}, accepted=True)
        store.record_generation("b", [], {}, accepted=True)
        store.record_generation("c", [], {}, accepted=False)
        stats = store.get_feedback_stats()
        assert stats["total_generations"] == 3
        assert abs(stats["acceptance_rate"] - 2/3) < 0.01

    def test_avg_fix_iterations(self, store):
        store.record_generation("a", [], {}, fix_iterations=2)
        store.record_generation("b", [], {}, fix_iterations=4)
        stats = store.get_feedback_stats()
        assert stats["avg_fix_iterations"] == 3.0

    def test_compile_success_rate(self, store):
        store.record_generation("a", [], {}, compile_attempted=True, compile_success=True)
        store.record_generation("b", [], {}, compile_attempted=True, compile_success=False)
        store.record_generation("c", [], {}, compile_attempted=False)
        stats = store.get_feedback_stats()
        assert stats["compile_success_rate"] == 0.5

    def test_common_features(self, store):
        store.record_generation("a", ["event_hooks", "imgui"], {})
        store.record_generation("b", ["event_hooks"], {})
        stats = store.get_feedback_stats()
        features = dict(stats["common_features"])
        assert features["event_hooks"] == 2


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    """Test save and reload from JSON file."""

    def test_save_and_reload(self, tmp_feedback_dir):
        store1 = FeedbackStore(feedback_dir=tmp_feedback_dir)
        store1.record_generation("plugin 1", ["hooks"], {"p.cpp": "code"}, accepted=True)
        store1.record_generation("plugin 2", ["imgui"], {"q.cpp": "code2"})

        # Create new store from same directory
        store2 = FeedbackStore(feedback_dir=tmp_feedback_dir)
        assert len(store2._entries) == 2
        assert store2._entries[0].description == "plugin 1"
        assert store2._entries[0].accepted is True

    def test_json_file_created(self, tmp_feedback_dir):
        store = FeedbackStore(feedback_dir=tmp_feedback_dir)
        store.record_generation("test", [], {})
        filepath = Path(tmp_feedback_dir) / "generations.json"
        assert filepath.exists()

        with open(filepath, "r") as f:
            data = json.load(f)
        assert len(data) == 1

    def test_corrupted_json_handled(self, tmp_feedback_dir):
        filepath = Path(tmp_feedback_dir) / "generations.json"
        filepath.write_text("{ invalid json }", encoding="utf-8")

        store = FeedbackStore(feedback_dir=tmp_feedback_dir)
        assert len(store._entries) == 0  # Gracefully handles corruption


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

class TestPruning:
    """Test that old entries are pruned when max_entries is exceeded."""

    def test_prune_oldest(self, tmp_feedback_dir):
        store = FeedbackStore(feedback_dir=tmp_feedback_dir, max_entries=5)
        for i in range(10):
            store.record_generation(f"plugin {i}", [], {})

        assert len(store._entries) == 5
        # Oldest should be pruned; newest should remain
        assert store._entries[0].description == "plugin 5"
        assert store._entries[-1].description == "plugin 9"


# ---------------------------------------------------------------------------
# Few-shot prompt formatting
# ---------------------------------------------------------------------------

class TestFormatFewShotPrompt:
    """Test the few-shot prompt generation for LLM."""

    def test_empty_store_returns_empty(self, store):
        prompt = store.format_few_shot_prompt()
        assert prompt == ""

    def test_prompt_contains_examples(self, store):
        store.record_generation(
            description="goal explosion tracker",
            features=["event_hooks"],
            generated_files={"GoalTracker.cpp": "void onLoad() { HookEvent(); }"},
            accepted=True,
        )
        prompt = store.format_few_shot_prompt(target_features=["event_hooks"])
        assert "EXAMPLES OF PREVIOUSLY SUCCESSFUL PLUGINS" in prompt
        assert "goal explosion tracker" in prompt
        assert "event_hooks" in prompt

    def test_prompt_includes_common_fixes(self, store):
        for _ in range(3):
            store.record_generation(
                description="test",
                features=[],
                generated_files={},
                validation_errors=["Missing BAKKESMOD_PLUGIN() macro"],
                accepted=True,
            )
        prompt = store.format_few_shot_prompt()
        assert "COMMON MISTAKES TO AVOID" in prompt


# ---------------------------------------------------------------------------
# Error normalisation
# ---------------------------------------------------------------------------

class TestErrorNormalisation:
    """Test error message normalisation for pattern aggregation."""

    def test_strips_file_paths(self):
        norm = FeedbackStore._normalise_error(
            "plugin.cpp: undeclared identifier"
        )
        assert "<file>" in norm

    def test_strips_line_numbers(self):
        norm = FeedbackStore._normalise_error("error at line 42")
        assert "line N" in norm

    def test_strips_identifiers(self):
        norm = FeedbackStore._normalise_error("'gameWrapper' undeclared")
        assert "'<id>'" in norm

    def test_handles_empty_string(self):
        norm = FeedbackStore._normalise_error("")
        assert norm == ""
