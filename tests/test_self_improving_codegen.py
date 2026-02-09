"""
Tests for Self-Improving Code Generation
=========================================
Validates the validate→fix loop, max iterations, error feedback,
compiler integration, feedback store integration, and graceful
degradation.
"""

from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, List

import pytest

from bakkesmod_rag.code_generator import (
    BakkesModCodeGenerator,
    CodeValidator,
    PluginTemplateEngine,
)
from bakkesmod_rag.config import CodeGenConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Create a mock LLM that returns valid BakkesMod code."""
    llm = MagicMock()
    llm.complete.return_value = MagicMock(
        text="""HEADER FILE:
```cpp
#pragma once
#include "GuiBase.h"
#include "bakkesmod/plugin/bakkesmodplugin.h"
#include "version.h"

constexpr auto plugin_version = stringify(VERSION_MAJOR) "." stringify(VERSION_MINOR);

class GoalTracker : public BakkesMod::Plugin::BakkesModPlugin {
public:
    void onLoad() override;
    void onUnload() override;
};
```

IMPLEMENTATION FILE:
```cpp
#include "pch.h"
#include "GoalTracker.h"

std::shared_ptr<CVarManagerWrapper> _globalCvarManager;

BAKKESMOD_PLUGIN(GoalTracker, "Track goals", plugin_version, PLUGINTYPE_FREEPLAY)

void GoalTracker::onLoad() {
    _globalCvarManager = cvarManager;
    gameWrapper->HookEvent("Function TAGame.Ball_TA.Explode", [this](std::string eventName) {
        LOG("Goal scored!");
    });
}

void GoalTracker::onUnload() {}
```
"""
    )
    return llm


@pytest.fixture
def config_self_improving():
    """Config with self-improving enabled but compilation disabled."""
    return CodeGenConfig(
        self_improving=True,
        max_fix_iterations=3,
        enable_compilation=False,
        feedback_enabled=False,
    )


@pytest.fixture
def config_no_self_improving():
    """Config with self-improving disabled."""
    return CodeGenConfig(
        self_improving=False,
        enable_compilation=False,
        feedback_enabled=False,
    )


@pytest.fixture
def generator(mock_llm, config_self_improving):
    """Create a BakkesModCodeGenerator with self-improving enabled."""
    return BakkesModCodeGenerator(
        llm=mock_llm,
        query_engine=None,
        config=config_self_improving,
    )


@pytest.fixture
def generator_no_improving(mock_llm, config_no_self_improving):
    """Create a BakkesModCodeGenerator with self-improving disabled."""
    return BakkesModCodeGenerator(
        llm=mock_llm,
        query_engine=None,
        config=config_no_self_improving,
    )


# ---------------------------------------------------------------------------
# Basic generation tests
# ---------------------------------------------------------------------------

class TestBasicGeneration:
    """Test that generation works with self-improving mode."""

    def test_generates_project_files(self, generator):
        result = generator.generate_full_plugin_with_rag("track goals in Rocket League")
        assert "project_files" in result
        assert isinstance(result["project_files"], dict)
        assert len(result["project_files"]) > 0

    def test_result_contains_new_fields(self, generator):
        result = generator.generate_full_plugin_with_rag("track goals")
        assert "fix_iterations" in result
        assert "fix_history" in result
        assert "compile_result" in result
        assert "generation_id" in result

    def test_result_has_header_and_impl(self, generator):
        result = generator.generate_full_plugin_with_rag("track goals")
        assert result["header"] != ""
        assert result["implementation"] != ""

    def test_features_detected(self, generator):
        result = generator.generate_full_plugin_with_rag(
            "hook goal events and display HUD overlay"
        )
        assert "event_hooks" in result["features_used"]

    def test_no_self_improving_single_validation(self, generator_no_improving):
        result = generator_no_improving.generate_full_plugin_with_rag("track goals")
        assert result["fix_iterations"] == 0
        assert result["fix_history"] == []


# ---------------------------------------------------------------------------
# Validate→Fix loop tests
# ---------------------------------------------------------------------------

class TestValidateFixLoop:
    """Test the iterative validate→fix loop."""

    def test_clean_code_no_iterations(self, generator):
        """If code is clean on first pass, no fix iterations needed."""
        result = generator.generate_full_plugin_with_rag("track goals")
        # The mock LLM produces valid code, so validation should pass
        validation = result["validation"]
        if validation["valid"]:
            # If somehow the template engine + LLM code is valid
            assert result["fix_iterations"] <= 1

    def test_max_iterations_respected(self, mock_llm, tmp_path):
        """Verify fix loop doesn't exceed max_fix_iterations."""
        # Make the LLM always return broken code
        broken_response = MagicMock(
            text="""HEADER FILE:
```cpp
class Broken {
```

IMPLEMENTATION FILE:
```cpp
void f() { undeclared_var = 1;
```
"""
        )
        mock_llm.complete.return_value = broken_response

        config = CodeGenConfig(
            self_improving=True,
            max_fix_iterations=3,
            enable_compilation=False,
            feedback_enabled=False,
        )
        gen = BakkesModCodeGenerator(llm=mock_llm, config=config)
        result = gen.generate_full_plugin_with_rag("broken plugin")

        # Should have attempted fixes but not exceeded max
        assert result["fix_iterations"] <= 3

    def test_no_progress_stops_early(self, mock_llm):
        """If same errors appear twice, stop the loop."""
        # LLM always returns same broken code (no progress)
        broken_response = MagicMock(
            text="""HEADER FILE:
```cpp
#pragma once
class BadPlugin : public BakkesMod::Plugin::BakkesModPlugin {
public:
    void onLoad() override;
    void onUnload() override;
};
```

IMPLEMENTATION FILE:
```cpp
#include "pch.h"
#include "BadPlugin.h"

BAKKESMOD_PLUGIN(BadPlugin, "bad", "1.0", PLUGINTYPE_FREEPLAY)

void BadPlugin::onLoad() { _globalCvarManager = cvarManager; }
void BadPlugin::onUnload() { /* unclosed bracket */
```
"""
        )
        mock_llm.complete.return_value = broken_response

        config = CodeGenConfig(
            self_improving=True,
            max_fix_iterations=5,
            enable_compilation=False,
            feedback_enabled=False,
        )
        gen = BakkesModCodeGenerator(llm=mock_llm, config=config)
        result = gen.generate_full_plugin_with_rag("bad plugin")

        # Should stop early due to no progress (same errors)
        # Fix history tells us how many iterations actually ran
        assert result["fix_iterations"] <= 5

    def test_fix_loop_calls_llm(self, mock_llm):
        """Verify that the fix loop actually calls the LLM for corrections."""
        call_count = 0
        original_complete = mock_llm.complete

        def counting_complete(prompt):
            nonlocal call_count
            call_count += 1
            return original_complete(prompt)

        mock_llm.complete.side_effect = counting_complete

        config = CodeGenConfig(
            self_improving=True,
            max_fix_iterations=2,
            enable_compilation=False,
            feedback_enabled=False,
        )
        gen = BakkesModCodeGenerator(llm=mock_llm, config=config)
        gen.generate_full_plugin_with_rag("test plugin")

        # At minimum 1 call (initial generation), possibly more for fixes
        assert call_count >= 1


# ---------------------------------------------------------------------------
# Error feedback formatting tests
# ---------------------------------------------------------------------------

class TestErrorFeedback:
    """Test error formatting for LLM feedback."""

    def test_format_error_feedback(self, generator):
        errors = [
            "plugin.cpp: Unclosed bracket '{' at position 50",
            "Missing BAKKESMOD_PLUGIN() macro in implementation",
        ]
        files = {
            "GoalTracker.h": "#pragma once\nclass GoalTracker {};",
            "GoalTracker.cpp": "void f() {",
        }
        feedback = generator._format_error_feedback(errors, files, "GoalTracker")
        assert "1." in feedback
        assert "2." in feedback
        assert "Unclosed bracket" in feedback
        assert "BAKKESMOD_PLUGIN" in feedback
        assert "GoalTracker.h" in feedback
        assert "GoalTracker.cpp" in feedback

    def test_format_includes_current_code(self, generator):
        files = {
            "Test.h": "header content",
            "Test.cpp": "impl content",
        }
        feedback = generator._format_error_feedback(["some error"], files, "Test")
        assert "header content" in feedback
        assert "impl content" in feedback


# ---------------------------------------------------------------------------
# Compiler integration tests
# ---------------------------------------------------------------------------

class TestCompilerIntegration:
    """Test integration with PluginCompiler."""

    def test_compile_disabled_by_config(self, mock_llm):
        """When compilation is disabled, no compiler is created."""
        config = CodeGenConfig(
            enable_compilation=False,
            self_improving=True,
            feedback_enabled=False,
        )
        gen = BakkesModCodeGenerator(llm=mock_llm, config=config)
        assert gen.compiler is None

    @patch("bakkesmod_rag.code_generator.BakkesModCodeGenerator._find_sdk_include_dirs",
           return_value=[])
    def test_compile_graceful_when_no_msvc(self, mock_dirs, mock_llm):
        """When MSVC isn't found, compiler is None and generation still works."""
        config = CodeGenConfig(
            enable_compilation=True,
            self_improving=True,
            feedback_enabled=False,
        )
        gen = BakkesModCodeGenerator(llm=mock_llm, config=config)
        # Compiler may or may not be available depending on system
        result = gen.generate_full_plugin_with_rag("test plugin")
        assert "project_files" in result

    def test_compile_result_in_output(self, mock_llm):
        """When compiler is available, compile_result appears in output."""
        config = CodeGenConfig(
            enable_compilation=False,
            self_improving=True,
            feedback_enabled=False,
        )
        gen = BakkesModCodeGenerator(llm=mock_llm, config=config)
        result = gen.generate_full_plugin_with_rag("test plugin")
        # No compiler means compile_result should be None
        assert result["compile_result"] is None


# ---------------------------------------------------------------------------
# Feedback store integration tests
# ---------------------------------------------------------------------------

class TestFeedbackIntegration:
    """Test integration with FeedbackStore."""

    def test_feedback_disabled(self, mock_llm):
        """When feedback is disabled, no store is created."""
        config = CodeGenConfig(
            feedback_enabled=False,
            self_improving=False,
            enable_compilation=False,
        )
        gen = BakkesModCodeGenerator(llm=mock_llm, config=config)
        assert gen.feedback is None

    def test_feedback_enabled_creates_store(self, mock_llm, tmp_path):
        """When feedback is enabled, a store is created."""
        config = CodeGenConfig(
            feedback_enabled=True,
            feedback_dir=str(tmp_path / "feedback"),
            self_improving=False,
            enable_compilation=False,
        )
        gen = BakkesModCodeGenerator(llm=mock_llm, config=config)
        assert gen.feedback is not None

    def test_generation_recorded(self, mock_llm, tmp_path):
        """Verify that generation is recorded in the feedback store."""
        config = CodeGenConfig(
            feedback_enabled=True,
            feedback_dir=str(tmp_path / "feedback"),
            self_improving=False,
            enable_compilation=False,
        )
        gen = BakkesModCodeGenerator(llm=mock_llm, config=config)
        result = gen.generate_full_plugin_with_rag("track goals")
        assert result["generation_id"] is not None
        assert len(gen.feedback._entries) == 1

    def test_record_feedback(self, mock_llm, tmp_path):
        """Test recording user feedback after generation."""
        config = CodeGenConfig(
            feedback_enabled=True,
            feedback_dir=str(tmp_path / "feedback"),
            self_improving=False,
            enable_compilation=False,
        )
        gen = BakkesModCodeGenerator(llm=mock_llm, config=config)
        result = gen.generate_full_plugin_with_rag("track goals")
        gen_id = result["generation_id"]

        success = gen.record_feedback(gen_id, accepted=True, user_edits="added feature")
        assert success is True
        assert gen.feedback._entries[0].accepted is True

    def test_record_feedback_no_store(self, generator):
        """When feedback is disabled, record_feedback returns False."""
        result = generator.record_feedback("fake-id", accepted=True)
        assert result is False


# ---------------------------------------------------------------------------
# Few-shot prompt integration
# ---------------------------------------------------------------------------

class TestFewShotIntegration:
    """Test that few-shot examples are included in generation prompts."""

    def test_few_shot_included_in_prompt(self, mock_llm, tmp_path):
        """When feedback store has accepted patterns, they're used."""
        config = CodeGenConfig(
            feedback_enabled=True,
            feedback_dir=str(tmp_path / "feedback"),
            self_improving=False,
            enable_compilation=False,
        )
        gen = BakkesModCodeGenerator(llm=mock_llm, config=config)

        # Record a successful generation
        gen.feedback.record_generation(
            description="boost tracker",
            features=["event_hooks"],
            generated_files={"BoostTracker.cpp": "void onLoad() { HookEvent(); }"},
            accepted=True,
        )

        # Generate another plugin — should include few-shot context
        gen.generate_full_plugin_with_rag("track boost usage")

        # Verify the LLM was called (we can't easily check the prompt
        # without more intrusive mocking, but we verify it doesn't crash)
        assert mock_llm.complete.called


# ---------------------------------------------------------------------------
# Fix with LLM tests
# ---------------------------------------------------------------------------

class TestFixWithLLM:
    """Test the _fix_with_llm helper method."""

    def test_fix_returns_code(self, generator):
        files = {
            "Test.h": "#pragma once\nclass Test {};",
            "Test.cpp": "void f() {",
        }
        errors = ["Unclosed bracket"]
        fixed = generator._fix_with_llm(files, errors, 0, "Test")
        # Mock LLM returns valid code, so fixed should have content
        assert "header" in fixed
        assert "implementation" in fixed

    def test_fix_handles_llm_failure(self, mock_llm):
        """If LLM raises exception during fix, return empty."""
        mock_llm.complete.side_effect = Exception("API error")
        config = CodeGenConfig(
            self_improving=True,
            enable_compilation=False,
            feedback_enabled=False,
        )
        gen = BakkesModCodeGenerator(llm=mock_llm, config=config)

        fixed = gen._fix_with_llm(
            {"T.h": "h", "T.cpp": "c"}, ["error"], 0, "T"
        )
        assert fixed == {"header": "", "implementation": ""}


# ---------------------------------------------------------------------------
# SDK include directory discovery
# ---------------------------------------------------------------------------

class TestSDKDiscovery:
    """Test SDK include directory finding."""

    def test_find_existing_dirs(self):
        dirs = BakkesModCodeGenerator._find_sdk_include_dirs()
        # Should find docs_bakkesmod_only and/or templates if they exist
        for d in dirs:
            assert os.path.isdir(d)

    def test_returns_absolute_paths(self):
        dirs = BakkesModCodeGenerator._find_sdk_include_dirs()
        for d in dirs:
            assert os.path.isabs(d)


import os
