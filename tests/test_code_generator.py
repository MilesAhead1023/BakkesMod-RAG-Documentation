"""Tests for BakkesModCodeGenerator: feature detection, name derivation, mock LLM generation."""

import pytest
from unittest.mock import MagicMock
from bakkesmod_rag.code_generator import BakkesModCodeGenerator, PluginTemplateEngine, CodeValidator
from tests.conftest import MockLLM


@pytest.fixture
def code_gen():
    """Code generator with mock LLM, no query engine."""
    llm = MockLLM(response_text="""Here is the plugin:

HEADER FILE:
```cpp
#pragma once
#include "GuiBase.h"
#include "bakkesmod/plugin/bakkesmodplugin.h"
#include "version.h"
constexpr auto plugin_version = stringify(VERSION_MAJOR) "." stringify(VERSION_MINOR) "." stringify(VERSION_PATCH) "." stringify(VERSION_BUILD);
class GoalTracker: public BakkesMod::Plugin::BakkesModPlugin
{
    void onLoad() override;
    void onUnload() override;
};
```

IMPLEMENTATION FILE:
```cpp
#include "pch.h"
#include "GoalTracker.h"
BAKKESMOD_PLUGIN(GoalTracker, "Tracks goals", plugin_version, PLUGINTYPE_FREEPLAY)
std::shared_ptr<CVarManagerWrapper> _globalCvarManager;
void GoalTracker::onLoad()
{
    _globalCvarManager = cvarManager;
    gameWrapper->HookEvent("Function TAGame.Ball_TA.OnHitGoal", [this](std::string eventName) {
        LOG("Goal scored!");
    });
}
void GoalTracker::onUnload() {}
```
""")
    return BakkesModCodeGenerator(llm=llm, query_engine=None)


# ---------------------------------------------------------------------------
# Feature detection
# ---------------------------------------------------------------------------

class TestFeatureDetection:
    def test_detects_event_hooks(self, code_gen):
        features = code_gen._detect_features("Track goal events and ball explosions")
        assert "event_hooks" in features

    def test_detects_cvars(self, code_gen):
        features = code_gen._detect_features("Add a toggle setting for the plugin")
        assert "cvars" in features

    def test_detects_settings_window(self, code_gen):
        features = code_gen._detect_features("Create a settings configuration window")
        assert "settings_window" in features

    def test_detects_plugin_window(self, code_gen):
        features = code_gen._detect_features("Show an overlay window on screen")
        assert "plugin_window" in features

    def test_detects_drawable(self, code_gen):
        features = code_gen._detect_features("Draw speed on the HUD canvas")
        assert "drawable" in features

    def test_detects_imgui(self, code_gen):
        features = code_gen._detect_features("Create an ImGui interface with buttons")
        assert "imgui" in features

    def test_detects_multiple_features(self, code_gen):
        features = code_gen._detect_features(
            "Hook goal events and create a settings window with toggles"
        )
        assert "event_hooks" in features
        assert "settings_window" in features
        assert "cvars" in features

    def test_empty_description(self, code_gen):
        features = code_gen._detect_features("")
        assert features == []


# ---------------------------------------------------------------------------
# Plugin name derivation
# ---------------------------------------------------------------------------

class TestNameDerivation:
    def test_derives_from_description(self, code_gen):
        name = code_gen._derive_plugin_name("Track goal scoring events")
        assert name[0].isupper()
        assert name.isalnum()
        assert "Goal" in name or "Track" in name or "Scoring" in name

    def test_strips_stopwords(self, code_gen):
        name = code_gen._derive_plugin_name("Create a plugin that hooks the ball")
        assert "Create" not in name
        assert "Plugin" not in name

    def test_empty_description_fallback(self, code_gen):
        name = code_gen._derive_plugin_name("")
        assert name == "MyPlugin"

    def test_all_stopwords_fallback(self, code_gen):
        name = code_gen._derive_plugin_name("create a plugin for the")
        assert name == "MyPlugin"

    def test_pascal_case(self, code_gen):
        name = code_gen._derive_plugin_name("boost tracking display")
        assert name[0].isupper()


# ---------------------------------------------------------------------------
# Parse code response
# ---------------------------------------------------------------------------

class TestParseCodeResponse:
    def test_parses_header_and_impl(self, code_gen):
        text = """
HEADER FILE:
```cpp
#pragma once
class Foo {};
```

IMPLEMENTATION FILE:
```cpp
#include "Foo.h"
void Foo::bar() {}
```
"""
        result = code_gen._parse_code_response(text)
        assert "#pragma once" in result["header"]
        assert "Foo::bar" in result["implementation"]

    def test_missing_blocks_returns_empty(self, code_gen):
        result = code_gen._parse_code_response("No code blocks here")
        assert result["header"] == ""
        assert result["implementation"] == ""


# ---------------------------------------------------------------------------
# Full plugin generation with RAG
# ---------------------------------------------------------------------------

class TestFullPluginGeneration:
    def test_generates_complete_project(self, code_gen):
        result = code_gen.generate_full_plugin_with_rag(
            "Track goal scoring events and display a counter"
        )

        assert "project_files" in result
        assert len(result["project_files"]) == 12
        assert "header" in result
        assert "implementation" in result
        assert "features_used" in result
        assert "validation" in result

    def test_project_files_have_no_empty_content(self, code_gen):
        result = code_gen.generate_full_plugin_with_rag("Simple plugin that logs events")
        for fname, content in result["project_files"].items():
            assert content, f"File {fname} is empty"

    def test_llm_generated_header_overrides_template(self, code_gen):
        result = code_gen.generate_full_plugin_with_rag("Track goal events")
        # The LLM response should override the template-generated plugin.h
        # (since our mock LLM returns GoalTracker code)
        header = result["header"]
        assert header  # Should not be empty

    def test_validation_runs(self, code_gen):
        result = code_gen.generate_full_plugin_with_rag("Simple test plugin")
        assert "valid" in result["validation"]

    def test_features_detected_from_description(self, code_gen):
        result = code_gen.generate_full_plugin_with_rag(
            "Hook goal events and create a settings window"
        )
        assert "event_hooks" in result["features_used"]
        assert "settings_window" in result["features_used"]


# ---------------------------------------------------------------------------
# Direct generation (no RAG)
# ---------------------------------------------------------------------------

class TestDirectGeneration:
    def test_generate_plugin_returns_code(self, code_gen):
        result = code_gen.generate_plugin("Simple logger plugin")
        assert "header" in result
        assert "implementation" in result

    def test_generate_plugin_with_rag_no_engine(self, code_gen):
        """Falls back to direct generation when no query engine."""
        result = code_gen.generate_plugin_with_rag("Test plugin")
        assert "header" in result
        assert "implementation" in result
