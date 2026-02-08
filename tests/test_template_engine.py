"""Tests for PluginTemplateEngine: all 12 file generators, feature flags, substitution."""

import pytest
from bakkesmod_rag.code_generator import PluginTemplateEngine, CodeValidator


@pytest.fixture
def engine():
    return PluginTemplateEngine()


@pytest.fixture
def validator():
    return CodeValidator()


# ---------------------------------------------------------------------------
# Basic plugin generation (backward compat)
# ---------------------------------------------------------------------------

class TestBasicPlugin:
    def test_generates_header_and_implementation(self, engine):
        result = engine.generate_basic_plugin("TestPlugin", "A test plugin")
        assert "header" in result
        assert "implementation" in result
        assert "TestPlugin" in result["header"]
        assert "TestPlugin" in result["implementation"]

    def test_header_has_pragma_once(self, engine):
        result = engine.generate_basic_plugin("Foo", "desc")
        assert "#pragma once" in result["header"]

    def test_implementation_has_macro(self, engine):
        result = engine.generate_basic_plugin("Foo", "desc")
        assert "BAKKESMOD_PLUGIN(Foo" in result["implementation"]

    def test_event_hook_snippet(self, engine):
        code = engine.generate_event_hook("Function TAGame.Ball_TA.Explode", "OnBallExplode")
        assert "HookEvent" in code
        assert "TAGame.Ball_TA.Explode" in code
        assert "OnBallExplode" in code

    def test_imgui_window(self, engine):
        code = engine.generate_imgui_window("Settings", ["checkbox", "slider"])
        assert "ImGui::Begin" in code
        assert "ImGui::Checkbox" in code
        assert "ImGui::SliderFloat" in code


# ---------------------------------------------------------------------------
# Complete project generation
# ---------------------------------------------------------------------------

class TestCompleteProject:
    def test_generates_all_12_files(self, engine):
        files = engine.generate_complete_project("TestPlugin", "A test")
        assert len(files) == 12

        expected_files = [
            "TestPlugin.h", "TestPlugin.cpp", "pch.h", "pch.cpp",
            "version.h", "logging.h", "GuiBase.h", "GuiBase.cpp",
            "resource.h", "TestPlugin.rc", "BakkesMod.props",
            "TestPlugin.vcxproj",
        ]
        for f in expected_files:
            assert f in files, f"Missing file: {f}"

    def test_no_dollar_placeholders(self, engine):
        """Ensure $projectname$ is fully replaced."""
        files = engine.generate_complete_project("GoalTracker", "Tracks goals")
        for fname, content in files.items():
            assert "$projectname$" not in content, (
                f"Unreplaced $projectname$ in {fname}"
            )
            assert "$safeprojectname$" not in content, (
                f"Unreplaced $safeprojectname$ in {fname}"
            )

    def test_plugin_name_in_header(self, engine):
        files = engine.generate_complete_project("GoalTracker", "Tracks goals")
        header = files["GoalTracker.h"]
        assert "class GoalTracker" in header
        assert "BakkesMod::Plugin::BakkesModPlugin" in header

    def test_plugin_name_in_implementation(self, engine):
        files = engine.generate_complete_project("GoalTracker", "Tracks goals")
        impl = files["GoalTracker.cpp"]
        assert 'BAKKESMOD_PLUGIN(GoalTracker' in impl
        assert "_globalCvarManager" in impl
        assert '#include "pch.h"' in impl

    def test_pch_h_has_imgui(self, engine):
        files = engine.generate_complete_project("X", "test")
        assert "imgui.h" in files["pch.h"]
        assert "logging.h" in files["pch.h"]

    def test_version_h_has_macros(self, engine):
        files = engine.generate_complete_project("X", "test")
        assert "VERSION_MAJOR" in files["version.h"]
        assert "stringify" in files["version.h"]

    def test_vcxproj_references_plugin(self, engine):
        files = engine.generate_complete_project("MyMod", "mod")
        vcx = files["MyMod.vcxproj"]
        assert "MyMod.cpp" in vcx
        assert "MyMod.h" in vcx
        assert "pch.cpp" in vcx
        assert "GuiBase.cpp" in vcx

    def test_rc_file_has_version_info(self, engine):
        files = engine.generate_complete_project("MyMod", "mod")
        rc = files["MyMod.rc"]
        assert "VS_VERSION_INFO" in rc
        assert "MyMod.dll" in rc

    def test_bakkesmod_props_has_sdk_paths(self, engine):
        files = engine.generate_complete_project("X", "test")
        props = files["BakkesMod.props"]
        assert "BakkesModPath" in props
        assert "pluginsdk.lib" in props

    def test_guibase_h_has_classes(self, engine):
        files = engine.generate_complete_project("MyPlug", "test")
        assert "SettingsWindowBase" in files["GuiBase.h"]
        assert "PluginWindowBase" in files["GuiBase.h"]

    def test_guibase_cpp_has_plugin_name(self, engine):
        files = engine.generate_complete_project("MyPlug", "test")
        assert "MyPlug" in files["GuiBase.cpp"]

    def test_logging_h_has_log_macro(self, engine):
        files = engine.generate_complete_project("X", "test")
        assert "void LOG" in files["logging.h"]
        assert "void DEBUGLOG" in files["logging.h"]


# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

class TestFeatureFlags:
    def test_settings_window_feature(self, engine):
        files = engine.generate_complete_project(
            "TestPlugin", "test", features=["settings_window"]
        )
        header = files["TestPlugin.h"]
        impl = files["TestPlugin.cpp"]
        assert "SettingsWindowBase" in header
        assert "RenderSettings" in header
        assert "RenderSettings" in impl

    def test_plugin_window_feature(self, engine):
        files = engine.generate_complete_project(
            "TestPlugin", "test", features=["plugin_window"]
        )
        header = files["TestPlugin.h"]
        impl = files["TestPlugin.cpp"]
        assert "PluginWindowBase" in header
        assert "RenderWindow" in header
        assert "RenderWindow" in impl

    def test_cvars_feature(self, engine):
        files = engine.generate_complete_project(
            "TestPlugin", "test", features=["cvars"]
        )
        impl = files["TestPlugin.cpp"]
        assert "registerCvar" in impl

    def test_event_hooks_feature(self, engine):
        files = engine.generate_complete_project(
            "TestPlugin", "test", features=["event_hooks"]
        )
        impl = files["TestPlugin.cpp"]
        assert "HookEvent" in impl

    def test_drawable_feature(self, engine):
        files = engine.generate_complete_project(
            "TestPlugin", "test", features=["drawable"]
        )
        impl = files["TestPlugin.cpp"]
        assert "RegisterDrawable" in impl

    def test_no_features_basic_output(self, engine):
        files = engine.generate_complete_project("TestPlugin", "test", features=[])
        impl = files["TestPlugin.cpp"]
        # Should not have feature-specific code
        assert "registerCvar" not in impl
        assert "HookEvent" not in impl
        assert "RegisterDrawable" not in impl

    def test_all_features_combined(self, engine):
        files = engine.generate_complete_project(
            "TestPlugin", "test",
            features=["settings_window", "plugin_window", "cvars", "event_hooks", "drawable"],
        )
        header = files["TestPlugin.h"]
        impl = files["TestPlugin.cpp"]
        assert "SettingsWindowBase" in header
        assert "PluginWindowBase" in header
        assert "registerCvar" in impl
        assert "HookEvent" in impl
        assert "RegisterDrawable" in impl


# ---------------------------------------------------------------------------
# Generated code passes validation
# ---------------------------------------------------------------------------

class TestGeneratedCodeValidity:
    def test_complete_project_passes_validation(self, engine, validator):
        files = engine.generate_complete_project(
            "ValidPlugin", "A valid plugin",
            features=["cvars", "event_hooks"],
        )
        result = validator.validate_project(files)
        assert result["valid"] is True, f"Errors: {result['errors']}"

    def test_all_features_passes_validation(self, engine, validator):
        files = engine.generate_complete_project(
            "FullPlugin", "Full featured plugin",
            features=["settings_window", "plugin_window", "cvars", "event_hooks", "drawable", "imgui"],
        )
        result = validator.validate_project(files)
        assert result["valid"] is True, f"Errors: {result['errors']}"
