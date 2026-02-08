"""Tests for CodeValidator: brackets, strings, semantic, and project validation."""

import pytest
from bakkesmod_rag.code_generator import CodeValidator


@pytest.fixture
def validator():
    return CodeValidator()


# ---------------------------------------------------------------------------
# Bracket validation
# ---------------------------------------------------------------------------

class TestBracketValidation:
    def test_matched_brackets(self, validator):
        code = "void foo() { int x = arr[0]; }"
        result = validator.validate_syntax(code)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_unclosed_brace(self, validator):
        code = "void foo() { int x = 1;"
        result = validator.validate_syntax(code)
        assert result["valid"] is False
        assert any("Unclosed" in e for e in result["errors"])

    def test_mismatched_bracket(self, validator):
        code = "void foo() { int x = arr(0]; }"
        result = validator.validate_syntax(code)
        assert result["valid"] is False
        assert any("Mismatched" in e for e in result["errors"])

    def test_extra_closing_bracket(self, validator):
        code = "void foo() {} }"
        result = validator.validate_syntax(code)
        assert result["valid"] is False
        assert any("Unmatched closing" in e for e in result["errors"])

    def test_nested_brackets(self, validator):
        code = "void foo() { if (x) { bar(arr[0]); } }"
        result = validator.validate_syntax(code)
        assert result["valid"] is True

    def test_empty_code(self, validator):
        result = validator.validate_syntax("")
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# String literal validation
# ---------------------------------------------------------------------------

class TestStringValidation:
    def test_closed_string(self, validator):
        code = 'LOG("hello world");'
        result = validator.validate_syntax(code)
        assert result["valid"] is True

    def test_unclosed_string(self, validator):
        code = 'LOG("hello world);'
        result = validator.validate_syntax(code)
        assert result["valid"] is False
        assert any("Unclosed string" in e for e in result["errors"])

    def test_escaped_quotes(self, validator):
        code = 'LOG("say \\"hello\\"");'
        result = validator.validate_syntax(code)
        assert result["valid"] is True


# ---------------------------------------------------------------------------
# API pattern detection
# ---------------------------------------------------------------------------

class TestAPIPatterns:
    def test_detects_gamewrapper(self, validator):
        code = "gameWrapper->GetOnlineGame();"
        result = validator.validate_bakkesmod_api(code)
        assert result["uses_gamewrapper"] is True

    def test_detects_hook_event(self, validator):
        code = 'gameWrapper->HookEvent("Function TAGame.Ball_TA.Explode", cb);'
        result = validator.validate_bakkesmod_api(code)
        assert result["hooks_events"] is True

    def test_detects_server_wrapper(self, validator):
        code = "ServerWrapper server = gameWrapper->GetOnlineGame();"
        result = validator.validate_bakkesmod_api(code)
        assert result["uses_server"] is True

    def test_detects_car_wrapper(self, validator):
        code = "CarWrapper car = gameWrapper->GetLocalCar();"
        result = validator.validate_bakkesmod_api(code)
        assert result["uses_car"] is True

    def test_empty_code_no_patterns(self, validator):
        result = validator.validate_bakkesmod_api("")
        assert all(v is False for v in result.values())


# ---------------------------------------------------------------------------
# Semantic validation
# ---------------------------------------------------------------------------

class TestSemanticValidation:
    def test_valid_plugin_files(self, validator):
        files = {
            "MyPlugin.h": '''#pragma once
#include "GuiBase.h"
#include "bakkesmod/plugin/bakkesmodplugin.h"
class MyPlugin: public BakkesMod::Plugin::BakkesModPlugin
{
    void onLoad() override;
};
''',
            "MyPlugin.cpp": '''#include "pch.h"
#include "MyPlugin.h"
BAKKESMOD_PLUGIN(MyPlugin, "test", plugin_version, PLUGINTYPE_FREEPLAY)
std::shared_ptr<CVarManagerWrapper> _globalCvarManager;
void MyPlugin::onLoad()
{
    _globalCvarManager = cvarManager;
}
''',
        }
        result = validator.validate_semantic(files)
        assert result["valid"] is True
        assert result["errors"] == []

    def test_missing_bakkesmod_plugin_macro(self, validator):
        files = {
            "MyPlugin.cpp": '''#include "pch.h"
void MyPlugin::onLoad() {}
''',
        }
        result = validator.validate_semantic(files)
        assert result["valid"] is False
        assert any("BAKKESMOD_PLUGIN" in e for e in result["errors"])

    def test_missing_onload(self, validator):
        files = {
            "MyPlugin.cpp": '''#include "pch.h"
BAKKESMOD_PLUGIN(MyPlugin, "test", "1.0", PLUGINTYPE_FREEPLAY)
''',
        }
        result = validator.validate_semantic(files)
        assert result["valid"] is False
        assert any("onLoad" in e for e in result["errors"])

    def test_warns_missing_global_cvar_manager(self, validator):
        files = {
            "MyPlugin.h": '''#pragma once
class MyPlugin: public BakkesMod::Plugin::BakkesModPlugin
{
    void onLoad() override;
};
''',
            "MyPlugin.cpp": '''#include "pch.h"
BAKKESMOD_PLUGIN(MyPlugin, "test", "1.0", PLUGINTYPE_FREEPLAY)
void MyPlugin::onLoad() {}
''',
        }
        result = validator.validate_semantic(files)
        assert any("_globalCvarManager" in w for w in result["warnings"])

    def test_warns_missing_pragma_once(self, validator):
        files = {
            "MyPlugin.h": '''
class MyPlugin: public BakkesMod::Plugin::BakkesModPlugin
{
    void onLoad() override;
};
''',
            "MyPlugin.cpp": '''#include "pch.h"
BAKKESMOD_PLUGIN(MyPlugin, "test", "1.0", PLUGINTYPE_FREEPLAY)
std::shared_ptr<CVarManagerWrapper> _globalCvarManager;
void MyPlugin::onLoad() {}
''',
        }
        result = validator.validate_semantic(files)
        assert any("pragma once" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# Full project validation
# ---------------------------------------------------------------------------

class TestProjectValidation:
    def test_full_valid_project(self, validator):
        files = {
            "MyPlugin.h": '''#pragma once
#include "bakkesmod/plugin/bakkesmodplugin.h"
class MyPlugin: public BakkesMod::Plugin::BakkesModPlugin
{
    void onLoad() override;
};
''',
            "MyPlugin.cpp": '''#include "pch.h"
#include "MyPlugin.h"
BAKKESMOD_PLUGIN(MyPlugin, "test", plugin_version, PLUGINTYPE_FREEPLAY)
std::shared_ptr<CVarManagerWrapper> _globalCvarManager;
void MyPlugin::onLoad() { _globalCvarManager = cvarManager; }
''',
            "pch.h": '#pragma once\n#include "bakkesmod/plugin/bakkesmodplugin.h"\n',
            "pch.cpp": '#include "pch.h"\n',
            "version.h": "#pragma once\n#define VERSION_MAJOR 1\n",
        }
        result = validator.validate_project(files)
        assert result["valid"] is True
        assert "syntax_results" in result

    def test_project_missing_pch(self, validator):
        files = {
            "MyPlugin.h": '#pragma once\nclass MyPlugin: public BakkesMod::Plugin::BakkesModPlugin { void onLoad() override; };',
            "MyPlugin.cpp": '#include "pch.h"\nBAKKESMOD_PLUGIN(MyPlugin, "x", "1.0", PLUGINTYPE_FREEPLAY)\nstd::shared_ptr<CVarManagerWrapper> _globalCvarManager;\nvoid MyPlugin::onLoad() {}',
        }
        result = validator.validate_project(files)
        assert any("pch.h" in w for w in result["warnings"])

    def test_project_syntax_error_in_cpp(self, validator):
        files = {
            "MyPlugin.cpp": '#include "pch.h"\nBAKKESMOD_PLUGIN(MyPlugin, "x", "1.0", PLUGINTYPE_FREEPLAY)\nstd::shared_ptr<CVarManagerWrapper> _globalCvarManager;\nvoid MyPlugin::onLoad() { /* unclosed',
        }
        result = validator.validate_project(files)
        assert result["valid"] is False
