"""Integration tests for code generation with real LLM calls.

Run with: pytest tests/test_code_gen_integration.py -v -m integration
"""

import os
import pytest

pytestmark = pytest.mark.integration


def _has_api_keys() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture(scope="module")
def engine():
    if not _has_api_keys():
        pytest.skip("No API keys available for integration tests")
    from bakkesmod_rag import RAGEngine
    return RAGEngine()


class TestLLMCodeGeneration:
    def test_simple_plugin_generation(self, engine):
        result = engine.generate_code("Log a message when the plugin loads")
        assert result.header
        assert result.implementation
        assert "onLoad" in result.implementation or "onLoad" in result.header

    def test_event_hook_plugin(self, engine):
        result = engine.generate_code(
            "Hook the ball explosion event and log a message each time"
        )
        assert result.implementation
        assert (
            "HookEvent" in result.implementation
            or "Ball" in result.implementation
        )

    def test_complete_project_structure(self, engine):
        result = engine.generate_code("Create a boost tracker plugin with settings")
        assert len(result.project_files) == 12

        # Check essential files exist
        has_h = any(f.endswith(".h") for f in result.project_files if f not in (
            "pch.h", "version.h", "logging.h", "resource.h", "GuiBase.h"
        ))
        has_cpp = any(f.endswith(".cpp") for f in result.project_files if f not in (
            "pch.cpp", "GuiBase.cpp"
        ))
        assert has_h, "Missing plugin header file"
        assert has_cpp, "Missing plugin implementation file"

    def test_generated_project_validates(self, engine):
        result = engine.generate_code("Simple speed display plugin")
        # Validation should at minimum have a 'valid' key
        assert "valid" in result.validation

    def test_features_influence_output(self, engine):
        result = engine.generate_code(
            "Create a plugin with a settings window, event hooks for goals, "
            "and CVars for toggling features"
        )
        assert "event_hooks" in result.features_used
        assert "settings_window" in result.features_used
        assert "cvars" in result.features_used
