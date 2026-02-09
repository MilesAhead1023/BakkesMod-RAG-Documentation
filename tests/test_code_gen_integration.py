"""Lightweight integration tests for code generation.

Validates the code generation pipeline without building a full
RAGEngine.  Tests use a mock LLM with real CodeValidator and
PluginTemplateEngine to verify end-to-end flow.

Auto-skips when no API keys are present (the LLM connectivity
test needs a real key).

Run only integration tests:  pytest -m integration -v
"""

import os
import pytest

_HAS_KEYS = bool(os.getenv("OPENAI_API_KEY"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_KEYS, reason="No API keys available"),
]


class TestLLMCodeGeneration:
    """Verify end-to-end code generation with real LLM (single call)."""

    def test_llm_generates_cpp_code(self):
        """One real LLM call to verify code generation returns something."""
        from bakkesmod_rag.config import get_config
        from bakkesmod_rag.llm_provider import get_llm
        from bakkesmod_rag.code_generator import BakkesModCodeGenerator

        llm = get_llm(get_config())
        gen = BakkesModCodeGenerator(llm=llm, query_engine=None)
        result = gen.generate_plugin("Log a message when the plugin loads")
        # LLM response format varies by provider — just verify it returned
        # something or didn't crash
        assert isinstance(result, dict)
        assert "header" in result and "implementation" in result

    def test_template_engine_produces_12_files(self):
        """PluginTemplateEngine should scaffold a full project."""
        from bakkesmod_rag.code_generator import PluginTemplateEngine
        te = PluginTemplateEngine()
        files = te.generate_complete_project(
            plugin_name="TestPlugin",
            description="A test plugin",
            features=["event_hooks"],
        )
        assert len(files) == 12, f"Expected 12 files, got {len(files)}"

    def test_feature_detection(self):
        """Verify feature detection from description works."""
        from bakkesmod_rag.code_generator import BakkesModCodeGenerator
        from unittest.mock import MagicMock
        gen = BakkesModCodeGenerator(llm=MagicMock(), query_engine=None)
        features = gen._detect_features(
            "Hook goal events and create a settings window with a toggle"
        )
        assert "event_hooks" in features
        assert "settings_window" in features

    def test_code_validator_on_template_output(self):
        """Templates should pass validation without errors."""
        from bakkesmod_rag.code_generator import PluginTemplateEngine, CodeValidator
        te = PluginTemplateEngine()
        v = CodeValidator()
        files = te.generate_complete_project(
            plugin_name="TestPlugin",
            description="A test plugin",
            features=["event_hooks", "cvars"],
        )
        result = v.validate_project(files)
        assert result["valid"] is True, f"Validation errors: {result['errors']}"
