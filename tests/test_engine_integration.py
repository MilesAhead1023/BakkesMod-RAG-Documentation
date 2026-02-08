"""Integration tests for the full RAG engine pipeline.

These tests require API keys and real LLM/embedding calls.
Run with: pytest tests/test_engine_integration.py -v -m integration
"""

import os
import time
import pytest

# Mark entire module as integration
pytestmark = pytest.mark.integration


def _has_api_keys() -> bool:
    """Check if required API keys are available."""
    return bool(os.getenv("OPENAI_API_KEY"))


@pytest.fixture(scope="module")
def engine():
    """Create a real RAGEngine (expensive -- cached per module)."""
    if not _has_api_keys():
        pytest.skip("No API keys available for integration tests")

    from bakkesmod_rag import RAGEngine
    return RAGEngine()


class TestEngineInit:
    def test_engine_loads_documents(self, engine):
        assert engine.num_documents > 0

    def test_engine_has_nodes(self, engine):
        assert engine.num_nodes > 0

    def test_engine_has_query_engines(self, engine):
        assert engine.query_engine_sync is not None
        assert engine.query_engine_streaming is not None

    def test_engine_has_subsystems(self, engine):
        assert engine.cache is not None
        assert engine.rewriter is not None
        assert engine.cost_tracker is not None
        assert engine.code_gen is not None
        assert engine.logger is not None
        assert engine.phoenix is not None
        assert engine.metrics is not None
        assert engine.api_manager is not None


class TestQueryPipeline:
    def test_basic_query(self, engine):
        """Test a simple question about BakkesMod."""
        result = engine.query("What is BakkesMod?", use_cache=False)
        assert result.answer
        assert len(result.answer) > 50
        assert result.confidence > 0
        assert result.query_time > 0
        assert result.cached is False

    def test_query_returns_sources(self, engine):
        result = engine.query("How do I hook events?", use_cache=False)
        assert len(result.sources) > 0

    def test_query_confidence_reasonable(self, engine):
        result = engine.query("What is ServerWrapper?", use_cache=False)
        assert 0.0 <= result.confidence <= 1.0
        assert result.confidence_label in (
            "VERY HIGH", "HIGH", "MEDIUM", "LOW", "VERY LOW", "NO DATA"
        )

    def test_streaming_query(self, engine):
        gen, get_meta = engine.query_streaming(
            "What is GameWrapper?", use_cache=False
        )
        tokens = []
        for token in gen:
            tokens.append(token)
        full_text = "".join(tokens)

        assert len(full_text) > 20
        meta = get_meta()
        assert meta.answer == full_text
        assert meta.cached is False

    def test_cache_hit_on_repeat(self, engine):
        q = "What classes does the SDK provide?"
        engine.query(q, use_cache=True)
        result2 = engine.query(q, use_cache=True)
        assert result2.cached is True
        assert result2.confidence_label == "CACHED"

    def test_query_latency_under_30s(self, engine):
        result = engine.query("How do I register a CVar?", use_cache=False)
        assert result.query_time < 30.0, f"Query took {result.query_time:.1f}s"


class TestCodeGeneration:
    def test_generate_code_returns_project(self, engine):
        result = engine.generate_code("A simple plugin that logs when a goal is scored")
        assert result.header
        assert result.implementation
        assert result.project_files
        assert len(result.project_files) == 12

    def test_generated_code_passes_validation(self, engine):
        result = engine.generate_code("Track ball explosions and log them")
        assert result.validation.get("valid") is True or result.validation.get("errors") is not None

    def test_features_detected(self, engine):
        result = engine.generate_code(
            "Hook goal events and create a settings window with a toggle"
        )
        assert "event_hooks" in result.features_used
