"""Lightweight integration tests for the RAG engine pipeline.

These tests verify that core subsystems initialise and return data
without running the full (expensive) RAG pipeline.  They auto-skip
when API keys are absent.

Run only integration tests:  pytest -m integration -v
"""

import os
import pytest

_HAS_KEYS = bool(os.getenv("OPENAI_API_KEY"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _HAS_KEYS, reason="No API keys available"),
]


# ---------------------------------------------------------------------------
# LLM / Embedding connectivity (no RAGEngine needed)
# ---------------------------------------------------------------------------

class TestProviderConnectivity:
    """Verify that at least one LLM provider responds."""

    def test_llm_responds(self):
        from bakkesmod_rag.config import get_config
        from bakkesmod_rag.llm_provider import get_llm
        llm = get_llm(get_config())
        response = llm.complete("Say OK")
        assert response and len(str(response).strip()) > 0

    def test_embed_model_responds(self):
        from bakkesmod_rag.config import get_config
        from bakkesmod_rag.llm_provider import get_embed_model
        embed = get_embed_model(get_config())
        vec = embed.get_text_embedding("test")
        assert isinstance(vec, list)
        assert len(vec) > 0


# ---------------------------------------------------------------------------
# Document loading (no API calls)
# ---------------------------------------------------------------------------

class TestDocumentLoading:
    """Verify documents load from disk without building full indexes."""

    def test_loads_documents(self):
        from bakkesmod_rag.config import get_config
        from bakkesmod_rag.document_loader import load_documents
        docs = load_documents(get_config())
        assert len(docs) > 100, f"Expected 100+ docs, got {len(docs)}"

    def test_parses_nodes(self):
        from bakkesmod_rag.config import get_config
        from bakkesmod_rag.document_loader import load_documents, parse_nodes
        config = get_config()
        docs = load_documents(config)
        nodes = parse_nodes(docs, config)
        assert len(nodes) > 0


# ---------------------------------------------------------------------------
# Subsystem smoke tests (no full query pipeline)
# ---------------------------------------------------------------------------

class TestSubsystemSmoke:
    """Quick checks that subsystems initialise without errors."""

    def test_query_rewriter(self):
        from bakkesmod_rag.query_rewriter import QueryRewriter
        rw = QueryRewriter(llm=None, use_llm=False)
        expanded = rw.rewrite("How do I hook events?")
        assert len(expanded) > 0

    def test_query_decomposer(self):
        from bakkesmod_rag.query_decomposer import QueryDecomposer
        d = QueryDecomposer(llm=None, enable_decomposition=True)
        subs = d.decompose("What is BakkesMod?")
        assert len(subs) >= 1

    def test_semantic_cache_roundtrip(self):
        from bakkesmod_rag.config import get_config
        from bakkesmod_rag.llm_provider import get_embed_model
        from bakkesmod_rag.cache import SemanticCache
        embed = get_embed_model(get_config())
        cache = SemanticCache(
            cache_dir=".cache/test_smoke",
            embed_model=embed,
            ttl_seconds=60,
        )
        cache.set("test question", "test answer", [])
        hit = cache.get("test question")
        assert hit is not None

    def test_code_validator(self):
        from bakkesmod_rag.code_generator import CodeValidator
        v = CodeValidator()
        result = v.validate_syntax("int main() { return 0; }")
        assert result["valid"] is True

    def test_confidence_scoring(self):
        from bakkesmod_rag.confidence import calculate_confidence
        conf, label, expl = calculate_confidence([])
        assert 0.0 <= conf <= 1.0
