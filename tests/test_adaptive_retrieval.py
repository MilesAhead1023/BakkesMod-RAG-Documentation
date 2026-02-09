"""Tests for adaptive retrieval — dynamic top_k escalation."""

import pytest
from unittest.mock import MagicMock, PropertyMock

from bakkesmod_rag.retrieval import adjust_retriever_top_k, reset_retriever_top_k
from bakkesmod_rag.config import RAGConfig, RetrieverConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_retriever(name: str, top_k: int = 5):
    """Create a mock retriever with similarity_top_k."""
    r = MagicMock()
    r.__class__.__name__ = name
    type(r).__name__ = name
    r._similarity_top_k = top_k
    return r


def _make_mock_fusion(retrievers):
    """Create a mock QueryFusionRetriever."""
    fusion = MagicMock()
    fusion.retrievers = retrievers
    return fusion


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAdjustTopK:
    """Tests for adjust_retriever_top_k."""

    def test_attempt_0_uses_first_value(self):
        """Attempt 0 sets top_k to first escalation value."""
        r1 = _make_mock_retriever("VectorRetriever")
        r2 = _make_mock_retriever("BM25Retriever")
        fusion = _make_mock_fusion([r1, r2])
        config = RAGConfig(retriever=RetrieverConfig(
            adaptive_top_k=True,
            top_k_escalation=[5, 8, 12],
        ))

        adjust_retriever_top_k(fusion, attempt=0, config=config)
        assert r1._similarity_top_k == 5
        assert r2._similarity_top_k == 5

    def test_attempt_1_escalates(self):
        """Attempt 1 escalates top_k."""
        r1 = _make_mock_retriever("VectorRetriever")
        r2 = _make_mock_retriever("BM25Retriever")
        fusion = _make_mock_fusion([r1, r2])
        config = RAGConfig(retriever=RetrieverConfig(
            adaptive_top_k=True,
            top_k_escalation=[5, 8, 12],
        ))

        adjust_retriever_top_k(fusion, attempt=1, config=config)
        assert r1._similarity_top_k == 8
        assert r2._similarity_top_k == 8

    def test_attempt_2_max_escalation(self):
        """Attempt 2 hits maximum escalation."""
        r1 = _make_mock_retriever("VectorRetriever")
        fusion = _make_mock_fusion([r1])
        config = RAGConfig(retriever=RetrieverConfig(
            adaptive_top_k=True,
            top_k_escalation=[5, 8, 12],
        ))

        adjust_retriever_top_k(fusion, attempt=2, config=config)
        assert r1._similarity_top_k == 12

    def test_attempt_beyond_list_clamps(self):
        """Attempt beyond escalation list clamps to last value."""
        r1 = _make_mock_retriever("VectorRetriever")
        fusion = _make_mock_fusion([r1])
        config = RAGConfig(retriever=RetrieverConfig(
            adaptive_top_k=True,
            top_k_escalation=[5, 8, 12],
        ))

        adjust_retriever_top_k(fusion, attempt=10, config=config)
        assert r1._similarity_top_k == 12

    def test_kg_retriever_uses_kg_escalation(self):
        """KG retriever uses separate kg_top_k_escalation."""
        r1 = _make_mock_retriever("VectorRetriever")
        r2 = _make_mock_retriever("KnowledgeGraphRetriever")
        fusion = _make_mock_fusion([r1, r2])
        config = RAGConfig(retriever=RetrieverConfig(
            adaptive_top_k=True,
            top_k_escalation=[5, 8, 12],
            kg_top_k_escalation=[3, 5, 8],
        ))

        adjust_retriever_top_k(fusion, attempt=1, config=config)
        assert r1._similarity_top_k == 8
        assert r2._similarity_top_k == 5

    def test_disabled_does_nothing(self):
        """When adaptive_top_k=False, no changes made."""
        r1 = _make_mock_retriever("VectorRetriever", top_k=5)
        fusion = _make_mock_fusion([r1])
        config = RAGConfig(retriever=RetrieverConfig(adaptive_top_k=False))

        adjust_retriever_top_k(fusion, attempt=2, config=config)
        assert r1._similarity_top_k == 5  # unchanged


class TestResetTopK:
    """Tests for reset_retriever_top_k."""

    def test_reset_restores_baseline(self):
        """Reset brings top_k back to attempt=0 values."""
        r1 = _make_mock_retriever("VectorRetriever", top_k=12)
        r2 = _make_mock_retriever("KnowledgeGraphRetriever", top_k=8)
        fusion = _make_mock_fusion([r1, r2])
        config = RAGConfig(retriever=RetrieverConfig(
            adaptive_top_k=True,
            top_k_escalation=[5, 8, 12],
            kg_top_k_escalation=[3, 5, 8],
        ))

        reset_retriever_top_k(fusion, config=config)
        assert r1._similarity_top_k == 5
        assert r2._similarity_top_k == 3


class TestConfigDefaults:
    """Tests for adaptive retrieval config defaults."""

    def test_defaults(self):
        cfg = RetrieverConfig()
        assert cfg.adaptive_top_k is True
        assert cfg.top_k_escalation == [5, 8, 12]
        assert cfg.kg_top_k_escalation == [3, 5, 8]

    def test_custom_escalation(self):
        cfg = RetrieverConfig(
            top_k_escalation=[3, 6, 10],
            kg_top_k_escalation=[2, 4, 6],
        )
        assert cfg.top_k_escalation == [3, 6, 10]
        assert cfg.kg_top_k_escalation == [2, 4, 6]
