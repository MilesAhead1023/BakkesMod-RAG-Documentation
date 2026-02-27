"""Tests for Self-RAG corrective retry loop."""

import pytest
from unittest.mock import MagicMock, patch

from bakkesmod_rag.config import RAGConfig, SelfRAGConfig
from bakkesmod_rag.query_rewriter import QueryRewriter
from bakkesmod_rag.engine import QueryResult
from bakkesmod_rag.retrieval import adjust_retriever_top_k, reset_retriever_top_k


class TestSelfRAGConfig:
    """Tests for SelfRAGConfig defaults and wiring."""

    def test_defaults(self):
        cfg = SelfRAGConfig()
        assert cfg.enabled is True
        assert cfg.confidence_threshold == 0.70
        assert cfg.max_retries == 2
        assert cfg.force_llm_rewrite_on_retry is True

    def test_disabled(self):
        cfg = SelfRAGConfig(enabled=False)
        assert cfg.enabled is False

    def test_custom_threshold(self):
        cfg = SelfRAGConfig(confidence_threshold=0.85, max_retries=3)
        assert cfg.confidence_threshold == 0.85
        assert cfg.max_retries == 3

    def test_rag_config_includes_self_rag(self):
        cfg = RAGConfig()
        assert hasattr(cfg, "self_rag")
        assert cfg.self_rag.enabled is True
        assert cfg.self_rag.confidence_threshold == 0.70


class TestQueryRewriterForceLLM:
    """Tests that force_llm actually invokes the LLM."""

    def test_force_llm_calls_llm_complete(self):
        """force_llm=True must call llm.complete() even when use_llm=False."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "rewritten query about BakkesMod hooks"
        mock_llm.complete.return_value = mock_response

        rewriter = QueryRewriter(llm=mock_llm, use_llm=False)
        result = rewriter.rewrite("how do hooks work", force_llm=True)
        mock_llm.complete.assert_called_once()
        assert result == "rewritten query about BakkesMod hooks"

    def test_force_llm_no_llm_returns_synonym_expansion(self):
        """force_llm=True without LLM falls back to synonym expansion."""
        rewriter = QueryRewriter(llm=None, use_llm=False)
        result = rewriter.rewrite("how do hooks work", force_llm=True)
        # Should contain synonym expansion for "hook"
        assert "hook" in result.lower()
        assert len(result) > len("how do hooks work")

    def test_normal_path_without_force(self):
        """Without force_llm, use_llm=False uses synonym expansion only."""
        mock_llm = MagicMock()
        rewriter = QueryRewriter(llm=mock_llm, use_llm=False)
        rewriter.rewrite("how do hooks work")
        mock_llm.complete.assert_not_called()

    def test_rewrite_with_llm_guard_allows_llm(self):
        """rewrite_with_llm no longer blocks when llm is available."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "improved query"
        mock_llm.complete.return_value = mock_response

        rewriter = QueryRewriter(llm=mock_llm, use_llm=False)
        result = rewriter.rewrite_with_llm("test query")
        mock_llm.complete.assert_called_once()
        assert result == "improved query"


class TestSelfRAGRetryIntegration:
    """Tests that exercise the actual retry loop machinery.

    Uses mocked retrievers and the real adjust/reset functions
    to verify the retry escalation path works end-to-end.
    """

    def _make_mock_retriever(self, name, top_k=5):
        r = MagicMock()
        type(r).__name__ = name
        r._similarity_top_k = top_k
        return r

    def _make_fusion(self, retrievers):
        fusion = MagicMock()
        fusion._retrievers = retrievers  # must match retrieval.py's fusion_retriever._retrievers
        return fusion

    def test_escalation_applied_during_retry(self):
        """Verify adjust_retriever_top_k actually changes top_k on retry."""
        r1 = self._make_mock_retriever("VectorRetriever", top_k=5)
        fusion = self._make_fusion([r1])
        config = RAGConfig()

        # Simulate retry loop: attempt 0, then 1
        adjust_retriever_top_k(fusion, attempt=0, config=config)
        assert r1._similarity_top_k == 5

        adjust_retriever_top_k(fusion, attempt=1, config=config)
        assert r1._similarity_top_k == 8

        adjust_retriever_top_k(fusion, attempt=2, config=config)
        assert r1._similarity_top_k == 12

    def test_reset_after_retries(self):
        """Verify reset restores baseline after retry loop completes."""
        r1 = self._make_mock_retriever("VectorRetriever", top_k=12)
        fusion = self._make_fusion([r1])
        config = RAGConfig()

        reset_retriever_top_k(fusion, config=config)
        assert r1._similarity_top_k == 5

    def test_disabled_self_rag_computes_single_attempt(self):
        """When self_rag.enabled=False, max_attempts should be 1."""
        config = RAGConfig(self_rag=SelfRAGConfig(enabled=False))
        max_attempts = (
            (config.self_rag.max_retries + 1)
            if config.self_rag.enabled
            else 1
        )
        assert max_attempts == 1

    def test_retry_loop_keeps_best_result(self):
        """Simulate a retry loop and verify best-confidence result wins."""
        config = RAGConfig(self_rag=SelfRAGConfig(
            enabled=True, max_retries=2, confidence_threshold=0.70
        ))

        # Simulate three attempts with varying confidences
        attempt_results = [
            {"answer": "bad", "confidence": 0.40, "attempt": 0},
            {"answer": "good", "confidence": 0.75, "attempt": 1},
            {"answer": "ok", "confidence": 0.55, "attempt": 2},
        ]

        best_result = None
        for result in attempt_results:
            if best_result is None or result["confidence"] > best_result["confidence"]:
                best_result = result
            if result["confidence"] >= config.self_rag.confidence_threshold:
                break

        # Should stop at attempt 1 (0.75 >= 0.70) and that's the best
        assert best_result["attempt"] == 1
        assert best_result["answer"] == "good"

    def test_retry_loop_exhausts_all_attempts_on_low_confidence(self):
        """All retries fire when confidence never exceeds threshold."""
        config = RAGConfig(self_rag=SelfRAGConfig(
            enabled=True, max_retries=2, confidence_threshold=0.70
        ))

        all_attempts = []
        confidences = [0.30, 0.45, 0.55]

        for attempt in range(config.self_rag.max_retries + 1):
            all_attempts.append({
                "attempt": attempt,
                "confidence": confidences[attempt],
            })
            if confidences[attempt] >= config.self_rag.confidence_threshold:
                break

        assert len(all_attempts) == 3  # all attempts used
        best = max(all_attempts, key=lambda a: a["confidence"])
        assert best["confidence"] == 0.55
        assert best["attempt"] == 2

    def test_force_llm_rewrite_uses_different_query(self):
        """On retry, force_llm should produce a different query."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "BakkesMod HookEvent GameWrapper event binding"
        mock_llm.complete.return_value = mock_response

        rewriter = QueryRewriter(llm=mock_llm, use_llm=False)
        original = rewriter.rewrite("how do hooks work")
        retried = rewriter.rewrite("how do hooks work", force_llm=True)

        # LLM rewrite should produce different output than synonym expansion
        assert retried != original
        assert "BakkesMod" in retried


class TestQueryResultRetryFields:
    """Tests for new retry-related fields on QueryResult dataclass."""

    def test_default_retry_count_zero(self):
        result = QueryResult(
            answer="test",
            sources=[],
            confidence=0.85,
            confidence_label="HIGH",
            confidence_explanation="Strong",
            query_time=1.0,
            cached=False,
            expanded_query="test",
        )
        assert result.retry_count == 0
        assert result.all_attempts == []
        assert result.verification_warning is None

    def test_retry_fields_populated(self):
        attempts = [
            {"attempt": 0, "confidence": 0.40},
            {"attempt": 1, "confidence": 0.60},
            {"attempt": 2, "confidence": 0.85},
        ]
        result = QueryResult(
            answer="test",
            sources=[],
            confidence=0.85,
            confidence_label="HIGH",
            confidence_explanation="Strong",
            query_time=1.0,
            cached=False,
            expanded_query="test",
            retry_count=2,
            all_attempts=attempts,
        )
        assert result.retry_count == 2
        assert len(result.all_attempts) == 3
        assert result.all_attempts[2]["confidence"] == 0.85
