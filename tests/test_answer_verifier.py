"""Tests for answer verification (embedding + LLM grounding check)."""

import json
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from bakkesmod_rag.answer_verifier import AnswerVerifier, VerificationResult
from bakkesmod_rag.config import RAGConfig, VerificationConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embed_model(similarity: float = 0.85):
    """Create a mock embed model returning controlled embeddings.

    When similarity=0.85, the embeddings will produce cosine
    similarity ~0.85 between answer and source.
    """
    model = MagicMock()

    # Fixed base vector
    base = np.random.RandomState(42).randn(384)
    base = base / np.linalg.norm(base)

    # Orthogonal vector for mixing
    orth = np.random.RandomState(99).randn(384)
    orth = orth - np.dot(orth, base) * base
    orth = orth / np.linalg.norm(orth)

    # Answer embedding is the base vector
    answer_emb = base.tolist()
    # Source embedding is mixed to achieve target similarity
    source_emb = (similarity * base + np.sqrt(1 - similarity**2) * orth).tolist()

    call_count = [0]

    def get_embedding(text):
        call_count[0] += 1
        if call_count[0] == 1:
            return answer_emb
        return source_emb

    model.get_text_embedding = MagicMock(side_effect=get_embedding)
    return model


def _make_source_node(text: str, score: float = 0.8):
    """Create a mock source node."""
    node = MagicMock()
    node.text = text
    node.node.text = text
    node.score = score
    return node


def _make_llm(grounded: bool = True, claims: list = None):
    """Create a mock LLM returning verification JSON."""
    llm = MagicMock()
    result = {
        "grounded": grounded,
        "unsupported_claims": claims or [],
    }
    response = MagicMock()
    response.text = json.dumps(result)
    llm.complete.return_value = response
    return llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVerificationResult:
    """Test VerificationResult defaults."""

    def test_defaults(self):
        vr = VerificationResult()
        assert vr.grounded is True
        assert vr.grounding_score == 1.0
        assert vr.confidence_penalty == 0.0
        assert vr.warning is None
        assert vr.ungrounded_claims == []


class TestAnswerVerifier:
    """Tests for AnswerVerifier."""

    def test_disabled_returns_default(self):
        """Disabled verifier returns default (grounded) result."""
        v = AnswerVerifier(enabled=False)
        result = v.verify("answer", [_make_source_node("source")], "query")
        assert result.grounded is True
        assert result.confidence_penalty == 0.0

    def test_empty_answer_returns_default(self):
        v = AnswerVerifier(enabled=True)
        result = v.verify("", [_make_source_node("source")], "query")
        assert result.grounded is True

    def test_no_sources_returns_ungrounded(self):
        v = AnswerVerifier(enabled=True)
        result = v.verify("some answer", [], "query")
        assert result.grounded is False
        assert result.confidence_penalty == 0.30
        assert result.warning is not None

    def test_grounded_high_similarity(self):
        """High similarity → grounded, no penalty."""
        embed = _make_embed_model(similarity=0.85)
        v = AnswerVerifier(embed_model=embed, enabled=True)
        source = _make_source_node("BakkesMod SDK documentation text")

        result = v.verify("Answer about BakkesMod", [source], "query")
        assert result.grounded is True
        assert result.confidence_penalty == 0.0
        assert result.warning is None

    def test_ungrounded_low_similarity(self):
        """Low similarity → ungrounded, penalty applied."""
        embed = _make_embed_model(similarity=0.40)
        v = AnswerVerifier(embed_model=embed, enabled=True)
        source = _make_source_node("Unrelated text")

        result = v.verify("Very different answer", [source], "query")
        assert result.grounded is False
        assert result.confidence_penalty == 0.30
        assert result.warning is not None

    def test_borderline_with_llm_grounded(self):
        """Borderline similarity triggers LLM check — LLM says grounded."""
        embed = _make_embed_model(similarity=0.65)
        llm = _make_llm(grounded=True)
        v = AnswerVerifier(embed_model=embed, llm=llm, enabled=True)
        source = _make_source_node("Source text")

        result = v.verify("Borderline answer", [source], "query")
        assert result.grounded is True
        assert result.confidence_penalty == 0.0
        llm.complete.assert_called_once()

    def test_borderline_with_llm_ungrounded(self):
        """Borderline similarity triggers LLM check — LLM says ungrounded."""
        embed = _make_embed_model(similarity=0.65)
        llm = _make_llm(grounded=False, claims=["Claim X not in sources"])
        v = AnswerVerifier(embed_model=embed, llm=llm, enabled=True)
        source = _make_source_node("Source text")

        result = v.verify("Borderline answer", [source], "query")
        assert result.grounded is False
        assert result.confidence_penalty == 0.15
        assert len(result.ungrounded_claims) == 1

    def test_borderline_no_llm_cautionary(self):
        """Borderline without LLM → mild penalty + cautionary warning."""
        embed = _make_embed_model(similarity=0.65)
        v = AnswerVerifier(embed_model=embed, llm=None, enabled=True)
        source = _make_source_node("Source text")

        result = v.verify("Borderline answer", [source], "query")
        assert result.grounded is True
        assert result.confidence_penalty == 0.15
        assert result.warning is not None

    def test_no_embed_model_assumes_grounded(self):
        """No embed model → embedding check returns 1.0 (grounded)."""
        v = AnswerVerifier(embed_model=None, enabled=True)
        source = _make_source_node("Source text")

        result = v.verify("Some answer", [source], "query")
        assert result.grounded is True

    def test_embed_error_fails_open(self):
        """Embedding error → fail open (assume grounded)."""
        embed = MagicMock()
        embed.get_text_embedding.side_effect = RuntimeError("API error")
        v = AnswerVerifier(embed_model=embed, enabled=True)
        source = _make_source_node("Source text")

        result = v.verify("Some answer", [source], "query")
        assert result.grounded is True
        assert result.confidence_penalty == 0.0

    def test_llm_error_fails_open(self):
        """LLM verification error → fail open with mild penalty."""
        embed = _make_embed_model(similarity=0.65)
        llm = MagicMock()
        llm.complete.side_effect = RuntimeError("LLM error")
        v = AnswerVerifier(embed_model=embed, llm=llm, enabled=True)
        source = _make_source_node("Source text")

        result = v.verify("Borderline answer", [source], "query")
        assert result.grounded is True
        assert result.confidence_penalty < 0.15  # Half of borderline penalty

    def test_confidence_penalty_never_negative(self):
        """Applying penalty doesn't result in negative confidence."""
        embed = _make_embed_model(similarity=0.40)
        v = AnswerVerifier(
            embed_model=embed,
            enabled=True,
            ungrounded_penalty=0.30,
        )
        source = _make_source_node("Source")

        result = v.verify("Ungrounded", [source], "query")
        # The penalty itself is 0.30 — caller must clamp confidence
        assert result.confidence_penalty == 0.30

    def test_custom_thresholds(self):
        """Custom threshold values are respected."""
        embed = _make_embed_model(similarity=0.80)
        v = AnswerVerifier(
            embed_model=embed,
            enabled=True,
            grounded_threshold=0.90,  # Higher bar
        )
        source = _make_source_node("Source")

        # 0.80 is below custom grounded threshold of 0.90
        result = v.verify("Answer", [source], "query")
        # Should be borderline (between 0.55 and 0.90)
        assert result.confidence_penalty > 0


class TestParseLLMJson:
    """Tests for _parse_llm_json."""

    def test_valid_json(self):
        text = '{"grounded": true, "unsupported_claims": []}'
        result = AnswerVerifier._parse_llm_json(text)
        assert result["grounded"] is True

    def test_json_in_code_block(self):
        text = '```json\n{"grounded": false, "unsupported_claims": ["a"]}\n```'
        result = AnswerVerifier._parse_llm_json(text)
        assert result["grounded"] is False

    def test_json_with_surrounding_text(self):
        text = 'Here is my analysis: {"grounded": true, "unsupported_claims": []} Done.'
        result = AnswerVerifier._parse_llm_json(text)
        assert result["grounded"] is True

    def test_invalid_json_returns_default(self):
        text = "This is not JSON at all."
        result = AnswerVerifier._parse_llm_json(text)
        assert result["grounded"] is True
        assert result["unsupported_claims"] == []


class TestCosineSimilarity:
    """Tests for _cosine_similarity."""

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        sim = AnswerVerifier._cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        sim = AnswerVerifier._cosine_similarity(a, b)
        assert abs(sim) < 1e-6

    def test_zero_vector(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        sim = AnswerVerifier._cosine_similarity(a, b)
        assert sim == 0.0


class TestConfigDefaults:
    """Test VerificationConfig defaults."""

    def test_defaults(self):
        cfg = VerificationConfig()
        assert cfg.enabled is True
        assert cfg.grounded_threshold == 0.75
        assert cfg.borderline_threshold == 0.55
        assert cfg.borderline_confidence_penalty == 0.15
        assert cfg.ungrounded_confidence_penalty == 0.30

    def test_rag_config_includes_verification(self):
        cfg = RAGConfig()
        assert hasattr(cfg, "verification")
        assert cfg.verification.enabled is True
