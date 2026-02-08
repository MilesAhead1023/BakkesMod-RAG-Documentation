"""Tests for confidence scoring logic."""

import pytest
from bakkesmod_rag.confidence import calculate_confidence


class TestConfidenceScoring:
    def test_no_nodes_returns_zero(self):
        confidence, label, explanation = calculate_confidence([])
        assert confidence == 0.0
        assert label == "NO DATA"

    def test_high_score_nodes(self, source_nodes_high):
        confidence, label, explanation = calculate_confidence(source_nodes_high)
        assert confidence >= 0.70
        assert label in ("HIGH", "VERY HIGH")

    def test_low_score_nodes(self, source_nodes_low):
        confidence, label, explanation = calculate_confidence(source_nodes_low)
        assert confidence < 0.50
        assert label in ("LOW", "VERY LOW")

    def test_single_perfect_node(self):
        from tests.conftest import MockSourceNode
        nodes = [MockSourceNode(score=1.0)]
        confidence, label, _ = calculate_confidence(nodes)
        assert confidence >= 0.80

    def test_nodes_without_scores(self):
        from tests.conftest import MockSourceNode
        node = MockSourceNode(score=None)
        node.score = None
        confidence, label, _ = calculate_confidence([node])
        assert confidence == 0.5
        assert label == "MEDIUM"

    def test_confidence_bounded_0_1(self, source_nodes_high):
        confidence, _, _ = calculate_confidence(source_nodes_high)
        assert 0.0 <= confidence <= 1.0

    def test_varied_scores(self):
        from tests.conftest import MockSourceNode
        nodes = [
            MockSourceNode(score=0.95),
            MockSourceNode(score=0.30),
            MockSourceNode(score=0.50),
        ]
        confidence, label, _ = calculate_confidence(nodes)
        assert 0.30 <= confidence <= 0.80

    def test_label_tiers(self):
        from tests.conftest import MockSourceNode

        # Very high
        nodes_vh = [MockSourceNode(score=0.99) for _ in range(5)]
        _, label_vh, _ = calculate_confidence(nodes_vh)
        assert label_vh == "VERY HIGH"

        # Very low
        nodes_vl = [MockSourceNode(score=0.05)]
        _, label_vl, _ = calculate_confidence(nodes_vl)
        assert label_vl in ("LOW", "VERY LOW")
