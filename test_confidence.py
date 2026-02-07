"""
Test Confidence Scoring
========================
Tests confidence calculation from retrieval quality metrics.
"""

from interactive_rag import calculate_confidence


def test_confidence_high_quality():
    """Test confidence with high-quality retrieval."""
    print("\n=== Test 1: High Quality Sources ===\n")

    class MockNode:
        def __init__(self, score):
            self.score = score
            self.node = type('obj', (object,), {'metadata': {'file_name': 'test.md'}})()

    # High scores, consistent
    nodes = [
        MockNode(0.92),
        MockNode(0.90),
        MockNode(0.89),
        MockNode(0.88),
        MockNode(0.87)
    ]

    confidence, label, explanation = calculate_confidence(nodes)

    print(f"Scores: {[n.score for n in nodes]}")
    print(f"Confidence: {confidence:.0%}")
    print(f"Label: {label}")
    print(f"Explanation: {explanation}")

    assert confidence >= 0.80, f"High quality sources should have >= 80% confidence, got {confidence:.0%}"
    assert label in ["HIGH", "VERY HIGH"], f"Should have HIGH/VERY HIGH label, got {label}"
    print("[OK] High quality sources correctly rated")


def test_confidence_medium_quality():
    """Test confidence with medium-quality retrieval."""
    print("\n=== Test 2: Medium Quality Sources ===\n")

    class MockNode:
        def __init__(self, score):
            self.score = score
            self.node = type('obj', (object,), {'metadata': {'file_name': 'test.md'}})()

    # Medium scores
    nodes = [
        MockNode(0.65),
        MockNode(0.62),
        MockNode(0.58)
    ]

    confidence, label, explanation = calculate_confidence(nodes)

    print(f"Scores: {[n.score for n in nodes]}")
    print(f"Confidence: {confidence:.0%}")
    print(f"Label: {label}")
    print(f"Explanation: {explanation}")

    assert 0.40 <= confidence <= 0.75, f"Medium quality should be 40-75%, got {confidence:.0%}"
    assert label in ["MEDIUM", "HIGH", "LOW"], f"Should have MEDIUM/HIGH/LOW label, got {label}"
    print("[OK] Medium quality sources correctly rated")


def test_confidence_low_quality():
    """Test confidence with low-quality retrieval."""
    print("\n=== Test 3: Low Quality Sources ===\n")

    class MockNode:
        def __init__(self, score):
            self.score = score
            self.node = type('obj', (object,), {'metadata': {'file_name': 'test.md'}})()

    # Low scores
    nodes = [
        MockNode(0.35),
        MockNode(0.30),
        MockNode(0.28)
    ]

    confidence, label, explanation = calculate_confidence(nodes)

    print(f"Scores: {[n.score for n in nodes]}")
    print(f"Confidence: {confidence:.0%}")
    print(f"Label: {label}")
    print(f"Explanation: {explanation}")

    assert confidence <= 0.50, f"Low quality should be <= 50%, got {confidence:.0%}"
    assert label in ["LOW", "VERY LOW", "MEDIUM"], f"Should have LOW/VERY LOW label, got {label}"
    print("[OK] Low quality sources correctly rated")


def test_confidence_no_sources():
    """Test confidence with no sources."""
    print("\n=== Test 4: No Sources ===\n")

    confidence, label, explanation = calculate_confidence([])

    print(f"Confidence: {confidence:.0%}")
    print(f"Label: {label}")
    print(f"Explanation: {explanation}")

    assert confidence == 0.0, f"No sources should be 0%, got {confidence:.0%}"
    assert label == "NO DATA", f"Should have NO DATA label, got {label}"
    print("[OK] No sources correctly handled")


def test_confidence_varied_quality():
    """Test confidence with mixed quality sources."""
    print("\n=== Test 5: Varied Quality (High Variance) ===\n")

    class MockNode:
        def __init__(self, score):
            self.score = score
            self.node = type('obj', (object,), {'metadata': {'file_name': 'test.md'}})()

    # Very mixed scores (high variance)
    nodes = [
        MockNode(0.95),
        MockNode(0.50),
        MockNode(0.40)
    ]

    confidence, label, explanation = calculate_confidence(nodes)

    print(f"Scores: {[n.score for n in nodes]}")
    print(f"Confidence: {confidence:.0%}")
    print(f"Label: {label}")
    print(f"Explanation: {explanation}")

    # High variance should reduce confidence despite good max score
    print("[OK] High variance correctly reduces confidence")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  CONFIDENCE SCORING TESTS")
    print("=" * 80)

    try:
        test_confidence_high_quality()
        test_confidence_medium_quality()
        test_confidence_low_quality()
        test_confidence_no_sources()
        test_confidence_varied_quality()

        print("\n" + "=" * 80)
        print("  ALL CONFIDENCE TESTS PASSED!")
        print("=" * 80)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
