"""
Confidence Scoring
==================
Calculates confidence scores from retrieval quality metrics.
"""

from typing import Tuple


def calculate_confidence(source_nodes: list) -> Tuple[float, str, str]:
    """Calculate confidence score from retrieval quality.

    Args:
        source_nodes: List of retrieved source nodes with scores.

    Returns:
        Tuple of (confidence_score, confidence_label, explanation).
    """
    if not source_nodes:
        return 0.0, "NO DATA", "No sources retrieved"

    scores = [
        node.score
        for node in source_nodes
        if hasattr(node, "score") and node.score is not None
    ]

    if not scores:
        return 0.5, "MEDIUM", "Sources retrieved but no similarity scores available"

    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    num_sources = len(source_nodes)

    if len(scores) > 1:
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
    else:
        std_dev = 0.0

    # Weighted confidence calculation:
    #   Average score (50%) + Max score (20%) + Source count bonus (10%) + Consistency (20%)
    score_component = avg_score * 50
    max_component = max_score * 20
    source_bonus = min(num_sources / 5.0, 1.0) * 10
    consistency_component = max(0, (1 - std_dev) * 20)

    confidence = (
        score_component + max_component + source_bonus + consistency_component
    ) / 100.0
    confidence = max(0.0, min(1.0, confidence))

    if confidence >= 0.85:
        label = "VERY HIGH"
        explanation = "Excellent source match with high consistency"
    elif confidence >= 0.70:
        label = "HIGH"
        explanation = "Strong source match with good relevance"
    elif confidence >= 0.50:
        label = "MEDIUM"
        explanation = "Moderate source match, answer should be helpful"
    elif confidence >= 0.30:
        label = "LOW"
        explanation = "Weak source match, verify answer carefully"
    else:
        label = "VERY LOW"
        explanation = "Poor source match, answer may be unreliable"

    return confidence, label, explanation
