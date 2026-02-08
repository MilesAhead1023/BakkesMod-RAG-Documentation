"""
RAG Evaluator -- Automated Quality Testing
============================================
Runs a set of golden test queries through the RAG engine and reports
confidence scores, source counts, latency, and cache behaviour.

Rewrite of the legacy ``evaluator.py`` using the unified
``bakkesmod_rag`` package.  The ``ragas`` library is imported
optionally -- if it is not installed the evaluator still works,
it just skips the RAGAS metrics.

Usage::

    python -m bakkesmod_rag.evaluator
"""

import sys
import time
from typing import Dict, List, Optional

# Golden test cases -- representative BakkesMod SDK questions that exercise
# different areas of the documentation (wrappers, events, permissions, etc.).
DEFAULT_TEST_QUERIES = [
    "How do I get the velocity of the ball?",
    "What wrapper is used for player cars?",
    "How do I hook the goal scored event?",
    "What is the difference between PERMISSION_ALL and PERMISSION_MENU?",
    "How do I set the rotation of an ActorWrapper?",
]


def _try_ragas_evaluation(results: List[Dict]) -> Optional[Dict]:
    """Attempt RAGAS evaluation if the library is installed.

    Args:
        results: List of result dicts from ``run_evaluation``.

    Returns:
        Dict of RAGAS metric scores, or None if ragas is not available.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        from datasets import Dataset
    except ImportError:
        return None

    try:
        # Build a HuggingFace Dataset in the format RAGAS expects
        data = {
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [
                [s.get("file_name", "") for s in r["sources"]]
                for r in results
            ],
        }
        dataset = Dataset.from_dict(data)

        ragas_result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )
        return dict(ragas_result)
    except Exception as e:
        print(f"  [RAGAS] Evaluation failed: {e}")
        return None


def run_evaluation(
    queries: Optional[List[str]] = None,
    verbose: bool = True,
) -> List[Dict]:
    """Run evaluation against the RAG system.

    Initialises a ``RAGEngine``, sends each query through it, and
    collects timing, confidence, and source information.

    Args:
        queries: Test queries to run (uses ``DEFAULT_TEST_QUERIES`` if None).
        verbose: Print results to stdout as they are produced.

    Returns:
        List of result dicts with question, answer, sources, confidence,
        confidence_label, latency, and cached fields.
    """
    queries = queries or DEFAULT_TEST_QUERIES

    from bakkesmod_rag.engine import RAGEngine

    if verbose:
        print("Initialising RAG Engine for evaluation...")

    engine = RAGEngine()

    if verbose:
        print(f"Running {len(queries)} test queries...\n")

    results: List[Dict] = []

    for i, query in enumerate(queries, 1):
        start = time.time()
        result = engine.query(query)
        latency = time.time() - start

        entry = {
            "question": query,
            "answer": result.answer,
            "sources": result.sources,
            "confidence": result.confidence,
            "confidence_label": result.confidence_label,
            "confidence_explanation": result.confidence_explanation,
            "latency": latency,
            "cached": result.cached,
            "expanded_query": result.expanded_query,
        }
        results.append(entry)

        if verbose:
            print(f"[{i}/{len(queries)}] Q: {query}")
            print(f"  Confidence: {result.confidence:.0%} ({result.confidence_label})")
            print(f"  Sources: {len(result.sources)}")
            print(f"  Latency: {latency:.2f}s")
            print(f"  Cached: {result.cached}")
            # Truncate long answers for readability
            preview = result.answer[:120].replace("\n", " ")
            print(f"  Answer: {preview}...")
            print()

    # Summary statistics
    if verbose and results:
        avg_confidence = sum(r["confidence"] for r in results) / len(results)
        avg_latency = sum(r["latency"] for r in results) / len(results)
        total_sources = sum(len(r["sources"]) for r in results)
        cached_count = sum(1 for r in results if r["cached"])

        print("=" * 60)
        print(f"  Evaluation Summary: {len(results)} queries")
        print(f"  Avg confidence:  {avg_confidence:.0%}")
        print(f"  Avg latency:     {avg_latency:.2f}s")
        print(f"  Total sources:   {total_sources}")
        print(f"  Cache hits:      {cached_count}/{len(results)}")
        print("=" * 60)

        # Optional RAGAS metrics
        ragas_scores = _try_ragas_evaluation(results)
        if ragas_scores:
            print("\n  RAGAS Metrics:")
            for metric, score in ragas_scores.items():
                print(f"    {metric}: {score:.3f}")
            print("=" * 60)
        else:
            print("\n  (RAGAS not installed -- install 'ragas' for "
                  "faithfulness/relevancy/precision metrics)")

    return results


if __name__ == "__main__":
    run_evaluation()
