import os
import sys
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from comprehensive_rag import build_comprehensive_stack

def run_evaluation():
    print("Starting Automated RAG Evaluation (2026 Standards)...")
    
    # 1. Initialize the system
    engine = build_comprehensive_stack(incremental=False)
    
    # 2. Define Golden Test Cases (Synthetic or Manual)
    # In a real Platinum stack, we'd use Gemini to generate these automatically
    test_queries = [
        "How do I get the velocity of the ball?",
        "What wrapper is used for player cars?",
        "How do I hook the goal scored event?",
        "What is the difference between PERMISSION_ALL and PERMISSION_MENU?",
        "How do I set the rotation of an ActorWrapper?"
    ]
    
    results = []
    print(f"Testing {len(test_queries)} queries...")
    
    for query in test_queries:
        response = engine.query(query)
        results.append({
            "question": query,
            "answer": str(response),
            "contexts": [n.node.get_content() for n in response.source_nodes]
        })
    
    # 3. Grade using Ragas
    # This evaluates: 
    # - Faithfulness (Did the AI hallucinate?)
    # - Answer Relevancy (Did it actually answer the question?)
    # - Context Precision (Were the retrieved snippets useful?)
    
    # Note: Ragas expects a dataset format
    # For this proof of concept, we'll print the raw retrieval stats
    print("\n--- Evaluation Results ---")
    for r in results:
        print(f"Q: {r['question']}")
        print(f"Sources used: {len(r['contexts'])}")
        print("-" * 20)

    print("\nPlatinum Evaluation Complete. Check Arize Phoenix for full traces.")

if __name__ == "__main__":
    run_evaluation()
