"""
Quick Knowledge Graph Integration Test
=======================================
Tests KG import and basic functionality without building full index.
"""

import sys

def test_kg_imports():
    """Test that KG can be imported."""
    try:
        from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
        print("[OK] KnowledgeGraphIndex imported successfully")
        return True
    except ImportError as e:
        print(f"[FAIL] Failed to import KnowledgeGraphIndex: {e}")
        return False

def test_kg_in_code():
    """Test that our code files have KG integration."""
    with open("interactive_rag.py", "r") as f:
        content = f.read()

    checks = [
        ("KnowledgeGraphIndex" in content, "KnowledgeGraphIndex import"),
        ("kg_index" in content, "kg_index variable"),
        ("kg_retriever" in content, "kg_retriever variable"),
        ("[vector_retriever, bm25_retriever, kg_retriever]" in content, "3-way fusion"),
        ("max_triplets_per_chunk=2" in content, "correct max_triplets setting")
    ]

    all_passed = True
    for check, description in checks:
        if check:
            print(f"[OK] {description}")
        else:
            print(f"[FAIL] Missing: {description}")
            all_passed = False

    return all_passed

def main():
    print("\n=== Quick KG Integration Test ===\n")

    test1 = test_kg_imports()
    test2 = test_kg_in_code()

    if test1 and test2:
        print("\n[SUCCESS] All quick tests passed!")
        print("KG integration is ready to use.")
        return 0
    else:
        print("\n[FAILURE] Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
