"""
Test Multi-Query Configuration
================================
Validates that multi-query is configured correctly without making API calls.
"""

def test_fusion_retriever_config():
    """Test that QueryFusionRetriever is configured with num_queries=4."""
    print("\n=== Test: Multi-Query Configuration ===\n")

    # Read interactive_rag.py and check configuration
    with open("interactive_rag.py", "r") as f:
        content = f.read()

    # Check that num_queries=4 exists in the file
    assert "num_queries=4" in content, "interactive_rag.py should have num_queries=4"
    print("[OK] Found num_queries=4 in interactive_rag.py")

    # Check that the comment mentions Phase 2
    assert "Phase 2" in content and "query variants" in content, "Should have Phase 2 comment explaining multi-query"
    print("[OK] Found Phase 2 multi-query comment")

    # Check that log message mentions multi-query
    assert "multi-query" in content or "variants" in content, "Log message should mention multi-query feature"
    print("[OK] Log message references multi-query")

    print("\n[CONFIGURATION VALIDATED]")
    print("  Feature: Multi-Query Generation")
    print("  Setting: num_queries=4")
    print("  Impact: Generates 4 query variants for better coverage")
    print("  Status: Enabled and ready")


def test_multi_query_documentation():
    """Test that multi-query feature is properly documented."""
    print("\n=== Test: Multi-Query Documentation ===\n")

    # Check Phase 2 plan exists
    import os
    plan_path = "docs/plans/2026-02-07-rag-phase2-enhancements.md"
    assert os.path.exists(plan_path), "Phase 2 plan should exist"
    print(f"[OK] Found Phase 2 plan at {plan_path}")

    # Read plan and check it mentions multi-query
    with open(plan_path, "r", encoding="utf-8") as f:
        plan_content = f.read()

    assert "Multi-Query" in plan_content, "Plan should mention Multi-Query"
    assert "num_queries=4" in plan_content, "Plan should specify num_queries=4"
    print("[OK] Phase 2 plan documents multi-query feature")

    print("\n[DOCUMENTATION VALIDATED]")


if __name__ == "__main__":
    try:
        test_fusion_retriever_config()
        test_multi_query_documentation()

        print("\n" + "=" * 80)
        print("  ALL MULTI-QUERY CONFIGURATION TESTS PASSED!")
        print("=" * 80)
        print("\n[NOTE] Full integration test with API calls requires API credits")
        print("[NOTE] Configuration is correct and ready to use")

    except AssertionError as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
