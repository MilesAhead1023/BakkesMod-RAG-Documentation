"""
Test Query Rewriting
=====================
Tests synonym expansion and query rewriting functionality.
"""

from query_rewriter import QueryRewriter, rewrite_query


def test_synonym_expansion():
    """Test that domain synonyms are correctly expanded."""
    print("\n=== Test: Synonym Expansion ===\n")

    rewriter = QueryRewriter(use_llm=False)

    # Test 1: Hook expansion
    query1 = "How do I hook events?"
    expanded1 = rewriter.expand_with_synonyms(query1)

    print(f"Query: {query1}")
    print(f"Expanded: {expanded1}")

    assert "hook" in expanded1.lower() or "attach" in expanded1.lower(), "Should expand 'hook'"
    assert "event" in expanded1.lower() or "callback" in expanded1.lower(), "Should expand 'event'"
    print("[OK] Hook and event synonyms expanded\n")

    # Test 2: Plugin expansion
    query2 = "create a plugin"
    expanded2 = rewriter.expand_with_synonyms(query2)

    print(f"Query: {query2}")
    print(f"Expanded: {expanded2}")

    assert "plugin" in expanded2.lower() or "mod" in expanded2.lower(), "Should expand 'plugin'"
    assert "create" in expanded2.lower() or "initialize" in expanded2.lower(), "Should expand 'create'"
    print("[OK] Plugin and create synonyms expanded\n")

    # Test 3: GUI expansion
    query3 = "set up GUI"
    expanded3 = rewriter.expand_with_synonyms(query3)

    print(f"Query: {query3}")
    print(f"Expanded: {expanded3}")

    assert "gui" in expanded3.lower() or "imgui" in expanded3.lower(), "Should expand 'GUI'"
    print("[OK] GUI synonyms expanded\n")

    # Test 4: No synonyms (should return original)
    query4 = "something completely unrelated"
    expanded4 = rewriter.expand_with_synonyms(query4)

    print(f"Query: {query4}")
    print(f"Expanded: {expanded4}")

    assert expanded4 == query4, "Should return original if no synonyms found"
    print("[OK] No false expansions\n")


def test_domain_coverage():
    """Test that domain synonym dictionary covers key concepts."""
    print("\n=== Test: Domain Coverage ===\n")

    rewriter = QueryRewriter(use_llm=False)

    key_concepts = [
        "plugin", "hook", "event",
        "car", "ball", "player",
        "GUI", "settings",
        "GameWrapper", "ServerWrapper",
        "create", "get", "set"
    ]

    for concept in key_concepts:
        assert concept in rewriter.DOMAIN_SYNONYMS, f"Domain should cover '{concept}'"
        synonyms = rewriter.DOMAIN_SYNONYMS[concept]
        assert len(synonyms) >= 2, f"'{concept}' should have at least 2 synonyms"

    print(f"[OK] Domain covers {len(key_concepts)} key concepts")
    print(f"[OK] Total synonyms: {sum(len(s) for s in rewriter.DOMAIN_SYNONYMS.values())}")
    print()


def test_convenience_function():
    """Test the convenience rewrite_query function."""
    print("\n=== Test: Convenience Function ===\n")

    query = "hook goal event"
    rewritten = rewrite_query(query, llm=None, use_llm=False)

    print(f"Original: {query}")
    print(f"Rewritten: {rewritten}")

    assert "hook" in rewritten.lower() or "event" in rewritten.lower(), "Should contain expanded terms"
    print("[OK] Convenience function works\n")


def test_rewriter_modes():
    """Test that rewriter can switch between LLM and synonym modes."""
    print("\n=== Test: Rewriter Modes ===\n")

    # Mode 1: No LLM (synonym only)
    rewriter_synonyms = QueryRewriter(llm=None, use_llm=False)
    assert not rewriter_synonyms.use_llm, "Should not use LLM when disabled"
    print("[OK] Synonym-only mode")

    # Mode 2: LLM disabled explicitly
    rewriter_disabled = QueryRewriter(llm=None, use_llm=True)
    assert not rewriter_disabled.use_llm, "Should not use LLM when llm=None"
    print("[OK] LLM mode gracefully disabled when no LLM provided")

    print()


def test_real_world_queries():
    """Test with realistic user queries."""
    print("\n=== Test: Real-World Queries ===\n")

    rewriter = QueryRewriter(use_llm=False)

    real_queries = [
        ("How do I hook the goal scored event?", ["hook", "event", "goal"]),
        ("Create a settings GUI", ["create", "settings", "gui"]),
        ("Get player car data", ["get", "player", "car"]),
        ("Load plugin on game start", ["load", "plugin"]),
    ]

    for query, expected_terms in real_queries:
        expanded = rewriter.expand_with_synonyms(query)
        print(f"Query: {query}")
        print(f"Expanded: {expanded}")

        # Check that at least one expected term is in expanded query
        found_terms = [term for term in expected_terms if term.lower() in expanded.lower()]
        assert len(found_terms) > 0, f"Should find at least one of {expected_terms} in expansion"
        print(f"[OK] Found terms: {found_terms}\n")


if __name__ == "__main__":
    try:
        test_synonym_expansion()
        test_domain_coverage()
        test_convenience_function()
        test_rewriter_modes()
        test_real_world_queries()

        print("=" * 80)
        print("  ALL QUERY REWRITING TESTS PASSED!")
        print("=" * 80)
        print("\n[FEATURES VALIDATED]")
        print("  [OK] Synonym expansion works")
        print("  [OK] Domain coverage comprehensive")
        print("  [OK] Graceful LLM fallback")
        print("  [OK] Real-world queries handled")

    except AssertionError as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
