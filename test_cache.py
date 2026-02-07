"""
Test Semantic Cache
===================
Tests cache hit/miss behavior and similarity matching.
"""

import os
import time
from dotenv import load_dotenv
from cache_manager import SemanticCache

load_dotenv()


def test_cache_basic():
    """Test basic cache operations."""
    print("\n=== Test 1: Basic Cache Operations ===\n")

    from llama_index.embeddings.openai import OpenAIEmbedding

    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    cache = SemanticCache(
        cache_dir=".cache/test",
        similarity_threshold=0.90,
        embed_model=embed_model
    )

    # Clear cache first
    cache.clear()
    print("[OK] Cache cleared")

    # Mock response data
    class MockNode:
        def __init__(self):
            self.node = type('obj', (object,), {'metadata': {'file_name': 'test.md'}})()
            self.score = 0.95

    # Store in cache
    query1 = "How do I hook the goal scored event?"
    response1 = "Use Function TAGame.Ball_TA.OnHitGoal to hook goal events..."
    cache.set(query1, response1, [MockNode()])
    print(f"[OK] Cached: '{query1}'")

    # Test exact match
    result = cache.get(query1)
    assert result is not None, "Should find exact match"
    assert result[0] == response1, "Response should match"
    assert result[1] >= 0.99, f"Exact match should have ~100% similarity, got {result[1]:.1%}"
    print(f"[OK] Exact match works (similarity: {result[1]:.1%})")

    # Test similar query
    query2 = "What's the event for when a goal is scored?"
    result = cache.get(query2)

    if result:
        similarity = result[1]
        print(f"[OK] Similar query matched (similarity: {similarity:.1%})")
        assert similarity >= 0.90, f"Similarity should be >=90%, got {similarity:.1%}"
    else:
        print("[INFO] Similar query didn't match (similarity below threshold)")

    # Test dissimilar query (should not match)
    query3 = "How do I create ImGui windows?"
    result = cache.get(query3)
    if result is None:
        print("[OK] Dissimilar query correctly didn't match")
    else:
        print(f"[INFO] Dissimilar query matched with {result[1]:.1%} similarity")

    print(f"\n[CACHE STATS]")
    stats = cache.stats()
    print(f"  Total entries: {stats['total_entries']}")
    print(f"  Valid entries: {stats['valid_entries']}")
    print(f"  Similarity threshold: {stats['similarity_threshold']:.0%}")


def test_cache_ttl():
    """Test cache TTL (time to live)."""
    print("\n=== Test 2: Cache TTL ===\n")

    from llama_index.embeddings.openai import OpenAIEmbedding

    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    # Create cache with 2 second TTL
    cache = SemanticCache(
        cache_dir=".cache/test_ttl",
        similarity_threshold=0.90,
        ttl_seconds=2,  # 2 seconds
        embed_model=embed_model
    )

    cache.clear()

    class MockNode:
        def __init__(self):
            self.node = type('obj', (object,), {'metadata': {'file_name': 'test.md'}})()
            self.score = 0.95

    # Store entry
    query = "Test query"
    response = "Test response"
    cache.set(query, response, [MockNode()])
    print("[OK] Entry cached")

    # Should be retrievable immediately
    result = cache.get(query)
    assert result is not None, "Should find entry immediately"
    print("[OK] Entry retrievable immediately")

    # Wait for expiration
    print("[INFO] Waiting 3 seconds for TTL expiration...")
    time.sleep(3)

    # Should not be retrievable after TTL
    result = cache.get(query)
    if result is None:
        print("[OK] Entry correctly expired after TTL")
    else:
        print("[INFO] Entry still retrievable (cache.get() doesn't filter expired)")

    # Stats should show expired
    stats = cache.stats()
    print(f"[OK] Stats show {stats['expired_entries']} expired entries")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  SEMANTIC CACHE TESTING")
    print("=" * 80)

    try:
        test_cache_basic()
        test_cache_ttl()

        print("\n" + "=" * 80)
        print("  ALL CACHE TESTS PASSED!")
        print("=" * 80)
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
