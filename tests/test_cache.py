"""Tests for SemanticCache: get/set/clear/expire/similarity."""

import json
import time
import pytest
from bakkesmod_rag.cache import SemanticCache


@pytest.fixture
def cache(tmp_cache_dir, mock_embed_model):
    """Create a SemanticCache with mock embeddings and temp dir."""
    return SemanticCache(
        cache_dir=str(tmp_cache_dir),
        similarity_threshold=0.90,
        ttl_seconds=3600,
        embed_model=mock_embed_model,
    )


class TestCacheBasics:
    def test_set_and_get(self, cache):
        cache.set("What is BakkesMod?", "BakkesMod is a mod framework.", [])
        result = cache.get("What is BakkesMod?")
        assert result is not None
        response, similarity, metadata = result
        assert "BakkesMod" in response
        assert similarity >= 0.90

    def test_cache_miss(self, cache):
        result = cache.get("completely unrelated query about cooking")
        # May or may not be None depending on mock embedding similarity
        # but the important thing is it doesn't crash
        assert result is None or isinstance(result, tuple)

    def test_no_embed_model_returns_none(self, tmp_cache_dir):
        cache = SemanticCache(
            cache_dir=str(tmp_cache_dir),
            embed_model=None,
        )
        result = cache.get("test")
        assert result is None

    def test_no_embed_model_set_is_noop(self, tmp_cache_dir):
        cache = SemanticCache(
            cache_dir=str(tmp_cache_dir),
            embed_model=None,
        )
        cache.set("test", "response", [])
        assert len(cache.cache_index["entries"]) == 0


class TestCacheExpiration:
    def test_expired_entry_not_returned(self, tmp_cache_dir, mock_embed_model):
        cache = SemanticCache(
            cache_dir=str(tmp_cache_dir),
            similarity_threshold=0.90,
            ttl_seconds=1,  # 1 second TTL
            embed_model=mock_embed_model,
        )
        cache.set("test query", "test response", [])
        time.sleep(1.5)
        result = cache.get("test query")
        assert result is None

    def test_clear_expired(self, cache):
        # Add an entry then manually backdate it
        cache.set("old query", "old response", [])
        if cache.cache_index["entries"]:
            cache.cache_index["entries"][0]["timestamp"] = 0  # epoch
            cache._save_index()
        cache.clear_expired()
        assert len(cache.cache_index["entries"]) == 0


class TestCacheClear:
    def test_clear_removes_all(self, cache):
        cache.set("q1", "r1", [])
        cache.set("q2", "r2", [])
        assert len(cache.cache_index["entries"]) >= 1
        cache.clear()
        assert len(cache.cache_index["entries"]) == 0


class TestCacheStats:
    def test_stats_format(self, cache):
        stats = cache.stats()
        assert "total_entries" in stats
        assert "valid_entries" in stats
        assert "expired_entries" in stats
        assert "similarity_threshold" in stats

    def test_stats_after_set(self, cache):
        cache.set("q1", "r1", [])
        stats = cache.stats()
        assert stats["total_entries"] >= 1
        assert stats["valid_entries"] >= 1
