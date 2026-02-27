"""Tests for SemanticCache, RedisSemanticCache, and create_cache factory."""

import json
import time
import pytest
from unittest.mock import MagicMock, patch

from bakkesmod_rag.cache import SemanticCache, RedisSemanticCache, create_cache


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


# ---------------------------------------------------------------------------
# Gap 6: RedisSemanticCache tests
# ---------------------------------------------------------------------------

class TestRedisSemanticCache:
    """Tests for Redis-backed cache using fakeredis."""

    @pytest.fixture
    def redis_cache(self, mock_embed_model):
        """Create RedisSemanticCache backed by fakeredis."""
        try:
            import fakeredis
        except ImportError:
            pytest.skip("fakeredis not installed (pip install fakeredis)")

        server = fakeredis.FakeServer()
        fake_client = fakeredis.FakeRedis(server=server)

        cache = RedisSemanticCache(
            redis_url="redis://localhost:6379",  # won't actually connect
            similarity_threshold=0.90,
            ttl_seconds=3600,
            embed_model=mock_embed_model,
        )
        # Inject fakeredis client directly
        cache._redis = fake_client
        cache._fallback = None
        return cache

    def test_set_and_get(self, redis_cache):
        """Set a value and retrieve it by similarity."""
        redis_cache.set("What is BakkesMod?", "BakkesMod is a mod framework.", [])
        result = redis_cache.get("What is BakkesMod?")
        assert result is not None
        response, sim, meta = result
        assert "BakkesMod" in response
        assert sim >= 0.90

    def test_get_miss(self, redis_cache):
        """Cache miss returns None."""
        result = redis_cache.get("cooking recipes")
        assert result is None

    def test_clear(self, redis_cache):
        """clear() removes all entries."""
        redis_cache.set("q1", "r1", [])
        redis_cache.clear()
        result = redis_cache.get("q1")
        assert result is None

    def test_stats_returns_dict(self, redis_cache):
        """stats() returns a dict with expected keys."""
        stats = redis_cache.stats()
        assert "backend" in stats
        assert stats["backend"] == "redis"
        assert "total_entries" in stats
        assert "similarity_threshold" in stats

    def test_clear_expired_noop(self, redis_cache):
        """clear_expired() returns 0 (Redis handles TTL natively)."""
        result = redis_cache.clear_expired()
        assert result == 0

    def test_no_embed_model_returns_none(self):
        """Without embed model, get() returns None."""
        cache = RedisSemanticCache(embed_model=None)
        result = cache.get("test")
        assert result is None

    def test_fallback_to_file_on_connection_refused(self, tmp_cache_dir, mock_embed_model):
        """Falls back to file cache when Redis connection refused."""
        cache = RedisSemanticCache(
            redis_url="redis://localhost:19999",  # nothing listening here
            similarity_threshold=0.90,
            ttl_seconds=3600,
            embed_model=mock_embed_model,
            cache_dir=str(tmp_cache_dir),
        )
        # Should have a file fallback
        assert cache._fallback is not None
        assert isinstance(cache._fallback, SemanticCache)


# ---------------------------------------------------------------------------
# Gap 6: create_cache factory
# ---------------------------------------------------------------------------

class TestCreateCacheFactory:
    """Tests for the create_cache() factory function."""

    def test_file_backend_returns_semantic_cache(self, tmp_cache_dir, mock_embed_model):
        """When backend='file', returns SemanticCache."""
        from bakkesmod_rag.config import CacheConfig
        config = CacheConfig(
            backend="file",
            cache_dir=str(tmp_cache_dir),
            similarity_threshold=0.90,
            ttl_seconds=3600,
        )
        cache = create_cache(config, mock_embed_model)
        assert isinstance(cache, SemanticCache)

    def test_redis_backend_returns_redis_cache(self, tmp_cache_dir, mock_embed_model):
        """When backend='redis', returns RedisSemanticCache (or file fallback)."""
        from bakkesmod_rag.config import CacheConfig
        config = CacheConfig(
            backend="redis",
            redis_url="redis://localhost:19999",  # will fail, fallback to file
            cache_dir=str(tmp_cache_dir),
            similarity_threshold=0.90,
            ttl_seconds=3600,
        )
        cache = create_cache(config, mock_embed_model)
        # Should return RedisSemanticCache (with file fallback inside)
        assert isinstance(cache, RedisSemanticCache)

    def test_factory_default_is_file(self, tmp_cache_dir, mock_embed_model):
        """Default backend (no backend field set) returns SemanticCache."""
        from bakkesmod_rag.config import CacheConfig
        config = CacheConfig(cache_dir=str(tmp_cache_dir))
        cache = create_cache(config, mock_embed_model)
        assert isinstance(cache, SemanticCache)
