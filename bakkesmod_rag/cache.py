"""
Semantic Cache
==============
Caches query-response pairs with embedding-based similarity matching.
Reduces costs by ~35% on repeated or similar queries.

Backends:
  - FileSemanticCache (default): JSON files in .cache/semantic/ — single process
  - RedisSemanticCache (optional): Redis hash per entry — multi-process safe

Use create_cache() factory to get the right backend based on config.
"""

import json
import time
import hashlib
import logging
import struct
from pathlib import Path
from typing import Optional, Tuple, List

logger = logging.getLogger("bakkesmod_rag.cache")


class SemanticCache:
    """Cache for RAG responses using semantic similarity matching."""

    def __init__(
        self,
        cache_dir: str = ".cache/semantic",
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 86400 * 7,
        embed_model=None,
    ):
        """Initialize semantic cache.

        Args:
            cache_dir: Directory to store cache files.
            similarity_threshold: Minimum similarity for cache hit (0.92 = 92%).
            ttl_seconds: Time to live for cache entries (default 7 days).
            embed_model: Embedding model from LlamaIndex Settings.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.embed_model = embed_model

        self.index_file = self.cache_dir / "cache_index.json"
        self.cache_index = self._load_index()

    def _load_index(self) -> dict:
        """Load cache index from disk."""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"entries": []}

    def _save_index(self):
        """Save cache index to disk."""
        with open(self.index_file, "w") as f:
            json.dump(self.cache_index, f, indent=2)

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def get(self, query: str) -> Optional[Tuple[str, float, dict]]:
        """Get cached response for query if similarity exceeds threshold.

        Args:
            query: User query string.

        Returns:
            Tuple of (response_text, similarity_score, metadata) if cache hit,
            else None.
        """
        if not self.embed_model:
            return None

        query_embedding = self.embed_model.get_text_embedding(query)

        best_match = None
        best_similarity = 0.0
        current_time = time.time()

        for entry in self.cache_index["entries"]:
            if current_time - entry["timestamp"] > self.ttl_seconds:
                continue

            similarity = self._cosine_similarity(query_embedding, entry["embedding"])

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        if best_match and best_similarity >= self.similarity_threshold:
            response_file = self.cache_dir / best_match["response_file"]

            if response_file.exists():
                with open(response_file, "r") as f:
                    response_data = json.load(f)

                metadata = {
                    "similarity": best_similarity,
                    "cached_query": best_match["query"],
                    "cache_age_seconds": int(current_time - best_match["timestamp"]),
                }

                return response_data["response"], best_similarity, metadata

        return None

    def set(self, query: str, response: str, source_nodes: list):
        """Cache a query-response pair.

        Args:
            query: User query string.
            response: Response text to cache.
            source_nodes: List of source nodes from retrieval.
        """
        if not self.embed_model:
            return

        query_embedding = self.embed_model.get_text_embedding(query)

        query_hash = hashlib.md5(query.encode()).hexdigest()[:16]
        timestamp = int(time.time())

        response_file = f"response_{timestamp}_{query_hash}.json"
        response_path = self.cache_dir / response_file

        response_data = {
            "query": query,
            "response": response,
            "sources": [
                {
                    "file_name": node.node.metadata.get("file_name", "unknown"),
                    "score": node.score if hasattr(node, "score") else None,
                }
                for node in source_nodes
            ],
            "timestamp": timestamp,
        }

        with open(response_path, "w") as f:
            json.dump(response_data, f, indent=2)

        self.cache_index["entries"].append(
            {
                "query": query,
                "embedding": query_embedding,
                "response_file": response_file,
                "timestamp": timestamp,
            }
        )

        self._save_index()

    def clear_expired(self):
        """Remove expired cache entries."""
        current_time = time.time()

        valid_entries = []
        for entry in self.cache_index["entries"]:
            if current_time - entry["timestamp"] <= self.ttl_seconds:
                valid_entries.append(entry)
            else:
                response_file = self.cache_dir / entry["response_file"]
                if response_file.exists():
                    response_file.unlink()

        self.cache_index["entries"] = valid_entries
        self._save_index()

    def stats(self) -> dict:
        """Get cache statistics."""
        total = len(self.cache_index["entries"])
        current_time = time.time()
        valid = sum(
            1
            for entry in self.cache_index["entries"]
            if current_time - entry["timestamp"] <= self.ttl_seconds
        )

        return {
            "total_entries": total,
            "valid_entries": valid,
            "expired_entries": total - valid,
            "cache_dir": str(self.cache_dir),
            "similarity_threshold": self.similarity_threshold,
        }

    def clear(self):
        """Clear all cache entries."""
        for entry in self.cache_index["entries"]:
            response_file = self.cache_dir / entry["response_file"]
            if response_file.exists():
                response_file.unlink()

        self.cache_index["entries"] = []
        self._save_index()


# Keep backward-compatible alias
FileSemanticCache = SemanticCache


# ---------------------------------------------------------------------------
# Redis-backed semantic cache
# ---------------------------------------------------------------------------

class RedisSemanticCache:
    """Redis-backed semantic cache for multi-process / containerized deployments.

    Stores each cache entry as a Redis hash with:
      - embedding: packed floats (struct)
      - response: JSON-encoded response text
      - query: original query string
      - timestamp: Unix epoch seconds (for stats only)

    TTL is managed by Redis EXPIRE — no manual cleanup needed.

    Falls back to FileSemanticCache if Redis connection fails at init.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        redis_db: int = 0,
        similarity_threshold: float = 0.92,
        ttl_seconds: int = 86400 * 7,
        embed_model=None,
        cache_dir: str = ".cache/semantic",
    ):
        """Initialise Redis semantic cache.

        Args:
            redis_url: Redis connection URL.
            redis_db: Redis database index.
            similarity_threshold: Cosine similarity threshold for cache hits.
            ttl_seconds: Entry TTL in seconds.
            embed_model: Embedding model for similarity computation.
            cache_dir: Fallback file cache directory if Redis unavailable.
        """
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.embed_model = embed_model
        self._redis = None
        self._fallback: Optional[SemanticCache] = None

        try:
            import redis as _redis

            client = _redis.from_url(redis_url, db=redis_db, socket_connect_timeout=2)
            client.ping()
            self._redis = client
            logger.info("RedisSemanticCache connected: %s db=%d", redis_url, redis_db)
        except Exception as e:
            logger.warning(
                "Redis unavailable (%s), falling back to file-based cache", e
            )
            self._fallback = SemanticCache(
                cache_dir=cache_dir,
                similarity_threshold=similarity_threshold,
                ttl_seconds=ttl_seconds,
                embed_model=embed_model,
            )

    # ---- encoding helpers -----------------------------------------------

    @staticmethod
    def _pack_embedding(vec: List[float]) -> bytes:
        return struct.pack(f"{len(vec)}f", *vec)

    @staticmethod
    def _unpack_embedding(data: bytes) -> List[float]:
        n = len(data) // 4  # 4 bytes per float
        return list(struct.unpack(f"{n}f", data))

    @staticmethod
    def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
        dot = sum(a * b for a, b in zip(v1, v2))
        m1 = sum(a * a for a in v1) ** 0.5
        m2 = sum(b * b for b in v2) ** 0.5
        if m1 == 0 or m2 == 0:
            return 0.0
        return dot / (m1 * m2)

    def _entry_key(self, embedding_bytes: bytes) -> str:
        return "ragcache:" + hashlib.sha256(embedding_bytes).hexdigest()

    # ---- public interface -----------------------------------------------

    def get(self, query: str) -> Optional[Tuple[str, float, dict]]:
        """Get cached response for query if similarity exceeds threshold."""
        if self._fallback is not None:
            return self._fallback.get(query)
        if not self.embed_model or self._redis is None:
            return None

        query_emb = self.embed_model.get_text_embedding(query)
        query_bytes = self._pack_embedding(query_emb)

        # Scan all cache entries for best match
        best_sim = 0.0
        best_entry = None
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match="ragcache:*", count=100)
                for key in keys:
                    raw = self._redis.hgetall(key)
                    if not raw or b"embedding" not in raw:
                        continue
                    stored_emb = self._unpack_embedding(raw[b"embedding"])
                    sim = self._cosine_similarity(query_emb, stored_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_entry = raw
                if cursor == 0:
                    break

            if best_entry and best_sim >= self.similarity_threshold:
                response = best_entry[b"response"].decode("utf-8")
                cached_query = best_entry.get(b"query", b"").decode("utf-8")
                return response, best_sim, {
                    "similarity": best_sim,
                    "cached_query": cached_query,
                }
        except Exception as e:
            logger.warning("Redis cache get failed: %s", e)
        return None

    def set(self, query: str, response: str, source_nodes: list) -> None:
        """Cache a query-response pair."""
        if self._fallback is not None:
            return self._fallback.set(query, response, source_nodes)
        if not self.embed_model or self._redis is None:
            return

        try:
            query_emb = self.embed_model.get_text_embedding(query)
            emb_bytes = self._pack_embedding(query_emb)
            key = self._entry_key(emb_bytes)
            self._redis.hset(key, mapping={
                "embedding": emb_bytes,
                "response": response,
                "query": query,
                "timestamp": int(time.time()),
            })
            self._redis.expire(key, self.ttl_seconds)
        except Exception as e:
            logger.warning("Redis cache set failed: %s", e)

    def clear_expired(self) -> int:
        """Redis handles TTL natively; returns 0 (no manual sweep needed)."""
        if self._fallback is not None:
            return self._fallback.clear_expired()
        return 0

    def clear(self) -> None:
        """Remove all cache entries."""
        if self._fallback is not None:
            return self._fallback.clear()
        if self._redis is None:
            return
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match="ragcache:*", count=100)
                if keys:
                    self._redis.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.warning("Redis cache clear failed: %s", e)

    def stats(self) -> dict:
        """Get cache statistics."""
        if self._fallback is not None:
            return self._fallback.stats()
        count = 0
        try:
            cursor = 0
            while True:
                cursor, keys = self._redis.scan(cursor, match="ragcache:*", count=100)
                count += len(keys)
                if cursor == 0:
                    break
        except Exception:
            pass
        return {
            "backend": "redis",
            "total_entries": count,
            "valid_entries": count,  # Redis handles TTL natively
            "expired_entries": 0,
            "similarity_threshold": self.similarity_threshold,
        }


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_cache(config, embed_model) -> "SemanticCache | RedisSemanticCache":
    """Create the appropriate semantic cache backend based on config.

    Returns Redis backend if config.backend == "redis" and connection succeeds.
    Falls back to file-based cache otherwise.

    Args:
        config: CacheConfig instance.
        embed_model: Embedding model for similarity computation.

    Returns:
        SemanticCache (file) or RedisSemanticCache.
    """
    if getattr(config, "backend", "file") == "redis":
        return RedisSemanticCache(
            redis_url=getattr(config, "redis_url", "redis://localhost:6379"),
            redis_db=getattr(config, "redis_db", 0),
            similarity_threshold=config.similarity_threshold,
            ttl_seconds=config.ttl_seconds,
            embed_model=embed_model,
            cache_dir=config.cache_dir,
        )
    return SemanticCache(
        cache_dir=config.cache_dir,
        similarity_threshold=config.similarity_threshold,
        ttl_seconds=config.ttl_seconds,
        embed_model=embed_model,
    )
