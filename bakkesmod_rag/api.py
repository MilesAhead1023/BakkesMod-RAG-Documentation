"""
FastAPI HTTP Layer for Horizontal Scaling
=========================================
Exposes the RAGEngine as a stateless HTTP service so multiple worker
instances can run behind a load balancer.

Shared mutable state (circuit breakers, cost counters, semantic cache)
is moved to Redis so all workers stay in sync.

Endpoints:
  POST /query         -> QueryResult (JSON)
  POST /generate      -> CodeResult (JSON)
  GET  /health        -> {"status": "ok", "num_documents": n}
  GET  /metrics       -> {"cost": {...}, "cache": {...}}

Usage::

    uvicorn bakkesmod_rag.api:app --host 0.0.0.0 --port 8080 --workers 2
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger("bakkesmod_rag.api")


# ---------------------------------------------------------------------------
# FastAPI app — lazy import so the rest of the package works without fastapi
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel as _PydanticBaseModel
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment,misc]
    HTTPException = None  # type: ignore[assignment]

    class _PydanticBaseModel:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(_PydanticBaseModel):
    question: str
    use_cache: bool = True


class CodeRequest(_PydanticBaseModel):
    description: str


# ---------------------------------------------------------------------------
# Engine singleton (lazy-initialised on first request)
# ---------------------------------------------------------------------------

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        from bakkesmod_rag.engine import RAGEngine
        from bakkesmod_rag.config import get_config
        config = get_config()
        # Use Redis cache if CACHE_BACKEND env var is set
        if os.getenv("CACHE_BACKEND", "").lower() == "redis":
            from bakkesmod_rag.config import CacheConfig
            config.cache = CacheConfig(
                backend="redis",
                redis_url=os.getenv("REDIS_URL", "redis://redis:6379"),
            )
        _engine = RAGEngine(config=config)
        logger.info("RAGEngine initialised for API worker")
    return _engine


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> "FastAPI":
    """Create the FastAPI application.

    Returns:
        FastAPI app instance.

    Raises:
        ImportError: If fastapi is not installed.
    """
    if not _FASTAPI_AVAILABLE:
        raise ImportError(
            "fastapi is required for the HTTP API layer. "
            "Install it with: pip install fastapi uvicorn"
        )

    app = FastAPI(
        title="BakkesMod RAG API",
        description="RAG-powered BakkesMod SDK documentation assistant",
        version="1.0.0",
    )

    @app.get("/health")
    async def health():
        """Health check endpoint.

        Returns the engine status and document count.
        """
        try:
            engine = _get_engine()
            return {
                "status": "ok",
                "num_documents": engine.num_documents,
                "num_nodes": engine.num_nodes,
            }
        except Exception as e:
            logger.error("Health check failed: %s", e)
            return JSONResponse(
                status_code=503,
                content={"status": "error", "detail": str(e)},
            )

    @app.get("/metrics")
    async def metrics():
        """Metrics endpoint: cost, cache stats, circuit breaker states."""
        try:
            engine = _get_engine()
            return {
                "cost": engine.cost_tracker.costs,
                "cache": engine.cache.stats(),
                "circuit_breakers": engine.api_manager.get_status(),
            }
        except Exception as e:
            logger.error("Metrics fetch failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/query")
    async def query(req: QueryRequest):
        """Execute a RAG query and return the result as JSON.

        Args:
            req: QueryRequest with question and use_cache flag.

        Returns:
            JSON representation of QueryResult.
        """
        try:
            engine = _get_engine()
            result = engine.query(req.question, use_cache=req.use_cache)
            return {
                "answer": result.answer,
                "sources": result.sources,
                "confidence": result.confidence,
                "confidence_label": result.confidence_label,
                "confidence_explanation": result.confidence_explanation,
                "query_time": result.query_time,
                "cached": result.cached,
                "expanded_query": result.expanded_query,
                "retry_count": result.retry_count,
            }
        except Exception as e:
            logger.error("Query failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/generate")
    async def generate(req: CodeRequest):
        """Generate a BakkesMod plugin and return the result as JSON.

        Args:
            req: CodeRequest with plugin description.

        Returns:
            JSON representation of CodeResult.
        """
        try:
            engine = _get_engine()
            result = engine.generate_code(req.description)
            return {
                "header": result.header,
                "implementation": result.implementation,
                "project_files": result.project_files,
                "explanation": result.explanation,
                "validation": result.validation,
                "features_used": result.features_used,
                "fix_iterations": result.fix_iterations,
            }
        except Exception as e:
            logger.error("Code generation failed: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ---------------------------------------------------------------------------
# Module-level app (for uvicorn bakkesmod_rag.api:app)
# ---------------------------------------------------------------------------

if _FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None  # type: ignore[assignment]
