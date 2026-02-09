"""
BakkesMod RAG - Unified Package
================================
RAG system for BakkesMod SDK documentation using LlamaIndex.
Provides query interface and code generation for plugin development.
"""

from bakkesmod_rag.config import RAGConfig, get_config, reload_config
from bakkesmod_rag.engine import RAGEngine, QueryResult, CodeResult
from bakkesmod_rag.resilience import (
    CircuitBreaker,
    CircuitBreakerOpen,
    RateLimiter,
    FallbackChain,
    APICallManager,
    RateLimitedCaller,
    resilient_api_call,
)
from bakkesmod_rag.observability import (
    StructuredLogger,
    PhoenixObserver,
    MetricsCollector,
    get_logger,
    get_phoenix,
    get_metrics,
    initialize_observability,
)
from bakkesmod_rag.cost_tracker import CostTracker, get_tracker
from bakkesmod_rag.setup_keys import ensure_api_keys
from bakkesmod_rag.query_decomposer import QueryDecomposer


__all__ = [
    # Core
    "RAGEngine",
    "RAGConfig",
    "QueryResult",
    "CodeResult",
    "get_config",
    "reload_config",
    # Setup
    "ensure_api_keys",
    # Query processing
    "QueryDecomposer",
    # Resilience
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "RateLimiter",
    "FallbackChain",
    "APICallManager",
    "RateLimitedCaller",
    "resilient_api_call",
    # Observability
    "StructuredLogger",
    "PhoenixObserver",
    "MetricsCollector",
    "get_logger",
    "get_phoenix",
    "get_metrics",
    "initialize_observability",
    # Cost tracking
    "CostTracker",
    "get_tracker",
]
