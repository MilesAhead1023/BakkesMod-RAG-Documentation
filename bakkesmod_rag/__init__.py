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
from bakkesmod_rag.answer_verifier import AnswerVerifier, VerificationResult
from bakkesmod_rag.compiler import PluginCompiler, CompileResult, CompilerError
from bakkesmod_rag.feedback_store import FeedbackStore, FeedbackEntry


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
    # Verification
    "AnswerVerifier",
    "VerificationResult",
    # Compiler
    "PluginCompiler",
    "CompileResult",
    "CompilerError",
    # Feedback
    "FeedbackStore",
    "FeedbackEntry",
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
