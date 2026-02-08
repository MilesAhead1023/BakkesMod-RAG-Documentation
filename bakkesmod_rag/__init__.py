"""
BakkesMod RAG - Unified Package
================================
RAG system for BakkesMod SDK documentation using LlamaIndex.
Provides query interface and code generation for plugin development.
"""

from bakkesmod_rag.config import RAGConfig, get_config, reload_config
from bakkesmod_rag.engine import RAGEngine, QueryResult, CodeResult


__all__ = [
    "RAGEngine",
    "RAGConfig",
    "QueryResult",
    "CodeResult",
    "get_config",
    "reload_config",
]
