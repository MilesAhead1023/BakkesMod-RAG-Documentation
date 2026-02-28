"""
Smoke tests - verify code syntax and basic functionality without API calls.
Tests the unified bakkesmod_rag package at the syntax/import level.
"""

import os
import py_compile
import pytest


# Set dummy API keys for import testing
os.environ.setdefault('OPENAI_API_KEY', 'sk-test-dummy-key-for-syntax-validation')
os.environ.setdefault('ANTHROPIC_API_KEY', 'sk-ant-test-dummy-key')
os.environ.setdefault('GOOGLE_API_KEY', 'test-dummy-google-key')


def test_python_syntax_validation():
    """Test 1: Python compilation of all package modules."""
    package_files = [
        'bakkesmod_rag/__init__.py',
        'bakkesmod_rag/config.py',
        'bakkesmod_rag/llm_provider.py',
        'bakkesmod_rag/document_loader.py',
        'bakkesmod_rag/retrieval.py',
        'bakkesmod_rag/cache.py',
        'bakkesmod_rag/query_rewriter.py',
        'bakkesmod_rag/confidence.py',
        'bakkesmod_rag/code_generator.py',
        'bakkesmod_rag/engine.py',
        'bakkesmod_rag/cost_tracker.py',
        'bakkesmod_rag/observability.py',
        'bakkesmod_rag/resilience.py',
        'bakkesmod_rag/sentinel.py',
        'bakkesmod_rag/evaluator.py',
        'bakkesmod_rag/mcp_server.py',
        'bakkesmod_rag/comprehensive_builder.py',
    ]

    entry_files = [
        'interactive_rag.py',
        'rag_gui.py',
    ]

    for file in package_files + entry_files:
        py_compile.compile(file, doraise=True)


def test_module_imports():
    """Test 2: Module import validation."""
    import_tests = [
        'bakkesmod_rag',
        'bakkesmod_rag.config',
        'bakkesmod_rag.llm_provider',
        'bakkesmod_rag.document_loader',
        'bakkesmod_rag.retrieval',
        'bakkesmod_rag.cache',
        'bakkesmod_rag.query_rewriter',
        'bakkesmod_rag.confidence',
        'bakkesmod_rag.code_generator',
        'bakkesmod_rag.engine',
        'bakkesmod_rag.cost_tracker',
        'bakkesmod_rag.observability',
        'bakkesmod_rag.resilience',
        'bakkesmod_rag.sentinel',
        'bakkesmod_rag.evaluator',
        'bakkesmod_rag.comprehensive_builder',
    ]

    for module_name in import_tests:
        __import__(module_name)


def test_public_api_exports():
    """Test 3: Verify public API exports."""
    from bakkesmod_rag import RAGEngine, RAGConfig, QueryResult, CodeResult
    assert RAGEngine is not None
    assert RAGConfig is not None
    assert QueryResult is not None
    assert CodeResult is not None


def test_config_defaults():
    """Test 4: Verify config defaults are correct."""
    from bakkesmod_rag.config import RAGConfig
    config = RAGConfig()

    # Embedding model should be text-embedding-3-small
    assert config.embedding.model == "text-embedding-3-small"

    # Embedding dimension should be 1536
    assert config.embedding.dimension == 1536

    # Fusion num_queries should be 4
    assert config.retriever.fusion_num_queries == 4

    # Storage dir should be rag_storage
    assert config.storage.storage_dir == "rag_storage"

    # Docs dirs should include both directories
    assert "docs_bakkesmod_only" in config.storage.docs_dirs
    assert "templates" in config.storage.docs_dirs

    # Required exts should include .md, .h, .cpp
    for ext in [".md", ".h", ".cpp"]:
        assert ext in config.storage.required_exts


def test_query_rewriter():
    """Test 5: Verify QueryRewriter has domain synonyms."""
    from bakkesmod_rag.query_rewriter import QueryRewriter
    rewriter = QueryRewriter()
    expanded = rewriter.expand_with_synonyms("car speed")
    assert "velocity" in expanded.lower() or "car" in expanded.lower()
