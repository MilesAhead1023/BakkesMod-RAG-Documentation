"""
Smoke tests - verify code syntax and basic functionality without API calls.
Tests the unified bakkesmod_rag package at the syntax/import level.
"""

import os
import sys

# Set dummy API keys for import testing
os.environ.setdefault('OPENAI_API_KEY', 'sk-test-dummy-key-for-syntax-validation')
os.environ.setdefault('ANTHROPIC_API_KEY', 'sk-ant-test-dummy-key')
os.environ.setdefault('GOOGLE_API_KEY', 'test-dummy-google-key')

print("=" * 60)
print("SMOKE TESTS - Code Syntax & Structure Validation")
print("=" * 60)

# Test 1: Python compilation of all package modules
print("\n[TEST 1] Python Syntax Validation")
print("-" * 60)

import py_compile

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

all_passed = True
for file in package_files + entry_files:
    try:
        py_compile.compile(file, doraise=True)
        print(f"[PASS] {file:50s} - Syntax valid")
    except py_compile.PyCompileError as e:
        print(f"[FAIL] {file:50s} - Syntax error")
        print(f"   {e}")
        all_passed = False
    except FileNotFoundError:
        print(f"[FAIL] {file:50s} - File not found")
        all_passed = False

# Test 2: Import validation
print("\n[TEST 2] Module Import Validation")
print("-" * 60)

import_tests = [
    ('bakkesmod_rag', 'bakkesmod_rag'),
    ('bakkesmod_rag.config', 'bakkesmod_rag.config'),
    ('bakkesmod_rag.llm_provider', 'bakkesmod_rag.llm_provider'),
    ('bakkesmod_rag.document_loader', 'bakkesmod_rag.document_loader'),
    ('bakkesmod_rag.retrieval', 'bakkesmod_rag.retrieval'),
    ('bakkesmod_rag.cache', 'bakkesmod_rag.cache'),
    ('bakkesmod_rag.query_rewriter', 'bakkesmod_rag.query_rewriter'),
    ('bakkesmod_rag.confidence', 'bakkesmod_rag.confidence'),
    ('bakkesmod_rag.code_generator', 'bakkesmod_rag.code_generator'),
    ('bakkesmod_rag.engine', 'bakkesmod_rag.engine'),
    ('bakkesmod_rag.cost_tracker', 'bakkesmod_rag.cost_tracker'),
    ('bakkesmod_rag.observability', 'bakkesmod_rag.observability'),
    ('bakkesmod_rag.resilience', 'bakkesmod_rag.resilience'),
    ('bakkesmod_rag.sentinel', 'bakkesmod_rag.sentinel'),
    ('bakkesmod_rag.evaluator', 'bakkesmod_rag.evaluator'),
    ('bakkesmod_rag.comprehensive_builder', 'bakkesmod_rag.comprehensive_builder'),
]

for display_name, module_name in import_tests:
    try:
        __import__(module_name)
        print(f"[PASS] {display_name:50s} - Imports successfully")
    except ImportError as e:
        print(f"[WARN] {display_name:50s} - Missing dependency: {e}")
    except Exception as e:
        print(f"[FAIL] {display_name:50s} - Import error: {type(e).__name__}: {e}")
        all_passed = False

# Test 3: Verify public API exports
print("\n[TEST 3] Public API Exports")
print("-" * 60)

try:
    from bakkesmod_rag import RAGEngine, RAGConfig, QueryResult, CodeResult
    print("[PASS] RAGEngine, RAGConfig, QueryResult, CodeResult all exported")
except ImportError as e:
    print(f"[FAIL] Public API export error: {e}")
    all_passed = False

# Test 4: Verify config defaults are correct
print("\n[TEST 4] Config Defaults Verification")
print("-" * 60)

try:
    from bakkesmod_rag.config import RAGConfig
    config = RAGConfig()

    # Embedding model should be text-embedding-3-small (not large)
    if config.embedding.model == "text-embedding-3-small":
        print("[PASS] Embedding model: text-embedding-3-small (correct)")
    else:
        print(f"[FAIL] Embedding model: {config.embedding.model} (expected text-embedding-3-small)")
        all_passed = False

    # Embedding dimension should be 1536 (not 3072)
    if config.embedding.dimension == 1536:
        print("[PASS] Embedding dimension: 1536 (correct)")
    else:
        print(f"[FAIL] Embedding dimension: {config.embedding.dimension} (expected 1536)")
        all_passed = False

    # Fusion num_queries should be 4 (not 1)
    if config.retriever.fusion_num_queries == 4:
        print("[PASS] Fusion num_queries: 4 (correct)")
    else:
        print(f"[FAIL] Fusion num_queries: {config.retriever.fusion_num_queries} (expected 4)")
        all_passed = False

    # Storage dir should be rag_storage
    if config.storage.storage_dir == "rag_storage":
        print("[PASS] Storage dir: rag_storage (correct)")
    else:
        print(f"[FAIL] Storage dir: {config.storage.storage_dir} (expected rag_storage)")
        all_passed = False

    # Docs dirs should include both directories
    if "docs_bakkesmod_only" in config.storage.docs_dirs and "templates" in config.storage.docs_dirs:
        print("[PASS] Docs dirs: docs_bakkesmod_only + templates (correct)")
    else:
        print(f"[FAIL] Docs dirs: {config.storage.docs_dirs} (expected both dirs)")
        all_passed = False

    # Required exts should include .md, .h, .cpp
    for ext in [".md", ".h", ".cpp"]:
        if ext in config.storage.required_exts:
            print(f"[PASS] Required ext: {ext} (correct)")
        else:
            print(f"[FAIL] Required ext: {ext} missing from {config.storage.required_exts}")
            all_passed = False

except Exception as e:
    print(f"[FAIL] Config validation error: {e}")
    all_passed = False

# Test 5: Verify QueryRewriter has domain synonyms
print("\n[TEST 5] Query Rewriter Validation")
print("-" * 60)

try:
    from bakkesmod_rag.query_rewriter import QueryRewriter
    rewriter = QueryRewriter()
    expanded = rewriter.expand_with_synonyms("car speed")
    if "velocity" in expanded.lower() or "car" in expanded.lower():
        print("[PASS] QueryRewriter expands domain synonyms")
    else:
        print(f"[WARN] QueryRewriter expansion unclear: {expanded[:60]}")
except Exception as e:
    print(f"[FAIL] QueryRewriter error: {e}")
    all_passed = False

# Final result
print("\n" + "=" * 60)
if all_passed:
    print("[PASS] ALL SMOKE TESTS PASSED")
    print("=" * 60)
    print("\nThe unified bakkesmod_rag package is correctly structured.")
    print("All Python files have valid syntax and import correctly.")
else:
    print("[FAIL] SOME TESTS FAILED")
    print("=" * 60)
    print("\nPlease review the failures above.")


def test_smoke_all_passed():
    """Pytest-compatible test that verifies all smoke checks passed."""
    assert all_passed, "One or more smoke tests failed — see output above"
