"""
System Sentinel -- Diagnostics & Health Checks
================================================
Validates environment, API health, index integrity, and end-to-end
query functionality for the BakkesMod RAG system.

Rewrite of the legacy ``rag_sentinel.py`` using the unified
``bakkesmod_rag`` package.  All checks are synchronous and report
PASS/FAIL status with actionable details.

Usage::

    python -m bakkesmod_rag.sentinel
"""

import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_pass_count = 0
_fail_count = 0


def print_step(msg):
    """Print a labelled step header."""
    print(f"\n[SENTINEL] {msg}...")


def print_result(success, details=""):
    """Print a PASS/FAIL result line and update global counters."""
    global _pass_count, _fail_count
    status = "PASS" if success else "FAIL"
    print(f"  [{status}] {details}")
    if success:
        _pass_count += 1
    else:
        _fail_count += 1


# ---------------------------------------------------------------------------
# Diagnostic checks
# ---------------------------------------------------------------------------

def _check_environment():
    """Check that required and optional environment variables are set."""
    print_step("Checking environment variables")

    # Required
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        masked = openai_key[:8] + "..." + openai_key[-4:]
        print_result(True, f"OPENAI_API_KEY set ({masked})")
    else:
        print_result(False, "OPENAI_API_KEY is missing (required for embeddings)")
        return False

    # Optional -- just report presence
    optional_keys = {
        "ANTHROPIC_API_KEY": "Anthropic Claude (premium LLM)",
        "OPENROUTER_API_KEY": "OpenRouter / DeepSeek V3 (free LLM)",
        "GOOGLE_API_KEY": "Google Gemini 2.5 Flash (free LLM)",
        "COHERE_API_KEY": "Cohere neural reranker (optional)",
    }
    for key, desc in optional_keys.items():
        val = os.getenv(key)
        if val:
            print(f"    {key}: set ({desc})")
        else:
            print(f"    {key}: not set ({desc})")

    return True


def _check_documentation_dirs(config):
    """Check that documentation source directories exist and count files."""
    print_step("Checking documentation directories")

    all_ok = True
    for dir_name in config.storage.docs_dirs:
        dir_path = Path(dir_name)
        if dir_path.is_dir():
            # Count files by extension
            counts = {}
            for ext in config.storage.required_exts:
                files = list(dir_path.rglob(f"*{ext}"))
                counts[ext] = len(files)
            total = sum(counts.values())
            detail = ", ".join(f"{ext}: {c}" for ext, c in counts.items())
            print_result(True, f"{dir_name}/ exists ({total} files: {detail})")
        else:
            print_result(False, f"{dir_name}/ directory not found")
            all_ok = False

    return all_ok


def _check_config(config):
    """Validate and display key configuration settings."""
    print_step("Validating configuration")

    print(f"    Embedding model: {config.embedding.model}")
    print(f"    Primary LLM: {config.llm.primary_provider} / {config.llm.primary_model}")
    print(f"    KG enabled: {config.retriever.enable_kg}")
    print(f"    Reranker enabled: {config.retriever.enable_reranker}")
    print(f"    Storage dir: {config.storage.storage_dir}")
    print(f"    Cache enabled: {config.cache.enabled}")
    print_result(True, "Configuration loaded successfully")
    return True


def _check_llm(config):
    """Test that at least one LLM provider responds."""
    print_step("Testing LLM providers")

    try:
        from bakkesmod_rag.llm_provider import get_llm

        llm = get_llm(config)
        model_name = getattr(llm, "model", "unknown")
        print_result(True, f"LLM provider available (model: {model_name})")
        return True
    except Exception as e:
        print_result(False, f"No LLM provider available: {e}")
        return False


def _check_embedding(config):
    """Test that the OpenAI embedding endpoint responds."""
    print_step("Testing embedding model")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=config.openai_api_key)
        client.embeddings.create(input=["sentinel test"], model=config.embedding.model)
        print_result(True, f"Embedding model '{config.embedding.model}' responding")
        return True
    except Exception as e:
        print_result(False, f"Embedding test failed: {e}")
        return False


def _check_index_storage(config):
    """Check that persisted index files exist on disk."""
    print_step("Checking index storage")

    storage_path = Path(config.storage.storage_dir)
    if not storage_path.exists():
        print_result(False, f"{storage_path}/ does not exist (indexes not built yet)")
        return False

    expected_files = [
        "docstore.json",
        "index_store.json",
    ]
    # Vector store may have a prefix
    vector_candidates = [
        "default__vector_store.json",
        "vector_store.json",
    ]

    all_found = True
    for fname in expected_files:
        fpath = storage_path / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"    {fname}: {size_kb:.0f} KB")
        else:
            print(f"    {fname}: MISSING")
            all_found = False

    vector_found = False
    for candidate in vector_candidates:
        fpath = storage_path / candidate
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            print(f"    {candidate}: {size_kb:.0f} KB")
            vector_found = True
            break

    if not vector_found:
        print("    vector store file: MISSING")
        all_found = False

    # Check for graph store (optional)
    graph_path = storage_path / "graph_store.json"
    if graph_path.exists():
        size_kb = graph_path.stat().st_size / 1024
        print(f"    graph_store.json: {size_kb:.0f} KB (KG index present)")
    else:
        print("    graph_store.json: not found (KG index not built)")

    print_result(all_found, "Index storage integrity check")
    return all_found


def _check_end_to_end(config):
    """Run a test query through the full RAG engine."""
    print_step("Running end-to-end test query")

    try:
        from bakkesmod_rag.engine import RAGEngine

        start = time.time()
        engine = RAGEngine(config)
        init_time = time.time() - start
        print(f"    Engine initialised in {init_time:.1f}s")

        start = time.time()
        result = engine.query("What is BakkesMod?")
        query_time = time.time() - start

        print_result(
            True,
            f"Query returned {len(result.answer)} chars, "
            f"{len(result.sources)} sources, "
            f"confidence {result.confidence:.0%} ({result.confidence_label}), "
            f"latency {query_time:.2f}s"
        )
        return True
    except Exception as e:
        print_result(False, f"End-to-end query failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_diagnostics():
    """Run all diagnostic checks and return overall success."""
    global _pass_count, _fail_count
    _pass_count = 0
    _fail_count = 0

    print("=" * 60)
    print("  BAKKESMOD RAG - SYSTEM DIAGNOSTICS")
    print("=" * 60)

    # 1. Environment
    env_ok = _check_environment()

    # 2. Config
    from bakkesmod_rag.config import RAGConfig
    config = RAGConfig()
    _check_config(config)

    # 3. Documentation directories
    _check_documentation_dirs(config)

    # 4. Embedding health (needs OPENAI_API_KEY)
    if env_ok:
        _check_embedding(config)

    # 5. LLM provider
    _check_llm(config)

    # 6. Index storage
    indexes_ok = _check_index_storage(config)

    # 7. End-to-end (only if indexes exist)
    if indexes_ok:
        _check_end_to_end(config)
    else:
        print_step("Skipping end-to-end test (indexes not built)")
        print_result(False, "Cannot run queries without indexes")

    # Summary
    total = _pass_count + _fail_count
    print()
    print("=" * 60)
    print(f"  RESULTS: {_pass_count}/{total} checks passed")
    if _fail_count == 0:
        print("  STATUS: ALL SYSTEMS GREEN")
    else:
        print(f"  STATUS: {_fail_count} CHECK(S) FAILED")
    print("=" * 60)

    return _fail_count == 0


if __name__ == "__main__":
    success = run_diagnostics()
    sys.exit(0 if success else 1)
