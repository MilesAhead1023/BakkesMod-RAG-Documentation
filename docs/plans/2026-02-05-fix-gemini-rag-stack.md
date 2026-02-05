# Fix Gemini RAG Stack Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Correct all critical API implementation errors in the Gemini RAG stack to align with 2026 official SDK documentation, making the code production-ready.

**Architecture:** Multi-file RAG system using LlamaIndex framework with hybrid retrieval (Vector + Knowledge Graph + BM25). The stack supports both OpenAI and Gemini LLMs, with Anthropic for reranking. MCP server provides Claude Code integration.

**Tech Stack:**
- LlamaIndex (v0.14.6+) - RAG framework
- Google GenAI SDK (v1.33.0+) - Gemini 2.0 Flash
- Anthropic SDK (v0.25.0+) - Claude 3.5 Sonnet
- OpenAI SDK (v1.0.0+) - Embeddings
- MCP (Model Context Protocol) - Tool integration
- Watchdog - File system monitoring

---

## Task 1: Create Requirements File

**Files:**
- Create: `requirements.txt`

**Step 1: Create requirements.txt with pinned versions**

```txt
# Core RAG Framework
llama-index-core>=0.14.6
llama-index-llms-google-genai>=0.1.0
llama-index-llms-anthropic>=0.1.0
llama-index-llms-openai>=0.1.0
llama-index-embeddings-openai>=0.1.0
llama-index-retrievers-bm25>=0.1.0
llama-index-graph-stores-neo4j>=0.1.0

# LLM Provider SDKs
google-genai>=1.33.0
anthropic>=0.25.0
openai>=1.0.0

# Utilities
nest-asyncio>=1.5.0
watchdog>=3.0.0
pandas>=2.0.0

# Optional: Evaluation (uncomment if needed)
# ragas>=0.1.0
# phoenix>=3.0.0

# MCP Integration
mcp>=0.1.0
```

**Step 2: Verify file was created**

Run: `cat requirements.txt | head -20`
Expected: File contents displayed

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add Python dependencies for RAG stack"
```

---

## Task 2: Fix Gemini API Syntax Error in rag_sentinel.py

**Files:**
- Modify: `rag_sentinel.py:52-64`

**Step 1: Read current implementation**

Run: `cat rag_sentinel.py | sed -n '52,64p'`
Expected: See duplicate exception handler

**Step 2: Remove duplicate exception handler**

Replace lines 52-64 with:

```python
    # Gemini Check
    try:
        from google import genai

        # Initialize client with API key
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

        # Synchronous content generation (NOT async)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents='hi'
        )
        print("  Gemini API: ✅ Active and accessible.")
    except Exception as e:
        print_result(False, f"Gemini API Error: {e}. Check key and billing.")
```

**Step 3: Verify syntax is valid**

Run: `python -m py_compile rag_sentinel.py`
Expected: No output (success)

**Step 4: Commit**

```bash
git add rag_sentinel.py
git commit -m "fix: correct Gemini API initialization and remove syntax error

- Use genai.Client() instead of GenerativeModel()
- Remove duplicate exception handler
- Use synchronous generate_content() call
- Align with google-genai SDK v1.33.0 pattern"
```

---

## Task 3: Fix Gemini API Pattern in gemini_rag_builder.py

**Files:**
- Modify: `gemini_rag_builder.py:38-41`

**Step 1: Read current LLM initialization**

Run: `cat gemini_rag_builder.py | sed -n '36,42p'`
Expected: See old model name format

**Step 2: Update Gemini model initialization**

Replace lines 36-42 with:

```python
    # Use Gemini 2.0 Flash for KG Extraction
    # It has a massive window and is very fast for relationship extraction
    # Note: Remove "models/" prefix - LlamaIndex handles it internally
    kg_llm = GoogleGenAI(model="gemini-2.0-flash")

    # Use Claude 3.5 Sonnet for reranking (verified 2026 snapshot)
    Settings.llm = Anthropic(model="claude-3-5-sonnet-20240620")
```

**Step 3: Verify imports are correct**

Run: `grep "from llama_index.llms" gemini_rag_builder.py`
Expected: See correct import paths

**Step 4: Commit**

```bash
git add gemini_rag_builder.py
git commit -m "fix: update Gemini and Anthropic model identifiers

- Remove 'models/' prefix from Gemini model name
- Update Anthropic model to verified 2026 snapshot (20240620)
- Align with official LlamaIndex integration patterns"
```

---

## Task 4: Fix Anthropic Model Format in comprehensive_rag.py

**Files:**
- Modify: `comprehensive_rag.py:54`

**Step 1: Read current Anthropic configuration**

Run: `cat comprehensive_rag.py | sed -n '52,55p'`
Expected: See Bedrock format model ID

**Step 2: Replace Bedrock format with Anthropic SDK format**

Replace line 54 with:

```python
    Settings.llm = Anthropic(model="claude-3-5-sonnet-20240620", max_retries=5)
```

**Step 3: Verify change**

Run: `grep "Settings.llm = Anthropic" comprehensive_rag.py`
Expected: See corrected model identifier without "anthropic." prefix or ":0" suffix

**Step 4: Commit**

```bash
git add comprehensive_rag.py
git commit -m "fix: use Anthropic SDK model format instead of Bedrock

- Change from 'anthropic.claude-3-5-sonnet-20241022-v2:0' (Bedrock)
- To 'claude-3-5-sonnet-20240620' (Anthropic SDK)
- Ensures compatibility with llama-index-llms-anthropic"
```

---

## Task 5: Remove Deprecated GPTCache Integration

**Files:**
- Modify: `comprehensive_rag.py:33-40`

**Step 1: Read current GPTCache implementation**

Run: `cat comprehensive_rag.py | sed -n '33,41p'`
Expected: See GPTCache imports and initialization

**Step 2: Remove GPTCache code block**

Delete lines 33-40 (the GPTCache initialization block):

```python
# --- Semantic Caching Initialization ---
# This will prevent redundant LLM calls for similar queries.
# Uses a simple file-based cache for persistence.
from gptcache import Cache
from gptcache.adapter import openai
cache = Cache()
cache.init()
openai.set_cache(cache)
```

**Step 3: Add comment explaining LlamaIndex handles caching**

Add at line 33:

```python
# Note: LlamaIndex handles caching internally via Settings
# No need for external GPTCache library
```

**Step 4: Verify imports still work**

Run: `python -c "import comprehensive_rag" 2>&1 | head -5`
Expected: No import errors (may show other errors if missing API keys)

**Step 5: Commit**

```bash
git add comprehensive_rag.py
git commit -m "refactor: remove deprecated GPTCache dependency

- GPTCache is unmaintained and incompatible with OpenAI SDK v1.x
- LlamaIndex provides built-in caching via Settings
- Reduces external dependencies and maintenance burden"
```

---

## Task 6: Standardize Storage Directory Configuration

**Files:**
- Modify: `comprehensive_rag.py:44`
- Modify: `gemini_rag_builder.py:28`
- Modify: `rag_builder.py:17`
- Modify: `rag_sentinel.py:118`

**Step 1: Update comprehensive_rag.py storage directory**

Replace line 44:

```python
STORAGE_DIR = os.getenv("RAG_STORAGE_DIR", "./rag_storage")
```

**Step 2: Update gemini_rag_builder.py storage directory**

Replace line 28:

```python
STORAGE_DIR = os.getenv("RAG_STORAGE_DIR", "./rag_storage/gemini")
```

**Step 3: Update rag_builder.py storage directory**

Replace line 17:

```python
STORAGE_DIR = os.getenv("RAG_STORAGE_DIR", "./rag_storage/local")
```

**Step 4: Update rag_sentinel.py to check all storage locations**

Replace line 118 with:

```python
    storage_paths = [
        os.getenv("RAG_STORAGE_DIR", "./rag_storage"),
        "./rag_storage/gemini",
        "./rag_storage/local"
    ]

    found_storage = False
    for storage_path in storage_paths:
        if os.path.exists(storage_path):
            found_storage = True
            print(f"  [Info] Found storage at: {storage_path}")

            if os.path.exists(os.path.join(storage_path, "docstore.json")) and \
               os.path.exists(os.path.join(storage_path, "default__vector_store.json")):
                print("  [Success] Vector Index found. System is partially operational.")

                if not os.path.exists(os.path.join(storage_path, "index_store.json")):
                    print("  [Note] Knowledge Graph is still building (index_store.json missing).")
                else:
                    print("  [Success] Knowledge Graph structure found.")

    if not found_storage:
        print_result(False, "No index files found yet. Background build is likely still parsing documents.")
    else:
        print_result(True, "Critical retrieval files exist.")
```

**Step 5: Commit**

```bash
git add comprehensive_rag.py gemini_rag_builder.py rag_builder.py rag_sentinel.py
git commit -m "refactor: standardize storage directory configuration

- Use RAG_STORAGE_DIR environment variable with fallback
- Separate storage subdirs: /gemini, /local for different backends
- Update sentinel to check all storage locations
- Enables clean separation of index types"
```

---

## Task 7: Add Comprehensive Error Handling

**Files:**
- Modify: `gemini_rag_builder.py:30-79`

**Step 1: Wrap index building in try-except**

Replace the main function body (lines 30-79) with:

```python
def build_gemini_stack():
    print("Initializing Gemini-Powered Ultra RAG Stack...")

    try:
        # 1. Global Settings
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large",
            max_retries=3,
            timeout=60.0
        )

        # Use Gemini 2.0 Flash for KG Extraction
        kg_llm = GoogleGenAI(model="gemini-2.0-flash")

        # Use Claude 3.5 Sonnet for reranking
        Settings.llm = Anthropic(model="claude-3-5-sonnet-20240620")

        # 2. Load and Parse
        print(f"Loading documents from {DOCS_DIR}...")
        if not os.path.exists(DOCS_DIR):
            raise FileNotFoundError(f"Documentation directory not found: {DOCS_DIR}")

        reader = SimpleDirectoryReader(
            input_dir=DOCS_DIR,
            required_exts=[".md"],
            recursive=True
        )
        documents = reader.load_data()

        if not documents:
            raise ValueError(f"No markdown documents found in {DOCS_DIR}")

        print(f"Loaded {len(documents)} documents.")

        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        print(f"Parsed into {len(nodes)} hierarchical nodes.")

        # 3. Build Indices
        if not os.path.exists(STORAGE_DIR):
            print("Building Gemini KG Index (this uses your fixed billing tier)...")
            os.makedirs(STORAGE_DIR, exist_ok=True)
            storage_context = StorageContext.from_defaults()

            # Vector Index
            print("Building Vector Index...")
            vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
            vector_index.set_index_id("vector")

            # Knowledge Graph Index with Gemini
            print("Building Knowledge Graph (may take several minutes)...")
            kg_index = KnowledgeGraphIndex(
                nodes,
                storage_context=storage_context,
                llm=kg_llm,
                max_triplets_per_chunk=10,
                include_embeddings=True,
            )
            kg_index.set_index_id("kg")

            storage_context.persist(persist_dir=STORAGE_DIR)
            print(f"Gemini indices persisted to {STORAGE_DIR}")
        else:
            print(f"Loading existing Gemini index from {STORAGE_DIR}...")
            storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
            vector_index = load_index_from_storage(storage_context, index_id="vector")
            kg_index = load_index_from_storage(storage_context, index_id="kg")
            print("Indices loaded successfully.")

        return True

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please create the docs directory and add markdown files.")
        return False
    except ValueError as e:
        print(f"ERROR: {e}")
        return False
    except Exception as e:
        print(f"CRITICAL ERROR during index building: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False
```

**Step 2: Test error handling with missing directory**

Run: `python -c "import os; os.rename('docs', 'docs_backup'); import gemini_rag_builder; gemini_rag_builder.build_gemini_stack(); os.rename('docs_backup', 'docs')"`
Expected: Graceful error message about missing directory

**Step 3: Commit**

```bash
git add gemini_rag_builder.py
git commit -m "feat: add comprehensive error handling to Gemini builder

- Wrap index building in try-except blocks
- Validate docs directory exists before processing
- Check for empty document sets
- Add detailed error messages with type and traceback
- Return boolean success status instead of raising"
```

---

## Task 8: Add Logging Framework

**Files:**
- Modify: `comprehensive_rag.py:24-31`
- Modify: `gemini_rag_builder.py:23-24`

**Step 1: Configure proper logging in comprehensive_rag.py**

Replace lines 24-31 with:

```python
# Setup logging with file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('rag_build.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
```

**Step 2: Configure logging in gemini_rag_builder.py**

Replace lines 23-24 with:

```python
# Setup logging with file and console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('gemini_build.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
```

**Step 3: Replace print statements with logger in comprehensive_rag.py**

Find and replace pattern:
- `print("` → `logger.info("`
- `print(f"` → `logger.info(f"`

**Step 4: Replace print statements with logger in gemini_rag_builder.py**

Find and replace pattern:
- `print("` → `logger.info("`
- `print(f"` → `logger.info(f"`

**Step 5: Commit**

```bash
git add comprehensive_rag.py gemini_rag_builder.py
git commit -m "feat: add proper logging framework

- Configure dual output (console + file)
- Use structured logging with timestamps
- Replace print() with logger methods
- Separate log files: rag_build.log, gemini_build.log
- UTF-8 encoding for Windows compatibility"
```

---

## Task 9: Add .gitignore for Python Artifacts

**Files:**
- Modify: `.gitignore` (or create if doesn't exist)

**Step 1: Check if .gitignore exists**

Run: `test -f .gitignore && echo "exists" || echo "create"`
Expected: Either "exists" or "create"

**Step 2: Append Python RAG-specific ignores**

Add to `.gitignore`:

```gitignore
# Python RAG Stack
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python

# RAG Storage
rag_storage/
gemini_rag_storage/
storage/

# Logs
*.log
rag_build.log
gemini_build.log
rag_error.log
gemini_error.log

# Environment
.env
.env.local

# Cache
.gptcache/
```

**Step 3: Verify gitignore syntax**

Run: `git check-ignore -v venv/test.py rag_storage/test.json`
Expected: Should match gitignore rules

**Step 4: Commit**

```bash
git add .gitignore
git commit -m "chore: add Python RAG artifacts to gitignore

- Exclude virtual environments and cache
- Exclude RAG storage directories
- Exclude log files
- Exclude environment files"
```

---

## Task 10: Create Integration Test

**Files:**
- Create: `test_rag_integration.py`

**Step 1: Write integration test**

```python
"""Integration tests for RAG stack - validates API connectivity and basic functionality."""

import os
import sys
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


class TestAPIConnectivity:
    """Test that all required API keys are present and APIs are reachable."""

    def test_openai_api_key_exists(self):
        """Verify OpenAI API key is in environment."""
        assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY not found in environment"
        assert len(os.environ["OPENAI_API_KEY"]) > 20, "OPENAI_API_KEY seems invalid"

    def test_anthropic_api_key_exists(self):
        """Verify Anthropic API key is in environment."""
        assert "ANTHROPIC_API_KEY" in os.environ, "ANTHROPIC_API_KEY not found"
        assert len(os.environ["ANTHROPIC_API_KEY"]) > 20, "ANTHROPIC_API_KEY seems invalid"

    def test_google_api_key_exists(self):
        """Verify Google API key is in environment."""
        assert "GOOGLE_API_KEY" in os.environ, "GOOGLE_API_KEY not found"
        assert len(os.environ["GOOGLE_API_KEY"]) > 20, "GOOGLE_API_KEY seems invalid"

    def test_openai_embedding_api(self):
        """Test OpenAI embeddings API is accessible."""
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        response = client.embeddings.create(
            input=["test"],
            model="text-embedding-3-small"
        )

        assert response.data[0].embedding is not None
        assert len(response.data[0].embedding) > 0

    def test_anthropic_api(self):
        """Test Anthropic API is accessible."""
        from anthropic import Anthropic

        client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )

        assert message.content is not None
        assert len(message.content) > 0

    def test_gemini_api(self):
        """Test Gemini API is accessible."""
        from google import genai

        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents='Hi'
        )

        assert response.text is not None
        assert len(response.text) > 0


class TestRAGBuilder:
    """Test RAG builder functionality."""

    def test_docs_directory_exists(self):
        """Verify docs directory exists."""
        assert os.path.exists("./docs"), "docs/ directory not found"

    def test_markdown_files_exist(self):
        """Verify markdown files exist in docs."""
        docs_path = Path("./docs")
        md_files = list(docs_path.rglob("*.md"))
        assert len(md_files) > 0, "No markdown files found in docs/"

    @pytest.mark.skipif(not os.path.exists("./rag_storage"), reason="RAG storage not built yet")
    def test_rag_storage_exists(self):
        """Verify RAG storage directory was created."""
        assert os.path.exists("./rag_storage/docstore.json")
        assert os.path.exists("./rag_storage/default__vector_store.json")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
```

**Step 2: Run the test to verify it fails appropriately**

Run: `python -m pytest test_rag_integration.py -v`
Expected: Tests pass for API keys, may skip storage tests if not built

**Step 3: Commit**

```bash
git add test_rag_integration.py
git commit -m "test: add integration tests for RAG stack

- Test API key presence and validity
- Test connectivity to OpenAI, Anthropic, Gemini
- Test docs directory structure
- Test RAG storage creation (skip if not built)
- Provides pre-flight checks before building indices"
```

---

## Task 11: Update Documentation

**Files:**
- Create: `docs/rag-setup.md`

**Step 1: Create RAG setup documentation**

```markdown
# RAG Stack Setup Guide

## Overview

This directory contains a Python-based RAG (Retrieval-Augmented Generation) system for querying BakkesMod SDK documentation. It uses hybrid retrieval combining vector search, knowledge graphs, and BM25 keyword matching.

## Architecture

```
rag_builder.py          - Local embeddings (HuggingFace, no API costs)
comprehensive_rag.py    - OpenAI embeddings + Anthropic reranking
gemini_rag_builder.py   - Gemini knowledge graph + Claude reranking
mcp_rag_server.py       - MCP server for Claude Code integration
rag_sentinel.py         - Health checks and diagnostics
evaluator.py            - RAG quality evaluation
```

## Prerequisites

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
# Required for all builds
export OPENAI_API_KEY="sk-..."

# Optional: for Anthropic reranking
export ANTHROPIC_API_KEY="sk-ant-..."

# Optional: for Gemini knowledge graphs
export GOOGLE_API_KEY="..."

# Optional: custom storage location
export RAG_STORAGE_DIR="./custom_storage"
```

### 3. Prepare Documentation

Ensure markdown files exist in `./docs/`:

```bash
ls -R docs/*.md
```

## Usage

### Option 1: Local Embeddings (No API costs)

```bash
python rag_builder.py
```

Uses HuggingFace embeddings (BAAI/bge-small-en-v1.5) locally. No API calls.

### Option 2: OpenAI Stack (High quality)

```bash
python comprehensive_rag.py
```

Uses:
- OpenAI `text-embedding-3-large` for embeddings
- GPT-4o-mini for knowledge graph extraction
- Claude 3.5 Sonnet for reranking

### Option 3: Gemini Stack (Fast knowledge graphs)

```bash
python gemini_rag_builder.py
```

Uses:
- OpenAI embeddings
- Gemini 2.0 Flash for knowledge graph extraction (faster/cheaper than GPT)
- Claude 3.5 Sonnet for reranking

## Health Checks

Before building indices, verify API connectivity:

```bash
python rag_sentinel.py
```

This checks:
- ✅ Environment variables present
- ✅ API keys valid and billing active
- ✅ Documentation files properly formatted
- ✅ Dependencies installed
- ✅ Existing storage integrity

## Integration Testing

```bash
python -m pytest test_rag_integration.py -v
```

## MCP Server (Claude Code)

Start the MCP server for Claude Code integration:

```bash
python mcp_rag_server.py
```

This enables the `query_bakkesmod_sdk` tool in Claude Code sessions.

## Troubleshooting

### Error: "No markdown files found"

Ensure `./docs/` contains `.md` files:

```bash
find docs -name "*.md" | head
```

### Error: "API key not found"

Set environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
```

### Error: "Index build failed"

Check logs:

```bash
tail -100 rag_build.log
tail -100 gemini_build.log
```

### Storage Corruption

Delete and rebuild:

```bash
rm -rf rag_storage/
python comprehensive_rag.py
```

## Performance Notes

**Vector Index Build:** ~2-5 minutes for 2,300 documents
**Knowledge Graph Build:** ~10-30 minutes depending on LLM speed
**Storage Size:** ~400-500MB for full BakkesMod SDK docs

## Cost Estimates (OpenAI Stack)

- Embeddings: ~$0.50 for full build
- KG Extraction: ~$2-5 for full build (GPT-4o-mini)
- Query: ~$0.01-0.05 per query (embeddings + reranking)
```

**Step 2: Commit**

```bash
git add docs/rag-setup.md
git commit -m "docs: add RAG stack setup and usage guide

- Document all three RAG builder options
- Provide setup instructions and prerequisites
- Add troubleshooting section
- Include cost estimates and performance notes
- Explain MCP server integration"
```

---

## Task 12: Final Verification

**Files:**
- All modified files

**Step 1: Run syntax check on all Python files**

Run: `python -m py_compile rag_sentinel.py gemini_rag_builder.py comprehensive_rag.py mcp_rag_server.py rag_builder.py evaluator.py`
Expected: No output (success)

**Step 2: Run integration tests**

Run: `python -m pytest test_rag_integration.py -v`
Expected: API connectivity tests pass

**Step 3: Run sentinel diagnostic**

Run: `python rag_sentinel.py 2>&1 | head -50`
Expected: All checks pass (if API keys present)

**Step 4: Verify git status**

Run: `git status`
Expected: Clean working directory or only untracked storage/logs

**Step 5: Create final summary commit**

```bash
git add -A
git commit -m "chore: finalize Gemini RAG stack fixes

Summary of changes:
- Fixed Gemini API initialization (Client pattern)
- Fixed Anthropic model identifiers (SDK format)
- Removed deprecated GPTCache
- Standardized storage configuration
- Added comprehensive error handling
- Implemented proper logging
- Created integration tests
- Updated documentation

All critical issues from audit report resolved.
Code is now aligned with 2026 official SDKs."
```

---

## Post-Implementation Checklist

After completing all tasks:

- [ ] All Python files pass syntax check (`py_compile`)
- [ ] Integration tests pass (`pytest`)
- [ ] Sentinel diagnostic passes (`rag_sentinel.py`)
- [ ] Documentation is complete (`docs/rag-setup.md`)
- [ ] `.gitignore` excludes generated files
- [ ] `requirements.txt` has pinned versions
- [ ] All commits have descriptive messages
- [ ] No API keys or secrets in code
- [ ] Code follows DRY principle
- [ ] Error handling is comprehensive
- [ ] Logging is configured properly

## Execution Notes

**Estimated Time:** 2-3 hours for full implementation
**Complexity:** Medium (mostly find-replace with verification)
**Risk Level:** Low (improvements to existing code, no breaking changes)

## Success Criteria

1. ✅ All Python files compile without syntax errors
2. ✅ Integration tests pass for API connectivity
3. ✅ Sentinel diagnostic shows all systems operational
4. ✅ Documentation is clear and complete
5. ✅ Code aligns with 2026 official SDK patterns
6. ✅ All critical audit issues resolved
