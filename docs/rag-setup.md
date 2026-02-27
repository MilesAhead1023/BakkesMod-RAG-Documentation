# RAG Stack Setup Guide

## Overview

This directory contains a Python-based RAG (Retrieval-Augmented Generation) system for querying BakkesMod SDK documentation. It uses hybrid retrieval combining vector search, knowledge graphs, and BM25 keyword matching.

> **Platform Note:** Optimized for **Windows 11** (primary platform for BakkesMod development). Cross-platform support available for Linux/Mac.

## Architecture

```
bakkesmod_rag/
  engine.py              # RAGEngine: central orchestrator
  config.py              # Pydantic config (loads .env automatically)
  llm_provider.py        # LLM fallback chain with live verification
  retrieval.py           # 3-way fusion: Vector + BM25 + Knowledge Graph
  document_loader.py     # Loads .md, .h, .cpp from docs + templates
  sentinel.py            # Health checks and diagnostics
  evaluator.py           # RAG quality evaluation
  mcp_server.py          # MCP server for Claude Code integration
  comprehensive_builder.py  # Full index builder with KG checkpoints
```

## Prerequisites

### 1. Install Python Dependencies

**Windows:**
```cmd
pip install -r requirements.txt
```

**Linux/Mac:**
```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

**Windows PowerShell:**
```powershell
# Required API keys (all three are required for full functionality)
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
$env:GOOGLE_API_KEY="..."

# Optional: custom storage location
$env:RAG_STORAGE_DIR="./custom_storage"
```

**Linux/Mac:**
```bash
# Required API keys (all three are required for full functionality)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."

# Optional: custom storage location
export RAG_STORAGE_DIR="./custom_storage"
```

**Note:** All three API keys (OpenAI, Anthropic, and Google) are required by `python -m bakkesmod_rag.sentinel` and integration tests. Individual scripts may work with a subset, but for complete functionality, configure all three keys.

### 3. Prepare Documentation

Ensure markdown files exist in `./docs/`:

**Windows:**
```cmd
dir docs\*.md /s
```

**Linux/Mac:**
```bash
find docs -name "*.md"
```

## Usage

### Option 1: Build Indices

```bash
python -m bakkesmod_rag.comprehensive_builder
```

Builds vector index and optionally knowledge graph. Uses OpenAI `text-embedding-3-small` for embeddings, with configurable LLM for KG extraction.

### Option 2: Interactive CLI

```bash
python interactive_rag.py
```

Launches the interactive query interface. Will build indices on first run if `rag_storage/` doesn't exist.

## Health Checks

Before building indices, verify API connectivity:

```bash
python -m bakkesmod_rag.sentinel
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
python -m bakkesmod_rag.mcp_server
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

**Windows PowerShell:**
```powershell
$env:OPENAI_API_KEY="your-key"
$env:ANTHROPIC_API_KEY="your-key"
$env:GOOGLE_API_KEY="your-key"
```

**Linux/Mac:**
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

**Windows:**
```cmd
rmdir /s /q rag_storage
python -m bakkesmod_rag.comprehensive_builder
```

**Linux/Mac:**
```bash
rm -rf rag_storage/
python -m bakkesmod_rag.comprehensive_builder
```

## Performance Notes

**Vector Index Build:** ~2-5 minutes for 2,300 documents
**Knowledge Graph Build:** ~10-30 minutes depending on LLM speed
**Storage Size:** ~400-500MB for full BakkesMod SDK docs

## Cost Estimates (OpenAI Stack)

- Embeddings: ~$0.50 for full build
- KG Extraction: ~$2-5 for full build (GPT-4o-mini)
- Query: ~$0.01-0.05 per query (embeddings + reranking)

## Recent Fixes (2026-02-05)

This RAG stack was updated to align with 2026 official SDK documentation:

- ✅ Fixed Gemini API initialization (Client pattern)
- ✅ Fixed Anthropic model identifiers (SDK format, not Bedrock)
- ✅ Removed deprecated GPTCache library
- ✅ Updated to verified model snapshots
- ✅ Switched reranker to OpenAI for cost-effectiveness
- ✅ Added comprehensive error handling
- ✅ Created integration tests

See `docs/plans/2026-02-05-fix-gemini-rag-stack.md` for detailed implementation plan.
