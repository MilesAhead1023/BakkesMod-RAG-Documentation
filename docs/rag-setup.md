# RAG Stack Setup Guide

## Overview

This directory contains a Python-based RAG (Retrieval-Augmented Generation) system for querying BakkesMod SDK documentation. It uses hybrid retrieval combining vector search, knowledge graphs, and BM25 keyword matching.

> **Platform Note:** Optimized for **Windows 11** (primary platform for BakkesMod development). Cross-platform support available for Linux/Mac.

## Architecture

```
rag_builder.py          - Local embeddings (HuggingFace, no API costs)
comprehensive_rag.py    - OpenAI embeddings + OpenAI reranking
gemini_rag_builder.py   - Gemini knowledge graph + Claude reranking
mcp_rag_server.py       - MCP server for Claude Code integration
rag_sentinel.py         - Health checks and diagnostics
evaluator.py            - RAG quality evaluation
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

**Note:** All three API keys (OpenAI, Anthropic, and Google) are required by `rag_sentinel.py` and integration tests. Individual scripts may work with a subset, but for complete functionality, configure all three keys.

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
- GPT-4o-mini for reranking (fast and cost-effective)

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
python comprehensive_rag.py
```

**Linux/Mac:**
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
