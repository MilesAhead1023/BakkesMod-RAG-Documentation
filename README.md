# BakkesMod RAG Documentation System

A Python-based Retrieval-Augmented Generation (RAG) system for querying BakkesMod SDK documentation and generating BakkesMod plugin code. Built with LlamaIndex.

> **Platform Note:** Optimized for **Windows 11** (BakkesMod is Windows-only). The RAG system itself is cross-platform.

## What It Does

**Two modes:**

1. **Q&A** — Ask questions about the BakkesMod SDK in plain English. Answers are grounded in the actual SDK docs with source citations and confidence scores.
2. **Code Generation** — Describe a plugin and get complete, validated C++ project files (`.h`, `.cpp`, CMake, README) using correct BakkesMod API patterns.

```bash
# Q&A mode
python interactive_rag.py

# Code generation (inside interactive mode)
[QUERY] > /generate Create a plugin that hooks goal events and logs scorer info
```

See [CODE_GENERATION_GUIDE.md](docs/CODE_GENERATION_GUIDE.md) for details.

## Quick Start

### Prerequisites

- Python 3.8+
- `OPENAI_API_KEY` (required for embeddings)
- At least one LLM key: `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`

### Installation

**Windows:**
```cmd
git clone https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation.git
cd BakkesMod-RAG-Documentation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

**Linux/Mac:**
```bash
git clone https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation.git
cd BakkesMod-RAG-Documentation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your API keys, then:

```bash
# Validate your environment
python -m bakkesmod_rag.sentinel

# Build RAG indices (first run takes 10-30 min)
python -m bakkesmod_rag.comprehensive_builder

# Start the CLI
python interactive_rag.py
```

## Web GUI

**Windows executable (no Python required):**

Download `BakkesMod_RAG_GUI.zip` from [Releases](https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation/releases).

**From source:**

```bash
python rag_gui.py          # opens at http://localhost:7860
```

Features: documentation Q&A, plugin code generation, session statistics, code export, confidence scores, source citations.

See [GUI_USER_GUIDE.md](GUI_USER_GUIDE.md) and [EXE_USER_GUIDE.md](EXE_USER_GUIDE.md).

## Architecture

All logic lives in the `bakkesmod_rag/` package. Entry points (`interactive_rag.py`, `rag_gui.py`) are thin I/O wrappers.

```
bakkesmod_rag/
  engine.py              # RAGEngine: central orchestrator
  config.py              # Pydantic config (loads .env automatically)
  llm_provider.py        # LLM fallback chain with live verification
  retrieval.py           # 3-way fusion: Vector + BM25 + Knowledge Graph
  document_loader.py     # Loads .md, .h, .cpp from docs + templates
  cache.py               # Semantic cache (92% threshold, 7-day TTL)
  code_generator.py      # Plugin code gen + validation
  resilience.py          # Circuit breakers, rate limiters, retries
  cost_tracker.py        # Token-level cost tracking + budget alerts
  query_rewriter.py      # BakkesMod domain synonym expansion
  confidence.py          # 5-tier confidence scoring
  observability.py       # Structured logging + Phoenix + Prometheus
  sentinel.py            # Environment & API key health checks
  evaluator.py           # RAG quality evaluation
  mcp_server.py          # MCP server for Claude Code integration
  comprehensive_builder.py  # Full index builder with KG checkpoints
```

### LLM Fallback Chain

Each provider is tested with a live API call at startup. If one fails, the next is tried:

1. Anthropic Claude Sonnet (premium)
2. OpenRouter / DeepSeek V3 (free)
3. Google Gemini 2.5 Flash (free)
4. OpenAI GPT-4o-mini (cheap)

### Retrieval Pipeline

Query → Semantic Cache → 3-way Fusion (Vector + BM25 + KG) → Reciprocal Rank Fusion → Reranking → LLM Response

### Index Storage

`rag_storage/` — single canonical index directory (gitignored, ~250MB). Delete it to force a full rebuild.

## Testing

```bash
# Run all tests (testpaths = tests/)
pytest -v

# Run a single test file
pytest tests/test_cache.py -v

# Run a single test
pytest tests/test_confidence.py::test_high_confidence -v

# Smoke tests (fast, no API keys needed)
pytest tests/test_smoke.py -v
```

Unit tests use mocks from `tests/conftest.py` and don't require API keys.

## Documentation

- [Setup Guide](docs/rag-setup.md) — Installation and configuration
- [Architecture](CLAUDE.md) — System design and architecture overview
- [Code Generation Guide](docs/CODE_GENERATION_GUIDE.md) — Plugin code gen docs
- [BakkesMod SDK Reference](docs/bakkesmod-sdk-reference.md) — SDK documentation
- [ImGui Signatures](docs/bakkesmod_imgui_signatures_annotated.md) — UI framework reference
- [Deployment Guide](DEPLOYMENT_GUIDE.md) — Docker and production deployment

## AI Agent Integration

Designed as a **source of truth** for AI coding agents working on BakkesMod plugins.

**Python:**
```python
from bakkesmod_rag import RAGEngine

engine = RAGEngine()
result = engine.query("How do I hook the goal scored event?")
print(result.answer)
```

**MCP (Claude Code, VSCode):**
```bash
python -m bakkesmod_rag.mcp_server
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see LICENSE file.

## Related Projects

- [BakkesMod](https://bakkesmod.com/) — Rocket League mod framework
