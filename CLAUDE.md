# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

RAG system for BakkesMod SDK documentation using LlamaIndex. Provides an interactive query interface and code generation mode for BakkesMod Rocket League plugin development. All functionality is in the unified `bakkesmod_rag/` Python package.

## Commands

```bash
# Run the interactive RAG system (CLI)
python interactive_rag.py

# Run the GUI (Gradio web interface)
python rag_gui.py

# Run all tests
pytest -v

# Run a specific test file
pytest test_smoke.py -v

# Build KG index separately using OpenAI (faster than free tier)
python build_kg_openai.py

# Run comprehensive builder with incremental KG checkpointing
python -m bakkesmod_rag.comprehensive_builder

# Validate environment and API keys
python -m bakkesmod_rag.sentinel

# Run RAG quality evaluation
python -m bakkesmod_rag.evaluator

# Start MCP server for Claude Code integration
python -m bakkesmod_rag.mcp_server

# Install dependencies
pip install -r requirements.txt

# Docker deployment
docker-compose up -d
```

## Architecture

### Unified Package: `bakkesmod_rag/`

All RAG logic lives in the `bakkesmod_rag/` package. The two entry points (`interactive_rag.py` and `rag_gui.py`) are thin wrappers that handle I/O while delegating to `RAGEngine`.

```
bakkesmod_rag/
  __init__.py            # Public API: RAGEngine, QueryResult, CodeResult, RAGConfig
  config.py              # Pydantic config with all settings
  llm_provider.py        # Single LLM fallback chain + live verification
  document_loader.py     # Load all 215 docs (.md, .h, .cpp) from both dirs
  retrieval.py           # 3-way fusion: Vector + BM25 + KG
  cache.py               # Semantic cache (embedding-based, 92% threshold, 7-day TTL)
  query_rewriter.py      # Domain synonym expansion (60+ mappings, zero API cost)
  confidence.py          # Confidence scoring logic
  code_generator.py      # Plugin code gen (templates + validator + LLM)
  observability.py       # Structured logging + optional Phoenix/Prometheus
  cost_tracker.py        # Token counting + budget enforcement
  resilience.py          # Circuit breaker + retry + rate limiter
  engine.py              # RAGEngine: central orchestrator wiring everything
  sentinel.py            # System diagnostics and health checks
  evaluator.py           # RAG quality evaluation with optional RAGAS
  mcp_server.py          # MCP server for Claude Code IDE integration
  comprehensive_builder.py  # Full index builder with incremental KG checkpoints

interactive_rag.py       # CLI entry point — thin wrapper (~260 lines)
rag_gui.py               # GUI entry point — thin wrapper (~500 lines)
```

### LLM Fallback Chain

Defined once in `bakkesmod_rag/llm_provider.py`. Priority order, each verified with a live `"Say OK"` test call at startup:
1. Anthropic Claude Sonnet (premium)
2. OpenRouter / DeepSeek V3 (FREE)
3. Google Gemini 2.5 Flash (FREE)
4. OpenAI GPT-4o-mini (cheap)

If one provider's credits are exhausted, the system automatically falls through to the next.

### 3-Way Fusion Retrieval

`QueryFusionRetriever` combines:
- **Vector search** (OpenAI `text-embedding-3-small` embeddings)
- **BM25** keyword search
- **Knowledge Graph** (relationship triplets extracted by LLM)

Results are fused via reciprocal rank fusion with 4 query variants, then optionally reranked by Cohere neural reranker.

### Document Sources

- `docs_bakkesmod_only/` — 179 files: SDK header files (`.h`), markdown reference docs
- `templates/` — 36 files: BakkesMod plugin template (`.h`, `.cpp`, `.props`, etc.)
- Markdown files parsed with `MarkdownNodeParser`; code files with `SentenceSplitter(chunk_size=1024, chunk_overlap=128)`

### Index Storage

- `rag_storage/` — Single canonical persisted index location (gitignored, ~250MB)
  - Vector index ID: `"vector"`
  - KG index ID: `"knowledge_graph"`
  - Delete this directory to force full rebuild

### Package Modules

| Module | Purpose |
|---|---|
| `bakkesmod_rag/config.py` | Pydantic `RAGConfig` with all settings (LLM, embedding, retriever, cache, cost, etc.) |
| `bakkesmod_rag/llm_provider.py` | Single LLM fallback chain with live verification + embedding model |
| `bakkesmod_rag/document_loader.py` | Load and parse all docs from configured directories |
| `bakkesmod_rag/retrieval.py` | Index building, caching, and fusion retriever creation |
| `bakkesmod_rag/cache.py` | `SemanticCache` — embedding-based response caching |
| `bakkesmod_rag/query_rewriter.py` | `QueryRewriter` — BakkesMod domain synonym expansion |
| `bakkesmod_rag/confidence.py` | Confidence scoring based on retrieval quality metrics |
| `bakkesmod_rag/code_generator.py` | `BakkesModCodeGenerator` + `PluginTemplateEngine` + `CodeValidator` |
| `bakkesmod_rag/engine.py` | `RAGEngine` — central orchestrator wiring all subsystems |
| `bakkesmod_rag/cost_tracker.py` | Token-level cost tracking with budget alerts |
| `bakkesmod_rag/observability.py` | Structured logging + optional Phoenix/Prometheus |
| `bakkesmod_rag/resilience.py` | Circuit breakers, retry strategies, fallback chains |
| `bakkesmod_rag/sentinel.py` | System diagnostics and health checks |
| `bakkesmod_rag/evaluator.py` | RAG quality evaluation with golden test queries |
| `bakkesmod_rag/mcp_server.py` | MCP server for Claude Code IDE integration |
| `bakkesmod_rag/comprehensive_builder.py` | Full index build with incremental KG checkpointing |
| `build_kg_openai.py` | Standalone KG builder using GPT-4o-mini |

## Key Patterns

### Live LLM Verification

Every LLM provider is tested with a real API call (`llm.complete("Say OK")`) before being accepted. This catches expired credits, invalid keys, and rate limits. Defined once in `bakkesmod_rag/llm_provider.py`.

### Document Text Sanitization

All loaded documents are cleaned of non-printable characters while preserving `\n\r\t`:
```python
clean_text = "".join(filter(lambda x: x.isprintable() or x in "\n\r\t", doc.text))
```

### Environment Variables

Required in `.env`:
- `OPENAI_API_KEY` — Required for embeddings (text-embedding-3-small) and optional LLM
- At least one LLM key: `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`

Optional:
- `COHERE_API_KEY` — Neural reranking
- `DAILY_BUDGET_USD` — Cost limit

## Coding Conventions

From `.github/copilot-instructions.md`:
- PEP 8 style, ~100 char line length
- Google-style docstrings for public functions
- `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Group imports: stdlib, third-party, local (separated by blank lines)
- Use `try-except` for all API calls; prefer specific exceptions over bare `except:`
- Always use UTF-8 encoding; on Windows, wrap stdout/stderr with UTF-8 `TextIOWrapper`

## User Context

The primary user is not a developer. Explain things in plain English. Anthropic API credits may be exhausted — the system is designed to fall back to free providers automatically.
