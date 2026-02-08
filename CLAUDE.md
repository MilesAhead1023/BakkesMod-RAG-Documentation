# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

RAG system for BakkesMod SDK documentation using LlamaIndex. Provides an interactive query interface and code generation mode for BakkesMod Rocket League plugin development.

## Commands

```bash
# Run the interactive RAG system (primary entry point)
python interactive_rag.py

# Run the config-driven "Gold Standard" RAG variant
python rag_2026.py

# Run all tests
pytest -v

# Run a specific test file
pytest test_smoke.py -v

# Build KG index separately using OpenAI (faster than free tier)
python build_kg_openai.py

# Validate environment and API keys
python rag_sentinel.py

# Install dependencies
pip install -r requirements.txt

# Docker deployment
docker-compose up -d
```

## Architecture

### Two RAG Entry Points

- **`interactive_rag.py`** — The actively-used, battle-tested entry point. Inline LLM fallback chain with live verification, semantic cache, query rewriting, streaming responses, syntax highlighting, confidence scoring, and `/generate` code generation mode. Loads from `docs_bakkesmod_only/` and `templates/` directories.
- **`rag_2026.py`** + **`config.py`** — Config-driven "Gold Standard" variant using Pydantic `RAGConfig` singleton. Has observability (Phoenix tracing, Prometheus metrics), cost tracking, and circuit breakers. Loads from `docs/` directory only. Uses different index IDs (`"kg"` vs `"knowledge_graph"`).

These two systems are **not integrated**. They have separate document loading, different storage paths, and different LLM configuration approaches. `interactive_rag.py` is the one actively maintained and used.

### LLM Fallback Chain (interactive_rag.py)

Priority order, each verified with a live `"Say OK"` test call at startup:
1. Anthropic Claude Sonnet 4.5 (premium)
2. OpenRouter / DeepSeek V3 (FREE)
3. Google Gemini 2.5 Flash (FREE)
4. OpenAI GPT-4o-mini (cheap)

This same pattern is duplicated in `code_generator.py`. If one provider's credits are exhausted, the system automatically falls through to the next.

### 3-Way Fusion Retrieval

`QueryFusionRetriever` combines:
- **Vector search** (OpenAI `text-embedding-3-small` embeddings)
- **BM25** keyword search
- **Knowledge Graph** (relationship triplets extracted by LLM)

Results are fused via reciprocal rank fusion, then optionally reranked by Cohere neural reranker.

### Document Sources

- `docs_bakkesmod_only/` — 179 files: SDK header files (`.h`), markdown reference docs
- `templates/` — 36 files: BakkesMod plugin template (`.h`, `.cpp`, `.props`, etc.)
- Markdown files parsed with `MarkdownNodeParser`; code files with `SentenceSplitter(chunk_size=1024, chunk_overlap=128)`

### Index Storage

- `rag_storage/` — Persisted indexes used by `interactive_rag.py` (gitignored, ~250MB)
  - Vector index ID: `"vector"`
  - KG index ID: `"knowledge_graph"`
  - Delete this directory to force full rebuild
- `rag_storage_bakkesmod/` — Older index used by `code_generator.py`

### Supporting Modules

| Module | Purpose |
|---|---|
| `cache_manager.py` | `SemanticCache` — embedding-based response caching (92% similarity threshold, 7-day TTL) |
| `query_rewriter.py` | `QueryRewriter` — BakkesMod domain synonym expansion (60+ mappings, zero API cost) |
| `code_generator.py` | `CodeGenerator` — RAG-enhanced plugin code generation (`.h` + `.cpp`) |
| `code_templates.py` | `PluginTemplateEngine` — structural templates for plugin scaffolding |
| `code_validator.py` | `CodeValidator` — C++ syntax and BakkesMod API pattern validation |
| `build_kg_openai.py` | Standalone KG builder using GPT-4o-mini (~13 min vs ~80 min on free tier) |
| `cost_tracker.py` | Token-level cost tracking with budget alerts |
| `observability.py` | Phoenix tracing + Prometheus metrics + structured logging |
| `resilience.py` | Circuit breakers, retry strategies, fallback chains |
| `mcp_rag_server.py` | MCP server for Claude Code IDE integration |

## Key Patterns

### Live LLM Verification

Every LLM provider is tested with a real API call (`llm.complete("Say OK")`) before being accepted. This catches expired credits, invalid keys, and rate limits that wouldn't surface from just initializing the client. Pattern used in both `interactive_rag.py` and `code_generator.py`.

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
