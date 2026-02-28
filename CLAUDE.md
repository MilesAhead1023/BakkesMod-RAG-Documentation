# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

RAG system for BakkesMod SDK documentation using LlamaIndex. Provides an interactive query interface and code generation mode for BakkesMod Rocket League plugin development. All functionality is in the unified `bakkesmod_rag/` Python package.

## Commands

```bash
# Run the NiceGUI desktop app (native exe)
python nicegui_app.py

# Build the native exe (PyInstaller)
pyinstaller --clean --noconfirm nicegui_app.spec

# Run the interactive RAG system (CLI)
python interactive_rag.py

# Run the Gradio web GUI (used by Docker)
python rag_gui.py

# Run all tests
pytest -v

# Run comprehensive builder with incremental KG checkpointing
python -m bakkesmod_rag.comprehensive_builder

# Validate environment and API keys
python -m bakkesmod_rag.sentinel

# Run RAG quality evaluation
python -m bakkesmod_rag.evaluator

# Start MCP server for Claude Code integration
python -m bakkesmod_rag.mcp_server

# Install core dependencies
pip install -r requirements.txt

# Install with dev/testing tools
pip install -r requirements.txt -r requirements-dev.txt

# Install optional features (observability, evaluation, API server, etc.)
pip install -r requirements-optional.txt

# Docker deployment
docker-compose up -d
```

## Architecture

### Unified Package: `bakkesmod_rag/`

All RAG logic lives in the `bakkesmod_rag/` package. The entry points (`nicegui_app.py`, `interactive_rag.py`, and `rag_gui.py`) are thin wrappers that handle I/O while delegating to `RAGEngine`.

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

nicegui_app.py           # Native desktop app (NiceGUI, 7 tabs)
nicegui_app.spec         # PyInstaller spec for building native exe
interactive_rag.py       # CLI entry point — thin wrapper (288 lines)
rag_gui.py               # Gradio web GUI — used by Docker (1174 lines)
```

### LLM Fallback Chain

Defined once in `bakkesmod_rag/llm_provider.py`. Priority order, each verified with a live `"Say OK"` test call at startup:
1. Anthropic Claude Sonnet (`ANTHROPIC_API_KEY`) -- premium
2. OpenAI GPT-4o (`OPENAI_API_KEY`) -- best non-Anthropic quality
3. Google Gemini 2.5 Pro (`GOOGLE_API_KEY`) -- high quality paid
4. OpenRouter / DeepSeek V3 (`OPENROUTER_API_KEY`) -- FREE
5. Google Gemini 2.5 Flash (`GOOGLE_API_KEY`) -- FREE tier, fast
6. Ollama local model -- auto-detected, no config, works offline

If one provider's credits are exhausted or unavailable, the system automatically falls through to the next.

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
| `bakkesmod_rag/answer_verifier.py` | Answer verification and fact-checking logic |
| `bakkesmod_rag/api.py` | HTTP REST API endpoints and request handling |
| `bakkesmod_rag/compiler.py` | C++ plugin code compilation and validation |
| `bakkesmod_rag/cpp_analyzer.py` | C++ code analysis and semantic extraction |
| `bakkesmod_rag/feedback_store.py` | User feedback persistence and retrieval |
| `bakkesmod_rag/guardrails.py` | Content safety and policy enforcement |
| `bakkesmod_rag/intent_router.py` | Query intent classification and routing |
| `bakkesmod_rag/query_decomposer.py` | Complex query decomposition into sub-queries |
| `bakkesmod_rag/setup_keys.py` | API key setup and environment configuration |

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
