# Copilot Instructions for BakkesMod RAG Documentation

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests (testpaths = tests/)
pytest -v

# Run a single test file
pytest tests/test_cache.py -v

# Run a single test by name
pytest tests/test_confidence.py::test_high_confidence -v

# Smoke tests (quick sanity check, root-level)
pytest test_smoke.py -v

# Validate environment & API keys before building indices
python -m bakkesmod_rag.sentinel

# Build/rebuild RAG indices (persisted to rag_storage/)
python -m bakkesmod_rag.comprehensive_builder

# Run CLI
python interactive_rag.py

# Run GUI (Gradio, port 7860)
python rag_gui.py
```

There is no linter or formatter configured in the project.

## Architecture

RAG system for BakkesMod SDK documentation using LlamaIndex. Provides an interactive query interface and code generation mode for BakkesMod Rocket League plugin development.

### Package layout

All logic lives in the `bakkesmod_rag/` package. Entry points (`interactive_rag.py`, `rag_gui.py`) are thin I/O wrappers that delegate to `RAGEngine`.

- **`engine.py`** — `RAGEngine`: central orchestrator. Wires LLM, cache, retrieval, code generation. Exposes `query()`, `query_streaming()`, `generate_code()`.
- **`config.py`** — Pydantic `RAGConfig` with nested config models (`LLMConfig`, `RetrieverConfig`, `CacheConfig`, `CostConfig`, etc.). Calls `load_dotenv()` at import.
- **`llm_provider.py`** — Single LLM fallback chain. Each provider is verified with a live `llm.complete("Say OK")` call before use.
- **`retrieval.py`** — 3-way fusion: `QueryFusionRetriever` combining Vector (OpenAI `text-embedding-3-small`), BM25, and Knowledge Graph. Results fused via reciprocal rank fusion.
- **`document_loader.py`** — Loads `.md`, `.h`, `.cpp` from `docs_bakkesmod_only/` and `templates/`. Markdown → `MarkdownNodeParser`; code → `CodeSplitter` (tree-sitter AST), falls back to `SentenceSplitter`.
- **`cache.py`** — `SemanticCache`: embedding-based response cache (92% similarity threshold, 7-day TTL).
- **`resilience.py`** — `CircuitBreaker`, `RateLimiter`, `FallbackChain`, `APICallManager`. `CircuitBreakerOpen` must be excluded from retry logic to allow fallback chains.
- **`code_generator.py`** — `BakkesModCodeGenerator` + `PluginTemplateEngine` + `CodeValidator` for BakkesMod plugin scaffolding.

### LLM fallback chain

Defined in `llm_provider.py`. Priority order (each verified live at startup):
1. Anthropic Claude Sonnet (premium)
2. OpenRouter / DeepSeek V3 (free)
3. Google Gemini 2.5 Flash (free)
4. OpenAI GPT-4o-mini (cheap)

### Index storage

`rag_storage/` is the single canonical index directory (gitignored, ~250MB). Delete it to force a full rebuild. Vector index ID: `"vector"`, KG index ID: `"knowledge_graph"`.

## Key Conventions

### Coding style
- PEP 8, ~100 char lines, Google-style docstrings
- `snake_case` functions/variables, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- Group imports: stdlib → third-party → local, separated by blank lines. No wildcard imports.
- Always UTF-8 encoding. On Windows, wrap stdout/stderr with `TextIOWrapper` for Unicode.

### Resilience patterns
- Resilience decorators (e.g. `resilient_api_call`) must create singleton state (circuit breakers, rate limiters) **outside** the wrapper function to preserve state across invocations.
- `RateLimiter.acquire()` releases its lock before sleeping to avoid blocking other threads.
- `CircuitBreakerOpen` is excluded from retry logic so fallback chains fire without delay.

### Observability
- Use `logging.getLogger("bakkesmod_rag.<module>")` — each module gets its own logger under the package namespace.
- `StructuredLogger` checks for existing handlers before adding new ones to prevent log duplication.
- Never log API keys or sensitive data.

### Environment variables
- Stored in `.env` (gitignored). `config.py` calls `load_dotenv()` automatically.
- Required: `OPENAI_API_KEY` (for embeddings). Plus at least one LLM key: `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `GOOGLE_API_KEY`, or `OPENAI_API_KEY`.
- Optional: `COHERE_API_KEY` (neural reranking), `DAILY_BUDGET_USD` (cost limit).
- Use `GOOGLE_API_KEY` (not `GEMINI_API_KEY`) for Gemini credentials.

### Testing
- `pytest.ini` sets `testpaths = tests`. Integration marker: `@pytest.mark.integration`.
- `tests/conftest.py` provides `MockLLM`, `MockEmbedModel`, `MockSourceNode`, and fixtures (`mock_llm`, `test_config`, `source_nodes_high`, etc.). Unit tests should never need API keys.
- Use `@pytest.mark.skipif` to gate tests that require API keys or network access.

### Document text sanitization
All loaded documents are cleaned: `"".join(filter(lambda x: x.isprintable() or x in "\n\r\t", text))`.

### User context
The primary user is not a developer. Explain things in plain English when writing user-facing text.
