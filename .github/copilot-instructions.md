# Copilot Instructions for BakkesMod RAG Documentation

## Project Context

This repository implements a Python-based Retrieval-Augmented Generation (RAG) system for querying BakkesMod SDK documentation. The system uses hybrid retrieval strategies combining:

- **Vector search** using sentence transformers (OpenAI embeddings)
- **Knowledge graph traversal** for relationship-based queries
- **BM25** for keyword-based retrieval
- **Semantic caching** with GPTCache for cost reduction (30-40% savings)

**Related Projects:**
- This was split from [SuiteSpot](https://github.com/MilesAhead1023/SuiteSpot) on 2026-02-05
- SuiteSpot is a C++20 BakkesMod plugin for Rocket League
- This RAG system provides documentation assistance for developing BakkesMod plugins

## Tech Stack

### Core Framework
- **LlamaIndex** (v0.14.6+) - RAG orchestration framework
- **Python 3.8+** - Required runtime

### LLM Providers (Multi-Provider Support)
- **OpenAI** - GPT-4o-mini (KG extraction), GPT-4 (queries), text-embedding-3-large
- **Anthropic** - Claude 3.5 Sonnet (primary response generation)
- **Google Gemini** - Gemini 2.0 Flash (fast KG extraction alternative)

### Storage & Retrieval
- **FAISS** - Vector similarity search (CPU variant)
- **Neo4j** - Knowledge graph storage (optional)
- **GPTCache** - Semantic caching layer with SQLite backend

### Testing
- **pytest** - Test framework
- **pytest-asyncio** - Async test support

### Integration
- **MCP (Model Context Protocol)** - Claude Code IDE integration

## Coding Standards

### Python Style
- Follow **PEP 8** style guidelines
- Use **type hints** for all function signatures when practical
- Maximum line length: **100 characters** (not strict, but preferred)
- Use **docstrings** for all public functions and classes (Google-style format)

### Code Organization
- Keep related functionality in dedicated modules (e.g., `rag_builder.py`, `gemini_rag_builder.py`)
- Avoid circular imports - use explicit imports, never wildcard imports (`from module import *`)
- Group imports: standard library, third-party, local modules (separated by blank lines)

### Naming Conventions
- **Variables/Functions**: `snake_case` (e.g., `build_comprehensive_stack`, `total_nodes`)
- **Classes**: `PascalCase` (e.g., `MarkdownNodeParser`, `VectorStoreIndex`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `DOCS_DIR`, `STORAGE_DIR`, `CHECKPOINT_INTERVAL`)
- **Environment Variables**: `UPPER_SNAKE_CASE` with descriptive names (e.g., `OPENAI_API_KEY`, `RAG_STORAGE_DIR`)

### Error Handling
- Use try-except blocks for API calls and file operations
- Log errors with context using the `logging` module
- Never expose API keys or sensitive data in error messages or logs
- Prefer specific exception types over bare `except:`

### Text Encoding
- Always use **UTF-8** encoding for file operations
- On Windows, wrap stdout/stderr with UTF-8 TextIOWrapper to handle special characters
- Sanitize document text to remove non-printable characters while preserving newlines/tabs

## LLM Provider Configuration

### Cost Optimization Strategy
- Use **GPT-4o-mini** for knowledge graph extraction and reranking (cheaper, faster)
- Use **Claude 3.5 Sonnet** for final response generation (higher quality)
- Target cost: **$0.01-0.05 per query**
- Enable **GPTCache** semantic caching for 30-40% cost reduction on repeated queries

### Provider Selection
All three provider types support multiple backends:
- `primary_provider`: openai, anthropic, gemini
- `kg_provider`: openai, anthropic, gemini
- `rerank_provider`: openai, anthropic (gemini not supported for reranking)

### API Key Management
- Store all API keys in `.env` file (NEVER commit to git)
- Use `os.environ` to access keys: `os.environ["OPENAI_API_KEY"]`
- Required keys: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY` (or `GEMINI_API_KEY`)
- All API credentials must be kept secure and never logged or printed

### Retry Logic
- Configure `max_retries` for all LLM/embedding calls (default: 10 for embeddings, 5 for LLMs)
- Use exponential backoff for rate limit errors
- Handle `CircuitBreakerOpen` exceptions separately from retry logic

## Testing Guidelines

### Test Organization
- Integration tests: `test_rag_integration.py` - validates API connectivity and functionality
- Smoke tests: `test_smoke.py` - quick sanity checks
- Use **pytest markers** for conditional tests (e.g., `@pytest.mark.skipif` for missing API keys)

### Running Tests
```bash
# Run all tests
pytest -v

# Run specific test file
pytest test_rag_integration.py -v

# Run with API key checks
pytest test_rag_integration.py -v --tb=short
```

### Test Requirements
- Tests should verify API connectivity before expensive operations
- Mock external API calls when testing logic (not integration)
- Validate environment variables exist before making API calls
- Clean up temporary files/indices after tests

## Documentation Management

### File Structure
- All documentation stored in `./docs/` directory
- Use **Markdown** format (`.md` files)
- Recursive directory scanning supported
- Documents are parsed using `MarkdownNodeParser` for hierarchical structure

### Documentation Standards
- Keep files under 50KB for optimal processing
- Use proper Markdown headings for document structure (# for titles, ## for sections)
- Include code examples in fenced code blocks with language identifiers
- Avoid non-printable characters (they are filtered during ingestion)

## Dependency Management

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Adding New Dependencies
- Update `requirements.txt` with version constraints
- Use **>= version** for flexibility unless specific version required
- Group related dependencies with comments (e.g., `# Core RAG Framework`, `# LLM Provider SDKs`)
- Test compatibility before committing

### Optional Dependencies
- Comment out optional dependencies (e.g., `# ragas>=0.1.0` for evaluation)
- Document when they should be uncommented

## Storage & Persistence

### Index Storage
- Vector indices saved to `./rag_storage/` or `./gemini_rag_storage/`
- Storage directories are **gitignored** (large binary files)
- Use checkpointing for large knowledge graph builds (`CHECKPOINT_INTERVAL = 500`)

### Cache Management
- GPTCache stores data in `.gptcache/` and `gptcache.db`
- Similarity threshold: **0.9** (90% similarity = cache hit)
- Cache is persistent across runs (SQLite backend)

## Security Best Practices

### API Keys
- **NEVER commit API keys** to the repository
- Store keys in `.env` file (gitignored)
- Use environment variables for all sensitive data
- Validate key presence before making API calls

### Logging
- Use structured logging with appropriate log levels
- **Never log API keys or sensitive user data**
- Log API errors without exposing credentials
- Use `logging.INFO` for general progress, `logging.DEBUG` for detailed traces

## MCP Server Integration

### Claude Code Integration
The `mcp_rag_server.py` provides Model Context Protocol integration for Claude Code:
- Exposes documentation queries as MCP tools
- Enables context-aware code suggestions in IDE
- Configure in Claude Code settings

### Local Settings
- MCP local settings stored in `.claude/settings.local.json` (gitignored)
- Server runs locally and connects to Claude Code

## Common Patterns

### Building Indices
1. Load documents from `./docs/` using `SimpleDirectoryReader`
2. Parse with `MarkdownNodeParser` for hierarchical nodes
3. Build vector index and knowledge graph in parallel when possible
4. Save to persistent storage with checkpointing
5. Use incremental builds to avoid rebuilding from scratch

### Query Execution
1. Check semantic cache first (GPTCache)
2. Route query to appropriate retrieval strategy (vector/graph/BM25)
3. Use query fusion for multi-strategy retrieval
4. Apply LLM reranking to top results
5. Generate final response with context

### Error Recovery
- Implement retry logic with exponential backoff
- Use fallback providers if primary fails
- Log errors with context for debugging
- Gracefully degrade functionality rather than crash

## Development Workflow

### Before Committing
1. Run tests: `pytest -v`
2. Verify no API keys in code
3. Update documentation if changing interfaces
4. Check .gitignore for new temporary files

### Health Checks
Run `rag_sentinel.py` before building indices to verify:
- Environment variables present
- API keys valid and billing active
- Documentation files properly formatted
- Dependencies installed

## Performance Considerations

### Optimization Tips
- Use parallel batching for knowledge graph extraction
- Implement checkpointing for long-running builds (every 500 nodes)
- Enable semantic caching to reduce API costs
- Use `ImGuiListClipper` pattern for large list rendering (if applicable)
- Prefer async operations for concurrent API calls

### Resource Management
- Clean up temporary files after processing
- Use `filename_as_id=True` to enable incremental updates
- Monitor API rate limits and implement backoff
- Cache embedding results to avoid recomputation

## Additional Notes

### Platform Compatibility
- Primary development on Windows (UTF-8 handling for console output)
- Cross-platform support for Linux/macOS
- Use `sys.platform` checks when platform-specific code needed

### IDE Integration
- VS Code recommended (with Python extension)
- `.vscode/` settings are gitignored for personalization
- Use virtual environments to isolate dependencies
