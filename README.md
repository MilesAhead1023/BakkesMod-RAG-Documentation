# BakkesMod RAG Documentation System

A Python-based Retrieval-Augmented Generation (RAG) system for querying BakkesMod SDK documentation using hybrid retrieval strategies and semantic caching.

## Features

- **Hybrid Retrieval Architecture**
  - Vector search using sentence transformers
  - Knowledge graph traversal for relationship queries
  - BM25 for keyword-based retrieval
  - Intelligent query routing and result fusion

- **Semantic Caching with GPTCache**
  - 30-40% cost reduction on repeated queries
  - Embedding similarity-based cache matching
  - Automatic cache management

- **Multiple LLM Backends**
  - OpenAI GPT-4
  - Google Gemini 2.0 Flash
  - Anthropic Claude (via API)
  - Configurable fallback strategies

- **MCP Server Integration**
  - Claude Code integration via Model Context Protocol
  - Interactive documentation queries in your IDE
  - Context-aware code suggestions

- **Comprehensive Documentation Database**
  - BakkesMod SDK reference documentation
  - ImGui integration guides
  - Code examples and best practices
  - 2,300+ training pack metadata

## Quick Start

### Prerequisites

- Python 3.8+
- API keys for your chosen LLM provider (OpenAI, Gemini, or Anthropic)

### Installation

```bash
# Clone the repository
git clone https://github.com/MilesAhead1023/BakkesMod-RAG-Documentation.git
cd BakkesMod-RAG-Documentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file with your API keys:

```env
OPENAI_API_KEY=your_openai_key_here
GEMINI_API_KEY=your_gemini_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

### Usage

**Build the RAG index:**

```bash
# Using OpenAI embeddings
python rag_builder.py

# Using Gemini
python gemini_rag_builder.py
```

**Query the documentation:**

```bash
# Interactive query mode
python comprehensive_rag.py

# MCP server for Claude Code
python mcp_rag_server.py
```

**Run tests:**

```bash
pytest test_rag_integration.py -v
pytest test_smoke.py -v
```

## Documentation

- [Setup Guide](docs/rag-setup.md) - Detailed installation and configuration
- [Architecture](docs/architecture.md) - System design and components
- [BakkesMod SDK Reference](docs/bakkesmod-sdk-reference.md) - SDK documentation
- [ImGui Signatures](docs/bakkesmod_imgui_signatures_annotated.md) - UI framework reference

## Project Structure

```
.
├── comprehensive_rag.py          # Main RAG query interface
├── gemini_rag_builder.py         # Gemini-based index builder
├── rag_builder.py                # OpenAI-based index builder
├── mcp_rag_server.py             # MCP server for Claude Code
├── rag_sentinel.py               # Monitoring and health checks
├── evaluator.py                  # RAG performance evaluation
├── test_rag_integration.py       # Integration tests
├── test_smoke.py                 # Smoke tests
├── requirements.txt              # Python dependencies
└── docs/                         # Documentation
    ├── rag-setup.md
    ├── architecture.md
    ├── bakkesmod-sdk-reference.md
    └── ...
```

## History

This project was split from the [SuiteSpot](https://github.com/MilesAhead1023/SuiteSpot) BakkesMod plugin repository on 2026-02-05 to maintain independent development cycles and clearer project boundaries.

**SuiteSpot** is a C++20 BakkesMod plugin for Rocket League that auto-loads training content after matches. This RAG system was originally built to help develop that plugin but has evolved into a standalone documentation tool.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Related Projects

- [SuiteSpot](https://github.com/MilesAhead1023/SuiteSpot) - BakkesMod plugin for Rocket League
- [BakkesMod](https://bakkesmod.com/) - Rocket League mod framework
