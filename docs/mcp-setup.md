# MCP Server Setup for Claude Code

The BakkesMod RAG system includes an MCP (Model Context Protocol) server that integrates directly with Claude Code. Once set up, you can ask Claude Code questions about the BakkesMod SDK and it will retrieve answers directly from the documentation.

## How It Works

1. You ask Claude Code a question (e.g., "How do I hook the goal scored event?")
2. The MCP server searches the BakkesMod SDK documentation using RAG retrieval
3. Claude Code uses **your own Claude subscription** to generate an answer from those docs
4. No extra API key needed for generation -- only an OpenAI key is needed for document retrieval (embeddings)

## Prerequisites

- Python 3.11+ installed
- `OPENAI_API_KEY` set in your `.env` file (for embeddings/retrieval)
- Claude Code CLI installed (`npm install -g @anthropic-ai/claude-code`)

## One-Time Setup

```bash
# Register the MCP server with Claude Code
claude mcp add bakkesmod-rag python -m bakkesmod_rag.mcp_server
```

That's it. The server registers globally and is available in every Claude Code session.

## Manual Configuration (Alternative)

If you prefer to configure manually, add this to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "bakkesmod-rag": {
      "command": "python",
      "args": ["-m", "bakkesmod_rag.mcp_server"],
      "cwd": "/path/to/BakkesMod-RAG-Documentation"
    }
  }
}
```

Replace `/path/to/BakkesMod-RAG-Documentation` with the actual path to this project.

## Available Tools

Once connected, Claude Code has access to these tools:

| Tool | What it does |
|---|---|
| `query_bakkesmod_sdk` | Ask any question about the BakkesMod SDK -- returns an answer with sources |
| `generate_plugin_code` | Generate a complete BakkesMod plugin from a description |
| `browse_sdk_classes` | List all SDK classes with categories and method counts |
| `get_sdk_class_details` | Get full method signatures and inheritance for a specific class |

## Example Prompts in Claude Code

Once the MCP server is registered, try these in Claude Code:

- "How do I hook the goal scored event?"
- "What methods does CarWrapper have?"
- "Generate a plugin that shows the boost amount in an ImGui overlay"
- "Show me all SDK classes related to the ball"
- "How do I access the game state in a BakkesMod plugin?"

## Troubleshooting

**"Tool not found" error**
Run `claude mcp list` to confirm the server is registered. Re-run the setup command if needed.

**"No documents found" responses**
The RAG index may not be built yet. Run `python -m bakkesmod_rag.comprehensive_builder` to build it.

**Server crashes on startup**
Check that `OPENAI_API_KEY` is set in your `.env` file. Run `python -m bakkesmod_rag.sentinel` to diagnose.

**Slow first response**
The first query loads the index into memory (~30 seconds). Subsequent queries are fast.
