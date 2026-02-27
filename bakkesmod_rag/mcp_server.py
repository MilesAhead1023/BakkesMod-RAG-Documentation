"""
MCP Server for Claude Code Integration
========================================
Exposes BakkesMod RAG as tools for Claude Code IDE via the Model
Context Protocol (MCP).

Provides two tools:

* ``query_bakkesmod_sdk`` -- Ask questions about the BakkesMod SDK
* ``generate_plugin_code`` -- Generate C++ plugin code from a description

Rewrite of the legacy ``mcp_rag_server.py`` using the unified
``bakkesmod_rag`` package.  The watchdog file-watcher dependency has
been removed (can be re-added later if needed).

Usage::

    python -m bakkesmod_rag.mcp_server
"""

import sys
import asyncio
import logging

logger = logging.getLogger("bakkesmod_rag.mcp_server")

# Global engine instance (initialised in main)
engine = None
server = None


async def main():
    """Initialise the RAG engine and start the MCP stdio server."""
    global engine, server

    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        import mcp.types as types
    except ImportError:
        print(
            "The 'mcp' package is required for the MCP server.\n"
            "Install it with:  pip install mcp",
            file=sys.stderr,
        )
        sys.exit(1)

    server = Server("bakkesmod-rag")

    print("Initializing BakkesMod RAG Engine...", file=sys.stderr)
    try:
        from bakkesmod_rag.engine import RAGEngine

        # Run the (potentially slow) engine init in a thread so we
        # don't block the event loop.
        loop = asyncio.get_running_loop()
        engine = await loop.run_in_executor(None, RAGEngine)
        print(
            f"RAG Engine ready: {engine.num_documents} docs, "
            f"{engine.num_nodes} nodes",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"Failed to initialize RAG Engine: {e}", file=sys.stderr)
        sys.exit(1)

    # ----- Tool definitions ------------------------------------------------

    @server.list_tools()
    async def handle_list_tools():
        return [
            types.Tool(
                name="query_bakkesmod_sdk",
                description=(
                    "Retrieve relevant BakkesMod SDK documentation chunks using RAG. "
                    "Returns raw context chunks with confidence scores - you (Claude) "
                    "generate the answer from the retrieved context. No LLM API cost."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Technical question about the BakkesMod SDK, "
                                "wrappers, events, or plugin development"
                            ),
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="generate_plugin_code",
                description=(
                    "Generate BakkesMod plugin C++ code (.h and .cpp) "
                    "from a natural-language description. Uses RAG context "
                    "from SDK documentation for accurate API usage."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "description": {
                            "type": "string",
                            "description": (
                                "Natural-language description of the plugin "
                                "requirements and desired functionality"
                            ),
                        },
                    },
                    "required": ["description"],
                },
            ),
        ]

    # ----- Tool execution --------------------------------------------------

    @server.call_tool()
    async def handle_call_tool(name, arguments):
        loop = asyncio.get_running_loop()

        if name == "query_bakkesmod_sdk":
            query = arguments.get("query", "")
            if not query:
                return [types.TextContent(
                    type="text",
                    text="Error: 'query' parameter is required.",
                )]

            # Retrieve context chunks only (no LLM answer generation)
            result = await loop.run_in_executor(None, engine.retrieve_context, query)

            # Format response with chunks and metadata
            response = f"**Query:** {query}\n"
            if result["expanded_query"] != query:
                response += f"**Expanded Query:** {result['expanded_query']}\n"
            response += f"**Confidence:** {result['confidence']:.0%} ({result['confidence_label']})\n"
            response += f"**Retrieval Time:** {result['query_time']:.2f}s\n\n"
            
            response += f"**Retrieved {len(result['chunks'])} Chunks:**\n\n"
            for i, chunk in enumerate(result['chunks'], 1):
                score_str = f" (score: {chunk['score']:.3f})" if chunk['score'] is not None else ""
                response += f"### Chunk {i} - {chunk['file_name']}{score_str}\n"
                response += f"```\n{chunk['text']}\n```\n\n"
            
            response += "**Sources:**\n"
            for src in result['sources']:
                score_str = f" (score: {src['score']:.3f})" if src.get('score') is not None else ""
                response += f"- {src['file_name']}{score_str}\n"

            return [types.TextContent(type="text", text=response)]

        elif name == "generate_plugin_code":
            description = arguments.get("description", "")
            if not description:
                return [types.TextContent(
                    type="text",
                    text="Error: 'description' parameter is required.",
                )]

            result = await loop.run_in_executor(
                None, engine.generate_code, description
            )

            response_parts = []

            if result.header:
                response_parts.append("// ===== Header File (.h) =====")
                response_parts.append(f"```cpp\n{result.header}\n```")

            if result.implementation:
                response_parts.append("\n// ===== Implementation File (.cpp) =====")
                response_parts.append(f"```cpp\n{result.implementation}\n```")

            if result.validation and not result.validation.get("valid", True):
                response_parts.append("\n// ===== Validation Warnings =====")
                for err in result.validation.get("errors", []):
                    response_parts.append(f"// WARNING: {err}")

            if not response_parts:
                response_parts.append(
                    "Code generation returned empty results. "
                    "Try rephrasing the description."
                )

            return [types.TextContent(
                type="text", text="\n".join(response_parts)
            )]

        raise ValueError(f"Tool not found: {name}")

    # ----- Start server ----------------------------------------------------

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
