"""
MCP Server for Claude Code Integration
========================================
Exposes BakkesMod RAG as tools for Claude Code IDE via the Model
Context Protocol (MCP).

Provides four tools:

* ``query_bakkesmod_sdk`` -- Ask questions about the BakkesMod SDK
* ``generate_plugin_code`` -- Generate C++ plugin code from a description
* ``browse_sdk_classes`` -- List all SDK classes with categories and method counts
* ``get_sdk_class_details`` -- Get full details for a specific SDK class

Rewrite of the legacy ``mcp_rag_server.py`` using the unified
``bakkesmod_rag`` package.  The watchdog file-watcher dependency has
been removed (can be re-added later if needed).

Usage::

    python -m bakkesmod_rag.mcp_server
"""

import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger("bakkesmod_rag.mcp_server")

# Global engine instance (initialised in main)
engine = None
server = None

# Lazy-cached CppAnalyzer instance and class hierarchy
_cpp_analyzer = None
_sdk_classes: Optional[Dict] = None

SDK_HEADER_DIR = "docs_bakkesmod_only"


def _get_sdk_classes() -> Dict:
    """Lazily initialise CppAnalyzer and scan SDK headers once.

    Returns:
        Dict mapping class name to CppClassInfo.

    Raises:
        FileNotFoundError: If the SDK header directory does not exist.
        RuntimeError: If analysis fails for any reason.
    """
    global _cpp_analyzer, _sdk_classes

    if _sdk_classes is not None:
        return _sdk_classes

    sdk_path = Path(SDK_HEADER_DIR)
    if not sdk_path.is_dir():
        raise FileNotFoundError(
            f"SDK header directory '{SDK_HEADER_DIR}/' not found. "
            f"Ensure the BakkesMod documentation files are present."
        )

    from bakkesmod_rag.cpp_analyzer import CppAnalyzer

    _cpp_analyzer = CppAnalyzer()
    _sdk_classes = _cpp_analyzer.analyze_directory(SDK_HEADER_DIR)
    return _sdk_classes


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
            types.Tool(
                name="browse_sdk_classes",
                description=(
                    "List all BakkesMod SDK classes with their categories "
                    "and method counts. Analyzes SDK header files in "
                    "docs_bakkesmod_only/ to extract C++ classes, their "
                    "inheritance hierarchy, and method counts. Note: Full "
                    "method signatures require tree-sitter-language-pack "
                    "to be installed; falls back to regex extraction "
                    "automatically otherwise."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
            types.Tool(
                name="get_sdk_class_details",
                description=(
                    "Get full method signatures, inheritance chain, and "
                    "related types for a specific BakkesMod SDK class. "
                    "Note: Full method signatures require "
                    "tree-sitter-language-pack to be installed; falls "
                    "back to regex extraction automatically otherwise."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "class_name": {
                            "type": "string",
                            "description": (
                                "The SDK class name to look up "
                                "(e.g., 'CarWrapper', 'BallWrapper')"
                            ),
                        },
                    },
                    "required": ["class_name"],
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

        elif name == "browse_sdk_classes":
            try:
                sdk_classes = await loop.run_in_executor(
                    None, _get_sdk_classes
                )
            except FileNotFoundError as e:
                return [types.TextContent(type="text", text=str(e))]
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"Error analyzing SDK headers: {e}",
                )]

            if not sdk_classes:
                return [types.TextContent(
                    type="text",
                    text="No SDK classes found in docs_bakkesmod_only/.",
                )]

            # Sort by category then name
            sorted_classes = sorted(
                sdk_classes.values(),
                key=lambda c: (c.category or "zzz", c.name),
            )

            lines = [f"BakkesMod SDK Classes ({len(sorted_classes)} total)\n"]
            current_cat = None
            for cls in sorted_classes:
                cat = cls.category or "other"
                if cat != current_cat:
                    current_cat = cat
                    lines.append(f"\n--- {cat.upper()} ---")
                parent = cls.base_classes[0] if cls.base_classes else "none"
                lines.append(
                    f"  {cls.name}"
                    f"\n    Methods: {len(cls.methods)}"
                    f"\n    Inherits from: {parent}"
                )

            return [types.TextContent(type="text", text="\n".join(lines))]

        elif name == "get_sdk_class_details":
            class_name = arguments.get("class_name", "")
            if not class_name:
                return [types.TextContent(
                    type="text",
                    text="Error: 'class_name' parameter is required.",
                )]

            try:
                sdk_classes = await loop.run_in_executor(
                    None, _get_sdk_classes
                )
            except FileNotFoundError as e:
                return [types.TextContent(type="text", text=str(e))]
            except Exception as e:
                return [types.TextContent(
                    type="text",
                    text=f"Error analyzing SDK headers: {e}",
                )]

            cls = sdk_classes.get(class_name)
            if cls is None:
                return [types.TextContent(
                    type="text",
                    text=(
                        f"Class '{class_name}' not found in the SDK. "
                        f"Use browse_sdk_classes to see available classes."
                    ),
                )]

            # Build inheritance chain
            chain = _cpp_analyzer.build_inheritance_chain(
                class_name, sdk_classes
            )

            lines = [f"Class: {cls.name}"]
            lines.append(f"File: {cls.file}")
            lines.append(f"Category: {cls.category or 'other'}")
            lines.append(f"Is Wrapper: {cls.is_wrapper}")

            if cls.base_classes:
                lines.append(
                    f"Direct Base Classes: {', '.join(cls.base_classes)}"
                )
            if chain:
                lines.append(
                    f"Full Inheritance Chain: {cls.name} -> "
                    + " -> ".join(chain)
                )

            lines.append(f"\nMethods ({len(cls.methods)}):")
            for m in cls.methods:
                sig = f"  {m.return_type} {m.name}({m.parameters})"
                if m.is_const:
                    sig += " const"
                if m.is_virtual:
                    sig += "  [virtual]"
                if m.is_override:
                    sig += "  [override]"
                lines.append(sig)

            if cls.forward_declarations:
                lines.append(
                    f"\nRelated Types: "
                    + ", ".join(cls.forward_declarations)
                )

            return [types.TextContent(type="text", text="\n".join(lines))]

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
