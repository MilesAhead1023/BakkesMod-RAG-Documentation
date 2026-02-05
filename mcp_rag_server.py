import os
import sys
import asyncio
import logging
from mcp.server import Server
from mcp.server.stdio import stdio_server
import mcp.types as types
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import our platinum logic
from comprehensive_rag import build_comprehensive_stack, DOCS_DIR

# --- Globals ---
engine = None
server = Server("bakkesmod-platinum-rag")

class DocsWatcher(FileSystemEventHandler):
    """Watches the docs directory and triggers incremental re-indexing."""
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(".md"):
            print(f"File changed: {event.src_path}. Triggering incremental re-index...", file=sys.stderr)
            asyncio.run_coroutine_threadsafe(reindex(), loop)

async def reindex():
    """Performs the incremental re-indexing."""
    global engine
    print("Re-indexing changed documents...", file=sys.stderr)
    engine = build_comprehensive_stack(incremental=True)
    print("Incremental re-index complete.", file=sys.stderr)

async def main():
    global engine, loop
    loop = asyncio.get_running_loop()
    
    print("Initializing Platinum RAG Engine...", file=sys.stderr)
    try:
        # Build initial stack (loads existing or builds fresh)
        engine = build_comprehensive_stack(incremental=True)
    except Exception as e:
        print(f"Failed to initialize RAG: {e}", file=sys.stderr)
        sys.exit(1)

    # Setup File Watcher for Continuous Ingestion
    observer = Observer()
    observer.schedule(DocsWatcher(), path=DOCS_DIR, recursive=True)
    observer.start()
    print(f"Live Observer active on {DOCS_DIR}", file=sys.stderr)

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="query_bakkesmod_sdk",
                description="Query the BakkesMod SDK using Platinum 2026 RAG (Hybrid + Knowledge Graph + Live Ingestion).",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The technical question."},
                    },
                    "required": ["query"],
                },
            )
        ]

    @server.call_tool()
    async def handle_call_tool(
        name: str, arguments: dict | None
    ) -> list[types.TextContent]:
        if name == "query_bakkesmod_sdk":
            query = arguments.get("query")
            print(f"Querying: {query}", file=sys.stderr)
            
            # Synchronous query wrapped in executor
            response = await loop.run_in_executor(None, engine.query, query)
            
            return [types.TextContent(type="text", text=str(response))]
        
        raise ValueError(f"Tool not found: {name}")

    async with stdio_server() as (read_stream, write_server):
        await server.run(read_stream, write_server, server.create_initialization_options())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass