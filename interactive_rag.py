"""
Interactive RAG Query Interface
================================
Ask questions about BakkesMod SDK in an interactive loop.
Provides detailed logging and formatted responses.
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def log(message, level="INFO"):
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level:5s}] {message}")

def log_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def build_rag_system():
    """Build and return the RAG system."""
    log("Building RAG system...")

    from llama_index.core import (
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
        Settings,
        load_index_from_storage,
        Document
    )
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.llms.anthropic import Anthropic
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex

    # Configure
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", max_retries=3)
    Settings.llm = Anthropic(model="claude-sonnet-4-5", max_retries=3, temperature=0)

    # Load documents - BakkesMod SDK only
    reader = SimpleDirectoryReader(
        input_dir="docs_bakkesmod_only",
        required_exts=[".md"],
        recursive=True,
        filename_as_id=True
    )
    documents = reader.load_data()

    # Clean
    cleaned_docs = []
    for doc in documents:
        clean_text = "".join(filter(lambda x: x.isprintable() or x in "\n\r\t", doc.text))
        cleaned_docs.append(Document(text=clean_text, metadata=doc.metadata))
    documents = cleaned_docs

    # Parse
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    # Build/load index
    storage_dir = "rag_storage_bakkesmod"  # New storage for BakkesMod-only docs
    from pathlib import Path
    storage_path = Path(storage_dir)

    if storage_path.exists():
        log("Loading indexes from cache...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        vector_index = load_index_from_storage(storage_context, index_id="vector")

        # Load KG index if it exists
        try:
            kg_index = load_index_from_storage(storage_context, index_id="knowledge_graph")
            log("Loaded cached KG index")
        except:
            log("Building new KG index (this will take several minutes)...")
            kg_index = KnowledgeGraphIndex.from_documents(
                documents,
                max_triplets_per_chunk=2,
                show_progress=True
            )
            kg_index.set_index_id("knowledge_graph")
            kg_index.storage_context.persist(persist_dir=storage_dir)
            log("KG index built and cached")
    else:
        log("Building new indexes (this will take several minutes)...")
        vector_index = VectorStoreIndex(nodes, show_progress=True)
        vector_index.set_index_id("vector")
        storage_path.mkdir(parents=True, exist_ok=True)
        vector_index.storage_context.persist(persist_dir=storage_dir)
        log("Vector index built and cached")

        # Build KG index
        log("Building Knowledge Graph index...")
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            max_triplets_per_chunk=2,
            show_progress=True
        )
        kg_index.set_index_id("knowledge_graph")
        kg_index.storage_context.persist(persist_dir=storage_dir)
        log("KG index built and cached")

    # Create retrievers
    vector_retriever = vector_index.as_retriever(similarity_top_k=5)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    kg_retriever = kg_index.as_retriever(similarity_top_k=3)

    # Create query engine with 3-way fusion (Vector + BM25 + KG)
    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever, kg_retriever],
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True
    )
    query_engine = RetrieverQueryEngine.from_args(fusion_retriever, streaming=True)

    log(f"System ready! ({len(documents)} docs, {len(nodes)} nodes, 3-way fusion: Vector+BM25+KG)")
    return query_engine, len(documents), len(nodes)

def stream_response(response):
    """Display streaming response token by token.

    Args:
        response: StreamingResponse from query engine

    Returns:
        str: Complete response text
    """
    print("\n" + "-" * 80)
    print("[ANSWER]")
    print("-" * 80)

    tokens = []
    for token in response.response_gen:
        print(token, end="", flush=True)
        tokens.append(token)

    print()  # Newline after streaming
    print("-" * 80)

    return "".join(tokens)

def display_help():
    """Display help information."""
    print("\n" + "=" * 80)
    print("  HELP - BakkesMod RAG System")
    print("=" * 80)
    print("\nThis system can answer questions about:")
    print("  - BakkesMod SDK documentation and API reference")
    print("  - Plugin development and architecture")
    print("  - ImGui UI integration")
    print("  - Event hooking and game events")
    print("  - Car physics and player data access")
    print("\nExample questions:")
    print("  - What is BakkesMod?")
    print("  - How do I create a plugin?")
    print("  - How do I hook the goal scored event?")
    print("  - What are the main classes in the SDK?")
    print("  - How do I access player car velocity?")
    print("  - How do I use ImGui to create a settings window?")
    print("\nCommands:")
    print("  help   - Show this help message")
    print("  stats  - Show session statistics")
    print("  quit   - Exit the program (or Ctrl+C)")
    print("=" * 80)

def main():
    """Main interactive loop."""
    log_section("BAKKESMOD RAG - INTERACTIVE MODE")

    # Check API keys
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        log("Missing API keys! Check your .env file.", "ERROR")
        sys.exit(1)

    # Build system
    try:
        query_engine, num_docs, num_nodes = build_rag_system()
    except Exception as e:
        log(f"Failed to build system: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Show welcome
    print("\nWelcome to the BakkesMod RAG System!")
    print(f"Loaded {num_docs} documents with {num_nodes} searchable chunks.")
    print("\nType your question and press Enter.")
    print("Type 'help' for examples, 'stats' for statistics, or 'quit' to exit.\n")

    query_count = 0
    total_time = 0.0
    successful = 0

    # Main loop
    while True:
        try:
            # Get input
            query = input("[QUERY] > ").strip()

            if not query:
                continue

            # Handle commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting...")
                if query_count > 0:
                    print(f"\nSession summary:")
                    print(f"  Total queries: {query_count}")
                    print(f"  Successful: {successful}")
                    print(f"  Average time: {total_time/query_count:.2f}s")
                print("\nThank you for using BakkesMod RAG!\n")
                break

            if query.lower() == 'help':
                display_help()
                continue

            if query.lower() == 'stats':
                print(f"\n[SESSION STATISTICS]")
                print(f"  Total queries: {query_count}")
                print(f"  Successful: {successful}")
                if query_count > 0:
                    print(f"  Success rate: {(successful/query_count*100):.1f}%")
                    print(f"  Average time: {total_time/query_count:.2f}s")
                else:
                    print(f"  No queries yet!")
                continue

            # Process query
            log(f"Processing: {query[:60]}{'...' if len(query) > 60 else ''}")
            start_time = time.time()

            try:
                response = query_engine.query(query)
                query_time = time.time() - start_time

                query_count += 1
                total_time += query_time
                successful += 1

                # Stream the response
                full_text = stream_response(response)

                print(f"\n[METADATA]")
                print(f"  Query time: {query_time:.2f}s")
                print(f"  Sources: {len(response.source_nodes)}")

                if response.source_nodes:
                    print(f"\n[SOURCE FILES]")
                    seen_files = set()
                    for node in response.source_nodes:
                        filename = node.node.metadata.get("file_name", "unknown")
                        if filename not in seen_files:
                            seen_files.add(filename)
                            print(f"  - {filename}")

            except Exception as e:
                query_time = time.time() - start_time
                query_count += 1
                total_time += query_time

                log(f"Query failed: {e}", "ERROR")
                print(f"\n[ERROR] {e}")
                print("Please try rephrasing your question or check the logs.")

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            print(f"\nSession summary:")
            print(f"  Total queries: {query_count}")
            print(f"  Successful: {successful}")
            if query_count > 0:
                print(f"  Average time: {total_time/query_count:.2f}s")
            print("\nThank you for using BakkesMod RAG!\n")
            break
        except Exception as e:
            log(f"Unexpected error: {e}", "ERROR")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
