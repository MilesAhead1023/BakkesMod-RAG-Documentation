"""
Comprehensive RAG Testing Suite
================================
Runs multiple test queries and provides interactive mode.
Shows verbose logging of all operations.
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
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] [{level:5s}] {message}")

def log_section(title):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def display_response(query, response, query_time):
    """Display query results in a formatted way."""
    log_section("QUERY RESULT")
    print(f"\n[QUESTION]")
    print(f"  {query}")
    print(f"\n[ANSWER]")
    print("-" * 80)
    print(response)
    print("-" * 80)

    print(f"\n[METADATA]")
    print(f"  Query time: {query_time:.2f}s")
    print(f"  Sources: {len(response.source_nodes)}")

    if response.source_nodes:
        print(f"\n[SOURCES]")
        for i, node in enumerate(response.source_nodes[:5], 1):
            filename = node.node.metadata.get("file_name", "unknown")
            score = node.score if hasattr(node, "score") else "N/A"
            print(f"  [{i}] {filename} (score: {score})")

def build_rag_system():
    """Build and return the RAG system."""
    log_section("BUILDING RAG SYSTEM")
    log("Importing modules...")

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

    log("Configuring settings...")
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        max_retries=3
    )
    Settings.llm = Anthropic(
        model="claude-sonnet-4-5",
        max_retries=3,
        temperature=0
    )

    log("Loading documents...")
    docs_dir = "docs_bakkesmod_only"
    reader = SimpleDirectoryReader(
        input_dir=docs_dir,
        required_exts=[".md"],
        recursive=True,
        filename_as_id=True
    )
    documents = reader.load_data()

    # Clean text
    cleaned_docs = []
    for doc in documents:
        clean_text = "".join(filter(lambda x: x.isprintable() or x in "\n\r\t", doc.text))
        cleaned_docs.append(Document(text=clean_text, metadata=doc.metadata))
    documents = cleaned_docs

    log(f"Loaded {len(documents)} documents")

    log("Parsing into nodes...")
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    log(f"Created {len(nodes)} nodes")

    log("Building/loading vector index...")
    storage_dir = "rag_storage_bakkesmod"
    from pathlib import Path
    storage_path = Path(storage_dir)

    if storage_path.exists():
        log("  Loading from cache...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        vector_index = load_index_from_storage(storage_context, index_id="vector")
        log("  Loaded from cache")
    else:
        log("  Building new index...")
        vector_index = VectorStoreIndex(nodes, show_progress=True)
        vector_index.set_index_id("vector")
        storage_path.mkdir(parents=True, exist_ok=True)
        vector_index.storage_context.persist(persist_dir=storage_dir)
        log("  Built and cached")

    log("Creating retrievers...")
    vector_retriever = vector_index.as_retriever(similarity_top_k=5)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True
    )

    log("Creating query engine...")
    query_engine = RetrieverQueryEngine.from_args(fusion_retriever)

    log("[OK] RAG system ready!")
    return query_engine, len(documents), len(nodes)

def run_test_queries(query_engine):
    """Run a suite of test queries."""
    log_section("RUNNING TEST QUERIES")

    test_queries = [
        "What is BakkesMod?",
        "How do I create a BakkesMod plugin?",
        "What are the main classes in the BakkesMod SDK?",
        "How do I hook into game events?",
        "How do I access player car data?",
    ]

    results = []
    for i, query in enumerate(test_queries, 1):
        log(f"\n[TEST {i}/{len(test_queries)}] {query}")
        start_time = time.time()

        try:
            response = query_engine.query(query)
            query_time = time.time() - start_time

            log(f"  Completed in {query_time:.2f}s")
            log(f"  Retrieved {len(response.source_nodes)} sources")

            results.append({
                "query": query,
                "response": response,
                "time": query_time,
                "success": True
            })
        except Exception as e:
            query_time = time.time() - start_time
            log(f"  Failed: {e}", "ERROR")
            results.append({
                "query": query,
                "error": str(e),
                "time": query_time,
                "success": False
            })

    return results

def display_test_summary(results):
    """Display summary of test results."""
    log_section("TEST SUMMARY")

    successful = sum(1 for r in results if r["success"])
    total = len(results)
    avg_time = sum(r["time"] for r in results) / total if total > 0 else 0

    print(f"\n[STATISTICS]")
    print(f"  Total queries: {total}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {total - successful}")
    print(f"  Success rate: {(successful/total*100):.1f}%")
    print(f"  Average query time: {avg_time:.2f}s")

    print(f"\n[DETAILED RESULTS]")
    for i, result in enumerate(results, 1):
        status = "[OK]" if result["success"] else "[FAIL]"
        print(f"\n  {i}. {status} {result['query']}")
        print(f"     Time: {result['time']:.2f}s")
        if result["success"]:
            response_preview = str(result["response"])[:100].replace("\n", " ")
            print(f"     Preview: {response_preview}...")
        else:
            print(f"     Error: {result['error']}")

def interactive_mode(query_engine):
    """Run interactive query loop."""
    log_section("INTERACTIVE MODE")
    print("\nEnter your questions about BakkesMod.")
    print("Commands:")
    print("  - Type your question and press Enter")
    print("  - Type 'quit', 'exit', or 'q' to exit")
    print("  - Type 'help' for more information")
    print("")

    query_count = 0
    total_time = 0.0

    while True:
        try:
            query = input("\n[QUERY] > ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting interactive mode...")
                if query_count > 0:
                    print(f"Total queries: {query_count}")
                    print(f"Average time: {total_time/query_count:.2f}s")
                break

            if query.lower() == 'help':
                print("\n[HELP]")
                print("This RAG system can answer questions about:")
                print("  - BakkesMod SDK documentation")
                print("  - Plugin development")
                print("  - API reference")
                print("  - ImGui integration")
                print("  - Event hooking")
                print("\nExample questions:")
                print("  - What is BakkesMod?")
                print("  - How do I create a plugin?")
                print("  - How do I hook the goal scored event?")
                print("  - What are the main classes in the SDK?")
                continue

            log(f"Processing query: {query}")
            start_time = time.time()

            response = query_engine.query(query)
            query_time = time.time() - start_time

            query_count += 1
            total_time += query_time

            display_response(query, response, query_time)

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            log(f"Query failed: {e}", "ERROR")
            import traceback
            traceback.print_exc()

def main():
    """Main test runner."""
    log_section("COMPREHENSIVE RAG TEST SUITE")
    log("Starting comprehensive testing...")

    # Check API keys
    if not os.getenv("OPENAI_API_KEY") or not os.getenv("ANTHROPIC_API_KEY"):
        log("Missing API keys! Check your .env file.", "ERROR")
        sys.exit(1)

    log("API keys found")

    # Build system
    try:
        query_engine, num_docs, num_nodes = build_rag_system()
        log(f"System ready: {num_docs} docs, {num_nodes} nodes")
    except Exception as e:
        log(f"Failed to build system: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run test queries
    try:
        results = run_test_queries(query_engine)
        display_test_summary(results)

        # Display some full results
        log_section("SAMPLE DETAILED RESULTS")
        for i, result in enumerate(results[:2], 1):
            if result["success"]:
                display_response(result["query"], result["response"], result["time"])

    except Exception as e:
        log(f"Test queries failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()

    # Ask if user wants interactive mode
    log_section("INTERACTIVE MODE OPTION")
    print("\nWould you like to enter interactive mode? (y/n)")
    choice = input("> ").strip().lower()

    if choice in ['y', 'yes']:
        interactive_mode(query_engine)
    else:
        log("Skipping interactive mode")

    log_section("TEST COMPLETE")
    log("All tests finished successfully!")

if __name__ == "__main__":
    main()
