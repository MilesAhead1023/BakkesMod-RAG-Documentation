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
from cache_manager import SemanticCache

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

    # Initialize semantic cache
    log("Initializing semantic cache...")
    cache = SemanticCache(
        cache_dir=".cache/semantic",
        similarity_threshold=0.92,  # 92% similar = cache hit
        ttl_seconds=86400 * 7,  # 7 days
        embed_model=Settings.embed_model
    )
    cache_stats = cache.stats()
    log(f"  Cache: {cache_stats['valid_entries']} valid entries, threshold={cache.similarity_threshold:.0%}")

    log(f"System ready! ({len(documents)} docs, {len(nodes)} nodes, 3-way fusion: Vector+BM25+KG)")
    return query_engine, cache, len(documents), len(nodes)

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

def calculate_confidence(source_nodes: list) -> tuple:
    """Calculate confidence score from retrieval quality.

    Args:
        source_nodes: List of retrieved source nodes with scores

    Returns:
        Tuple of (confidence_score, confidence_label, explanation)
    """
    if not source_nodes:
        return 0.0, "NO DATA", "No sources retrieved"

    # Get scores
    scores = [node.score for node in source_nodes if hasattr(node, 'score') and node.score is not None]

    if not scores:
        return 0.5, "MEDIUM", "Sources retrieved but no similarity scores available"

    # Calculate metrics
    avg_score = sum(scores) / len(scores)
    max_score = max(scores)
    num_sources = len(source_nodes)

    # Score variance (lower is better - more consistent retrieval)
    if len(scores) > 1:
        mean = sum(scores) / len(scores)
        variance = sum((s - mean) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
    else:
        std_dev = 0.0

    # Calculate confidence (0-100%)
    # Factors:
    # - Average score (0-1): 50% weight
    # - Max score (0-1): 20% weight
    # - Number of sources (bonus): 10% weight
    # - Consistency (low variance): 20% weight

    score_component = avg_score * 50
    max_component = max_score * 20
    source_bonus = min(num_sources / 5.0, 1.0) * 10  # Max at 5 sources
    consistency_component = max(0, (1 - std_dev) * 20)  # Lower variance = higher confidence

    confidence = (score_component + max_component + source_bonus + consistency_component) / 100.0

    # Clamp to 0-1
    confidence = max(0.0, min(1.0, confidence))

    # Assign label
    if confidence >= 0.85:
        label = "VERY HIGH"
        explanation = "Excellent source match with high consistency"
    elif confidence >= 0.70:
        label = "HIGH"
        explanation = "Strong source match with good relevance"
    elif confidence >= 0.50:
        label = "MEDIUM"
        explanation = "Moderate source match, answer should be helpful"
    elif confidence >= 0.30:
        label = "LOW"
        explanation = "Weak source match, verify answer carefully"
    else:
        label = "VERY LOW"
        explanation = "Poor source match, answer may be unreliable"

    return confidence, label, explanation

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
        query_engine, cache, num_docs, num_nodes = build_rag_system()
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

            # Check cache first
            cache_result = cache.get(query)

            if cache_result:
                response_text, similarity, metadata = cache_result
                query_time = time.time() - start_time

                query_count += 1
                total_time += query_time
                successful += 1

                print("\n" + "-" * 80)
                print("[ANSWER] (from cache)")
                print("-" * 80)
                print(response_text)
                print("-" * 80)

                print(f"\n[METADATA]")
                print(f"  Cache hit! (similarity: {similarity:.1%})")
                print(f"  Cached query: '{metadata['cached_query'][:60]}...'")
                print(f"  Cache age: {metadata['cache_age_seconds']/3600:.1f} hours")
                print(f"  Query time: {query_time:.2f}s")
                print(f"  Cost savings: ~$0.02-0.04")

                continue  # Skip actual query

            # If not in cache, do normal query
            try:
                response = query_engine.query(query)
                query_time = time.time() - start_time

                query_count += 1
                total_time += query_time
                successful += 1

                # Stream the response
                full_text = stream_response(response)

                # Cache the response
                cache.set(query, full_text, response.source_nodes)

                # Calculate confidence
                confidence, conf_label, conf_explanation = calculate_confidence(response.source_nodes)

                print(f"\n[METADATA]")
                print(f"  Query time: {query_time:.2f}s")
                print(f"  Sources: {len(response.source_nodes)}")
                print(f"  Confidence: {confidence:.0%} ({conf_label}) - {conf_explanation}")
                print(f"  Cached for future queries")

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
