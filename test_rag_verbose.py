"""
Verbose RAG System Test with Debug Logging
==========================================
Provides detailed, timestamped logging of every step.
Shows exactly what's happening at each moment.
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

log_section("VERBOSE RAG SYSTEM TEST - START")
log("Test started")

# STEP 1: Environment Check
log_section("STEP 1: Environment Configuration Check")
log("Checking for API keys...")

api_keys = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY"),
    "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
}

configured_providers = []
for key_name, value in api_keys.items():
    if value and not value.startswith("your_"):
        provider = key_name.replace("_API_KEY", "")
        configured_providers.append(provider)
        # Show first 20 chars for verification
        masked_key = value[:20] + "..." if len(value) > 20 else value
        log(f"  {provider}: Found (starts with '{masked_key}')")
    else:
        log(f"  {key_name}: NOT CONFIGURED", "WARN")

if not configured_providers:
    log("No API keys configured! Cannot proceed.", "ERROR")
    sys.exit(1)

log(f"API check complete. Found {len(configured_providers)} provider(s): {', '.join(configured_providers)}")

# STEP 2: Import modules
log_section("STEP 2: Importing Required Modules")
start_time = time.time()

try:
    log("Importing llama_index core components...")
    from llama_index.core import (
        SimpleDirectoryReader,
        StorageContext,
        VectorStoreIndex,
        Settings,
        load_index_from_storage,
        Document
    )
    log("  - Core components imported")

    log("Importing node parser...")
    from llama_index.core.node_parser import MarkdownNodeParser
    log("  - Node parser imported")

    log("Importing LLM providers...")
    from llama_index.llms.anthropic import Anthropic
    from llama_index.llms.openai import OpenAI
    log("  - LLM providers imported")

    log("Importing embedding models...")
    from llama_index.embeddings.openai import OpenAIEmbedding
    log("  - Embedding models imported")

    log("Importing retrievers...")
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.retrievers import QueryFusionRetriever
    log("  - Retrievers imported")

    log("Importing query engine...")
    from llama_index.core.query_engine import RetrieverQueryEngine
    log("  - Query engine imported")

    elapsed = time.time() - start_time
    log(f"All imports successful ({elapsed:.2f}s)")

except ImportError as e:
    log(f"Import failed: {e}", "ERROR")
    log("Try: pip install -r requirements.txt", "ERROR")
    sys.exit(1)

# STEP 3: Configure LlamaIndex Settings
log_section("STEP 3: Configuring LlamaIndex Settings")
start_time = time.time()

try:
    log("Configuring embedding model...")
    log("  Model: text-embedding-3-small (faster, lower cost)")
    log("  Provider: OpenAI")
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        max_retries=3
    )
    log("  Embedding model configured")

    log("Configuring LLM...")
    log("  Model: claude-sonnet-4-5 (Sonnet 4.5)")
    log("  Provider: Anthropic")
    log("  Temperature: 0 (deterministic responses)")
    Settings.llm = Anthropic(
        model="claude-sonnet-4-5",
        max_retries=3,
        temperature=0
    )
    log("  LLM configured")

    elapsed = time.time() - start_time
    log(f"Configuration complete ({elapsed:.2f}s)")

except Exception as e:
    log(f"Configuration failed: {e}", "ERROR")
    sys.exit(1)

# STEP 4: Load Documents
log_section("STEP 4: Loading Documentation Files")
start_time = time.time()

docs_dir = "docs_bakkesmod_only"
log(f"Document directory: {docs_dir} (BakkesMod SDK only)")
log("Looking for .md files (recursive)...")

try:
    reader = SimpleDirectoryReader(
        input_dir=docs_dir,
        required_exts=[".md"],
        recursive=True,
        filename_as_id=True
    )
    log("SimpleDirectoryReader initialized")

    log("Loading documents...")
    documents = reader.load_data()
    log(f"  Loaded {len(documents)} document(s)")

    # Show document details
    total_chars = sum(len(doc.text) for doc in documents)
    log(f"  Total characters: {total_chars:,}")
    log(f"  Average size: {total_chars // len(documents) if documents else 0:,} chars/doc")

    # Clean text (remove non-printable characters)
    log("Sanitizing document text...")
    cleaned_docs = []
    for i, doc in enumerate(documents):
        clean_text = "".join(filter(lambda x: x.isprintable() or x in "\n\r\t", doc.text))
        new_doc = Document(text=clean_text, metadata=doc.metadata)
        cleaned_docs.append(new_doc)
        if (i + 1) % 5 == 0 or (i + 1) == len(documents):
            log(f"  Cleaned {i + 1}/{len(documents)} documents")

    documents = cleaned_docs
    elapsed = time.time() - start_time
    log(f"Document loading complete ({elapsed:.2f}s)")

except FileNotFoundError:
    log(f"Directory not found: {docs_dir}", "ERROR")
    log("Make sure the 'docs' directory exists with .md files", "ERROR")
    sys.exit(1)
except Exception as e:
    log(f"Document loading failed: {e}", "ERROR")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# STEP 5: Parse into Nodes
log_section("STEP 5: Parsing Documents into Nodes")
start_time = time.time()

try:
    log("Initializing MarkdownNodeParser...")
    parser = MarkdownNodeParser()

    log("Parsing documents into nodes (chunks)...")
    nodes = parser.get_nodes_from_documents(documents)
    log(f"  Created {len(nodes)} node(s)")

    # Node statistics
    node_sizes = [len(node.get_content()) for node in nodes]
    avg_size = sum(node_sizes) // len(node_sizes) if node_sizes else 0
    max_size = max(node_sizes) if node_sizes else 0
    min_size = min(node_sizes) if node_sizes else 0

    log(f"  Node statistics:")
    log(f"    Average size: {avg_size:,} chars")
    log(f"    Max size: {max_size:,} chars")
    log(f"    Min size: {min_size:,} chars")

    elapsed = time.time() - start_time
    log(f"Node parsing complete ({elapsed:.2f}s)")

except Exception as e:
    log(f"Node parsing failed: {e}", "ERROR")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# STEP 6: Build Vector Index
log_section("STEP 6: Building Vector Index")
start_time = time.time()

storage_dir = "rag_storage_bakkesmod"
log(f"Storage directory: {storage_dir}")

try:
    from pathlib import Path
    storage_path = Path(storage_dir)

    if storage_path.exists():
        log("Existing index found. Loading from cache...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        vector_index = load_index_from_storage(storage_context, index_id="vector")
        elapsed = time.time() - start_time
        log(f"Vector index loaded from cache ({elapsed:.2f}s)")
    else:
        log("No cached index found. Building new vector index...")
        log("  This will create embeddings for all nodes")
        log(f"  Estimated API calls: ~{len(nodes) // 100 + 1} batch(es)")
        log("  This may take 2-5 minutes...")

        # Show progress for embedding creation
        log("  Creating embeddings...")
        vector_index = VectorStoreIndex(nodes, show_progress=True)

        log("  Persisting index to disk...")
        vector_index.set_index_id("vector")
        storage_path.mkdir(parents=True, exist_ok=True)
        vector_index.storage_context.persist(persist_dir=storage_dir)

        elapsed = time.time() - start_time
        log(f"Vector index built and persisted ({elapsed:.2f}s)")
        log(f"  Cache saved to: {storage_path.absolute()}")

except Exception as e:
    log(f"Vector index building failed: {e}", "ERROR")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# STEP 7: Create Retrievers
log_section("STEP 7: Creating Retrievers")
start_time = time.time()

try:
    log("Creating vector retriever...")
    log("  Similarity top-k: 5")
    vector_retriever = vector_index.as_retriever(similarity_top_k=5)
    log("  Vector retriever created")

    log("Creating BM25 retriever (keyword-based)...")
    log("  Similarity top-k: 5")
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=5
    )
    log("  BM25 retriever created")

    log("Creating fusion retriever (combines both)...")
    log("  Mode: reciprocal_rerank")
    log("  Async: True")
    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True
    )
    log("  Fusion retriever created")

    elapsed = time.time() - start_time
    log(f"Retriever creation complete ({elapsed:.2f}s)")

except Exception as e:
    log(f"Retriever creation failed: {e}", "ERROR")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# STEP 8: Create Query Engine
log_section("STEP 8: Creating Query Engine")
start_time = time.time()

try:
    log("Building query engine with fusion retriever...")
    query_engine = RetrieverQueryEngine.from_args(fusion_retriever)

    elapsed = time.time() - start_time
    log(f"Query engine created ({elapsed:.2f}s)")

except Exception as e:
    log(f"Query engine creation failed: {e}", "ERROR")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# STEP 9: Test Query
log_section("STEP 9: Executing Test Query")

test_query = "What is BakkesMod?"
log(f"Test query: '{test_query}'")

start_time = time.time()

try:
    log("Retrieving relevant documents...")
    log("  Step 1: Vector similarity search")
    log("  Step 2: BM25 keyword search")
    log("  Step 3: Reciprocal rank fusion")
    log("  Step 4: Generating response with LLM")

    response = query_engine.query(test_query)

    elapsed = time.time() - start_time
    log(f"Query completed in {elapsed:.2f}s")

    # Display response
    log_section("QUERY RESPONSE")
    print("\nQuestion:")
    print(f"  {test_query}")
    print("\nAnswer:")
    print("-" * 80)
    print(response)
    print("-" * 80)

    # Show sources
    log_section("SOURCE DOCUMENTS")
    log(f"Retrieved {len(response.source_nodes)} source(s)")

    for i, node in enumerate(response.source_nodes, 1):
        filename = node.node.metadata.get("file_name", "unknown")
        score = node.score if hasattr(node, "score") else "N/A"
        content_preview = node.node.get_content()[:150].replace("\n", " ")
        log(f"  [{i}] {filename}")
        log(f"      Score: {score}")
        log(f"      Preview: {content_preview}...")

except Exception as e:
    log(f"Query failed: {e}", "ERROR")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success Summary
log_section("TEST COMPLETE - SUCCESS")
log("All components working correctly!")
log("")
log("System Summary:")
log(f"  - Documents: {len(documents)}")
log(f"  - Nodes: {len(nodes)}")
log(f"  - Retrievers: Vector + BM25 (Fusion)")
log(f"  - LLM: Anthropic Claude 3.5 Sonnet")
log(f"  - Storage: {storage_dir}")
log("")
log("Next steps:")
log("  1. Run again to test cache loading (should be faster)")
log("  2. Try different queries")
log("  3. Run: python rag_2026.py (for full Gold Standard with KG)")
log("")
log_section("END OF TEST")
