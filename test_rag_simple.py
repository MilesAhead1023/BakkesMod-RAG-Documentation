"""
Simple RAG Test (Without Knowledge Graph)
=========================================
Tests the RAG system with just vector + BM25 retrieval (no KG).
Much faster for initial testing.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("    Simple RAG Test (Vector + BM25 Only)")
print("=" * 60)

# Test 1: Check environment
print("\n[STEP 1] Checking API Keys...")
api_keys = {
    "OpenAI": os.getenv("OPENAI_API_KEY"),
    "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
}

configured = []
for provider, key in api_keys.items():
    if key and not key.startswith("your_"):
        configured.append(provider)
        print(f"  [OK] {provider}")

if not configured:
    print("\n[ERROR] No API keys found!")
    sys.exit(1)

print(f"\n  Found {len(configured)} provider(s)")

# Test 2: Import and configure
print("\n[STEP 2] Importing modules...")
try:
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
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine

    print("  [OK] All imports successful")
except Exception as e:
    print(f"  [ERROR] Import failed: {e}")
    sys.exit(1)

# Test 3: Configure settings
print("\n[STEP 3] Configuring LlamaIndex...")
try:
    # Use OpenAI embeddings
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",  # Smaller/faster model
        max_retries=3
    )

    # Use Anthropic for LLM
    Settings.llm = Anthropic(
        model="claude-3-5-sonnet-20240620",
        max_retries=3,
        temperature=0
    )

    print("  [OK] Settings configured")
    print(f"      Embedding: text-embedding-3-small")
    print(f"      LLM: claude-3-5-sonnet-20240620")
except Exception as e:
    print(f"  [ERROR] Configuration failed: {e}")
    sys.exit(1)

# Test 4: Load documents
print("\n[STEP 4] Loading documents...")
try:
    docs_dir = "docs"
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
        new_doc = Document(text=clean_text, metadata=doc.metadata)
        cleaned_docs.append(new_doc)

    documents = cleaned_docs

    print(f"  [OK] Loaded {len(documents)} documents")
except Exception as e:
    print(f"  [ERROR] Failed to load documents: {e}")
    sys.exit(1)

# Test 5: Parse nodes
print("\n[STEP 5] Parsing into nodes...")
try:
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(f"  [OK] Parsed {len(nodes)} nodes")
except Exception as e:
    print(f"  [ERROR] Parsing failed: {e}")
    sys.exit(1)

# Test 6: Build vector index
print("\n[STEP 6] Building vector index...")
storage_dir = "rag_storage_simple"
try:
    from pathlib import Path
    storage_path = Path(storage_dir)

    if storage_path.exists():
        print("  [INFO] Loading from cache...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        vector_index = load_index_from_storage(storage_context, index_id="vector")
        print("  [OK] Loaded cached index")
    else:
        print("  [INFO] Building new index (this may take 2-5 minutes)...")
        vector_index = VectorStoreIndex(nodes)
        vector_index.set_index_id("vector")
        storage_path.mkdir(parents=True, exist_ok=True)
        vector_index.storage_context.persist(persist_dir=storage_dir)
        print("  [OK] Vector index built and cached")
except Exception as e:
    print(f"  [ERROR] Index building failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Create retrievers and query engine
print("\n[STEP 7] Creating query engine...")
try:
    # Vector retriever
    vector_retriever = vector_index.as_retriever(similarity_top_k=5)

    # BM25 retriever
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=5
    )

    # Fusion retriever (combines both)
    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True
    )

    # Query engine
    query_engine = RetrieverQueryEngine.from_args(fusion_retriever)

    print("  [OK] Query engine created")
    print("      Retrievers: Vector + BM25")
except Exception as e:
    print(f"  [ERROR] Query engine creation failed: {e}")
    sys.exit(1)

# Test 8: Run test query
print("\n[STEP 8] Running test query...")
test_query = "What is BakkesMod?"
print(f"  Query: {test_query}")

try:
    response = query_engine.query(test_query)

    print("\n" + "=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(response)
    print("\n" + "=" * 60)
    print(f"Sources: {len(response.source_nodes)} documents used")

    if response.source_nodes:
        print("\nTop sources:")
        for i, node in enumerate(response.source_nodes[:3], 1):
            filename = node.node.metadata.get("file_name", "unknown")
            print(f"  {i}. {filename}")

    print("\n" + "=" * 60)
    print("SUCCESS! RAG System Working")
    print("=" * 60)
    print("\nYour RAG system is operational!")
    print("\nNext steps:")
    print("  - Run this script again (will use cache, ~10s)")
    print("  - Try: python rag_2026.py (includes KG for better results)")
    print("  - Interactive: python rag_2026.py")

except Exception as e:
    print(f"\n[ERROR] Query failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
