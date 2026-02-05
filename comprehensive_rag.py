import os
import sys
import logging
import time
import math
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    KnowledgeGraphIndex,
    Settings,
    load_index_from_storage,
    Document
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.openai import OpenAI
from llama_index.llms.anthropic import Anthropic
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank

# Force UTF-8 for Windows Console
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    # Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Note: LlamaIndex handles caching internally via Settings
# No need for external GPTCache library (deprecated/unmaintained)

# --- Configuration ---
DOCS_DIR = "./docs"
STORAGE_DIR = "./rag_storage"
CHECKPOINT_INTERVAL = 500 # Save graph every 500 nodes

def build_comprehensive_stack(incremental=True):
    print("Initializing Ultra-Comprehensive RAG Stack (Parallel Batching)...")

    # 1. Global Settings
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", max_retries=10)
    # Using GPT-4o-mini with parallel requests for speed
    kg_llm = OpenAI(model="gpt-4o-mini", max_retries=10, temperature=0)
    Settings.llm = Anthropic(model="claude-3-5-sonnet-20240620", max_retries=5)

    # 2. Parsing
    print(f"Ingesting documents from {DOCS_DIR}...")
    reader = SimpleDirectoryReader(
        input_dir=DOCS_DIR,
        required_exts=[".md"],
        recursive=True,
        filename_as_id=True 
    )
    documents = reader.load_data()
    
    # Sanitize content
    cleaned_docs = []
    for doc in documents:
        # Filter non-printable chars
        clean_text = "".join(filter(lambda x: x.isprintable() or x in "\n\r\t", doc.text))
        # Create new document with sanitized text to avoid immutable setter issues
        new_doc = Document(text=clean_text, metadata=doc.metadata)
        cleaned_docs.append(new_doc)
    
    documents = cleaned_docs

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    total_nodes = len(nodes)
    print(f"Parsed into {total_nodes} hierarchical nodes.")

    # 3. Vector Index Build (Fast Phase)
    if not os.path.exists(STORAGE_DIR):
        os.makedirs(STORAGE_DIR)
        print("Building Vector Index (Phase 1/2)...")
        vector_index = VectorStoreIndex(nodes)
        vector_index.set_index_id("vector")
        vector_index.storage_context.persist(persist_dir=STORAGE_DIR)
        print("Vector Index saved. Search is now partially active.")
    else:
        print("Loading existing Vector Index...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        vector_index = load_index_from_storage(storage_context, index_id="vector")

    # 4. Knowledge Graph Build (Long Phase with Checkpoints)
    print("Building Knowledge Graph (Phase 2/2)...")
    
    # Check if a partial KG exists
    try:
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        kg_index = load_index_from_storage(storage_context, index_id="kg")
        print("Existing KG Index loaded. Checking for updates...")
        if incremental:
            kg_index.refresh_ref(documents)
            kg_index.storage_context.persist(persist_dir=STORAGE_DIR)
    except Exception:
        print("No existing KG found. Starting fresh build with batching...")
        
        # Manually iterate to allow saving intermediate progress
        # LlamaIndex doesn't natively checkpoint inside the constructor, so we build iteratively
        kg_index = KnowledgeGraphIndex(
            [], # Start empty
            storage_context=storage_context,
            llm=kg_llm,
            max_triplets_per_chunk=5, # High fidelity
            include_embeddings=True,
        )
        kg_index.set_index_id("kg")

        # Process nodes in batches
        for i in range(0, total_nodes, CHECKPOINT_INTERVAL):
            batch = nodes[i : i + CHECKPOINT_INTERVAL]
            print(f"Processing KG Batch {i}/{total_nodes}...")
            
            # Insert batch into index (triggers extraction)
            kg_index.insert_nodes(batch)
            
            # CHECKPOINT SAVE
            kg_index.storage_context.persist(persist_dir=STORAGE_DIR)
            print(f"Checkpoint saved at node {i + len(batch)}.")

    # 5. Fusion Retrieval Setup
    print("Configuring Fusion Retriever...")
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=10)
    
    fusion_retriever = QueryFusionRetriever(
        [
            vector_index.as_retriever(similarity_top_k=10),
            kg_index.as_retriever(similarity_top_k=10),
            bm25_retriever
        ],
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True
    )

    # For the reranker, we pass an explicit OpenAI LLM for compatibility
    rerank_llm = OpenAI(model="gpt-4o-mini", max_retries=5, temperature=0) 

    reranker = LLMRerank(choice_batch_size=5, top_n=5, llm=rerank_llm) # Use rerank_llm here

    query_engine = RetrieverQueryEngine.from_args(
        fusion_retriever,
        node_postprocessors=[reranker]
    )

    print("Platinum RAG Stack Ready.")
    return query_engine

if __name__ == "__main__":
    build_comprehensive_stack()
