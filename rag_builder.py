import os
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# --- Configuration ---
DOCS_DIR = "./docs"
STORAGE_DIR = "./storage"
REFERENCE_FILE = "bakkesmod-sdk-reference.md"

def build_rag_index():
    print(f"Initializing RAG build for {DOCS_DIR}...")

    # 1. Setup Local Embeddings (No OpenAI API key needed)
    # Using a high-performance local model specialized for code/docs
    Settings.embedding_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5" 
    )
    # We'll use a mock LLM for indexing since we are just building retrieval structures
    # (The actual querying will happen via the MCP server later)
    Settings.llm = None 

    # 2. Parse Documents
    # We specifically target the reference file for hierarchical parsing
    reader = SimpleDirectoryReader(
        input_dir=DOCS_DIR, 
        required_exts=[".md"], 
        recursive=True
    )
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document chunks.")

    # 3. Hierarchical Splitting
    # This splits by H1, H2, H3 headers to preserve context (Class -> Method)
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Parsed into {len(nodes)} hierarchical nodes.")

    # 4. Build Indices
    if not os.path.exists(STORAGE_DIR):
        print("Creating new vector index...")
        # Create Vector Store (Semantic Search)
        storage_context = StorageContext.from_defaults()
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
        
        # Persist to disk
        vector_index.set_index_id("vector_index")
        vector_index.storage_context.persist(persist_dir=STORAGE_DIR)
        print("Vector index built and saved.")
    else:
        print("Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        vector_index = load_index_from_storage(storage_context, index_id="vector_index")

    # 5. Build BM25 Keyword Retriever (On-the-fly, fast to build)
    # We allow this to be rebuilt each run as it's lightweight
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=10,
        stemmer=None, # Keep technical terms exact
        language="english"
    )
    
    print("RAG System Ready.")
    return vector_index, bm25_retriever

if __name__ == "__main__":
    build_rag_index()
