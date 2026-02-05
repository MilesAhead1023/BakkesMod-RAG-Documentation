import os
import logging
import sys
from typing import List

from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    KnowledgeGraphIndex,
    Settings,
    load_index_from_storage,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import LLMRerank
from llama_index.llms.anthropic import Anthropic

# Setup logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# --- Configuration ---
DOCS_DIR = "./docs"
STORAGE_DIR = "./gemini_rag_storage"

def build_gemini_stack():
    print("Initializing Gemini-Powered Ultra RAG Stack...")

    # 1. Global Settings
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    
    # Use Gemini 2.0 Flash for KG Extraction
    # It has a massive window and is very fast for relationship extraction
    # Note: LlamaIndex handles "models/" prefix internally
    kg_llm = GoogleGenAI(model="gemini-2.0-flash")

    # Use Claude 3.5 Sonnet for reranking (verified 2026 snapshot)
    Settings.llm = Anthropic(model="claude-3-5-sonnet-20240620")

    # 2. Load and Parse
    print(f"Loading documents from {DOCS_DIR}...")
    reader = SimpleDirectoryReader(input_dir=DOCS_DIR, required_exts=[".md"], recursive=True)
    documents = reader.load_data()

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    print(f"Parsed into {len(nodes)} hierarchical nodes.")

    # 3. Build Indices
    if not os.path.exists(STORAGE_DIR):
        print("Building Gemini KG Index (this uses your fixed billing tier)...")
        storage_context = StorageContext.from_defaults()
        
        # Vector Index
        vector_index = VectorStoreIndex(nodes, storage_context=storage_context)
        vector_index.set_index_id("vector")
        
        # Knowledge Graph Index with Gemini
        kg_index = KnowledgeGraphIndex(
            nodes,
            storage_context=storage_context,
            llm=kg_llm,
            max_triplets_per_chunk=10, 
            include_embeddings=True,
        )
        kg_index.set_index_id("kg")
        
        storage_context.persist(persist_dir=STORAGE_DIR)
        print("Gemini indices persisted to disk.")
    else:
        print("Loading existing Gemini index...")
        storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
        vector_index = load_index_from_storage(storage_context, index_id="vector")
        kg_index = load_index_from_storage(storage_context, index_id="kg")

    return True

if __name__ == "__main__":
    build_gemini_stack()
    print("Gemini RAG Build Complete.")
