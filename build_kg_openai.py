"""
Build KG index using OpenAI GPT-4o-mini for speed.
Merges the result into existing rag_storage/ alongside the vector index.
"""
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

def log(msg, level="INFO"):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] [{level:5s}] {msg}")

log("Building KG index with OpenAI GPT-4o-mini (fast)...")

from llama_index.core import (
    SimpleDirectoryReader, StorageContext, Settings,
    load_index_from_storage, Document
)
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Use GPT-4o-mini for KG extraction - fast and cheap
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
log("Using OpenAI GPT-4o-mini for KG extraction")

# Load documents from both directories
input_dirs = ["docs_bakkesmod_only", "templates"]
all_documents = []
for input_dir in input_dirs:
    if not os.path.isdir(input_dir):
        log(f"Directory '{input_dir}' not found, skipping", "WARNING")
        continue
    reader = SimpleDirectoryReader(
        input_dir=input_dir,
        required_exts=[".md", ".h", ".cpp"],
        recursive=True,
        filename_as_id=True
    )
    docs = reader.load_data()
    log(f"Loaded {len(docs)} files from {input_dir}")
    all_documents.extend(docs)

# Clean
documents = []
for doc in all_documents:
    clean_text = "".join(filter(lambda x: x.isprintable() or x in "\n\r\t", doc.text))
    documents.append(Document(text=clean_text, metadata=doc.metadata))
log(f"Total documents: {len(documents)}")

# Load existing storage context so KG gets saved alongside vector index
storage_dir = "rag_storage"
storage_context = StorageContext.from_defaults(persist_dir=storage_dir)

# Build KG index
log("Building Knowledge Graph index (this may take 10-20 minutes with OpenAI)...")
kg_index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
    show_progress=True
)
kg_index.set_index_id("knowledge_graph")
storage_context.persist(persist_dir=storage_dir)
log("KG index built and saved to rag_storage/")
log("Done! The full 3-way fusion system is now ready.")
