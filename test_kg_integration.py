"""
Test Knowledge Graph Integration
=================================
Tests KG index building and retrieval.
"""

import os
from dotenv import load_dotenv
load_dotenv()

def test_kg_index_builds():
    """Test that KG index can be built from documents."""
    from llama_index.core import SimpleDirectoryReader, Settings, Document
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.llms.anthropic import Anthropic
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex

    Settings.llm = Anthropic(model="claude-sonnet-4-5", max_retries=3, temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", max_retries=3)

    reader = SimpleDirectoryReader(
        input_dir="docs_bakkesmod_only",
        required_exts=[".md"],
        recursive=True
    )
    documents = reader.load_data()

    # Clean text
    cleaned_docs = []
    for doc in documents:
        clean_text = "".join(filter(lambda x: x.isprintable() or x in "\n\r\t", doc.text))
        cleaned_docs.append(Document(text=clean_text, metadata=doc.metadata))

    # Build KG index
    kg_index = KnowledgeGraphIndex.from_documents(
        cleaned_docs,
        max_triplets_per_chunk=2,
        show_progress=True
    )

    assert kg_index is not None
    print("[OK] KG index built successfully")

def test_kg_retriever_works():
    """Test that KG retriever can retrieve relevant nodes."""
    from llama_index.core import SimpleDirectoryReader, Settings, Document
    from llama_index.llms.anthropic import Anthropic
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex

    Settings.llm = Anthropic(model="claude-sonnet-4-5", max_retries=3, temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", max_retries=3)

    reader = SimpleDirectoryReader(
        input_dir="docs_bakkesmod_only",
        required_exts=[".md"],
        recursive=True
    )
    documents = reader.load_data()

    cleaned_docs = []
    for doc in documents:
        clean_text = "".join(filter(lambda x: x.isprintable() or x in "\n\r\t", doc.text))
        cleaned_docs.append(Document(text=clean_text, metadata=doc.metadata))

    kg_index = KnowledgeGraphIndex.from_documents(
        cleaned_docs,
        max_triplets_per_chunk=2
    )

    # Test retrieval
    retriever = kg_index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve("CarWrapper physics")

    assert len(nodes) > 0
    print(f"[OK] KG retriever returned {len(nodes)} nodes")

if __name__ == "__main__":
    print("\n=== Testing KG Integration ===\n")
    test_kg_index_builds()
    test_kg_retriever_works()
    print("\n[OK] All KG tests passed!")
