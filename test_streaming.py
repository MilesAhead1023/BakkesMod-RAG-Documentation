"""
Test Streaming Responses
=========================
Tests that query engine supports token-by-token streaming.
"""

import os
import time
from dotenv import load_dotenv
load_dotenv()

def test_streaming_query_engine():
    """Test that streaming query works and produces tokens progressively."""
    from llama_index.core import SimpleDirectoryReader, Settings, Document
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.llms.anthropic import Anthropic
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core import VectorStoreIndex

    Settings.llm = Anthropic(model="claude-sonnet-4-5", max_retries=3, temperature=0)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", max_retries=3)

    # Load minimal documents
    reader = SimpleDirectoryReader(
        input_dir="docs_bakkesmod_only",
        required_exts=[".md"],
        recursive=True,
        filename_as_id=True
    )
    documents = reader.load_data()

    # Clean and parse
    cleaned_docs = []
    for doc in documents:
        clean_text = "".join(filter(lambda x: x.isprintable() or x in "\n\r\t", doc.text))
        cleaned_docs.append(Document(text=clean_text, metadata=doc.metadata))

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(cleaned_docs)

    # Build simple vector index
    vector_index = VectorStoreIndex(nodes, show_progress=False)

    # Get streaming query engine
    query_engine = vector_index.as_query_engine(streaming=True)

    # Test streaming
    start_time = time.time()
    response = query_engine.query("What is BakkesMod?")

    # Should get response object immediately
    first_token_time = time.time() - start_time
    assert first_token_time < 2.0, f"First token took {first_token_time:.2f}s (should be <2s)"
    print(f"[OK] Got response object in {first_token_time:.2f}s")

    # Check that response is a generator
    assert hasattr(response, 'response_gen'), "Response should have response_gen attribute"
    print("[OK] Response has response_gen for streaming")

    # Iterate through tokens
    tokens = []
    for token in response.response_gen:
        tokens.append(token)
        if len(tokens) == 1:
            print(f"[OK] First token received: '{token[:20]}...'")

    full_response = "".join(tokens)
    assert len(full_response) > 50, "Response should have content"
    print(f"[OK] Streamed {len(tokens)} tokens, {len(full_response)} chars total")

def test_streaming_with_source_nodes():
    """Test that streaming response still includes source nodes."""
    from llama_index.core import SimpleDirectoryReader, Settings, Document
    from llama_index.llms.anthropic import Anthropic
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.core import VectorStoreIndex

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

    from llama_index.core.node_parser import MarkdownNodeParser
    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(cleaned_docs)

    vector_index = VectorStoreIndex(nodes, show_progress=False)
    query_engine = vector_index.as_query_engine(streaming=True)

    response = query_engine.query("What is CarWrapper?")

    # Consume streaming tokens
    _ = "".join(response.response_gen)

    # Check source nodes are still available
    assert hasattr(response, 'source_nodes'), "Response should have source_nodes"
    assert len(response.source_nodes) > 0, "Should retrieve source nodes"
    print(f"[OK] Streaming response includes {len(response.source_nodes)} source nodes")

if __name__ == "__main__":
    print("\n=== Testing Streaming Responses ===\n")
    test_streaming_query_engine()
    print()
    test_streaming_with_source_nodes()
    print("\n[OK] All streaming tests passed!")
