"""
Developer Scenario Testing
===========================
Simulates a real BakkesMod plugin developer asking questions
during different stages of plugin development.
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def log(message, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] [{level:5s}] {message}")

def log_section(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

def display_answer(scenario, question, response, query_time):
    """Display the response in a clear format."""
    log_section(f"SCENARIO: {scenario}")
    print(f"\n[DEVELOPER ASKS]")
    print(f"  {question}")
    print(f"\n[RAG SYSTEM RESPONDS]")
    print("-" * 80)
    print(response)
    print("-" * 80)
    print(f"\n[QUERY TIME] {query_time:.2f}s")
    print(f"[SOURCES] {len(response.source_nodes)} documents")
    if response.source_nodes:
        for node in response.source_nodes[:3]:
            filename = node.node.metadata.get("file_name", "unknown")
            print(f"  - {filename}")

def build_rag():
    """Build the RAG system."""
    log("Building RAG system for BakkesMod plugin development...")

    from llama_index.core import (
        SimpleDirectoryReader, StorageContext, VectorStoreIndex,
        Settings, load_index_from_storage, Document
    )
    from llama_index.core.node_parser import MarkdownNodeParser
    from llama_index.llms.anthropic import Anthropic
    from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.retrievers.bm25 import BM25Retriever
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex

    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small", max_retries=3)
    Settings.llm = Anthropic(model="claude-sonnet-4-5", max_retries=3, temperature=0)

    reader = SimpleDirectoryReader(
        input_dir="docs_bakkesmod_only",
        required_exts=[".md"],
        recursive=True,
        filename_as_id=True
    )
    documents = reader.load_data()

    cleaned_docs = []
    for doc in documents:
        clean_text = "".join(filter(lambda x: x.isprintable() or x in "\n\r\t", doc.text))
        cleaned_docs.append(Document(text=clean_text, metadata=doc.metadata))
    documents = cleaned_docs

    parser = MarkdownNodeParser()
    nodes = parser.get_nodes_from_documents(documents)

    storage_dir = "rag_storage_bakkesmod"
    from pathlib import Path
    storage_path = Path(storage_dir)

    if storage_path.exists():
        log("Loading cached indexes...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        vector_index = load_index_from_storage(storage_context, index_id="vector")

        # Load KG index if it exists
        try:
            kg_index = load_index_from_storage(storage_context, index_id="knowledge_graph")
            log("Loaded cached KG index")
        except:
            log("Building new KG index...")
            kg_index = KnowledgeGraphIndex.from_documents(
                documents,
                max_triplets_per_chunk=2,
                show_progress=True
            )
            kg_index.set_index_id("knowledge_graph")
            kg_index.storage_context.persist(persist_dir=storage_dir)
            log("KG index built and cached")
    else:
        log("Building new indexes...")
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

    vector_retriever = vector_index.as_retriever(similarity_top_k=5)
    bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=5)
    kg_retriever = kg_index.as_retriever(similarity_top_k=3)
    fusion_retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever, kg_retriever],
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=True
    )
    query_engine = RetrieverQueryEngine.from_args(fusion_retriever)

    log(f"System ready! ({len(documents)} docs, {len(nodes)} nodes, 3-way fusion: Vector+BM25+KG)")
    return query_engine

def main():
    log_section("BAKKESMOD PLUGIN DEVELOPER - REALISTIC SCENARIO TESTING")

    # Build system
    query_engine = build_rag()

    # Developer questions across different development stages
    developer_questions = [
        # Stage 1: Getting Started
        {
            "scenario": "Starting a New Plugin Project",
            "question": "I'm starting my first BakkesMod plugin. What's the basic structure I need and what methods do I need to implement?"
        },

        # Stage 2: Event Hooking
        {
            "scenario": "Hooking Game Events",
            "question": "I want to detect when a goal is scored in the match. How do I hook into the goal scored event?"
        },

        # Stage 3: Accessing Game Data
        {
            "scenario": "Getting Player Car Information",
            "question": "How do I get the local player's car velocity and boost amount during a match?"
        },

        # Stage 4: Server vs Client
        {
            "scenario": "Understanding Wrappers",
            "question": "What's the difference between ServerWrapper and CarWrapper? When should I use each one?"
        },

        # Stage 5: UI Development
        {
            "scenario": "Creating Settings UI",
            "question": "I need to create a settings window with checkboxes and sliders for my plugin. How do I use ImGui to build this?"
        },

        # Stage 6: Ball Physics
        {
            "scenario": "Working with Ball Data",
            "question": "How can I get the ball's current position and velocity? I want to predict where it's going."
        },

        # Stage 7: Multiple Cars
        {
            "scenario": "Iterating Through Players",
            "question": "How do I get a list of all cars in the match and iterate through each player?"
        },

        # Stage 8: Hook Types
        {
            "scenario": "Understanding Hook Variants",
            "question": "What's the difference between HookEvent and HookEventWithCaller? When would I use HookEventWithCallerPost?"
        },
    ]

    results = []
    total_time = 0

    for i, scenario_data in enumerate(developer_questions, 1):
        scenario = scenario_data["scenario"]
        question = scenario_data["question"]

        log(f"\n[TEST {i}/{len(developer_questions)}] {scenario}")
        log(f"Question: {question[:80]}...")

        start_time = time.time()
        try:
            response = query_engine.query(question)
            query_time = time.time() - start_time
            total_time += query_time

            display_answer(scenario, question, response, query_time)

            results.append({
                "scenario": scenario,
                "question": question,
                "response": response,
                "time": query_time,
                "success": True
            })

            # Brief pause between queries
            time.sleep(1)

        except Exception as e:
            query_time = time.time() - start_time
            log(f"Failed: {e}", "ERROR")
            results.append({
                "scenario": scenario,
                "question": question,
                "error": str(e),
                "time": query_time,
                "success": False
            })

    # Summary
    log_section("DEVELOPER TESTING SUMMARY")
    successful = sum(1 for r in results if r["success"])
    print(f"\n[STATISTICS]")
    print(f"  Total Questions: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {len(results) - successful}")
    print(f"  Success Rate: {(successful/len(results)*100):.1f}%")
    print(f"  Average Query Time: {total_time/len(results):.2f}s")
    print(f"  Total Time: {total_time:.2f}s")

    print(f"\n[DEVELOPER EXPERIENCE]")
    if successful == len(results):
        print("  [EXCELLENT] All questions answered successfully!")
        print("  The RAG system can effectively support BakkesMod plugin development.")
    else:
        print(f"  [WARNING] {len(results) - successful} questions failed")
        print("  Some development scenarios may need additional documentation.")

    log_section("TEST COMPLETE")

if __name__ == "__main__":
    main()
