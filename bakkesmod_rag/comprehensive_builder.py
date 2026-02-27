"""
Comprehensive RAG Builder
==========================
Full index build with incremental KG construction and checkpointing.
Use this for initial setup or rebuilding indexes from scratch.

Rewrite of the legacy ``comprehensive_rag.py`` using the unified
``bakkesmod_rag`` package.  Removed GPTCache (replaced by
``SemanticCache``), fixed the embedding model (uses config defaults
instead of hard-coded ``text-embedding-3-large``), and wired into the
package's ``config``, ``llm_provider``, and ``document_loader`` modules.

The unique value of this builder is incremental KG construction with
periodic checkpoint saves, so a long build can be resumed if interrupted.

Usage::

    python -m bakkesmod_rag.comprehensive_builder
"""

import os
import sys
import logging
import time
from pathlib import Path

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
)
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from bakkesmod_rag.config import RAGConfig, get_config
from bakkesmod_rag.llm_provider import get_llm, get_embed_model
from bakkesmod_rag.document_loader import load_documents, parse_nodes

logger = logging.getLogger("bakkesmod_rag.comprehensive_builder")

# Save KG progress every N nodes so a long build can be resumed.
CHECKPOINT_INTERVAL = 500


def build_comprehensive_stack(config=None, incremental=True):
    """Build the full RAG stack with incremental KG support.

    This function performs the following steps:

    1. Initialise LLM and embedding model via the package's provider layer.
    2. Load and parse all BakkesMod documentation.
    3. Build (or reload) the **vector index**.
    4. Build (or reload) the **knowledge graph index** with checkpoint saves
       every ``CHECKPOINT_INTERVAL`` nodes.
    5. Assemble a 3-way fusion retriever (Vector + BM25 + KG) and return
       a ``RetrieverQueryEngine`` ready for queries.

    Args:
        config: ``RAGConfig`` instance.  When *None* the global singleton
            from ``get_config()`` is used.
        incremental: When *True* and an existing KG index is found, call
            ``refresh_ref`` to incorporate any new or changed documents
            instead of rebuilding from scratch.

    Returns:
        ``RetrieverQueryEngine`` ready for queries.
    """
    config = config or get_config()

    print("Initializing comprehensive RAG stack...")
    start_time = time.time()

    # ------------------------------------------------------------------
    # 1. LLM & embeddings
    # ------------------------------------------------------------------
    llm = get_llm(config)
    embed_model = get_embed_model(config)
    Settings.llm = llm
    Settings.embed_model = embed_model

    # ------------------------------------------------------------------
    # 2. Document loading & parsing
    # ------------------------------------------------------------------
    documents = load_documents(config)
    nodes = parse_nodes(documents, config)
    total_nodes = len(nodes)
    print(f"Parsed {len(documents)} documents into {total_nodes} nodes.")

    storage_dir = config.storage.storage_dir
    storage_path = Path(storage_dir)

    # ------------------------------------------------------------------
    # 3. Vector index
    # ------------------------------------------------------------------
    if storage_path.exists():
        try:
            print("Loading existing Vector Index...")
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            vector_index = load_index_from_storage(
                storage_context, index_id="vector"
            )
            print("Vector Index loaded from cache.")
        except Exception as e:
            logger.warning("Could not load vector index (%s), rebuilding...", e)
            print("Rebuilding Vector Index...")
            vector_index = VectorStoreIndex(nodes, show_progress=True)
            vector_index.set_index_id("vector")
            vector_index.storage_context.persist(persist_dir=storage_dir)
            print("Vector Index saved.")
    else:
        print("Building Vector Index (this may take a minute)...")
        storage_path.mkdir(parents=True, exist_ok=True)
        vector_index = VectorStoreIndex(nodes, show_progress=True)
        vector_index.set_index_id("vector")
        vector_index.storage_context.persist(persist_dir=storage_dir)
        print("Vector Index saved.")

    # ------------------------------------------------------------------
    # 4. Knowledge Graph index with checkpointing
    # ------------------------------------------------------------------
    print("Building/loading Knowledge Graph...")
    kg_index = None

    try:
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        kg_index = load_index_from_storage(
            storage_context, index_id="knowledge_graph"
        )
        print("Existing KG Index loaded.")

        if incremental:
            print("Running incremental KG refresh...")
            kg_index.refresh_ref(documents)
            kg_index.storage_context.persist(persist_dir=storage_dir)
            print("KG refresh complete and saved.")
    except Exception:
        print("No existing KG found. Building with checkpoints...")
        logger.info("Starting fresh KG build (%d nodes, checkpoint every %d)",
                     total_nodes, CHECKPOINT_INTERVAL)

        # Create a fresh storage context that shares the existing persist dir
        # so the vector index files are preserved alongside the KG files.
        try:
            kg_storage_context = StorageContext.from_defaults(
                persist_dir=storage_dir
            )
        except Exception:
            kg_storage_context = StorageContext.from_defaults()

        kg_index = KnowledgeGraphIndex(
            [],  # Start empty -- we insert in batches below
            storage_context=kg_storage_context,
            max_triplets_per_chunk=config.retriever.kg_max_triplets_per_chunk,
            include_embeddings=True,
        )
        kg_index.set_index_id("knowledge_graph")

        # Process nodes in batches with checkpoint saves
        for i in range(0, total_nodes, CHECKPOINT_INTERVAL):
            batch = nodes[i:i + CHECKPOINT_INTERVAL]
            batch_end = min(i + CHECKPOINT_INTERVAL, total_nodes)
            print(f"Processing KG batch {i + 1}-{batch_end} / {total_nodes}...")

            batch_start = time.time()
            kg_index.insert_nodes(batch)
            batch_elapsed = time.time() - batch_start

            # Checkpoint save
            kg_index.storage_context.persist(persist_dir=storage_dir)
            print(f"  Checkpoint saved at node {batch_end} "
                  f"({batch_elapsed:.1f}s for this batch).")

        print(f"KG build complete: {total_nodes} nodes processed.")

    # ------------------------------------------------------------------
    # 5. Fusion retrieval
    # ------------------------------------------------------------------
    print("Configuring fusion retriever...")

    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=config.retriever.bm25_top_k,
    )

    retrievers = [
        vector_index.as_retriever(
            similarity_top_k=config.retriever.vector_top_k
        ),
        bm25_retriever,
    ]

    mode_label = "2-way fusion (Vector + BM25)"
    if kg_index is not None:
        retrievers.append(
            kg_index.as_retriever(
                similarity_top_k=config.retriever.kg_similarity_top_k
            )
        )
        mode_label = "3-way fusion (Vector + BM25 + KG)"

    fusion_retriever = QueryFusionRetriever(
        retrievers,
        num_queries=config.retriever.fusion_num_queries,
        mode=config.retriever.fusion_mode,
        use_async=True,
    )

    # Optional Cohere neural reranker
    node_postprocessors = []
    if config.retriever.enable_reranker and config.cohere_api_key:
        try:
            from llama_index.postprocessor.cohere_rerank import CohereRerank

            reranker = CohereRerank(
                api_key=config.cohere_api_key,
                model=config.retriever.reranker_model,
                top_n=config.retriever.rerank_top_n,
            )
            node_postprocessors.append(reranker)
            print("Neural reranker enabled (Cohere).")
        except ImportError:
            logger.warning("Cohere reranker package not installed, skipping")
        except Exception as e:
            logger.warning("Failed to initialize Cohere reranker: %s", e)

    query_engine = RetrieverQueryEngine.from_args(
        fusion_retriever,
        node_postprocessors=node_postprocessors if node_postprocessors else None,
    )

    # ------------------------------------------------------------------
    # 6. ColBERT index (optional, requires use_colbert=True)
    # ------------------------------------------------------------------
    if config.retriever.use_colbert:
        print("Building ColBERT index...")
        from bakkesmod_rag.retrieval import build_colbert_retriever
        colbert_retriever = build_colbert_retriever(documents, config, storage_dir)
        if colbert_retriever is not None:
            print("ColBERT index built and ready.")
        else:
            print("ColBERT unavailable (ragatouille not installed).")

    elapsed = time.time() - start_time
    print(f"Comprehensive RAG stack ready ({mode_label}) in {elapsed:.1f}s.")
    return query_engine


if __name__ == "__main__":
    engine = build_comprehensive_stack()
    print("\nReady for queries. Running a quick test...")

    test_query = "How do I get the velocity of the ball?"
    start = time.time()
    response = engine.query(test_query)
    latency = time.time() - start

    print(f"\nQ: {test_query}")
    print(f"A: {str(response)[:300]}...")
    print(f"Sources: {len(response.source_nodes)}")
    print(f"Latency: {latency:.2f}s")
