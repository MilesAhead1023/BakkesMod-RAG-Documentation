"""
Retrieval Module
================
3-way fusion retrieval: Vector + BM25 + Knowledge Graph.
Handles index building, caching, and query engine creation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, List

from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
    Settings,
    load_index_from_storage,
)
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.schema import BaseNode
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from bakkesmod_rag.config import RAGConfig

logger = logging.getLogger("bakkesmod_rag.retrieval")


def build_or_load_indexes(
    nodes: List[BaseNode],
    documents: list,
    config: Optional[RAGConfig] = None,
) -> Dict[str, object]:
    """Build or load indexes from storage.

    Builds vector index and optionally knowledge graph index.
    Persists to rag_storage/ for fast reload.

    Args:
        nodes: Parsed document nodes.
        documents: Original documents (needed for KG build).
        config: RAGConfig instance.

    Returns:
        Dict with 'vector' and optionally 'kg' index objects.
    """
    if config is None:
        from bakkesmod_rag.config import get_config
        config = get_config()

    storage_dir = config.storage.storage_dir
    storage_path = Path(storage_dir)
    indexes = {}

    if storage_path.exists():
        print("[RETRIEVAL] Loading indexes from cache...")
        logger.info("Loading indexes from %s", storage_dir)
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)

        vector_index = load_index_from_storage(storage_context, index_id="vector")
        indexes["vector"] = vector_index

        # Try loading KG index
        if config.retriever.enable_kg:
            try:
                kg_index = load_index_from_storage(
                    storage_context, index_id="knowledge_graph"
                )
                indexes["kg"] = kg_index
                print("[RETRIEVAL] Loaded cached KG index")
                logger.info("Loaded cached KG index")
            except Exception as e:
                logger.warning("KG index not in cache: %s", e)
                print(f"[RETRIEVAL] KG index not in cache ({e})")
                _try_build_kg(documents, storage_dir, config, indexes)
    else:
        print("[RETRIEVAL] Building new indexes (this may take a while)...")
        logger.info("Building new indexes")

        vector_index = VectorStoreIndex(nodes, show_progress=True)
        vector_index.set_index_id("vector")
        storage_path.mkdir(parents=True, exist_ok=True)
        vector_index.storage_context.persist(persist_dir=storage_dir)
        indexes["vector"] = vector_index
        print("[RETRIEVAL] Vector index built and cached")

        if config.retriever.enable_kg:
            _try_build_kg(documents, storage_dir, config, indexes)

    return indexes


def _try_build_kg(documents, storage_dir, config, indexes):
    """Attempt to build KG index, gracefully degrading on failure."""
    print("[RETRIEVAL] Building Knowledge Graph index...")
    logger.info("Building KG index")
    try:
        kg_index = KnowledgeGraphIndex.from_documents(
            documents,
            max_triplets_per_chunk=config.retriever.kg_max_triplets_per_chunk,
            show_progress=True,
        )
        kg_index.set_index_id("knowledge_graph")
        kg_index.storage_context.persist(persist_dir=storage_dir)
        indexes["kg"] = kg_index
        print("[RETRIEVAL] KG index built and cached")
        logger.info("KG index built and cached")
    except Exception as kg_error:
        logger.error("Failed to build KG index: %s", str(kg_error)[:100])
        print(f"[RETRIEVAL] Failed to build KG: {str(kg_error)[:100]}")
        print("[RETRIEVAL] Using 2-way fusion (Vector + BM25) only")


def create_fusion_retriever(
    indexes: Dict[str, object],
    nodes: List[BaseNode],
    config: Optional[RAGConfig] = None,
) -> QueryFusionRetriever:
    """Create a fusion retriever combining Vector + BM25 + optional KG.

    Args:
        indexes: Dict from build_or_load_indexes().
        nodes: Parsed nodes (needed for BM25).
        config: RAGConfig instance.

    Returns:
        QueryFusionRetriever with reciprocal rank fusion.
    """
    if config is None:
        from bakkesmod_rag.config import get_config
        config = get_config()

    vector_retriever = indexes["vector"].as_retriever(
        similarity_top_k=config.retriever.vector_top_k
    )
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes, similarity_top_k=config.retriever.bm25_top_k
    )

    retrievers: List[BaseRetriever] = [vector_retriever, bm25_retriever]

    if "kg" in indexes:
        kg_retriever = indexes["kg"].as_retriever(
            similarity_top_k=config.retriever.kg_similarity_top_k
        )
        retrievers.append(kg_retriever)
        mode_label = "3-way fusion (Vector+BM25+KG)"
    else:
        mode_label = "2-way fusion (Vector+BM25)"

    fusion_retriever = QueryFusionRetriever(
        retrievers,
        num_queries=config.retriever.fusion_num_queries,
        mode=config.retriever.fusion_mode,
        use_async=True,
    )

    print(f"[RETRIEVAL] {mode_label} with {config.retriever.fusion_num_queries} query variants")
    logger.info("%s with %d query variants", mode_label, config.retriever.fusion_num_queries)
    return fusion_retriever


def create_query_engine(
    fusion_retriever: QueryFusionRetriever,
    config: Optional[RAGConfig] = None,
    streaming: bool = True,
) -> RetrieverQueryEngine:
    """Create query engine with optional neural reranking.

    Args:
        fusion_retriever: The fusion retriever to use.
        config: RAGConfig instance.
        streaming: Whether to enable streaming responses.

    Returns:
        RetrieverQueryEngine ready for queries.
    """
    if config is None:
        from bakkesmod_rag.config import get_config
        config = get_config()

    node_postprocessors = []

    # Optional Cohere reranker
    if config.retriever.enable_reranker and config.cohere_api_key:
        try:
            from llama_index.postprocessor.cohere_rerank import CohereRerank

            reranker = CohereRerank(
                api_key=config.cohere_api_key,
                model=config.retriever.reranker_model,
                top_n=config.retriever.rerank_top_n,
            )
            node_postprocessors.append(reranker)
            print("[RETRIEVAL] Neural reranker enabled (Cohere)")
            logger.info("Cohere reranker enabled")
        except ImportError:
            logger.warning("Cohere reranker not available")
        except Exception as e:
            logger.warning("Failed to initialize Cohere reranker: %s", e)

    query_engine = RetrieverQueryEngine.from_args(
        fusion_retriever,
        streaming=streaming,
        node_postprocessors=node_postprocessors if node_postprocessors else None,
    )

    return query_engine
