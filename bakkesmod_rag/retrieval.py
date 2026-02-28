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
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

# BM25Retriever import with fallback
try:
    from llama_index.retrievers.bm25 import BM25Retriever
except (ImportError, ModuleNotFoundError):
    try:
        from llama_index.legacy.retrievers.bm25 import BM25Retriever
    except (ImportError, ModuleNotFoundError):
        BM25Retriever = None

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
    """Create a fusion retriever combining Vector + BM25 + optional KG + optional ColBERT.

    When use_hierarchical_chunking=True, wraps the vector retriever with
    AutoMergingRetriever so that child chunk hits can merge up to parent nodes.

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

    # Wrap with AutoMergingRetriever for hierarchical chunking
    if config.retriever.use_hierarchical_chunking:
        vector_retriever = _wrap_auto_merging(
            vector_retriever, indexes["vector"].storage_context, config
        )

    # BM25 retriever (optional, may not be available in all installations)
    retrievers: List[BaseRetriever] = [vector_retriever]

    if BM25Retriever is not None:
        bm25_retriever = BM25Retriever.from_defaults(
            nodes=nodes, similarity_top_k=config.retriever.bm25_top_k
        )
        retrievers.append(bm25_retriever)
    else:
        logger.warning("BM25Retriever not available; using vector search only")

    if "kg" in indexes:
        kg_retriever = indexes["kg"].as_retriever(
            similarity_top_k=config.retriever.kg_similarity_top_k
        )
        retrievers.append(kg_retriever)
        if BM25Retriever is not None:
            mode_label = "3-way fusion (Vector+BM25+KG)"
        else:
            mode_label = "2-way fusion (Vector+KG)"
    else:
        if BM25Retriever is not None:
            mode_label = "2-way fusion (Vector+BM25)"
        else:
            mode_label = "Vector search only"

    # Optional ColBERT 4th retriever
    if config.retriever.use_colbert and "colbert" in indexes:
        retrievers.append(indexes["colbert"])
        mode_label = mode_label.replace(")", "+ColBERT)")

    fusion_retriever = QueryFusionRetriever(
        retrievers,
        num_queries=config.retriever.fusion_num_queries,
        mode=config.retriever.fusion_mode,
        use_async=True,
    )

    print(f"[RETRIEVAL] {mode_label} with {config.retriever.fusion_num_queries} query variants")
    logger.info("%s with %d query variants", mode_label, config.retriever.fusion_num_queries)
    return fusion_retriever


def _wrap_auto_merging(
    retriever: BaseRetriever,
    storage_context,
    config: RAGConfig,
) -> BaseRetriever:
    """Wrap a retriever with AutoMergingRetriever for parent-child merging.

    When >merge_threshold of a parent's children are retrieved, replaces
    them with the parent node for more complete context.

    Falls back to the original retriever if AutoMergingRetriever is unavailable.
    """
    try:
        from llama_index.core.retrievers import AutoMergingRetriever

        merged = AutoMergingRetriever(
            retriever,
            storage_context,
            simple_ratio_thresh=config.retriever.merge_threshold,
            verbose=False,
        )
        logger.info(
            "AutoMergingRetriever enabled (merge_threshold=%.2f)",
            config.retriever.merge_threshold,
        )
        return merged
    except (ImportError, Exception) as e:
        logger.warning(
            "AutoMergingRetriever unavailable (%s), using flat retriever", e
        )
        return retriever


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

    # Reranker fallback chain: tries each backend in preference order
    if config.retriever.enable_reranker:
        reranker = _get_reranker(config)
        if reranker is not None:
            node_postprocessors.append(reranker)

    # MMR diversity reranking (applied after neural reranker)
    if config.retriever.use_mmr:
        mmr_processor = _get_mmr_postprocessor(config)
        if mmr_processor is not None:
            node_postprocessors.append(mmr_processor)

    query_engine = RetrieverQueryEngine.from_args(
        fusion_retriever,
        streaming=streaming,
        node_postprocessors=node_postprocessors if node_postprocessors else None,
    )

    return query_engine


def _get_reranker(config: RAGConfig):
    """Try each reranker backend in preference order, return first that works.

    Fallback chain defined by config.retriever.reranker_preference.
    Default order: BAAI/BGE (free, local) → FlashRank (free, lightweight)
    → Cohere (paid API).

    Returns:
        A LlamaIndex node postprocessor, or None if all fail.
    """
    top_n = config.retriever.rerank_top_n

    for backend in config.retriever.reranker_preference:
        try:
            if backend == "bge":
                from llama_index.postprocessor.flag_embedding_reranker import (
                    FlagEmbeddingReranker,
                )

                reranker = FlagEmbeddingReranker(
                    model=config.retriever.bge_reranker_model,
                    top_n=top_n,
                )
                print(f"[RETRIEVAL] Neural reranker enabled (BGE: {config.retriever.bge_reranker_model})")
                logger.info("BGE reranker enabled: %s", config.retriever.bge_reranker_model)
                return reranker

            elif backend == "flashrank":
                from llama_index.postprocessor.flashrank_rerank import FlashRankRerank

                reranker = FlashRankRerank(
                    top_n=top_n,
                    model=config.retriever.flashrank_model,
                )
                print("[RETRIEVAL] Neural reranker enabled (FlashRank)")
                logger.info("FlashRank reranker enabled: %s", config.retriever.flashrank_model)
                return reranker

            elif backend == "cohere":
                if not config.cohere_api_key:
                    logger.info("Cohere reranker skipped — no API key")
                    continue
                from llama_index.postprocessor.cohere_rerank import CohereRerank

                reranker = CohereRerank(
                    api_key=config.cohere_api_key,
                    model=config.retriever.reranker_model,
                    top_n=top_n,
                )
                print("[RETRIEVAL] Neural reranker enabled (Cohere)")
                logger.info("Cohere reranker enabled")
                return reranker

        except ImportError:
            logger.warning("Reranker '%s' package not installed, trying next", backend)
        except Exception as e:
            logger.warning("Reranker '%s' failed to initialize: %s, trying next", backend, e)

    logger.warning("No reranker available — proceeding without reranking")
    print("[RETRIEVAL] No reranker available — proceeding without reranking")
    return None


def _get_mmr_postprocessor(config: RAGConfig):
    """Create an MMR node postprocessor for diversity-aware reranking.

    Maximal Marginal Relevance (MMR) reduces near-duplicate chunks from
    the same source by penalising nodes too similar to already-selected ones.

    Falls back gracefully if not available.

    Returns:
        MMR postprocessor instance, or None if unavailable.
    """
    try:
        # Try LlamaIndex built-in MMR postprocessor
        from llama_index.core.postprocessor import MMRNodePostprocessor

        mmr = MMRNodePostprocessor(
            similarity_cutoff=config.retriever.mmr_threshold,
        )
        logger.info(
            "MMR postprocessor enabled (threshold=%.2f)", config.retriever.mmr_threshold
        )
        print(f"[RETRIEVAL] MMR diversity reranking enabled (threshold={config.retriever.mmr_threshold})")
        return mmr
    except (ImportError, Exception):
        pass

    # Fallback: custom MMR implementation
    try:
        return _CustomMMRPostprocessor(
            similarity_cutoff=config.retriever.mmr_threshold,
        )
    except Exception as e:
        logger.warning("MMR postprocessor unavailable: %s", e)
        return None


class _CustomMMRPostprocessor:
    """Minimal MMR postprocessor for when LlamaIndex's is unavailable.

    Implements greedy MMR: select nodes that are relevant to the query
    but dissimilar from already-selected nodes.
    """

    def __init__(self, similarity_cutoff: float = 0.7):
        self.similarity_cutoff = similarity_cutoff

    def _cosine_similarity(self, v1, v2) -> float:
        if not v1 or not v2:
            return 0.0
        dot = sum(a * b for a, b in zip(v1, v2))
        m1 = sum(a * a for a in v1) ** 0.5
        m2 = sum(b * b for b in v2) ** 0.5
        if m1 == 0 or m2 == 0:
            return 0.0
        return dot / (m1 * m2)

    def postprocess_nodes(self, nodes, query_bundle=None):
        """Filter out near-duplicate nodes above the similarity cutoff."""
        selected = []
        selected_embeddings = []

        for node_with_score in nodes:
            node = node_with_score.node
            emb = getattr(node, "embedding", None)

            if emb is None:
                # No embedding: always include
                selected.append(node_with_score)
                selected_embeddings.append(None)
                continue

            # Check similarity against already-selected nodes
            too_similar = False
            for sel_emb in selected_embeddings:
                if sel_emb is not None:
                    sim = self._cosine_similarity(emb, sel_emb)
                    if sim > self.similarity_cutoff:
                        too_similar = True
                        break

            if not too_similar:
                selected.append(node_with_score)
                selected_embeddings.append(emb)

        logger.info(
            "MMR: %d nodes -> %d after diversity filtering (cutoff=%.2f)",
            len(nodes), len(selected), self.similarity_cutoff,
        )
        return selected


def build_colbert_retriever(
    documents: list,
    config: RAGConfig,
    storage_dir: str,
):
    """Build a ColBERT late-interaction retriever (optional).

    Uses ragatouille library with colbertv2.0 model for fine-grained
    token-level query-document matching. Falls back to None if ragatouille
    is not installed.

    Args:
        documents: List of Document objects to index.
        config: RAGConfig instance.
        storage_dir: Directory to persist ColBERT index.

    Returns:
        A LlamaIndex-compatible retriever, or None if unavailable.
    """
    if not config.retriever.use_colbert:
        return None

    try:
        from ragatouille import RAGPretrainedModel

        colbert_dir = str(Path(storage_dir) / "colbert")
        texts = [d.text for d in documents if d.text.strip()]
        ids = [
            d.metadata.get("file_path", f"doc_{i}")
            for i, d in enumerate(documents)
            if d.text.strip()
        ]

        model = RAGPretrainedModel.from_pretrained(config.retriever.colbert_model)
        model.index(
            collection=texts,
            document_ids=ids,
            index_name="bakkesmod",
            overwrite_index=True,
        )

        class _ColBERTRetriever:
            def __init__(self, colbert_model, top_k: int):
                self._model = colbert_model
                self._top_k = top_k

            def retrieve(self, query: str):
                results = self._model.search(query, k=self._top_k)
                from llama_index.core.schema import NodeWithScore, TextNode
                nodes = []
                for r in results:
                    n = TextNode(text=r.get("content", ""), id_=r.get("document_id", ""))
                    nodes.append(NodeWithScore(node=n, score=r.get("score", 0.0)))
                return nodes

        retriever = _ColBERTRetriever(model, top_k=config.retriever.vector_top_k)
        logger.info("ColBERT retriever built: %s", config.retriever.colbert_model)
        print(f"[RETRIEVAL] ColBERT retriever enabled ({config.retriever.colbert_model})")
        return retriever

    except ImportError:
        logger.info("ragatouille not installed, ColBERT retrieval skipped")
        return None
    except Exception as e:
        logger.warning("ColBERT retriever build failed: %s", e)
        return None


def adjust_retriever_top_k(
    fusion_retriever: QueryFusionRetriever,
    attempt: int,
    config: Optional[RAGConfig] = None,
) -> None:
    """Dynamically adjust top_k for all sub-retrievers based on retry attempt.

    Uses escalation lists from config. Attempt 0 = initial, 1 = retry 1, etc.
    Clamps to the last value if attempt exceeds the list length.

    Args:
        fusion_retriever: The fusion retriever whose sub-retrievers to adjust.
        attempt: The retry attempt number (0-based).
        config: RAGConfig instance.
    """
    if config is None:
        from bakkesmod_rag.config import get_config
        config = get_config()

    if not config.retriever.adaptive_top_k:
        return

    escalation = config.retriever.top_k_escalation
    kg_escalation = config.retriever.kg_top_k_escalation

    # Clamp attempt to valid range
    idx = min(attempt, len(escalation) - 1)
    kg_idx = min(attempt, len(kg_escalation) - 1)

    top_k = escalation[idx]
    kg_top_k = kg_escalation[kg_idx]

    for retriever in fusion_retriever._retrievers:
        class_name = type(retriever).__name__
        if "KnowledgeGraph" in class_name or "KG" in class_name:
            if hasattr(retriever, "_similarity_top_k"):
                retriever._similarity_top_k = kg_top_k
            elif hasattr(retriever, "similarity_top_k"):
                retriever.similarity_top_k = kg_top_k
        else:
            if hasattr(retriever, "_similarity_top_k"):
                retriever._similarity_top_k = top_k
            elif hasattr(retriever, "similarity_top_k"):
                retriever.similarity_top_k = top_k

    logger.info(
        "Adjusted top_k: attempt=%d, vector/bm25=%d, kg=%d",
        attempt, top_k, kg_top_k,
    )


def reset_retriever_top_k(
    fusion_retriever: QueryFusionRetriever,
    config: Optional[RAGConfig] = None,
) -> None:
    """Reset all sub-retriever top_k values to baseline config defaults.

    Args:
        fusion_retriever: The fusion retriever to reset.
        config: RAGConfig instance.
    """
    if config is None:
        from bakkesmod_rag.config import get_config
        config = get_config()

    adjust_retriever_top_k(fusion_retriever, attempt=0, config=config)
