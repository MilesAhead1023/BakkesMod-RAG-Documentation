"""
RAG Engine
==========
Central orchestrator that wires all subsystems together.

Provides the main ``RAGEngine`` class that initialises the LLM fallback chain,
embedding model, semantic cache, query rewriting, document loading, index
building, fusion retrieval, and code generation -- then exposes a clean
``query()`` / ``query_streaming()`` / ``generate_code()`` API.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Generator

from bakkesmod_rag.config import RAGConfig, get_config
from bakkesmod_rag.llm_provider import get_llm, get_embed_model
from bakkesmod_rag.cache import SemanticCache
from bakkesmod_rag.query_rewriter import QueryRewriter
from bakkesmod_rag.confidence import calculate_confidence
from bakkesmod_rag.cost_tracker import CostTracker
from bakkesmod_rag.observability import StructuredLogger
from bakkesmod_rag.document_loader import load_documents, parse_nodes
from bakkesmod_rag.retrieval import (
    build_or_load_indexes,
    create_fusion_retriever,
    create_query_engine,
)
from bakkesmod_rag.code_generator import BakkesModCodeGenerator

logger = logging.getLogger("bakkesmod_rag.engine")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Result container for a RAG query.

    Attributes:
        answer: The synthesised answer text.
        sources: List of source node metadata dicts (file_name, score).
        confidence: Numeric confidence score (0.0 -- 1.0).
        confidence_label: Human-readable confidence tier (e.g. ``"HIGH"``).
        confidence_explanation: Short explanation of the confidence rating.
        query_time: Wall-clock seconds the query took.
        cached: Whether the answer came from the semantic cache.
        expanded_query: The query after domain-synonym expansion.
    """

    answer: str
    sources: list  # list of source node metadata dicts
    confidence: float
    confidence_label: str
    confidence_explanation: str
    query_time: float
    cached: bool
    expanded_query: str


@dataclass
class CodeResult:
    """Result container for plugin code generation.

    Attributes:
        header: Generated C++ header file content.
        implementation: Generated C++ implementation file content.
        explanation: The original natural-language description that was used.
        validation: Validation dict from ``CodeValidator.validate_syntax``.
    """

    header: str
    implementation: str
    explanation: str
    validation: dict  # from CodeValidator


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class RAGEngine:
    """Central orchestrator for the BakkesMod RAG system.

    Initialises every subsystem on construction and exposes three main
    entry-points:

    * ``query()``           -- non-streaming RAG Q&A
    * ``query_streaming()`` -- streaming RAG Q&A (yields tokens)
    * ``generate_code()``   -- RAG-enhanced plugin code generation
    """

    def __init__(self, config: RAGConfig | None = None) -> None:
        """Initialise all subsystems.

        Args:
            config: Optional ``RAGConfig``.  Falls back to the global
                singleton returned by ``get_config()`` when *None*.
        """
        self.config: RAGConfig = config or get_config()

        # Observability -------------------------------------------------
        self.logger = StructuredLogger("rag_engine", self.config.observability)

        # LLM + Embeddings ---------------------------------------------
        self.llm = get_llm(self.config)
        self.embed_model = get_embed_model(self.config)

        # Set LlamaIndex global settings
        from llama_index.core import Settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # Semantic cache ------------------------------------------------
        self.cache = SemanticCache(
            cache_dir=self.config.cache.cache_dir,
            similarity_threshold=self.config.cache.similarity_threshold,
            ttl_seconds=self.config.cache.ttl_seconds,
            embed_model=self.embed_model,
        )

        # Query rewriter (synonym expansion only, no LLM cost) ---------
        self.rewriter = QueryRewriter(llm=self.llm, use_llm=False)

        # Cost tracking -------------------------------------------------
        self.cost_tracker = CostTracker(config=self.config)

        # Documents & indexes -------------------------------------------
        self.documents = load_documents(self.config)
        self.nodes = parse_nodes(self.documents, self.config)
        self.indexes = build_or_load_indexes(
            self.nodes, self.documents, self.config
        )
        self.fusion_retriever = create_fusion_retriever(
            self.indexes, self.nodes, self.config
        )
        self.query_engine = create_query_engine(
            self.fusion_retriever, self.config, streaming=True
        )

        # Code generator ------------------------------------------------
        self.code_gen = BakkesModCodeGenerator(
            llm=self.llm,
            query_engine=self.query_engine,
        )

        print(
            f"[ENGINE] Ready! {len(self.documents)} docs, "
            f"{len(self.nodes)} nodes"
        )

    # ---- convenience properties ----------------------------------------

    @property
    def num_documents(self) -> int:
        """Return the number of loaded documents."""
        return len(self.documents)

    @property
    def num_nodes(self) -> int:
        """Return the number of parsed nodes."""
        return len(self.nodes)

    # ---- non-streaming query -------------------------------------------

    def query(self, question: str, use_cache: bool = True) -> QueryResult:
        """Run a non-streaming RAG query.

        Flow: cache check -> synonym expansion -> retrieve -> synthesise
        -> cache store -> return.

        Args:
            question: The user's natural-language question.
            use_cache: Whether to check / populate the semantic cache.

        Returns:
            A ``QueryResult`` with the answer, sources, and metadata.
        """
        start_time = time.time()

        expanded_query = self.rewriter.expand_with_synonyms(question)

        # -- Cache check ------------------------------------------------
        if use_cache and self.config.cache.enabled:
            cache_result = self.cache.get(question)
            if cache_result:
                response_text, similarity, metadata = cache_result
                query_time = time.time() - start_time
                return QueryResult(
                    answer=response_text,
                    sources=[],
                    confidence=similarity,
                    confidence_label="CACHED",
                    confidence_explanation=(
                        f"Cache hit ({similarity:.1%} similarity)"
                    ),
                    query_time=query_time,
                    cached=True,
                    expanded_query=expanded_query,
                )

        # -- Execute query (non-streaming for programmatic use) ---------
        response = self.query_engine.query(expanded_query)
        query_time = time.time() - start_time

        answer = str(response)

        # -- Confidence scoring -----------------------------------------
        confidence, label, explanation = calculate_confidence(
            response.source_nodes
        )

        # -- Extract source metadata ------------------------------------
        sources: list[dict] = []
        for node in response.source_nodes:
            sources.append({
                "file_name": node.node.metadata.get("file_name", "unknown"),
                "score": node.score if hasattr(node, "score") else None,
            })

        # -- Cache the response -----------------------------------------
        if use_cache and self.config.cache.enabled:
            self.cache.set(question, answer, response.source_nodes)

        return QueryResult(
            answer=answer,
            sources=sources,
            confidence=confidence,
            confidence_label=label,
            confidence_explanation=explanation,
            query_time=query_time,
            cached=False,
            expanded_query=expanded_query,
        )

    # ---- streaming query -----------------------------------------------

    def query_streaming(
        self,
        question: str,
        use_cache: bool = True,
    ) -> Tuple[Generator[str, None, None], callable]:
        """Run a streaming RAG query that yields tokens.

        Returns a *(token_generator, get_metadata)* pair.  Consume the
        generator first, then call ``get_metadata()`` to obtain the full
        ``QueryResult`` with confidence scores and source information.

        Usage::

            gen, get_meta = engine.query_streaming("How do I hook an event?")
            for token in gen:
                print(token, end="")
            result = get_meta()  # QueryResult with full metadata

        Args:
            question: The user's natural-language question.
            use_cache: Whether to check / populate the semantic cache.

        Returns:
            Tuple of ``(token_generator, get_metadata_callable)``.
        """
        start_time = time.time()

        expanded_query = self.rewriter.expand_with_synonyms(question)

        # -- Cache check ------------------------------------------------
        if use_cache and self.config.cache.enabled:
            cache_result = self.cache.get(question)
            if cache_result:
                response_text, similarity, metadata = cache_result
                query_time = time.time() - start_time
                result = QueryResult(
                    answer=response_text,
                    sources=[],
                    confidence=similarity,
                    confidence_label="CACHED",
                    confidence_explanation=(
                        f"Cache hit ({similarity:.1%} similarity)"
                    ),
                    query_time=query_time,
                    cached=True,
                    expanded_query=expanded_query,
                )

                def cached_gen() -> Generator[str, None, None]:
                    yield response_text

                return cached_gen(), lambda: result

        # -- Execute streaming query ------------------------------------
        response = self.query_engine.query(expanded_query)

        tokens_collected: list[str] = []

        def token_generator() -> Generator[str, None, None]:
            for token in response.response_gen:
                tokens_collected.append(token)
                yield token

        def get_metadata() -> QueryResult:
            full_text = "".join(tokens_collected)
            query_time = time.time() - start_time

            confidence, label, explanation = calculate_confidence(
                response.source_nodes
            )

            sources: list[dict] = []
            for node in response.source_nodes:
                sources.append({
                    "file_name": node.node.metadata.get(
                        "file_name", "unknown"
                    ),
                    "score": (
                        node.score if hasattr(node, "score") else None
                    ),
                })

            # Cache the fully-collected response
            if use_cache and self.config.cache.enabled:
                self.cache.set(question, full_text, response.source_nodes)

            return QueryResult(
                answer=full_text,
                sources=sources,
                confidence=confidence,
                confidence_label=label,
                confidence_explanation=explanation,
                query_time=query_time,
                cached=False,
                expanded_query=expanded_query,
            )

        return token_generator(), get_metadata

    # ---- code generation -----------------------------------------------

    def generate_code(self, description: str) -> CodeResult:
        """Generate BakkesMod plugin code using RAG context.

        Args:
            description: Natural language description of the desired plugin.

        Returns:
            A ``CodeResult`` containing the header, implementation,
            explanation, and validation results.
        """
        result = self.code_gen.generate_plugin_with_rag(description)

        validation = self.code_gen.validator.validate_syntax(
            result.get("implementation", "")
        )

        return CodeResult(
            header=result.get("header", ""),
            implementation=result.get("implementation", ""),
            explanation=description,
            validation=validation,
        )
