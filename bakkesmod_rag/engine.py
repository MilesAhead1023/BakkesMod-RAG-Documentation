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
from bakkesmod_rag.query_decomposer import QueryDecomposer
from bakkesmod_rag.confidence import calculate_confidence
from bakkesmod_rag.answer_verifier import AnswerVerifier
from bakkesmod_rag.cost_tracker import CostTracker
from bakkesmod_rag.observability import (
    StructuredLogger,
    PhoenixObserver,
    MetricsCollector,
)
from bakkesmod_rag.resilience import APICallManager
from bakkesmod_rag.document_loader import load_documents, parse_nodes
from bakkesmod_rag.retrieval import (
    build_or_load_indexes,
    create_fusion_retriever,
    create_query_engine,
    adjust_retriever_top_k,
    reset_retriever_top_k,
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
    verification_warning: Optional[str] = None


@dataclass
class CodeResult:
    """Result container for plugin code generation.

    Attributes:
        header: Generated C++ header file content.
        implementation: Generated C++ implementation file content.
        project_files: Dict mapping filename to content for ALL project files.
        explanation: The original natural-language description that was used.
        validation: Validation dict from ``CodeValidator.validate_project``.
        features_used: List of feature flags that were detected/enabled.
    """

    header: str
    implementation: str
    project_files: Dict[str, str]
    explanation: str
    validation: dict
    features_used: List[str] = field(default_factory=list)


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
        self.phoenix = PhoenixObserver(self.config.observability)
        self.metrics = MetricsCollector(self.config.observability)

        # Resilience ----------------------------------------------------
        self.api_manager = APICallManager(self.config.production)

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

        # Query rewriter (synonym expansion + LLM rewriting) ------------
        self.rewriter = QueryRewriter(
            llm=self.llm,
            use_llm=self.config.retriever.enable_llm_rewrite,
        )

        # Query decomposer (hybrid rule + LLM decomposition) -----------
        self.decomposer = QueryDecomposer(
            llm=self.llm,
            max_sub_queries=self.config.retriever.max_sub_queries,
            complexity_threshold=self.config.retriever.decomposition_complexity_threshold,
            enable_decomposition=self.config.retriever.enable_query_decomposition,
        )

        # Cost tracking -------------------------------------------------
        self.cost_tracker = CostTracker(config=self.config)

        # Answer verifier (embedding + LLM grounding check) ------------
        self.verifier = AnswerVerifier(
            embed_model=self.embed_model,
            llm=self.llm,
            grounded_threshold=self.config.verification.grounded_threshold,
            borderline_threshold=self.config.verification.borderline_threshold,
            borderline_penalty=self.config.verification.borderline_confidence_penalty,
            ungrounded_penalty=self.config.verification.ungrounded_confidence_penalty,
            enabled=self.config.verification.enabled,
        )

        # Documents & indexes -------------------------------------------
        self.documents = load_documents(self.config)
        self.nodes = parse_nodes(self.documents, self.config, embed_model=self.embed_model)
        self.indexes = build_or_load_indexes(
            self.nodes, self.documents, self.config
        )
        self.fusion_retriever = create_fusion_retriever(
            self.indexes, self.nodes, self.config
        )

        # Two query engines: sync for query(), streaming for query_streaming()
        self.query_engine_sync = create_query_engine(
            self.fusion_retriever, self.config, streaming=False
        )
        self.query_engine_streaming = create_query_engine(
            self.fusion_retriever, self.config, streaming=True
        )

        # Code generator (uses sync engine for RAG lookups) -------------
        self.code_gen = BakkesModCodeGenerator(
            llm=self.llm,
            query_engine=self.query_engine_sync,
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
        self.logger.log_query(question)

        expanded_query = self.rewriter.rewrite(question)

        # -- Cache check ------------------------------------------------
        if use_cache and self.config.cache.enabled:
            cache_result = self.cache.get(question)
            if cache_result:
                response_text, similarity, metadata = cache_result
                query_time = time.time() - start_time
                self.logger.log_cache_hit(question, similarity)
                self.metrics.record_cache_hit()
                self.metrics.record_query("cache_hit", query_time)
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

        self.metrics.record_cache_miss()

        # -- Query decomposition ----------------------------------------
        sub_queries = self.decomposer.decompose(expanded_query)

        # -- Execute query (non-streaming via sync engine) --------------
        try:
            source_nodes = []
            if len(sub_queries) > 1:
                # Iterative sub-query retrieval + merge
                answer, sources, confidence, label, explanation = (
                    self._execute_decomposed_query(sub_queries)
                )
            else:
                answer, sources, confidence, label, explanation, source_nodes = (
                    self._execute_single_query(expanded_query)
                )

            # -- Answer verification ------------------------------------
            verification_warning = None
            if source_nodes:
                vr = self.verifier.verify(answer, source_nodes, question)
                if vr.confidence_penalty > 0:
                    confidence = max(0.0, confidence - vr.confidence_penalty)
                    label, _ = self._confidence_label(confidence)
                verification_warning = vr.warning

            query_time = time.time() - start_time

            # -- Log retrieval metrics ----------------------------------
            source_names = [s["file_name"] for s in sources]
            self.logger.log_retrieval(
                num_chunks=len(sources),
                sources=source_names,
                latency_ms=query_time * 1000,
            )
            self.metrics.record_retrieval(len(sources))
            self.metrics.record_query("success", query_time)

            # -- Cache the response -------------------------------------
            if use_cache and self.config.cache.enabled:
                self.cache.set(question, answer, [])

            return QueryResult(
                answer=answer,
                sources=sources,
                confidence=confidence,
                confidence_label=label,
                confidence_explanation=explanation,
                query_time=query_time,
                cached=False,
                expanded_query=expanded_query,
                verification_warning=verification_warning,
            )
        except Exception as e:
            query_time = time.time() - start_time
            self.logger.log_error(e, {"query": question})
            self.metrics.record_query("error", query_time)
            raise

    # ---- internal query helpers ----------------------------------------

    @staticmethod
    def _confidence_label(confidence: float) -> tuple[str, str]:
        """Return (label, explanation) for a confidence score."""
        if confidence >= 0.80:
            return "HIGH", "Strong retrieval support"
        elif confidence >= 0.50:
            return "MEDIUM", "Moderate retrieval support"
        else:
            return "LOW", "Weak retrieval support"

    def _execute_single_query(
        self, query: str
    ) -> tuple[str, list[dict], float, str, str, list]:
        """Execute a single query against the sync engine.

        Returns:
            Tuple of (answer, sources, confidence, label, explanation,
            source_nodes).
        """
        response = self.query_engine_sync.query(query)
        answer = str(response)

        confidence, label, explanation = calculate_confidence(
            response.source_nodes
        )

        sources: list[dict] = []
        for node in response.source_nodes:
            fname = node.node.metadata.get("file_name", "unknown")
            sources.append({
                "file_name": fname,
                "score": node.score if hasattr(node, "score") else None,
            })

        return answer, sources, confidence, label, explanation, response.source_nodes

    def _execute_decomposed_query(
        self, sub_queries: list[str]
    ) -> tuple[str, list[dict], float, str, str]:
        """Execute multiple sub-queries iteratively and merge results.

        Each sub-query's answer is prepended as context for the next,
        enabling iterative context building.

        Returns:
            Tuple of (merged_answer, all_sources, avg_confidence, label,
            explanation).
        """
        sub_answers: list[str] = []
        all_sources: list[dict] = []
        all_confidences: list[float] = []
        seen_files: set[str] = set()

        for i, sq in enumerate(sub_queries):
            logger.info("Sub-query %d/%d: %s", i + 1, len(sub_queries), sq[:80])
            answer, sources, conf, _, _, _ = self._execute_single_query(sq)
            sub_answers.append(answer)
            all_confidences.append(conf)

            for s in sources:
                fname = s["file_name"]
                if fname not in seen_files:
                    all_sources.append(s)
                    seen_files.add(fname)

        # Merge sub-answers
        merged = QueryDecomposer.merge_sub_answers(
            sub_queries, sub_answers, llm=self.llm
        )

        # Average confidence across sub-queries
        avg_conf = (
            sum(all_confidences) / len(all_confidences)
            if all_confidences
            else 0.0
        )

        if avg_conf >= 0.8:
            label, explanation = "HIGH", "High confidence across sub-queries"
        elif avg_conf >= 0.5:
            label, explanation = "MEDIUM", "Medium confidence across sub-queries"
        else:
            label, explanation = "LOW", "Low confidence across sub-queries"

        return merged, all_sources, avg_conf, label, explanation

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
        self.logger.log_query(question, {"mode": "streaming"})

        expanded_query = self.rewriter.rewrite(question)

        # -- Cache check ------------------------------------------------
        if use_cache and self.config.cache.enabled:
            cache_result = self.cache.get(question)
            if cache_result:
                response_text, similarity, metadata = cache_result
                query_time = time.time() - start_time
                self.logger.log_cache_hit(question, similarity)
                self.metrics.record_cache_hit()
                self.metrics.record_query("cache_hit", query_time)
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

        self.metrics.record_cache_miss()

        # -- Query decomposition ----------------------------------------
        sub_queries = self.decomposer.decompose(expanded_query)

        if len(sub_queries) > 1:
            # Decomposed: run sub-queries non-streaming, stream merged answer
            answer, sources, confidence, label, explanation = (
                self._execute_decomposed_query(sub_queries)
            )
            query_time = time.time() - start_time

            source_names = [s["file_name"] for s in sources]
            self.logger.log_retrieval(
                num_chunks=len(sources),
                sources=source_names,
                latency_ms=query_time * 1000,
            )
            self.metrics.record_retrieval(len(sources))
            self.metrics.record_query("success", query_time)

            if use_cache and self.config.cache.enabled:
                self.cache.set(question, answer, [])

            result = QueryResult(
                answer=answer,
                sources=sources,
                confidence=confidence,
                confidence_label=label,
                confidence_explanation=explanation,
                query_time=query_time,
                cached=False,
                expanded_query=expanded_query,
            )

            def decomposed_gen() -> Generator[str, None, None]:
                yield answer

            return decomposed_gen(), lambda: result

        # -- Execute streaming query (via streaming engine) -------------
        response = self.query_engine_streaming.query(expanded_query)

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
            source_names: list[str] = []
            for node in response.source_nodes:
                fname = node.node.metadata.get("file_name", "unknown")
                sources.append({
                    "file_name": fname,
                    "score": (
                        node.score if hasattr(node, "score") else None
                    ),
                })
                source_names.append(fname)

            # Log retrieval metrics
            self.logger.log_retrieval(
                num_chunks=len(sources),
                sources=source_names,
                latency_ms=query_time * 1000,
            )
            self.metrics.record_retrieval(len(sources))
            self.metrics.record_query("success", query_time)

            # Cache the fully-collected response
            if use_cache and self.config.cache.enabled:
                self.cache.set(question, full_text, response.source_nodes)

            # Answer verification
            verification_warning = None
            if response.source_nodes:
                vr = self.verifier.verify(
                    full_text, response.source_nodes, question
                )
                if vr.confidence_penalty > 0:
                    confidence = max(0.0, confidence - vr.confidence_penalty)
                    label, _ = self._confidence_label(confidence)
                verification_warning = vr.warning

            return QueryResult(
                answer=full_text,
                sources=sources,
                confidence=confidence,
                confidence_label=label,
                confidence_explanation=explanation,
                query_time=query_time,
                cached=False,
                expanded_query=expanded_query,
                verification_warning=verification_warning,
            )

        return token_generator(), get_metadata

    # ---- code generation -----------------------------------------------

    def generate_code(self, description: str) -> CodeResult:
        """Generate a complete BakkesMod plugin project using RAG context.

        Uses the full pipeline: RAG retrieval -> feature detection ->
        LLM generation -> project scaffolding -> validation.

        Args:
            description: Natural language description of the desired plugin.

        Returns:
            A ``CodeResult`` containing all project files, validation
            results, and detected features.
        """
        result = self.code_gen.generate_full_plugin_with_rag(description)

        return CodeResult(
            header=result.get("header", ""),
            implementation=result.get("implementation", ""),
            project_files=result.get("project_files", {}),
            explanation=description,
            validation=result.get("validation", {}),
            features_used=result.get("features_used", []),
        )
