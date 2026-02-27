"""
Document Loader
================
Loads and parses all BakkesMod SDK documents from configured directories.
Supports .md, .h, and .cpp files with appropriate parsers.

Hybrid chunking pipeline:
  - Markdown: MarkdownNodeParser → subsplit oversized nodes with SemanticSplitter
    OR HierarchicalNodeParser for parent-child retrieval
  - Code (.h/.cpp): CodeSplitter (tree-sitter AST) for structure-aware boundaries
    Splits at function, class, struct, and namespace boundaries — zero API calls.
    Falls back to SentenceSplitter if tree-sitter-language-pack is unavailable.

Hierarchical chunking (use_hierarchical_chunking=True):
  - Creates parent nodes (512 tokens markdown / 1024 code) and
    child nodes (128 tokens markdown / 256 code) in the same storage context.
  - AutoMergingRetriever in retrieval.py merges child hits back to their parent
    when >merge_threshold fraction of a parent's children are retrieved.

CodeSplitter verified against LlamaIndex source:
  github.com/run-llama/llama_index/.../node_parser/text/code.py
  Import: from llama_index.core.node_parser import CodeSplitter
  Params: language="cpp", chunk_lines=40, chunk_lines_overlap=15, max_chars=1500
  Requires: tree-sitter-language-pack (160+ languages including C++)

SemanticSplitterNodeParser verified against LlamaIndex source:
  github.com/run-llama/llama_index/.../node_parser/text/semantic_splitter.py
  Import: from llama_index.core.node_parser import SemanticSplitterNodeParser
  Params: embed_model (required), buffer_size=1, breakpoint_percentile_threshold=95
"""

import hashlib
import os
import logging
import pickle
from pathlib import Path
from typing import List, Optional

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode

from bakkesmod_rag.config import RAGConfig

logger = logging.getLogger("bakkesmod_rag.document_loader")

NODE_CACHE_FILE = "node_cache.pkl"


def _compute_docs_fingerprint(documents: List[Document]) -> str:
    """Compute a hash fingerprint from document paths and modification times.

    Used to detect when source files have changed and node cache is stale.
    """
    entries = []
    for doc in sorted(documents, key=lambda d: d.metadata.get("file_path", "")):
        fpath = doc.metadata.get("file_path", "")
        try:
            mtime = os.path.getmtime(fpath) if fpath and os.path.exists(fpath) else 0
        except OSError:
            mtime = 0
        entries.append(f"{fpath}:{mtime}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()[:16]


def _load_cached_nodes(
    storage_dir: str, fingerprint: str
) -> Optional[List[BaseNode]]:
    """Load cached nodes if fingerprint matches."""
    cache_path = Path(storage_dir) / NODE_CACHE_FILE
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if data.get("fingerprint") == fingerprint:
            nodes = data["nodes"]
            logger.info("Loaded %d cached nodes (fingerprint match)", len(nodes))
            print(f"[LOADER] Loaded {len(nodes)} cached nodes (skipping parse + embeddings)")
            return nodes
        logger.info("Node cache fingerprint mismatch, will re-parse")
    except Exception as e:
        logger.warning("Failed to load node cache: %s", e)
    return None


def _save_cached_nodes(
    storage_dir: str, fingerprint: str, nodes: List[BaseNode]
) -> None:
    """Persist parsed nodes with their fingerprint."""
    cache_path = Path(storage_dir) / NODE_CACHE_FILE
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"fingerprint": fingerprint, "nodes": nodes}, f)
        logger.info("Saved %d nodes to cache", len(nodes))
    except Exception as e:
        logger.warning("Failed to save node cache: %s", e)


def _sanitize_text(text: str) -> str:
    """Remove non-printable characters while preserving whitespace."""
    return "".join(filter(lambda x: x.isprintable() or x in "\n\r\t", text))


def load_documents(config: Optional[RAGConfig] = None) -> List[Document]:
    """Load documents from all configured source directories.

    Loads .md, .h, and .cpp files from docs_bakkesmod_only/ and templates/.
    Cleans non-printable characters from all documents.

    Args:
        config: RAGConfig instance (optional, uses defaults if None).

    Returns:
        List of cleaned LlamaIndex Document objects.
    """
    if config is None:
        from bakkesmod_rag.config import get_config
        config = get_config()

    input_dirs = config.storage.docs_dirs
    required_exts = config.storage.required_exts
    all_documents = []

    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            logger.warning("Directory '%s' not found, skipping", input_dir)
            print(f"[LOADER] Directory '{input_dir}' not found, skipping")
            continue

        reader = SimpleDirectoryReader(
            input_dir=input_dir,
            required_exts=required_exts,
            recursive=True,
            filename_as_id=True,
        )
        docs = reader.load_data()
        print(f"[LOADER] Loaded {len(docs)} files from {input_dir}")
        logger.info("Loaded %d files from %s", len(docs), input_dir)
        all_documents.extend(docs)

    # Sanitize all documents
    cleaned_docs = []
    for doc in all_documents:
        clean_text = _sanitize_text(doc.text)
        cleaned_docs.append(Document(text=clean_text, metadata=doc.metadata))

    print(f"[LOADER] Total documents loaded: {len(cleaned_docs)}")
    logger.info("Total documents loaded: %d", len(cleaned_docs))
    return cleaned_docs


def parse_nodes_hierarchical(
    documents: List[Document],
    config: RAGConfig,
) -> List[BaseNode]:
    """Parse documents using HierarchicalNodeParser for parent-child relationships.

    Creates two levels of chunking:
      - Parent nodes: larger chunks (512 tokens markdown / 1024 tokens code)
      - Child nodes: smaller chunks (128 tokens markdown / 256 tokens code)

    Parent-child relationships are stored in node metadata so that
    AutoMergingRetriever can merge child hits back to parent context.

    Falls back to flat chunking if HierarchicalNodeParser is unavailable.

    Args:
        documents: List of Document objects to parse.
        config: RAGConfig instance.

    Returns:
        List of parsed nodes (all levels) ready for indexing.
    """
    try:
        from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

        md_docs = [
            d for d in documents
            if d.metadata.get("file_path", "").endswith(".md")
        ]
        code_docs = [
            d for d in documents
            if d.metadata.get("file_path", "").endswith((".h", ".cpp"))
        ]

        all_nodes: List[BaseNode] = []

        if md_docs:
            # Parent: 512 tokens, Child: 128 tokens for markdown
            md_parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[512, 128]
            )
            md_nodes = md_parser.get_nodes_from_documents(md_docs)
            all_nodes.extend(md_nodes)
            leaf_count = len(get_leaf_nodes(md_nodes))
            print(
                f"[LOADER] Hierarchical: {len(md_docs)} markdown docs -> "
                f"{len(md_nodes)} nodes ({leaf_count} leaf)"
            )
            logger.info(
                "Hierarchical markdown: %d docs -> %d nodes (%d leaf)",
                len(md_docs), len(md_nodes), leaf_count,
            )

        if code_docs:
            # Parent: 1024 tokens, Child: 256 tokens for code
            code_parser = HierarchicalNodeParser.from_defaults(
                chunk_sizes=[1024, 256]
            )
            code_nodes = code_parser.get_nodes_from_documents(code_docs)
            all_nodes.extend(code_nodes)
            leaf_count = len(get_leaf_nodes(code_nodes))
            print(
                f"[LOADER] Hierarchical: {len(code_docs)} code docs -> "
                f"{len(code_nodes)} nodes ({leaf_count} leaf)"
            )
            logger.info(
                "Hierarchical code: %d docs -> %d nodes (%d leaf)",
                len(code_docs), len(code_nodes), leaf_count,
            )

        return all_nodes

    except ImportError as e:
        logger.warning(
            "HierarchicalNodeParser unavailable (%s), falling back to flat chunking", e
        )
        return []


def parse_nodes(
    documents: List[Document],
    config: Optional[RAGConfig] = None,
    embed_model=None,
) -> List[BaseNode]:
    """Parse documents into nodes using appropriate parsers.

    Hybrid chunking pipeline:
      - Markdown: MarkdownNodeParser → subsplit oversized nodes with SemanticSplitter
      - Code (.h/.cpp): CodeSplitter (tree-sitter AST) for structure-aware boundaries
      - Fallback: SentenceSplitter if CodeSplitter/SemanticSplitter unavailable

    When use_hierarchical_chunking=True (default), uses HierarchicalNodeParser
    to build parent-child node relationships for AutoMergingRetriever.

    Results are cached to ``rag_storage/node_cache.pkl`` keyed by a fingerprint
    of source-file paths and modification times.  On subsequent startups with
    unchanged docs the cache is loaded directly, skipping all parsing and the
    42+ OpenAI embedding calls from SemanticSplitter.

    Args:
        documents: List of Document objects to parse.
        config: RAGConfig instance (optional, uses defaults if None).
        embed_model: Embedding model for semantic chunking of markdown (optional).

    Returns:
        List of parsed nodes ready for indexing.
    """
    if config is None:
        from bakkesmod_rag.config import get_config
        config = get_config()

    # -- Try loading from node cache ------------------------------------
    storage_dir = config.storage.storage_dir
    fingerprint = _compute_docs_fingerprint(documents)
    cached = _load_cached_nodes(storage_dir, fingerprint)
    if cached is not None:
        return cached

    # -- Hierarchical chunking path -------------------------------------
    if config.retriever.use_hierarchical_chunking:
        hier_nodes = parse_nodes_hierarchical(documents, config)
        if hier_nodes:
            if config.cpp_intelligence.enabled:
                _inject_cpp_metadata(hier_nodes, documents, config)
            print(f"[LOADER] Total nodes (hierarchical): {len(hier_nodes)}")
            logger.info("Total nodes (hierarchical): %d", len(hier_nodes))
            _save_cached_nodes(storage_dir, fingerprint, hier_nodes)
            return hier_nodes
        # Fallback if hierarchical parsing returned empty (import error)
        logger.warning("Hierarchical chunking returned no nodes, falling back to flat")

    md_docs = [
        d for d in documents
        if d.metadata.get("file_path", "").endswith(".md")
    ]
    code_docs = [
        d for d in documents
        if d.metadata.get("file_path", "").endswith((".h", ".cpp"))
    ]

    use_semantic = (
        config.chunking.enable_semantic_chunking
        and embed_model is not None
    )

    nodes = []

    # -- Markdown files --------------------------------------------------
    if md_docs:
        md_parser = MarkdownNodeParser()
        md_nodes = md_parser.get_nodes_from_documents(md_docs)
        print(f"[LOADER] Parsed {len(md_docs)} markdown files -> {len(md_nodes)} nodes")
        logger.info("Parsed %d markdown files -> %d nodes", len(md_docs), len(md_nodes))

        if use_semantic:
            md_nodes = _semantic_subsplit(md_nodes, embed_model, config)

        nodes.extend(md_nodes)

    # -- Code files (.h, .cpp) -------------------------------------------
    if code_docs:
        code_nodes = _parse_code_ast(code_docs, config)
        nodes.extend(code_nodes)
        print(f"[LOADER] Parsed {len(code_docs)} code files (.h/.cpp) -> {len(code_nodes)} nodes")
        logger.info("Parsed %d code files -> %d nodes", len(code_docs), len(code_nodes))

    # -- C++ structural metadata injection --------------------------------
    if config.cpp_intelligence.enabled:
        _inject_cpp_metadata(nodes, documents, config)

    print(f"[LOADER] Total nodes: {len(nodes)}")
    logger.info("Total nodes: %d", len(nodes))

    # -- Persist to cache for fast reload --------------------------------
    _save_cached_nodes(storage_dir, fingerprint, nodes)

    return nodes


def _get_semantic_splitter(embed_model, config: RAGConfig):
    """Create a SemanticSplitterNodeParser with config settings.

    Args:
        embed_model: Embedding model for similarity computation.
        config: RAGConfig instance.

    Returns:
        SemanticSplitterNodeParser instance.
    """
    from llama_index.core.node_parser import SemanticSplitterNodeParser

    return SemanticSplitterNodeParser(
        embed_model=embed_model,
        buffer_size=1,
        breakpoint_percentile_threshold=config.chunking.semantic_breakpoint_percentile,
        include_metadata=config.chunking.include_metadata,
        include_prev_next_rel=config.chunking.include_prev_next_rel,
    )


def _semantic_subsplit(
    nodes: List[BaseNode],
    embed_model,
    config: RAGConfig,
) -> List[BaseNode]:
    """Subsplit oversized markdown nodes using semantic boundaries.

    Nodes within the chunk_size limit pass through unchanged.
    Oversized nodes are re-split at semantic boundaries using the
    embedding model to find natural break points.

    Falls back to SentenceSplitter if SemanticSplitter fails.

    Args:
        nodes: Nodes from MarkdownNodeParser.
        embed_model: Embedding model for semantic similarity.
        config: RAGConfig instance.

    Returns:
        List of nodes with oversized ones subsplit.
    """
    max_chars = config.chunking.chunk_size
    small_nodes = [n for n in nodes if len(n.get_content()) <= max_chars]
    oversized = [n for n in nodes if len(n.get_content()) > max_chars]

    if not oversized:
        return nodes

    logger.info(
        "Subsplitting %d oversized markdown nodes (>%d chars)",
        len(oversized), max_chars,
    )

    try:
        splitter = _get_semantic_splitter(embed_model, config)
        # _parse_nodes works on existing nodes (not documents)
        split_nodes = splitter._parse_nodes(oversized, show_progress=False)
        print(
            f"[LOADER] Semantic subsplit: {len(oversized)} oversized -> "
            f"{len(split_nodes)} nodes"
        )
        logger.info(
            "Semantic subsplit: %d oversized -> %d nodes",
            len(oversized), len(split_nodes),
        )
        return small_nodes + split_nodes
    except Exception as e:
        logger.warning("Semantic subsplit failed, using SentenceSplitter: %s", e)
        fallback = SentenceSplitter(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
        split_nodes = fallback._parse_nodes(oversized, show_progress=False)
        print(
            f"[LOADER] Fallback subsplit: {len(oversized)} oversized -> "
            f"{len(split_nodes)} nodes"
        )
        return small_nodes + split_nodes


def _parse_code_ast(
    code_docs: List[Document],
    config: RAGConfig,
) -> List[BaseNode]:
    """Parse code files using tree-sitter AST for structure-aware boundaries.

    Uses CodeSplitter (built into llama-index-core) which leverages tree-sitter
    to split C++ at function, class, and struct boundaries. Zero API calls —
    all parsing is local. Verified against LlamaIndex source:
      github.com/run-llama/llama_index/.../node_parser/text/code.py
      Import: from llama_index.core.node_parser import CodeSplitter
      Requires: tree-sitter-language-pack (pip install tree-sitter-language-pack)

    Falls back to SentenceSplitter if tree-sitter is unavailable.

    Args:
        code_docs: Code document objects (.h, .cpp).
        config: RAGConfig instance.

    Returns:
        List of parsed code nodes.
    """
    try:
        from llama_index.core.node_parser import CodeSplitter

        splitter = CodeSplitter(
            language="cpp",
            chunk_lines=config.chunking.code_chunk_lines,
            chunk_lines_overlap=config.chunking.code_chunk_lines_overlap,
            max_chars=config.chunking.chunk_size,
        )
        code_nodes = splitter.get_nodes_from_documents(code_docs)
        print(f"[LOADER] AST-based code chunking (tree-sitter C++) enabled")
        logger.info("Code files parsed with CodeSplitter (tree-sitter AST)")
        return code_nodes
    except Exception as e:
        logger.warning("CodeSplitter failed, falling back to SentenceSplitter: %s", e)
        fallback = SentenceSplitter(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
        return fallback.get_nodes_from_documents(code_docs)


def _inject_cpp_metadata(
    nodes: List[BaseNode],
    documents: List[Document],
    config,
) -> None:
    """Inject C++ structural metadata into nodes from .h files.

    Runs the CppAnalyzer over all .h source documents, builds the full
    inheritance hierarchy, and attaches typed metadata (class name,
    base class, inheritance chain, method signatures, related types) to
    every node whose ``file_path`` matches an analyzed header.

    This enables the LLM to understand C++ relationships (inheritance,
    return types, parameter types) at retrieval time — not just text
    similarity.

    Args:
        nodes: All parsed nodes (modified in-place).
        documents: Original documents (used for source paths).
        config: RAGConfig instance.
    """
    try:
        from bakkesmod_rag.cpp_analyzer import CppAnalyzer
    except ImportError as e:
        logger.warning("CppAnalyzer unavailable, skipping C++ metadata: %s", e)
        return

    # Collect unique directories containing .h files
    h_dirs: set = set()
    for doc in documents:
        fp = doc.metadata.get("file_path", "")
        if fp.endswith(".h"):
            import os as _os
            h_dirs.add(_os.path.dirname(fp))

    if not h_dirs:
        return

    analyzer = CppAnalyzer()
    all_classes = {}
    for d in h_dirs:
        all_classes.update(analyzer.analyze_directory(d))

    if not all_classes:
        logger.info("No C++ classes found, skipping metadata injection")
        return

    enriched_count = 0
    for node in nodes:
        fp = node.metadata.get("file_path", "")
        if not fp.endswith(".h"):
            continue

        # Find which class(es) this node's file defines
        file_classes = [
            cls for cls in all_classes.values()
            if _paths_match(cls.file, fp)
        ]

        if not file_classes:
            continue

        # Use the primary class (usually one per file)
        cls = file_classes[0]
        cpp_meta = analyzer.format_metadata_for_node(cls, all_classes)
        node.metadata.update(cpp_meta)
        enriched_count += 1

    print(f"[LOADER] C++ metadata injected into {enriched_count} nodes "
          f"({len(all_classes)} classes analyzed)")
    logger.info(
        "C++ metadata injected into %d nodes (%d classes analyzed)",
        enriched_count, len(all_classes),
    )


def _paths_match(path_a: str, path_b: str) -> bool:
    """Check if two file paths refer to the same file (cross-platform)."""
    import os as _os
    try:
        return _os.path.normpath(path_a) == _os.path.normpath(path_b)
    except (TypeError, ValueError):
        return str(path_a) == str(path_b)
