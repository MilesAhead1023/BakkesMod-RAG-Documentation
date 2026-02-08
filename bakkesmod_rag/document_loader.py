"""
Document Loader
================
Loads and parses all BakkesMod SDK documents from configured directories.
Supports .md, .h, and .cpp files with appropriate parsers.

Hybrid chunking pipeline:
  - Markdown: MarkdownNodeParser → subsplit oversized nodes with SemanticSplitter
  - Code (.h/.cpp): CodeSplitter (tree-sitter AST) for structure-aware boundaries
    Splits at function, class, struct, and namespace boundaries — zero API calls.
    Falls back to SentenceSplitter if tree-sitter-language-pack is unavailable.

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

import os
import logging
from typing import List, Optional

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode

from bakkesmod_rag.config import RAGConfig

logger = logging.getLogger("bakkesmod_rag.document_loader")


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

    print(f"[LOADER] Total nodes: {len(nodes)}")
    logger.info("Total nodes: %d", len(nodes))
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
