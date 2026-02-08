"""
Document Loader
================
Loads and parses all BakkesMod SDK documents from configured directories.
Supports .md, .h, and .cpp files with appropriate parsers.
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


def parse_nodes(documents: List[Document], config: Optional[RAGConfig] = None) -> List[BaseNode]:
    """Parse documents into nodes using appropriate parsers.

    Markdown files (.md) use MarkdownNodeParser for structure-aware chunking.
    Code files (.h, .cpp) use SentenceSplitter with 1024/128 chunk settings.

    Args:
        documents: List of Document objects to parse.
        config: RAGConfig instance (optional, uses defaults if None).

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

    nodes = []

    if md_docs:
        md_parser = MarkdownNodeParser()
        md_nodes = md_parser.get_nodes_from_documents(md_docs)
        nodes.extend(md_nodes)
        print(f"[LOADER] Parsed {len(md_docs)} markdown files -> {len(md_nodes)} nodes")
        logger.info("Parsed %d markdown files -> %d nodes", len(md_docs), len(md_nodes))

    if code_docs:
        code_parser = SentenceSplitter(
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap,
        )
        code_nodes = code_parser.get_nodes_from_documents(code_docs)
        nodes.extend(code_nodes)
        print(f"[LOADER] Parsed {len(code_docs)} code files (.h/.cpp) -> {len(code_nodes)} nodes")
        logger.info(
            "Parsed %d code files -> %d nodes", len(code_docs), len(code_nodes)
        )

    print(f"[LOADER] Total nodes: {len(nodes)}")
    logger.info("Total nodes: %d", len(nodes))
    return nodes
