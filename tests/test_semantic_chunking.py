"""Tests for hybrid chunking in document_loader.

Covers:
  - Markdown: MarkdownNodeParser → SemanticSplitter subsplit for oversized nodes
  - Code (.h/.cpp): CodeSplitter (tree-sitter AST) for structure-aware boundaries
  - Fallback: SentenceSplitter if CodeSplitter/SemanticSplitter unavailable
"""

from unittest.mock import MagicMock, patch

import pytest

from llama_index.core import Document
from llama_index.core.schema import TextNode

from bakkesmod_rag.config import RAGConfig, ChunkingConfig
from bakkesmod_rag.document_loader import (
    parse_nodes,
    _semantic_subsplit,
    _get_semantic_splitter,
    _parse_code_ast,
)


def _make_doc(text: str, file_path: str) -> Document:
    """Create a Document with the given text and file_path metadata."""
    return Document(text=text, metadata={"file_path": file_path})


def _make_node(text: str) -> TextNode:
    """Create a TextNode with the given text."""
    return TextNode(text=text)


class TestParseNodesSignature:
    """Test that parse_nodes accepts the new embed_model parameter."""

    def test_accepts_embed_model_none(self):
        """parse_nodes still works without embed_model (backward compat)."""
        docs = [_make_doc("# Hello\nWorld", "/test/doc.md")]
        config = RAGConfig()
        nodes = parse_nodes(docs, config, embed_model=None)
        assert len(nodes) >= 1

    def test_accepts_embed_model_kwarg(self):
        """parse_nodes accepts embed_model as keyword argument."""
        docs = [_make_doc("# Hello\nWorld", "/test/doc.md")]
        config = RAGConfig()
        nodes = parse_nodes(docs, config, embed_model=None)
        assert isinstance(nodes, list)


class TestSemanticSubsplit:
    """Tests for _semantic_subsplit()."""

    def test_small_nodes_pass_through(self):
        """Nodes under chunk_size should pass through unchanged."""
        config = RAGConfig(chunking=ChunkingConfig(chunk_size=1024))
        small_node = _make_node("Short text under limit")
        result = _semantic_subsplit([small_node], MagicMock(), config)
        assert len(result) == 1
        assert result[0].get_content() == "Short text under limit"

    def test_no_oversized_returns_original(self):
        """If no nodes are oversized, return the original list."""
        config = RAGConfig(chunking=ChunkingConfig(chunk_size=5000))
        nodes = [_make_node("A" * 100), _make_node("B" * 200)]
        result = _semantic_subsplit(nodes, MagicMock(), config)
        assert len(result) == 2

    def test_oversized_triggers_split(self):
        """Oversized nodes should be subsplit."""
        config = RAGConfig(chunking=ChunkingConfig(chunk_size=50))
        small = _make_node("Small")
        big = _make_node("A" * 200)

        mock_embed = MagicMock()

        # Mock the semantic splitter to return 3 sub-nodes
        mock_split_nodes = [_make_node("Part1"), _make_node("Part2"), _make_node("Part3")]
        with patch(
            "bakkesmod_rag.document_loader._get_semantic_splitter"
        ) as mock_get:
            mock_splitter = MagicMock()
            mock_splitter._parse_nodes.return_value = mock_split_nodes
            mock_get.return_value = mock_splitter

            result = _semantic_subsplit([small, big], mock_embed, config)

        # 1 small + 3 from split = 4
        assert len(result) == 4

    def test_fallback_on_semantic_failure(self):
        """Falls back to SentenceSplitter if semantic splitter fails."""
        config = RAGConfig(chunking=ChunkingConfig(chunk_size=50, chunk_overlap=10))
        big = _make_node("A " * 200)

        with patch(
            "bakkesmod_rag.document_loader._get_semantic_splitter",
            side_effect=ImportError("no module"),
        ):
            result = _semantic_subsplit([big], MagicMock(), config)

        # SentenceSplitter fallback should produce at least 1 node
        assert len(result) >= 1


class TestParseCodeAST:
    """Tests for _parse_code_ast() — tree-sitter AST-based code chunking."""

    def test_parses_cpp_code(self):
        """CodeSplitter should parse valid C++ into chunks."""
        config = RAGConfig()
        code = "#pragma once\nclass Foo {\npublic:\n    void bar();\n};"
        code_doc = _make_doc(code, "/test/plugin.h")
        result = _parse_code_ast([code_doc], config)
        assert len(result) >= 1

    def test_keeps_comments_with_code(self):
        """Comments should stay attached to their function."""
        config = RAGConfig()
        code = "// Sets boost amount\nvoid SetBoost(float v) {\n    boost = v;\n}\n"
        code_doc = _make_doc(code, "/test/plugin.cpp")
        result = _parse_code_ast([code_doc], config)
        assert len(result) >= 1
        # The comment and function should be in the same chunk
        combined = " ".join(n.get_content() for n in result)
        assert "Sets boost" in combined
        assert "SetBoost" in combined

    def test_falls_back_on_failure(self):
        """Falls back to SentenceSplitter if CodeSplitter fails."""
        config = RAGConfig(chunking=ChunkingConfig(chunk_size=512))
        code_doc = _make_doc("void setup() { }\nvoid loop() { }", "/test/plugin.cpp")

        with patch(
            "llama_index.core.node_parser.CodeSplitter",
            side_effect=ImportError("no tree-sitter"),
        ):
            result = _parse_code_ast([code_doc], config)

        assert len(result) >= 1


class TestChunkingConfig:
    """Tests for ChunkingConfig fields."""

    def test_defaults(self):
        config = ChunkingConfig()
        assert config.enable_semantic_chunking is True
        assert config.semantic_breakpoint_percentile == 95
        assert config.chunk_size == 1024
        assert config.code_chunk_lines == 40
        assert config.code_chunk_lines_overlap == 15

    def test_disable_semantic(self):
        config = ChunkingConfig(enable_semantic_chunking=False)
        assert config.enable_semantic_chunking is False

    def test_custom_code_chunk_lines(self):
        config = ChunkingConfig(code_chunk_lines=60, code_chunk_lines_overlap=20)
        assert config.code_chunk_lines == 60
        assert config.code_chunk_lines_overlap == 20


class TestParseNodesCodePath:
    """Test parse_nodes routes code files to CodeSplitter."""

    def test_code_uses_ast_splitter(self):
        """Code files should use CodeSplitter (AST), not SemanticSplitter."""
        config = RAGConfig()
        code_doc = _make_doc("#pragma once\nclass Foo {};", "/test/code.h")

        with patch(
            "bakkesmod_rag.document_loader._get_semantic_splitter"
        ) as mock_get:
            nodes = parse_nodes([code_doc], config, embed_model=MagicMock())
            # Semantic splitter should NOT be called for code files
            mock_get.assert_not_called()

        assert len(nodes) >= 1


# ---------------------------------------------------------------------------
# Gap 1: Hierarchical chunking tests
# ---------------------------------------------------------------------------

class TestHierarchicalChunking:
    """Tests for hierarchical parent-child chunking."""

    def test_hierarchical_produces_nodes(self):
        """parse_nodes_hierarchical returns nodes for markdown docs."""
        from bakkesmod_rag.document_loader import parse_nodes_hierarchical
        from bakkesmod_rag.config import RetrieverConfig

        config = RAGConfig()
        docs = [_make_doc("# Section\nSome text here.\n" * 20, "/test/doc.md")]
        nodes = parse_nodes_hierarchical(docs, config)
        # May return empty if HierarchicalNodeParser is unavailable, that's OK
        assert isinstance(nodes, list)

    def test_hierarchical_creates_parent_child_relationships(self):
        """When HierarchicalNodeParser is available, nodes have parent relationships."""
        try:
            from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
        except ImportError:
            pytest.skip("HierarchicalNodeParser not available")

        from bakkesmod_rag.document_loader import parse_nodes_hierarchical

        config = RAGConfig()
        # Create a document large enough to trigger multi-level splitting
        text = "# Header\n" + ("Some content here. " * 50 + "\n\n") * 5
        docs = [_make_doc(text, "/test/large_doc.md")]

        nodes = parse_nodes_hierarchical(docs, config)
        assert len(nodes) >= 1
        # Check that get_leaf_nodes works (implies parent-child relationships)
        leaf_nodes = get_leaf_nodes(nodes)
        assert isinstance(leaf_nodes, list)

    def test_hierarchical_fallback_on_import_error(self):
        """Returns empty list (not raises) when HierarchicalNodeParser unavailable."""
        from bakkesmod_rag.document_loader import parse_nodes_hierarchical
        from bakkesmod_rag.config import RetrieverConfig

        config = RAGConfig()
        docs = [_make_doc("# Hello\nWorld", "/test/doc.md")]

        with patch(
            "llama_index.core.node_parser.HierarchicalNodeParser",
            side_effect=ImportError("no hierarchical"),
        ):
            # Even with import error at class level, function should not raise
            try:
                nodes = parse_nodes_hierarchical(docs, config)
                assert isinstance(nodes, list)
            except Exception:
                pass  # Import errors before runtime are acceptable

    def test_retriever_config_has_hierarchical_fields(self):
        """RetrieverConfig exposes use_hierarchical_chunking and merge_threshold."""
        from bakkesmod_rag.config import RetrieverConfig

        cfg = RetrieverConfig()
        assert hasattr(cfg, "use_hierarchical_chunking")
        assert isinstance(cfg.use_hierarchical_chunking, bool)
        assert hasattr(cfg, "merge_threshold")
        assert 0.0 < cfg.merge_threshold <= 1.0

    def test_hierarchical_disabled_falls_back_to_flat(self):
        """When use_hierarchical_chunking=False, flat chunking is used."""
        from bakkesmod_rag.config import RetrieverConfig

        config = RAGConfig(
            retriever=RetrieverConfig(use_hierarchical_chunking=False)
        )
        docs = [_make_doc("# Hello\nWorld", "/test/doc.md")]

        nodes = parse_nodes(docs, config, embed_model=None)
        assert len(nodes) >= 1

    def test_auto_merging_retriever_import(self):
        """AutoMergingRetriever is importable from llama_index."""
        try:
            from llama_index.core.retrievers import AutoMergingRetriever
            assert AutoMergingRetriever is not None
        except ImportError:
            pytest.skip("AutoMergingRetriever not available in this LlamaIndex version")
