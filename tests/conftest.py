"""Shared test fixtures for the BakkesMod RAG test suite.

Provides mock LLMs, configs, embeddings, and source nodes that all test
files can reuse.  Unit tests should never need API keys.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Optional

from bakkesmod_rag.config import RAGConfig


# ---------------------------------------------------------------------------
# Mock LLM
# ---------------------------------------------------------------------------

class MockLLMResponse:
    """Simulates a LlamaIndex LLM completion response."""

    def __init__(self, text: str):
        self.text = text


class MockLLM:
    """A fake LLM that returns a predetermined response.

    Usage in tests::

        llm = MockLLM("Hello world")
        resp = llm.complete("anything")
        assert resp.text == "Hello world"
    """

    def __init__(self, response_text: str = "OK"):
        self.response_text = response_text
        self.call_count = 0

    def complete(self, prompt: str, **kwargs):
        self.call_count += 1
        return MockLLMResponse(self.response_text)


# ---------------------------------------------------------------------------
# Mock Embedding Model
# ---------------------------------------------------------------------------

class MockEmbedModel:
    """A fake embedding model that returns fixed-length vectors."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension

    def get_text_embedding(self, text: str) -> List[float]:
        """Return a deterministic embedding based on text hash."""
        import hashlib
        h = hashlib.md5(text.encode()).digest()
        # Generate a reproducible vector from hash bytes
        vec = [float(b) / 255.0 for b in h]
        # Pad or truncate to the right dimension
        while len(vec) < self.dimension:
            vec.extend(vec[:self.dimension - len(vec)])
        return vec[:self.dimension]


# ---------------------------------------------------------------------------
# Mock Source Nodes (for confidence scoring tests)
# ---------------------------------------------------------------------------

@dataclass
class MockNodeMetadata:
    file_name: str = "test_file.h"


@dataclass
class MockNode:
    metadata: dict

    def __init__(self, file_name: str = "test_file.h"):
        self.metadata = {"file_name": file_name}


@dataclass
class MockSourceNode:
    """Mimics a LlamaIndex NodeWithScore."""
    node: MockNode
    score: Optional[float] = None

    def __init__(self, score: float = 0.8, file_name: str = "test_file.h"):
        self.node = MockNode(file_name=file_name)
        self.score = score


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """Return a MockLLM with a simple OK response."""
    return MockLLM("OK")


@pytest.fixture
def mock_embed_model():
    """Return a MockEmbedModel with default dimensions."""
    return MockEmbedModel(dimension=1536)


@pytest.fixture
def test_config():
    """Return a RAGConfig with default settings (no API keys needed)."""
    return RAGConfig(
        openai_api_key="test-key",
        anthropic_api_key=None,
        google_api_key=None,
        openrouter_api_key=None,
        cohere_api_key=None,
    )


@pytest.fixture
def source_nodes_high():
    """Return mock source nodes with high scores."""
    return [
        MockSourceNode(score=0.95, file_name="plugin.h"),
        MockSourceNode(score=0.90, file_name="GameWrapper.h"),
        MockSourceNode(score=0.88, file_name="CarWrapper.h"),
        MockSourceNode(score=0.85, file_name="hooks_reference.md"),
        MockSourceNode(score=0.82, file_name="getting_started.md"),
    ]


@pytest.fixture
def source_nodes_low():
    """Return mock source nodes with low scores."""
    return [
        MockSourceNode(score=0.30, file_name="random_file.md"),
        MockSourceNode(score=0.25, file_name="unrelated.h"),
    ]


@pytest.fixture
def source_nodes_empty():
    """Return an empty list of source nodes."""
    return []


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Return a temporary cache directory."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir
