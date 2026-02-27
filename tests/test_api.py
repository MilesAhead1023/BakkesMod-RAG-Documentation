"""Tests for the FastAPI HTTP layer (Gap 8: Horizontal Scaling).

Uses fastapi.testclient.TestClient so no real server or RAGEngine is needed.
All RAGEngine calls are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Skip entire module if fastapi is not installed
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi", reason="fastapi not installed")
pytest.importorskip("httpx", reason="httpx not installed (required by TestClient)")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_engine():
    """Return a mock RAGEngine with realistic return values."""
    engine = MagicMock()
    engine.num_documents = 215
    engine.num_nodes = 1800

    # mock QueryResult
    qr = MagicMock()
    qr.answer = "BakkesMod is a Rocket League modding framework."
    qr.sources = [{"file_name": "README.md", "score": 0.9}]
    qr.confidence = 0.85
    qr.confidence_label = "HIGH"
    qr.confidence_explanation = "Strong retrieval support"
    qr.query_time = 0.5
    qr.cached = False
    qr.expanded_query = "BakkesMod framework"
    qr.retry_count = 0
    engine.query.return_value = qr

    # mock CodeResult
    cr = MagicMock()
    cr.header = "#pragma once\nclass MyPlugin : public BakkesMod::Plugin::BakkesModPlugin {};"
    cr.implementation = "#include \"MyPlugin.h\""
    cr.project_files = {"MyPlugin.h": cr.header, "MyPlugin.cpp": cr.implementation}
    cr.explanation = "Test plugin"
    cr.validation = {"valid": True, "errors": [], "warnings": []}
    cr.features_used = ["basic_plugin"]
    cr.fix_iterations = 0
    engine.generate_code.return_value = cr

    # mock cost_tracker, cache, api_manager
    engine.cost_tracker.costs = {"total": 0.001, "by_provider": {"openai": 0.001}}
    engine.cache.stats.return_value = {"total_entries": 5, "valid_entries": 5}
    engine.api_manager.get_status.return_value = {"openai": "closed"}

    return engine


@pytest.fixture
def client(mock_engine):
    """Create a TestClient with the mock engine injected."""
    from fastapi.testclient import TestClient
    from bakkesmod_rag.api import create_app
    import bakkesmod_rag.api as api_module

    app = create_app()

    # Inject mock engine
    api_module._engine = mock_engine

    yield TestClient(app)

    # Cleanup
    api_module._engine = None


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_returns_num_documents(self, client):
        data = client.get("/health").json()
        assert "num_documents" in data
        assert data["num_documents"] == 215

    def test_health_returns_num_nodes(self, client):
        data = client.get("/health").json()
        assert "num_nodes" in data
        assert data["num_nodes"] == 1800


# ---------------------------------------------------------------------------
# /metrics endpoint
# ---------------------------------------------------------------------------

class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_has_cost_key(self, client):
        data = client.get("/metrics").json()
        assert "cost" in data

    def test_metrics_has_cache_key(self, client):
        data = client.get("/metrics").json()
        assert "cache" in data

    def test_metrics_has_circuit_breakers_key(self, client):
        data = client.get("/metrics").json()
        assert "circuit_breakers" in data


# ---------------------------------------------------------------------------
# /query endpoint
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_query_returns_200(self, client):
        response = client.post("/query", json={"question": "What is BakkesMod?"})
        assert response.status_code == 200

    def test_query_returns_answer(self, client):
        data = client.post(
            "/query", json={"question": "What is BakkesMod?"}
        ).json()
        assert "answer" in data
        assert "BakkesMod" in data["answer"]

    def test_query_returns_confidence(self, client):
        data = client.post(
            "/query", json={"question": "What is BakkesMod?"}
        ).json()
        assert "confidence" in data
        assert data["confidence"] == 0.85

    def test_query_returns_sources(self, client):
        data = client.post(
            "/query", json={"question": "What is BakkesMod?"}
        ).json()
        assert "sources" in data
        assert isinstance(data["sources"], list)

    def test_query_use_cache_defaults_true(self, client, mock_engine):
        client.post("/query", json={"question": "test"})
        mock_engine.query.assert_called_once_with("test", use_cache=True)

    def test_query_use_cache_false(self, client, mock_engine):
        client.post("/query", json={"question": "test", "use_cache": False})
        mock_engine.query.assert_called_with("test", use_cache=False)


# ---------------------------------------------------------------------------
# /generate endpoint
# ---------------------------------------------------------------------------

class TestGenerateEndpoint:
    def test_generate_returns_200(self, client):
        response = client.post(
            "/generate", json={"description": "A plugin that shows boost"}
        )
        assert response.status_code == 200

    def test_generate_returns_header(self, client):
        data = client.post(
            "/generate", json={"description": "A plugin that shows boost"}
        ).json()
        assert "header" in data
        assert len(data["header"]) > 0

    def test_generate_returns_implementation(self, client):
        data = client.post(
            "/generate", json={"description": "A plugin that shows boost"}
        ).json()
        assert "implementation" in data

    def test_generate_returns_project_files(self, client):
        data = client.post(
            "/generate", json={"description": "A plugin that shows boost"}
        ).json()
        assert "project_files" in data
        assert isinstance(data["project_files"], dict)

    def test_generate_returns_validation(self, client):
        data = client.post(
            "/generate", json={"description": "A plugin that shows boost"}
        ).json()
        assert "validation" in data


# ---------------------------------------------------------------------------
# API module availability
# ---------------------------------------------------------------------------

class TestAPIModule:
    def test_api_module_importable(self):
        """bakkesmod_rag.api is importable."""
        import bakkesmod_rag.api as api
        assert api is not None

    def test_create_app_returns_fastapi_app(self):
        """create_app() returns a FastAPI instance."""
        from fastapi import FastAPI
        from bakkesmod_rag.api import create_app
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_redis_circuit_breaker_importable(self):
        """RedisCircuitBreaker is importable from resilience."""
        from bakkesmod_rag.resilience import RedisCircuitBreaker
        assert RedisCircuitBreaker is not None

    def test_redis_cost_tracker_importable(self):
        """RedisCostTracker is importable from cost_tracker."""
        from bakkesmod_rag.cost_tracker import RedisCostTracker
        assert RedisCostTracker is not None
