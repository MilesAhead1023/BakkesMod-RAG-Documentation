"""Tests for RAGConfig: defaults, validation, env var loading."""

import os
import pytest
from bakkesmod_rag.config import (
    RAGConfig,
    EmbeddingConfig,
    LLMConfig,
    RetrieverConfig,
    ChunkingConfig,
    CacheConfig,
    ObservabilityConfig,
    CostConfig,
    ProductionConfig,
    StorageConfig,
    CodeGenConfig,
    get_config,
    reload_config,
)


class TestDefaults:
    def test_default_embedding_config(self):
        cfg = EmbeddingConfig()
        assert cfg.provider == "openai"
        assert cfg.model == "text-embedding-3-small"
        assert cfg.dimension == 1536

    def test_default_llm_config(self):
        cfg = LLMConfig()
        assert cfg.primary_provider == "anthropic"
        assert cfg.temperature == 0.0
        assert "openrouter" in cfg.fallback_providers

    def test_default_retriever_config(self):
        cfg = RetrieverConfig()
        assert cfg.vector_top_k == 5
        assert cfg.enable_kg is True
        assert cfg.fusion_mode == "reciprocal_rerank"
        assert cfg.fusion_num_queries == 4

    def test_default_cache_config(self):
        cfg = CacheConfig()
        assert cfg.enabled is True
        assert cfg.similarity_threshold == 0.92
        assert cfg.ttl_seconds == 86400 * 7

    def test_default_chunking_config(self):
        cfg = ChunkingConfig()
        assert cfg.chunk_size == 1024
        assert cfg.chunk_overlap == 128

    def test_default_production_config(self):
        cfg = ProductionConfig()
        assert cfg.rate_limit_enabled is True
        assert cfg.circuit_breaker_enabled is True
        assert cfg.failure_threshold == 5

    def test_default_storage_config(self):
        cfg = StorageConfig()
        assert "docs_bakkesmod_only" in cfg.docs_dirs
        assert "templates" in cfg.docs_dirs
        assert ".md" in cfg.required_exts

    def test_default_codegen_config(self):
        cfg = CodeGenConfig()
        assert cfg.enabled is True
        assert cfg.validate_output is True


class TestRAGConfig:
    def test_creates_with_defaults(self):
        cfg = RAGConfig(openai_api_key="test")
        assert cfg.openai_api_key == "test"
        assert cfg.embedding.model == "text-embedding-3-small"
        assert cfg.llm.primary_provider == "anthropic"

    def test_embedding_dimension_auto_correction(self):
        cfg = EmbeddingConfig(model="text-embedding-3-small", dimension=999)
        assert cfg.dimension == 1536  # auto-corrected

    def test_large_embedding_dimension(self):
        cfg = EmbeddingConfig(model="text-embedding-3-large", dimension=999)
        assert cfg.dimension == 3072


class TestSingleton:
    def test_get_config_returns_instance(self):
        import bakkesmod_rag.config as cfg_mod
        cfg_mod._config = None  # reset singleton

        cfg = get_config()
        assert isinstance(cfg, RAGConfig)

        # Same instance returned
        cfg2 = get_config()
        assert cfg is cfg2

        cfg_mod._config = None  # cleanup

    def test_reload_config_creates_new_instance(self):
        import bakkesmod_rag.config as cfg_mod
        cfg_mod._config = None

        cfg1 = get_config()
        cfg2 = reload_config()
        assert cfg1 is not cfg2

        cfg_mod._config = None  # cleanup


class TestCostConfig:
    def test_default_cost_rates(self):
        cfg = CostConfig()
        assert cfg.openai_embedding_cost == 0.02
        assert cfg.openai_gpt4o_mini_input == 0.15
        assert cfg.anthropic_claude_sonnet_input == 3.0
        assert cfg.gemini_flash_input == 0.075

    def test_alert_threshold(self):
        cfg = CostConfig()
        assert cfg.alert_threshold_pct == 80.0
