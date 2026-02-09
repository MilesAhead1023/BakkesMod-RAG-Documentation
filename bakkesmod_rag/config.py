"""
RAG Configuration
=================
Centralized configuration management with validation.
"""

import os
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    provider: Literal["openai", "huggingface"] = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100
    max_retries: int = 10

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v, info):
        model = info.data.get("model", "")
        if "text-embedding-3-small" in model:
            return 1536
        elif "text-embedding-3-large" in model:
            return 3072
        return v


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    primary_provider: Literal["openai", "anthropic", "gemini", "openrouter"] = "anthropic"
    primary_model: str = "claude-sonnet-4-5"

    fallback_providers: list[Literal["openai", "anthropic", "gemini", "openrouter"]] = [
        "openrouter", "gemini", "openai"
    ]
    fallback_models: dict[str, str] = {
        "openrouter": "deepseek/deepseek-chat-v3-0324",
        "gemini": "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-5",
    }

    kg_provider: Literal["openai", "anthropic", "gemini"] = "openai"
    kg_model: str = "gpt-4o-mini"

    temperature: float = 0.0
    max_retries: int = 5
    timeout: int = 60


class RetrieverConfig(BaseModel):
    """Configuration for retrieval components."""
    vector_top_k: int = 5
    enable_kg: bool = True
    kg_similarity_top_k: int = 3
    kg_max_triplets_per_chunk: int = 2
    bm25_top_k: int = 5

    fusion_mode: Literal["reciprocal_rerank", "simple"] = "reciprocal_rerank"
    fusion_num_queries: int = 4

    enable_reranker: bool = True
    reranker_preference: list[Literal["bge", "flashrank", "cohere"]] = [
        "bge", "flashrank", "cohere"
    ]
    reranker_model: str = "rerank-english-v3.0"  # Cohere model
    bge_reranker_model: str = "BAAI/bge-reranker-v2-m3"
    flashrank_model: str = "ms-marco-MiniLM-L-12-v2"
    rerank_top_n: int = 5

    enable_llm_rewrite: bool = True

    # Query decomposition
    enable_query_decomposition: bool = True
    max_sub_queries: int = 4
    decomposition_complexity_threshold: int = 80


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""
    chunk_size: int = 1024
    chunk_overlap: int = 128
    include_metadata: bool = True
    include_prev_next_rel: bool = True
    enable_semantic_chunking: bool = True
    semantic_breakpoint_percentile: int = 95
    # CodeSplitter (tree-sitter AST) settings for .h/.cpp files
    code_chunk_lines: int = 40
    code_chunk_lines_overlap: int = 15


class CacheConfig(BaseModel):
    """Configuration for semantic caching."""
    enabled: bool = True
    similarity_threshold: float = 0.92
    ttl_seconds: int = 86400 * 7  # 7 days
    cache_dir: str = ".cache/semantic"


class ObservabilityConfig(BaseModel):
    """Configuration for monitoring and observability."""
    enabled: bool = True
    phoenix_enabled: bool = False
    phoenix_host: str = "127.0.0.1"
    phoenix_port: int = 6006
    prometheus_enabled: bool = False
    prometheus_port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: str = "json"


class CostConfig(BaseModel):
    """Configuration for cost tracking."""
    track_costs: bool = True
    daily_budget_usd: Optional[float] = Field(
        default_factory=lambda: (
            float(os.getenv("DAILY_BUDGET_USD"))
            if os.getenv("DAILY_BUDGET_USD")
            else None
        )
    )
    alert_threshold_pct: float = 80.0
    openai_embedding_cost: float = 0.02  # text-embedding-3-small
    openai_gpt4o_mini_input: float = 0.15
    openai_gpt4o_mini_output: float = 0.60
    anthropic_claude_sonnet_input: float = 3.0
    anthropic_claude_sonnet_output: float = 15.0
    gemini_flash_input: float = 0.075
    gemini_flash_output: float = 0.30


class ProductionConfig(BaseModel):
    """Configuration for production resilience."""
    rate_limit_enabled: bool = True
    requests_per_minute: int = 60
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5
    recovery_timeout: int = 60
    max_retries: int = 3
    retry_backoff_factor: float = 2.0
    retry_jitter: bool = True


class StorageConfig(BaseModel):
    """Configuration for storage paths."""
    docs_dirs: list[str] = ["docs_bakkesmod_only", "templates"]
    required_exts: list[str] = [".md", ".h", ".cpp"]
    storage_dir: str = "rag_storage"
    cache_dir: str = ".cache"
    logs_dir: str = "logs"


class CodeGenConfig(BaseModel):
    """Configuration for code generation."""
    enabled: bool = True
    max_context_chunks: int = 5
    validate_output: bool = True


class RAGConfig(BaseModel):
    """Complete RAG system configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    google_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    openrouter_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY"))
    cohere_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("COHERE_API_KEY"))

    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    production: ProductionConfig = Field(default_factory=ProductionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    codegen: CodeGenConfig = Field(default_factory=CodeGenConfig)


# Singleton
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def reload_config() -> RAGConfig:
    """Reload configuration from environment."""
    global _config
    load_dotenv(override=True)
    _config = RAGConfig()
    return _config
