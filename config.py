"""
2026 Gold Standard RAG Configuration
=====================================
Centralized configuration management with validation, cost tracking, and observability.
"""

import os
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""
    provider: Literal["openai", "huggingface"] = "openai"
    model: str = "text-embedding-3-large"
    dimension: int = 3072  # text-embedding-3-large produces 3072-dim vectors
    batch_size: int = 100
    max_retries: int = 10
    
    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v, info):
        """Ensure dimension matches model."""
        model = info.data.get("model", "")
        if "text-embedding-3-small" in model:
            return 1536
        elif "text-embedding-3-large" in model:
            return 3072
        return v


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    # Primary LLM for query engine
    primary_provider: Literal["openai", "anthropic", "gemini"] = "anthropic"
    primary_model: str = "claude-3-5-sonnet-20240620"

    # LLM for knowledge graph extraction (should be fast and cheap)
    kg_provider: Literal["openai", "anthropic", "gemini"] = "openai"
    kg_model: str = "gpt-4o-mini"

    # Reranker LLM
    rerank_provider: Literal["openai", "anthropic"] = "openai"
    rerank_model: str = "gpt-4o-mini"
    
    temperature: float = 0.0
    max_retries: int = 5
    timeout: int = 60  # seconds


class RetrieverConfig(BaseModel):
    """Configuration for retrieval components."""
    # Vector retrieval
    vector_top_k: int = 10

    # Knowledge graph retrieval
    enable_kg: bool = True
    kg_similarity_top_k: int = 3
    kg_max_triplets_per_chunk: int = 2
    kg_include_embeddings: bool = True
    
    # BM25 retrieval
    bm25_top_k: int = 10
    
    # Fusion retrieval
    fusion_mode: Literal["reciprocal_rerank", "simple"] = "reciprocal_rerank"
    fusion_num_queries: int = 1
    
    # Reranker (Phase 2: Neural reranking)
    enable_reranker: bool = True  # Enable Cohere reranker
    reranker_model: str = "rerank-english-v3.0"  # Cohere rerank model
    rerank_top_n: int = 5  # Return top 5 after reranking
    rerank_batch_size: int = 10  # Rerank top 10 from retrieval


class ChunkingConfig(BaseModel):
    """Configuration for document chunking strategy."""
    chunk_size: int = 1024  # tokens
    chunk_overlap: int = 200  # tokens
    parser: Literal["markdown", "sentence", "semantic"] = "markdown"
    
    # Markdown-specific settings
    include_metadata: bool = True
    include_prev_next_rel: bool = True


class CacheConfig(BaseModel):
    """Configuration for semantic caching."""
    enabled: bool = True
    similarity_threshold: float = 0.9  # 90% similarity = cache hit
    backend: Literal["faiss", "redis"] = "faiss"
    ttl: int = 86400  # 24 hours in seconds
    max_size: int = 10000  # maximum cache entries


class ObservabilityConfig(BaseModel):
    """Configuration for monitoring and observability."""
    phoenix_enabled: bool = True
    phoenix_host: str = "127.0.0.1"
    phoenix_port: int = 6006
    
    # Prometheus metrics
    prometheus_enabled: bool = True
    prometheus_port: int = 8000
    
    # Structured logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: str = "json"  # or "text"


class CostConfig(BaseModel):
    """Configuration for cost tracking and budgeting."""
    track_costs: bool = True
    daily_budget_usd: Optional[float] = None  # None = unlimited
    alert_threshold_pct: float = 80.0  # Alert at 80% of budget
    
    # Cost per 1M tokens (update these regularly)
    openai_embedding_cost: float = 0.13  # text-embedding-3-large
    openai_gpt4o_mini_input: float = 0.15
    openai_gpt4o_mini_output: float = 0.60
    anthropic_claude_sonnet_input: float = 3.0
    anthropic_claude_sonnet_output: float = 15.0
    gemini_flash_input: float = 0.075
    gemini_flash_output: float = 0.30


class ProductionConfig(BaseModel):
    """Configuration for production deployment."""
    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = 60
    
    # Circuit breaker
    circuit_breaker_enabled: bool = True
    failure_threshold: int = 5  # trips after 5 failures
    recovery_timeout: int = 60  # seconds
    
    # Retry strategy
    max_retries: int = 3
    retry_backoff_factor: float = 2.0  # exponential backoff
    retry_jitter: bool = True


class StorageConfig(BaseModel):
    """Configuration for storage paths."""
    docs_dir: Path = Path("./docs")
    storage_dir: Path = Path("./rag_storage")
    cache_dir: Path = Path("./.cache")
    logs_dir: Path = Path("./logs")
    
    checkpoint_interval: int = 100  # Save every N nodes (smaller = more frequent saves)
    
    @field_validator("docs_dir", "storage_dir", "cache_dir", "logs_dir")
    @classmethod
    def create_if_not_exists(cls, v):
        """Ensure directories exist."""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v


class RAGConfig(BaseModel):
    """Complete RAG system configuration - 2026 Gold Standard."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # API Keys (from environment)
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    google_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY"))
    cohere_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("COHERE_API_KEY"))  # Phase 2: Neural reranking
    
    # Component configurations
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    cost: CostConfig = Field(default_factory=CostConfig)
    production: ProductionConfig = Field(default_factory=ProductionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    
    @field_validator("openai_api_key", "anthropic_api_key", "google_api_key", "cohere_api_key")
    @classmethod
    def validate_api_keys(cls, v, info):
        """Warn if API keys are missing."""
        if not v:
            field_name = info.field_name
            print(f"⚠️  Warning: {field_name} not found in environment")
        return v


# Singleton instance
_config: Optional[RAGConfig] = None


def get_config() -> RAGConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = RAGConfig()
    return _config


def reload_config():
    """Reload configuration from environment."""
    global _config
    load_dotenv(override=True)
    _config = RAGConfig()
    return _config


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("✅ Configuration loaded successfully!")
    print(f"Primary LLM: {config.llm.primary_provider} - {config.llm.primary_model}")
    print(f"Embedding: {config.embedding.provider} - {config.embedding.model}")
    print(f"Phoenix enabled: {config.observability.phoenix_enabled}")
    print(f"Cost tracking: {config.cost.track_costs}")
