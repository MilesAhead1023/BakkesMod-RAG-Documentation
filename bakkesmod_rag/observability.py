"""
Observability & Monitoring
===========================
Structured logging with optional Phoenix/Prometheus integration.
"""

import logging
import json
import sys
from typing import Optional, Dict
from datetime import datetime


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if hasattr(record, "event"):
            log_data["event"] = record.event

        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName",
                "relativeCreated", "thread", "threadName", "exc_info",
                "exc_text", "stack_info",
            ):
                log_data[key] = value

        return json.dumps(log_data, default=str)


class StructuredLogger:
    """Structured logging for RAG operations."""

    def __init__(self, name: str = "bakkesmod_rag", config=None):
        """Initialize structured logger.

        Args:
            name: Logger name.
            config: ObservabilityConfig instance (optional).
        """
        log_level = "INFO"
        log_format = "json"

        if config is not None:
            log_level = config.log_level
            log_format = config.log_format

        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level))

        if not self.logger.handlers:
            handler = logging.StreamHandler(sys.stdout)

            if log_format == "json":
                handler.setFormatter(JsonFormatter())
            else:
                handler.setFormatter(
                    logging.Formatter(
                        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    )
                )

            self.logger.addHandler(handler)
            self.logger.propagate = False

    def log_query(self, query: str, metadata: Optional[Dict] = None):
        """Log a RAG query."""
        self.logger.info(
            "RAG Query",
            extra={"event": "query", "query": query, "metadata": metadata or {}},
        )

    def log_retrieval(self, num_chunks: int, sources: list, latency_ms: float):
        """Log retrieval results."""
        self.logger.info(
            "Retrieval Complete",
            extra={
                "event": "retrieval",
                "num_chunks": num_chunks,
                "sources": sources,
                "latency_ms": latency_ms,
            },
        )

    def log_llm_call(
        self, provider: str, model: str, tokens_in: int, tokens_out: int, latency_ms: float
    ):
        """Log LLM API call."""
        self.logger.info(
            "LLM Call",
            extra={
                "event": "llm_call",
                "provider": provider,
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "latency_ms": latency_ms,
            },
        )

    def log_cache_hit(self, query: str, similarity: float):
        """Log cache hit."""
        self.logger.info(
            "Cache Hit",
            extra={"event": "cache_hit", "query": query, "similarity": similarity},
        )

    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Log an error with context."""
        self.logger.error(
            "Error",
            extra={
                "event": "error",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
            },
            exc_info=True,
        )


class PhoenixObserver:
    """Phoenix/Arize integration for LLM tracing (optional)."""

    def __init__(self, config=None):
        self.enabled = False
        self.phoenix_session = None

        if config is not None and config.phoenix_enabled:
            self._initialize_phoenix(config)

    def _initialize_phoenix(self, config):
        """Initialize Phoenix tracing."""
        try:
            import phoenix as px
            from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

            self.phoenix_session = px.launch_app(
                host=config.phoenix_host, port=config.phoenix_port
            )
            LlamaIndexInstrumentor().instrument()
            self.enabled = True
            print(f"Phoenix UI: http://{config.phoenix_host}:{config.phoenix_port}")

        except ImportError:
            print("Phoenix not installed (pip install arize-phoenix)")
        except Exception as e:
            print(f"Could not initialize Phoenix: {e}")

    def close(self):
        """Close Phoenix session."""
        if self.phoenix_session:
            try:
                self.phoenix_session.close()
            except Exception:
                pass


class MetricsCollector:
    """Prometheus metrics collector (optional)."""

    def __init__(self, config=None):
        self.enabled = False

        if config is not None and config.prometheus_enabled:
            self._initialize_prometheus(config)

    def _initialize_prometheus(self, config):
        """Initialize Prometheus metrics."""
        try:
            from prometheus_client import Counter, Histogram, Gauge, start_http_server

            self.queries_total = Counter(
                "rag_queries_total", "Total RAG queries", ["status"]
            )
            self.query_latency = Histogram(
                "rag_query_latency_seconds",
                "Query latency",
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            )
            self.retrieval_chunks = Histogram(
                "rag_retrieval_chunks",
                "Chunks retrieved",
                buckets=[1, 5, 10, 20, 50],
            )
            self.llm_tokens = Counter(
                "rag_llm_tokens_total", "LLM tokens", ["provider", "type"]
            )
            self.cache_hits = Counter("rag_cache_hits_total", "Cache hits")
            self.cache_misses = Counter("rag_cache_misses_total", "Cache misses")
            self.daily_cost = Gauge("rag_daily_cost_usd", "Daily cost USD")

            start_http_server(config.prometheus_port)
            self.enabled = True
            print(f"Prometheus metrics: http://localhost:{config.prometheus_port}/metrics")

        except ImportError:
            print("Prometheus not installed (pip install prometheus-client)")
        except Exception as e:
            print(f"Could not initialize Prometheus: {e}")

    def record_query(self, status: str, latency: float):
        if not self.enabled:
            return
        self.queries_total.labels(status=status).inc()
        self.query_latency.observe(latency)

    def record_retrieval(self, num_chunks: int):
        if not self.enabled:
            return
        self.retrieval_chunks.observe(num_chunks)

    def record_llm_tokens(self, provider: str, input_tokens: int, output_tokens: int):
        if not self.enabled:
            return
        self.llm_tokens.labels(provider=provider, type="input").inc(input_tokens)
        self.llm_tokens.labels(provider=provider, type="output").inc(output_tokens)

    def record_cache_hit(self):
        if not self.enabled:
            return
        self.cache_hits.inc()

    def record_cache_miss(self):
        if not self.enabled:
            return
        self.cache_misses.inc()

    def update_daily_cost(self, cost: float):
        if not self.enabled:
            return
        self.daily_cost.set(cost)
