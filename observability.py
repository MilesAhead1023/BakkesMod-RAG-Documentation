"""
Observability & Monitoring
===========================
Phoenix integration for LLM tracing and comprehensive system monitoring.
"""

import logging
import json
import sys
from typing import Optional, Dict, Any
from datetime import datetime
from pathlib import Path
from config import get_config


class StructuredLogger:
    """Structured logging for RAG operations."""
    
    def __init__(self, name: str = "bakkesmod_rag"):
        self.config = get_config().observability
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, self.config.log_level))
        
        # Configure handler
        handler = logging.StreamHandler(sys.stdout)
        
        if self.config.log_format == "json":
            handler.setFormatter(JsonFormatter())
        else:
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        
        self.logger.addHandler(handler)
    
    def log_query(self, query: str, metadata: Optional[Dict] = None):
        """Log a RAG query."""
        self.logger.info("RAG Query", extra={
            "event": "query",
            "query": query,
            "metadata": metadata or {}
        })
    
    def log_retrieval(self, num_chunks: int, sources: list, latency_ms: float):
        """Log retrieval results."""
        self.logger.info("Retrieval Complete", extra={
            "event": "retrieval",
            "num_chunks": num_chunks,
            "sources": sources,
            "latency_ms": latency_ms
        })
    
    def log_llm_call(self, provider: str, model: str, tokens_in: int, tokens_out: int, latency_ms: float):
        """Log LLM API call."""
        self.logger.info("LLM Call", extra={
            "event": "llm_call",
            "provider": provider,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms
        })
    
    def log_cache_hit(self, query: str, similarity: float):
        """Log cache hit."""
        self.logger.info("Cache Hit", extra={
            "event": "cache_hit",
            "query": query,
            "similarity": similarity
        })
    
    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """Log an error with context."""
        self.logger.error("Error", extra={
            "event": "error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        }, exc_info=True)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, "event"):
            log_data["event"] = record.event
        
        # Add any other extra attributes
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "created", "filename", "funcName",
                          "levelname", "levelno", "lineno", "module", "msecs",
                          "message", "pathname", "process", "processName",
                          "relativeCreated", "thread", "threadName", "exc_info",
                          "exc_text", "stack_info"]:
                log_data[key] = value
        
        return json.dumps(log_data)


class PhoenixObserver:
    """Phoenix/Arize integration for LLM tracing and observability."""
    
    def __init__(self):
        self.config = get_config().observability
        self.enabled = self.config.phoenix_enabled
        self.phoenix_session = None
        
        if self.enabled:
            self._initialize_phoenix()
    
    def _initialize_phoenix(self):
        """Initialize Phoenix tracing."""
        try:
            import phoenix as px
            from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
            
            # Launch Phoenix
            self.phoenix_session = px.launch_app(
                host=self.config.phoenix_host,
                port=self.config.phoenix_port
            )
            
            # Enable auto-instrumentation for LlamaIndex
            LlamaIndexInstrumentor().instrument()
            
            print(f"✅ Phoenix UI: http://{self.config.phoenix_host}:{self.config.phoenix_port}")
            
        except ImportError:
            print("⚠️  Phoenix not installed. Install with: pip install arize-phoenix")
            self.enabled = False
        except Exception as e:
            print(f"⚠️  Could not initialize Phoenix: {e}")
            self.enabled = False
    
    def close(self):
        """Close Phoenix session."""
        if self.phoenix_session:
            try:
                self.phoenix_session.close()
            except:
                pass


class MetricsCollector:
    """Collect and expose Prometheus metrics."""
    
    def __init__(self):
        self.config = get_config().observability
        self.enabled = self.config.prometheus_enabled
        
        if self.enabled:
            self._initialize_prometheus()
    
    def _initialize_prometheus(self):
        """Initialize Prometheus metrics."""
        try:
            from prometheus_client import Counter, Histogram, Gauge, start_http_server
            
            # Define metrics
            self.queries_total = Counter(
                'rag_queries_total',
                'Total number of RAG queries',
                ['status']
            )
            
            self.query_latency = Histogram(
                'rag_query_latency_seconds',
                'Query latency in seconds',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            )
            
            self.retrieval_chunks = Histogram(
                'rag_retrieval_chunks',
                'Number of chunks retrieved',
                buckets=[1, 5, 10, 20, 50]
            )
            
            self.llm_tokens = Counter(
                'rag_llm_tokens_total',
                'Total LLM tokens used',
                ['provider', 'type']  # type: input/output
            )
            
            self.cache_hits = Counter(
                'rag_cache_hits_total',
                'Total cache hits'
            )
            
            self.cache_misses = Counter(
                'rag_cache_misses_total',
                'Total cache misses'
            )
            
            self.daily_cost = Gauge(
                'rag_daily_cost_usd',
                'Estimated daily cost in USD'
            )
            
            # Start HTTP server for metrics
            start_http_server(self.config.prometheus_port)
            print(f"✅ Prometheus metrics: http://localhost:{self.config.prometheus_port}/metrics")
            
        except ImportError:
            print("⚠️  Prometheus client not installed. Install with: pip install prometheus-client")
            self.enabled = False
        except Exception as e:
            print(f"⚠️  Could not initialize Prometheus: {e}")
            self.enabled = False
    
    def record_query(self, status: str, latency: float):
        """Record a query."""
        if not self.enabled:
            return
        
        self.queries_total.labels(status=status).inc()
        self.query_latency.observe(latency)
    
    def record_retrieval(self, num_chunks: int):
        """Record retrieval."""
        if not self.enabled:
            return
        
        self.retrieval_chunks.observe(num_chunks)
    
    def record_llm_tokens(self, provider: str, input_tokens: int, output_tokens: int):
        """Record LLM token usage."""
        if not self.enabled:
            return
        
        self.llm_tokens.labels(provider=provider, type="input").inc(input_tokens)
        self.llm_tokens.labels(provider=provider, type="output").inc(output_tokens)
    
    def record_cache_hit(self):
        """Record cache hit."""
        if not self.enabled:
            return
        
        self.cache_hits.inc()
    
    def record_cache_miss(self):
        """Record cache miss."""
        if not self.enabled:
            return
        
        self.cache_misses.inc()
    
    def update_daily_cost(self, cost: float):
        """Update daily cost gauge."""
        if not self.enabled:
            return
        
        self.daily_cost.set(cost)


# Global singletons
_logger: Optional[StructuredLogger] = None
_phoenix: Optional[PhoenixObserver] = None
_metrics: Optional[MetricsCollector] = None


def get_logger() -> StructuredLogger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger()
    return _logger


def get_phoenix() -> PhoenixObserver:
    """Get the global Phoenix observer."""
    global _phoenix
    if _phoenix is None:
        _phoenix = PhoenixObserver()
    return _phoenix


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsCollector()
    return _metrics


def initialize_observability():
    """Initialize all observability components."""
    logger = get_logger()
    phoenix = get_phoenix()
    metrics = get_metrics()
    
    logger.logger.info("Observability initialized", extra={
        "event": "observability_init",
        "phoenix_enabled": phoenix.enabled,
        "metrics_enabled": metrics.enabled
    })
    
    return logger, phoenix, metrics


if __name__ == "__main__":
    # Test observability
    logger, phoenix, metrics = initialize_observability()
    
    logger.log_query("test query", {"test": True})
    logger.log_retrieval(5, ["source1", "source2"], 123.45)
    logger.log_llm_call("openai", "gpt-4o-mini", 100, 50, 678.90)
    
    metrics.record_query("success", 1.23)
    metrics.record_retrieval(5)
    metrics.record_llm_tokens("openai", 100, 50)
    
    print("\n✅ Observability test complete!")
