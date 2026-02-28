"""
Observability & Monitoring
===========================
Structured logging with optional Phoenix/Prometheus integration,
and optional OpenTelemetry (OTel) tracing for Jaeger, Grafana Tempo,
Datadog, or any OTel-compatible backend.

OTel is fully optional: if opentelemetry-api is not installed, OTelTracer
is a no-op stub — no errors, no warnings, just silent pass-through.
"""

import logging
import json
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from contextlib import contextmanager
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


def setup_file_logging(
    log_dir: str = "./logs",
    log_name: str = "bakkesmod_rag.log",
    max_bytes: int = 10_000_000,  # 10 MB
    backup_count: int = 5,
) -> Path:
    """Setup centralized file logging for all RAG operations.

    Args:
        log_dir: Directory to store logs (created if doesn't exist).
        log_name: Name of the log file.
        max_bytes: Max size before rotation (default 10 MB).
        backup_count: Number of backup files to keep (default 5).

    Returns:
        Path to the log file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    log_file = log_path / log_name

    # Create rotating file handler
    handler = RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )

    # Format: timestamp | level | logger | message
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)

    # Add to bakkesmod_rag root logger
    root_logger = logging.getLogger("bakkesmod_rag")
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.DEBUG)

    # Also capture uncaught exceptions
    def log_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = log_exception

    return log_file


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


# ---------------------------------------------------------------------------
# Singleton accessors (convenience shortcuts for DI-primary pattern)
# ---------------------------------------------------------------------------

_logger_instance: Optional[StructuredLogger] = None
_phoenix_instance: Optional[PhoenixObserver] = None
_metrics_instance: Optional[MetricsCollector] = None


def get_logger(config=None) -> StructuredLogger:
    """Return the global ``StructuredLogger`` singleton.

    Creates one on first call. Passing *config* on the first call
    configures the logger; subsequent calls ignore *config* and return
    the cached instance.

    Args:
        config: Optional ``ObservabilityConfig``.

    Returns:
        The shared ``StructuredLogger`` instance.
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = StructuredLogger("bakkesmod_rag", config)
    return _logger_instance


def get_phoenix(config=None) -> PhoenixObserver:
    """Return the global ``PhoenixObserver`` singleton.

    Args:
        config: Optional ``ObservabilityConfig``.

    Returns:
        The shared ``PhoenixObserver`` instance.
    """
    global _phoenix_instance
    if _phoenix_instance is None:
        _phoenix_instance = PhoenixObserver(config)
    return _phoenix_instance


def get_metrics(config=None) -> MetricsCollector:
    """Return the global ``MetricsCollector`` singleton.

    Args:
        config: Optional ``ObservabilityConfig``.

    Returns:
        The shared ``MetricsCollector`` instance.
    """
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector(config)
    return _metrics_instance


def initialize_observability(config=None):
    """One-shot initialisation of all observability subsystems.

    Args:
        config: ``ObservabilityConfig`` (or the full ``RAGConfig``'s
            ``.observability`` attribute).

    Returns:
        Tuple of ``(StructuredLogger, PhoenixObserver, MetricsCollector)``.
    """
    return get_logger(config), get_phoenix(config), get_metrics(config)


# ---------------------------------------------------------------------------
# OpenTelemetry integration
# ---------------------------------------------------------------------------

class _NoOpSpan:
    """No-op span used when OTel is unavailable or disabled."""

    def set_attribute(self, key: str, value) -> None:
        pass

    def record_exception(self, exc: Exception) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class OTelTracer:
    """Optional OpenTelemetry tracer for the RAG pipeline.

    Instruments query, retrieval, LLM call, and cache spans.
    When opentelemetry-api is not installed, all methods become no-ops —
    no errors, no warnings, zero overhead.

    Usage::

        tracer = OTelTracer(config)
        with tracer.span("rag.query") as span:
            span.set_attribute("query_text", query)
            ...
    """

    def __init__(self, config=None):
        """Initialise the OTel tracer.

        Args:
            config: ObservabilityConfig with enable_otel, otel_endpoint,
                otel_service_name fields.
        """
        self._enabled = False
        self._tracer = None

        if config is None or not getattr(config, "enable_otel", False):
            return

        self.configure(
            endpoint=getattr(config, "otel_endpoint", "http://localhost:4317"),
            service_name=getattr(config, "otel_service_name", "bakkesmod-rag"),
        )

    def configure(self, endpoint: str, service_name: str) -> None:
        """Configure the OTel tracer with an exporter endpoint.

        Silently skips if opentelemetry-api is not installed.

        Args:
            endpoint: OTel collector gRPC endpoint (e.g. http://localhost:4317).
            service_name: Service name tag for all spans.
        """
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": service_name})
            provider = TracerProvider(resource=resource)

            # Try to configure OTLP gRPC exporter
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.trace.export import BatchSpanProcessor

                exporter = OTLPSpanExporter(endpoint=endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except ImportError:
                # opentelemetry-exporter-otlp not installed — tracer still works,
                # just no export
                pass

            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(service_name)
            self._enabled = True
            logging.getLogger("bakkesmod_rag.observability").info(
                "OTel tracer configured: endpoint=%s service=%s",
                endpoint, service_name,
            )

        except ImportError:
            # opentelemetry-api not installed — silently become a no-op
            pass
        except Exception as e:
            logging.getLogger("bakkesmod_rag.observability").warning(
                "OTel tracer setup failed (continuing without tracing): %s", e
            )

    @property
    def enabled(self) -> bool:
        """True if OTel tracing is active."""
        return self._enabled

    @contextmanager
    def span(self, name: str, attributes: Optional[Dict] = None):
        """Context manager that creates an OTel span (or no-op if disabled).

        Args:
            name: Span name (e.g. "rag.query", "rag.retrieval").
            attributes: Initial span attributes dict.

        Yields:
            An OTel Span or a _NoOpSpan.
        """
        if not self._enabled or self._tracer is None:
            noop = _NoOpSpan()
            yield noop
            return

        with self._tracer.start_as_current_span(name) as otel_span:
            if attributes:
                for k, v in attributes.items():
                    try:
                        otel_span.set_attribute(k, v)
                    except Exception:
                        pass
            yield otel_span


# ---------------------------------------------------------------------------
# OTelTracer singleton
# ---------------------------------------------------------------------------

_otel_tracer_instance: Optional[OTelTracer] = None


def get_otel_tracer(config=None) -> OTelTracer:
    """Return the global OTelTracer singleton.

    Args:
        config: ObservabilityConfig (used only on first call).

    Returns:
        The shared OTelTracer instance.
    """
    global _otel_tracer_instance
    if _otel_tracer_instance is None:
        _otel_tracer_instance = OTelTracer(config)
    return _otel_tracer_instance
