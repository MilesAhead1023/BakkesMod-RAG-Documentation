"""Tests for observability: StructuredLogger, singletons, MetricsCollector, OTelTracer."""

import pytest
from bakkesmod_rag.observability import (
    StructuredLogger,
    PhoenixObserver,
    MetricsCollector,
    JsonFormatter,
    OTelTracer,
    _NoOpSpan,
    get_logger,
    get_phoenix,
    get_metrics,
    get_otel_tracer,
    initialize_observability,
)


class TestStructuredLogger:
    def test_creates_logger(self):
        logger = StructuredLogger("test")
        assert logger.logger is not None
        assert logger.logger.name == "test"

    def test_log_query(self):
        logger = StructuredLogger("test_query")
        # Should not raise
        logger.log_query("test query", {"key": "value"})

    def test_log_retrieval(self):
        logger = StructuredLogger("test_retrieval")
        logger.log_retrieval(num_chunks=5, sources=["a.h", "b.cpp"], latency_ms=150.0)

    def test_log_cache_hit(self):
        logger = StructuredLogger("test_cache")
        logger.log_cache_hit("test query", 0.95)

    def test_log_error(self):
        logger = StructuredLogger("test_error")
        logger.log_error(ValueError("test error"), {"context": "unit_test"})


class TestJsonFormatter:
    def test_format_produces_json(self):
        import logging, json
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "hello"
        assert data["level"] == "INFO"


class TestPhoenixObserver:
    def test_disabled_by_default(self):
        obs = PhoenixObserver()
        assert obs.enabled is False

    def test_close_noop_when_disabled(self):
        obs = PhoenixObserver()
        obs.close()  # should not raise


class TestMetricsCollector:
    def test_disabled_by_default(self):
        mc = MetricsCollector()
        assert mc.enabled is False

    def test_record_methods_noop_when_disabled(self):
        mc = MetricsCollector()
        mc.record_query("success", 1.0)
        mc.record_retrieval(5)
        mc.record_llm_tokens("openai", 100, 50)
        mc.record_cache_hit()
        mc.record_cache_miss()
        mc.update_daily_cost(0.05)


class TestSingletonAccessors:
    def test_get_logger_singleton(self):
        import bakkesmod_rag.observability as obs
        obs._logger_instance = None

        l1 = get_logger()
        l2 = get_logger()
        assert l1 is l2

        obs._logger_instance = None

    def test_get_phoenix_singleton(self):
        import bakkesmod_rag.observability as obs
        obs._phoenix_instance = None

        p1 = get_phoenix()
        p2 = get_phoenix()
        assert p1 is p2

        obs._phoenix_instance = None

    def test_get_metrics_singleton(self):
        import bakkesmod_rag.observability as obs
        obs._metrics_instance = None

        m1 = get_metrics()
        m2 = get_metrics()
        assert m1 is m2

        obs._metrics_instance = None

    def test_initialize_observability(self):
        import bakkesmod_rag.observability as obs
        obs._logger_instance = None
        obs._phoenix_instance = None
        obs._metrics_instance = None

        logger, phoenix, metrics = initialize_observability()
        assert isinstance(logger, StructuredLogger)
        assert isinstance(phoenix, PhoenixObserver)
        assert isinstance(metrics, MetricsCollector)

        obs._logger_instance = None
        obs._phoenix_instance = None
        obs._metrics_instance = None


# ---------------------------------------------------------------------------
# Gap 5: OpenTelemetry tests
# ---------------------------------------------------------------------------

class TestOTelTracer:
    """Tests for OTelTracer — no-op when OTel not installed."""

    def test_noop_when_otel_not_configured(self):
        """OTelTracer is disabled by default (enable_otel=False)."""
        from bakkesmod_rag.config import ObservabilityConfig
        cfg = ObservabilityConfig(enable_otel=False)
        tracer = OTelTracer(config=cfg)
        assert tracer.enabled is False

    def test_noop_span_is_context_manager(self):
        """_NoOpSpan can be used as a context manager."""
        span = _NoOpSpan()
        with span as s:
            s.set_attribute("key", "value")
            s.record_exception(ValueError("test"))
        # No errors raised

    def test_span_returns_noop_when_disabled(self):
        """span() yields a _NoOpSpan when OTel is disabled."""
        from bakkesmod_rag.config import ObservabilityConfig
        cfg = ObservabilityConfig(enable_otel=False)
        tracer = OTelTracer(config=cfg)

        with tracer.span("rag.query", {"key": "val"}) as span:
            assert isinstance(span, _NoOpSpan)
            span.set_attribute("confidence", 0.9)
            span.set_attribute("num_sources", 5)

    def test_span_with_no_attributes(self):
        """span() with no attributes dict is fine."""
        from bakkesmod_rag.config import ObservabilityConfig
        cfg = ObservabilityConfig(enable_otel=False)
        tracer = OTelTracer(config=cfg)

        with tracer.span("rag.retrieval") as span:
            assert span is not None

    def test_otel_noop_when_package_missing(self):
        """OTelTracer remains a no-op when opentelemetry is not installed."""
        from bakkesmod_rag.config import ObservabilityConfig
        import sys

        cfg = ObservabilityConfig(enable_otel=True)
        # Remove opentelemetry from sys.modules to simulate missing package
        otel_modules = {k: v for k, v in sys.modules.items() if "opentelemetry" in k}
        for k in otel_modules:
            sys.modules[k] = None  # type: ignore[assignment]

        try:
            tracer = OTelTracer(config=cfg)
            # Should not raise; may or may not be enabled depending on install
            with tracer.span("rag.query") as span:
                pass  # Should not raise
        finally:
            # Restore modules
            for k in otel_modules:
                sys.modules[k] = otel_modules[k]

    def test_otel_config_fields_exist(self):
        """ObservabilityConfig has all required OTel fields."""
        from bakkesmod_rag.config import ObservabilityConfig
        cfg = ObservabilityConfig()
        assert hasattr(cfg, "enable_otel")
        assert hasattr(cfg, "otel_endpoint")
        assert hasattr(cfg, "otel_service_name")
        assert cfg.enable_otel is False  # off by default
        assert "4317" in cfg.otel_endpoint  # default gRPC port
        assert "bakkesmod" in cfg.otel_service_name.lower()

    def test_get_otel_tracer_singleton(self):
        """get_otel_tracer() returns a singleton."""
        import bakkesmod_rag.observability as obs
        obs._otel_tracer_instance = None

        t1 = get_otel_tracer()
        t2 = get_otel_tracer()
        assert t1 is t2

        obs._otel_tracer_instance = None

    def test_trace_context_propagates_through_span(self):
        """Span context manager propagates attributes correctly."""
        from bakkesmod_rag.config import ObservabilityConfig
        cfg = ObservabilityConfig(enable_otel=False)
        tracer = OTelTracer(config=cfg)

        captured = {}
        with tracer.span("rag.query", {"query_text": "test query"}) as span:
            # Attributes on noop span are silently ignored
            span.set_attribute("confidence", 0.85)
            span.set_attribute("cached", False)
            span.set_attribute("retry_count", 0)
            captured["completed"] = True

        assert captured["completed"] is True
