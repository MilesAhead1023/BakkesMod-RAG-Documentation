"""Tests for observability: StructuredLogger, singletons, MetricsCollector."""

import pytest
from bakkesmod_rag.observability import (
    StructuredLogger,
    PhoenixObserver,
    MetricsCollector,
    JsonFormatter,
    get_logger,
    get_phoenix,
    get_metrics,
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
