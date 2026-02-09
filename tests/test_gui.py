"""
Tests for RAG GUI
==================
Tests for GUI helper functions that don't require a running RAGEngine.
Covers: debug log handler, SDK explorer, dashboard formatting,
provider detection, feedback functions.
"""

import logging
import time

import pytest

from rag_gui import (
    GUILogHandler,
    _detect_provider_name,
    get_class_list,
    get_class_detail,
    get_inheritance_tree,
    get_dashboard,
    get_debug_logs,
    clear_debug_logs,
    _gui_log_handler,
)


# ---------------------------------------------------------------------------
# GUILogHandler
# ---------------------------------------------------------------------------

class TestGUILogHandler:
    """Tests for the debug panel log handler."""

    def test_handler_captures_logs(self):
        handler = GUILogHandler(max_entries=100)
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        logger = logging.getLogger("test.gui.capture")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("test message")
        logs = handler.get_logs()

        assert "test message" in logs
        logger.removeHandler(handler)

    def test_handler_max_entries(self):
        handler = GUILogHandler(max_entries=5)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger("test.gui.max")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        for i in range(10):
            logger.info(f"msg{i}")

        logs = handler.get_logs()
        # Should only have the last 5
        assert "msg5" in logs
        assert "msg9" in logs
        assert "msg0" not in logs
        logger.removeHandler(handler)

    def test_handler_clear(self):
        handler = GUILogHandler(max_entries=100)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger("test.gui.clear")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        logger.info("before clear")
        handler.clear()
        logs = handler.get_logs()

        assert "before clear" not in logs
        logger.removeHandler(handler)

    def test_handler_level_filter_info(self):
        handler = GUILogHandler(max_entries=100)
        handler.setFormatter(
            logging.Formatter("%(levelname)s %(message)s")
        )
        logger = logging.getLogger("test.gui.filter")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # prevent pytest capture interference

        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warn msg")

        # Filter at INFO level
        logs = handler.get_logs(min_level="INFO")
        assert "info msg" in logs
        assert "warn msg" in logs
        # DEBUG should be filtered out
        assert "debug msg" not in logs
        logger.removeHandler(handler)
        logger.propagate = True

    def test_handler_level_filter_warning(self):
        handler = GUILogHandler(max_entries=100)
        handler.setFormatter(
            logging.Formatter("%(levelname)s %(message)s")
        )
        logger = logging.getLogger("test.gui.filter2")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # prevent pytest capture interference

        logger.info("info msg")
        logger.warning("warn msg")
        logger.error("error msg")

        logs = handler.get_logs(min_level="WARNING")
        assert "info msg" not in logs
        assert "warn msg" in logs
        assert "error msg" in logs
        logger.removeHandler(handler)
        logger.propagate = True

    def test_handler_empty_returns_placeholder(self):
        handler = GUILogHandler(max_entries=100)
        logs = handler.get_logs()
        assert "no logs" in logs.lower()

    def test_handler_thread_safety(self):
        """Handler should not crash under concurrent access."""
        import threading

        handler = GUILogHandler(max_entries=100)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger = logging.getLogger("test.gui.thread")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        def write_logs():
            for i in range(50):
                logger.info(f"thread msg {i}")

        threads = [threading.Thread(target=write_logs) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        logs = handler.get_logs()
        assert len(logs) > 0
        logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

class TestProviderDetection:
    """Tests for LLM provider name detection."""

    def test_detect_anthropic(self):
        class FakeAnthropic:
            model = "claude-sonnet-4-5"
        FakeAnthropic.__name__ = "Anthropic"
        result = _detect_provider_name(FakeAnthropic())
        assert "Anthropic" in result

    def test_detect_gemini(self):
        class FakeGemini:
            model = "gemini-2.5-flash"
        FakeGemini.__name__ = "Gemini"
        result = _detect_provider_name(FakeGemini())
        assert "Google" in result

    def test_detect_openai(self):
        class FakeOpenAI:
            model = "gpt-4o-mini"
        FakeOpenAI.__name__ = "OpenAI"
        result = _detect_provider_name(FakeOpenAI())
        assert "OpenAI" in result

    def test_detect_unknown(self):
        class FakeLLM:
            pass
        result = _detect_provider_name(FakeLLM())
        assert "FakeLLM" in result


# ---------------------------------------------------------------------------
# SDK Explorer (uses real SDK headers if present)
# ---------------------------------------------------------------------------

import os

SDK_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "docs_bakkesmod_only"
)
HAS_SDK = os.path.isdir(SDK_DIR)


@pytest.mark.skipif(not HAS_SDK, reason="SDK headers not found")
class TestSDKExplorer:
    """Tests for SDK Explorer functions."""

    def test_class_list_returns_content(self):
        result = get_class_list("")
        assert "classes found" in result.lower() or "class" in result.lower()

    def test_class_list_search_filter(self):
        result = get_class_list("car")
        assert "CarWrapper" in result

    def test_class_list_no_match(self):
        result = get_class_list("zzzznonexistent")
        assert "no classes" in result.lower() or "No classes" in result

    def test_class_detail_car_wrapper(self):
        result = get_class_detail("CarWrapper")
        assert "CarWrapper" in result
        assert "VehicleWrapper" in result  # base class

    def test_class_detail_empty_input(self):
        result = get_class_detail("")
        assert "enter" in result.lower()

    def test_class_detail_unknown(self):
        result = get_class_detail("ZZZNotAClass")
        assert "not found" in result.lower()

    def test_inheritance_tree(self):
        result = get_inheritance_tree()
        assert "ObjectWrapper" in result or "Hierarchy" in result


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

class TestDashboard:
    """Tests for dashboard output."""

    def test_dashboard_returns_markdown(self):
        result = get_dashboard()
        assert "Dashboard" in result
        assert "Session Statistics" in result

    def test_dashboard_has_cost_section(self):
        result = get_dashboard()
        assert "Cost" in result

    def test_dashboard_has_provider_section(self):
        result = get_dashboard()
        assert "Provider" in result or "LLM" in result


# ---------------------------------------------------------------------------
# Debug log integration
# ---------------------------------------------------------------------------

class TestDebugLogIntegration:
    """Tests for the global debug log functions used by the GUI."""

    def test_get_debug_logs_returns_code_block(self):
        result = get_debug_logs("DEBUG")
        assert "```" in result

    def test_clear_debug_logs(self):
        result = clear_debug_logs()
        assert "cleared" in result.lower()

    def test_debug_captures_bakkesmod_logs(self):
        """Logs from bakkesmod_rag.* should be captured by the GUI handler."""
        test_logger = logging.getLogger("bakkesmod_rag.test_gui_capture")
        test_logger.info("gui_capture_test_message_12345")

        logs = _gui_log_handler.get_logs()
        assert "gui_capture_test_message_12345" in logs
