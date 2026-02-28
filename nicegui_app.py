"""BakkesMod RAG -- NiceGUI desktop app entry point.

Seven-tab native desktop interface:
  - Tab 1: Live Console (Python log output)
  - Tab 2: Query Documentation (RAG Q&A)
  - Tab 3: Generate Plugin Code (RAG code gen)
  - Tab 4: SDK Explorer (C++ class browser)
  - Tab 5: Dashboard (stats, cost, provider health)
  - Tab 6: Help (feature reference)
  - Tab 7: Settings (API keys, budget)

All RAG logic is delegated to ``bakkesmod_rag.RAGEngine``.
Runs as a native window via pywebview when bundled with PyInstaller,
or as a normal browser app in development.
"""

# Pre-load torch c10.dll before PyQt/NiceGUI to prevent WinError 1114 on Windows.
# PyTorch 2.9.x introduced a regression where c10.dll fails to initialize if
# imported after PyQt. Loading it explicitly first avoids the conflict.
import platform as _platform, ctypes as _ctypes, os as _os
if _platform.system() == "Windows":
    from importlib.util import find_spec as _find_spec
    try:
        if (_spec := _find_spec("torch")) and _spec.origin:
            _dll = _os.path.join(_os.path.dirname(_spec.origin), "lib", "c10.dll")
            if _os.path.exists(_dll):
                _ctypes.CDLL(_os.path.normpath(_dll))
    except Exception:
        pass
del _platform, _ctypes, _os

import asyncio
import io
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

SETTINGS_FILE = Path(__file__).parent / "nicegui_settings.json"

# Ensure UTF-8 output encoding on Windows
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding="utf-8", errors="replace"
    )

from nicegui import ui, app as nicegui_app

from bakkesmod_rag.llm_provider import get_llm, NullLLM

IS_EXE = getattr(sys, "frozen", False)

logger = logging.getLogger("bakkesmod_rag.nicegui_app")

# ---------------------------------------------------------------------------
# Model lists per LLM provider  (used by Settings dropdowns)
# ---------------------------------------------------------------------------

PROVIDER_MODELS: dict[str, list[str]] = {
    "anthropic": [
        "claude-sonnet-4-6",
        "claude-opus-4-6",
        "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
        "o3-mini",
    ],
    "gemini": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ],
    "openrouter": [
        "deepseek/deepseek-v3",
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet",
        "meta-llama/llama-3.1-70b-instruct",
        "google/gemini-2.5-flash",
        "mistralai/mistral-large",
    ],
    "ollama": [
        "llama3.2",
        "llama3.1",
        "mistral",
        "gemma2",
        "phi3",
        "qwen2.5",
    ],
}


async def _fetch_models_live(provider: str, api_key: str) -> list[str]:
    """Fetch available model IDs from the provider's live API.

    Falls back to ``PROVIDER_MODELS[provider]`` on any error so the UI always
    has a usable list.

    Args:
        provider: One of ``"openai"``, ``"anthropic"``, ``"gemini"``,
                  ``"openrouter"``, ``"ollama"``.
        api_key: The API key to use for authentication.

    Returns:
        Sorted list of model ID strings.
    """
    try:
        if provider == "openai":
            import openai as _oai
            client = _oai.AsyncOpenAI(api_key=api_key)
            resp = await client.models.list()
            chat_ids = sorted(
                m.id for m in resp.data
                if any(tag in m.id for tag in ("gpt-", "o1", "o3", "o4"))
            )
            return chat_ids or PROVIDER_MODELS.get(provider, [])

        if provider == "anthropic":
            import anthropic as _anth
            client = _anth.AsyncAnthropic(api_key=api_key)
            resp = await client.models.list()
            ids = sorted(m.id for m in resp.data)
            return ids or PROVIDER_MODELS.get(provider, [])

        if provider == "gemini":
            import google.genai as genai
            client = genai.Client(api_key=api_key)
            loop = asyncio.get_event_loop()
            models = await loop.run_in_executor(
                None, lambda: list(client.models.list())
            )
            ids = sorted(
                m.name.replace("models/", "")
                for m in models
                if "generateContent" in getattr(m, "supported_actions", [])
                or hasattr(m, "supported_generation_methods")
            )
            return ids or PROVIDER_MODELS.get(provider, [])

        if provider == "openrouter":
            import httpx
            async with httpx.AsyncClient(timeout=10) as http:
                resp = await http.get(
                    "https://openrouter.ai/api/v1/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                data = resp.json().get("data", [])
                ids = sorted(m.get("id", "") for m in data if m.get("id"))
                return ids[:100] or PROVIDER_MODELS.get(provider, [])

        if provider == "ollama":
            import httpx
            async with httpx.AsyncClient(timeout=5) as http:
                resp = await http.get("http://localhost:11434/api/tags")
                data = resp.json().get("models", [])
                ids = sorted(m.get("name", "") for m in data if m.get("name"))
                return ids or PROVIDER_MODELS.get(provider, [])

    except Exception as exc:
        logger.debug("Model fetch failed for %s: %s", provider, exc)

    return PROVIDER_MODELS.get(provider, [])


# ---------------------------------------------------------------------------
# Background process tracking + clean shutdown
# ---------------------------------------------------------------------------

_bg_procs: set = set()  # asyncio.subprocess.Process handles tracked for cleanup


def _shutdown_cleanup() -> None:
    """Kill all tracked background subprocesses and force-exit the Python process.

    Registered with ``nicegui_app.on_shutdown`` so it runs whenever the window
    is closed (native mode) or the server is asked to stop (Ctrl-C in dev mode).
    The ``os._exit(0)`` at the end is intentional: Uvicorn, asyncio, and any
    background threads would otherwise keep the process alive indefinitely.
    """
    for proc in list(_bg_procs):
        try:
            proc.terminate()
            logger.debug("Terminated background process PID %s", proc.pid)
        except Exception:
            pass
    _bg_procs.clear()
    logger.info("BakkesMod RAG shutdown — bye!")
    os._exit(0)


nicegui_app.on_shutdown(_shutdown_cleanup)


# ---------------------------------------------------------------------------
# Lazy RAGEngine singleton
# ---------------------------------------------------------------------------

_engine = None  # type: ignore[assignment]
_engine_error: Optional[str] = None
_repair_report = None  # RepairReport from last auto-repair run, or None
_repair_attempted: bool = False  # Only run auto-repair once per session


def _indexes_exist() -> bool:
    """Return True if the RAG storage indexes have been built."""
    storage = Path(__file__).parent / "rag_storage"
    if not storage.exists():
        return False
    docstore = (storage / "docstore.json").exists()
    vector = (
        (storage / "default__vector_store.json").exists()
        or (storage / "vector_store.json").exists()
    )
    return docstore and vector


async def _do_build_indexes_bg(log_widget=None) -> None:
    """Run comprehensive_builder as a subprocess, streaming output to log_widget.

    Also routes all output through the ``bakkesmod_rag`` logger so it appears
    in the Live Console regardless of whether a dedicated log widget is supplied.
    """
    _bg_logger = logging.getLogger("bakkesmod_rag.build")
    _bg_logger.info("Starting comprehensive_builder subprocess...")
    if log_widget is not None:
        try:
            log_widget.push("▶ Starting comprehensive builder…")
        except Exception:
            pass
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "bakkesmod_rag.comprehensive_builder",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(Path(__file__).parent),
        )
        _bg_procs.add(proc)
        assert proc.stdout is not None
        try:
            async for raw in proc.stdout:
                line = raw.decode("utf-8", errors="replace").rstrip()
                _bg_logger.info("[builder] %s", line)
                if log_widget is not None:
                    try:
                        log_widget.push(line)
                    except Exception:
                        pass
            await proc.wait()
        finally:
            _bg_procs.discard(proc)
        if proc.returncode == 0:
            _bg_logger.info("Comprehensive builder completed successfully.")
            if log_widget is not None:
                try:
                    log_widget.push("✅ Build complete — restart the app to load indexes.")
                except Exception:
                    pass
        else:
            _bg_logger.error(
                "Comprehensive builder exited with code %d", proc.returncode
            )
            if log_widget is not None:
                try:
                    log_widget.push(
                        f"❌ Builder exited with code {proc.returncode}"
                    )
                except Exception:
                    pass
    except Exception as exc:
        _bg_logger.error("Failed to run comprehensive_builder: %s", exc)
        if log_widget is not None:
            try:
                log_widget.push(f"❌ Error: {exc}")
            except Exception:
                pass


def _load_settings_dict() -> dict:
    """Load saved RAG backend settings from JSON file."""
    if SETTINGS_FILE.exists():
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _apply_sub(sub_cfg, data: dict, keys: list) -> None:
    """Set attributes on sub_cfg for each key found in data."""
    for key in keys:
        if key in data:
            setattr(sub_cfg, key, data[key])


def _apply_settings_to_config(settings: dict) -> None:
    """Apply loaded settings dict to the RAGConfig singleton (all 15 sub-configs)."""
    from bakkesmod_rag.config import get_config
    cfg = get_config()

    _apply_sub(cfg.llm, settings.get("llm", {}), [
        "primary_provider", "primary_model", "kg_provider", "kg_model",
        "temperature", "max_retries", "timeout",
    ])
    _apply_sub(cfg.embedding, settings.get("embedding", {}), [
        "provider", "model", "batch_size", "max_retries",
    ])
    _apply_sub(cfg.retriever, settings.get("retriever", {}), [
        "vector_top_k", "bm25_top_k", "kg_similarity_top_k", "kg_max_triplets_per_chunk",
        "fusion_mode", "fusion_num_queries", "enable_kg", "enable_reranker",
        "enable_llm_rewrite", "enable_query_decomposition", "adaptive_top_k",
        "use_hierarchical_chunking", "merge_threshold", "use_mmr", "mmr_threshold",
        "rerank_top_n", "use_colbert", "colbert_model", "max_sub_queries",
        "decomposition_complexity_threshold",
    ])
    _apply_sub(cfg.chunking, settings.get("chunking", {}), [
        "chunk_size", "chunk_overlap", "enable_semantic_chunking",
        "semantic_breakpoint_percentile", "code_chunk_lines", "code_chunk_lines_overlap",
        "include_metadata", "include_prev_next_rel",
    ])
    _apply_sub(cfg.cache, settings.get("cache", {}), [
        "enabled", "similarity_threshold", "ttl_seconds", "backend",
        "redis_url", "redis_db", "cache_dir",
    ])
    _apply_sub(cfg.self_rag, settings.get("self_rag", {}), [
        "enabled", "confidence_threshold", "max_retries", "force_llm_rewrite_on_retry",
    ])
    _apply_sub(cfg.verification, settings.get("verification", {}), [
        "enabled", "grounded_threshold", "borderline_threshold",
        "borderline_confidence_penalty", "ungrounded_confidence_penalty",
    ])
    _apply_sub(cfg.codegen, settings.get("codegen", {}), [
        "enabled", "validate_output", "self_improving", "max_fix_iterations",
        "enable_compilation", "msvc_path", "feedback_enabled", "max_context_chunks",
    ])
    _apply_sub(cfg.observability, settings.get("observability", {}), [
        "log_level", "log_format", "phoenix_enabled", "phoenix_host", "phoenix_port",
        "prometheus_enabled", "prometheus_port", "enable_otel", "otel_endpoint",
        "otel_service_name",
    ])
    _apply_sub(cfg.cost, settings.get("cost", {}), [
        "track_costs", "alert_threshold_pct", "openai_embedding_cost",
        "openai_gpt4o_mini_input", "openai_gpt4o_mini_output",
        "anthropic_claude_sonnet_input", "anthropic_claude_sonnet_output",
        "gemini_flash_input", "gemini_flash_output",
    ])
    _apply_sub(cfg.production, settings.get("production", {}), [
        "rate_limit_enabled", "requests_per_minute", "circuit_breaker_enabled",
        "failure_threshold", "recovery_timeout", "max_retries",
        "retry_backoff_factor", "retry_jitter",
    ])
    _apply_sub(cfg.storage, settings.get("storage", {}), [
        "storage_dir", "cache_dir", "logs_dir",
    ])
    _apply_sub(cfg.cpp_intelligence, settings.get("cpp_intelligence", {}), [
        "enabled", "max_methods_in_metadata", "max_signatures_in_metadata",
        "max_related_types", "include_inheritance_chain", "include_method_signatures",
        "include_forward_declarations",
    ])
    _apply_sub(cfg.intent_router, settings.get("intent_router", {}), [
        "enabled", "llm_confirmation_threshold",
    ])
    _apply_sub(cfg.guardrails, settings.get("guardrails", {}), [
        "enabled", "min_length", "max_length",
    ])


def get_engine():
    """Return the RAGEngine singleton, creating it on first call."""
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    if _engine_error is not None:
        return None
    try:
        settings = _load_settings_dict()
        if settings:
            _apply_settings_to_config(settings)
        from bakkesmod_rag.engine import RAGEngine
        _engine = RAGEngine()
        return _engine
    except Exception as exc:
        _engine_error = str(exc)
        logger.error("RAGEngine init failed: %s", exc)
        return None


async def _ensure_engine_running() -> None:
    """Auto-repair and retry engine initialisation if it failed at startup.

    Runs once per session.  Dispatches ``repair_for_error()`` on a thread-pool
    executor so pip subprocess calls don't block the NiceGUI event loop.  If
    at least one repair succeeds, resets ``_engine_error`` and retries
    ``get_engine()`` so the user never needs to do anything manually.
    """
    global _engine, _engine_error, _repair_report, _repair_attempted
    if _repair_attempted or _engine is not None:
        return
    if _engine_error is None:
        return
    _repair_attempted = True

    err = _engine_error
    logger.info("[auto-repair] Engine failed at startup — running auto-repair…")
    try:
        from bakkesmod_rag.self_repair import repair_for_error
        loop = asyncio.get_event_loop()
        report = await loop.run_in_executor(None, repair_for_error, err)
        _repair_report = report

        for line in report.as_lines():
            logger.info("[auto-repair] %s", line)

        if report.any_success:
            logger.info("[auto-repair] Repairs applied — retrying engine init…")
            _engine_error = None  # allow get_engine() to retry
            get_engine()
            if _engine is not None:
                logger.info("[auto-repair] Engine initialised successfully after auto-repair.")
                ui.notify(
                    "Auto-repair succeeded — RAG engine is now running!",
                    type="positive",
                    timeout=8000,
                )
            else:
                logger.warning("[auto-repair] Engine still failed after repair.")
                ui.notify(
                    "Auto-repair applied fixes but the engine still failed. "
                    "See Dashboard for details.",
                    type="warning",
                    timeout=8000,
                )
        else:
            if not report.is_empty():
                logger.warning("[auto-repair] All repair attempts failed — manual intervention needed.")
            else:
                logger.debug("[auto-repair] No applicable repair found for this error.")
    except Exception as exc:
        logger.warning("Auto-repair system encountered an error: %s", exc)


# ---------------------------------------------------------------------------
# SDK Explorer data (lazy)
# ---------------------------------------------------------------------------

_sdk_classes: Optional[dict] = None


def _get_sdk_classes() -> dict:
    """Analyze SDK headers and return class name -> CppClassInfo dict."""
    global _sdk_classes
    if _sdk_classes is not None:
        return _sdk_classes
    try:
        from bakkesmod_rag.cpp_analyzer import CppAnalyzer
        analyzer = CppAnalyzer()
        _sdk_classes = analyzer.analyze_directory("docs_bakkesmod_only")
    except Exception as exc:
        logger.warning("SDK analysis failed: %s", exc)
        _sdk_classes = {}
    return _sdk_classes


# ---------------------------------------------------------------------------
# Help text (matches rag_gui.py content)
# ---------------------------------------------------------------------------

HELP_TEXT = """\
## BakkesMod RAG System -- Feature Reference

### Query Pipeline
1. **Synonym expansion** -- 60+ BakkesMod domain mappings (zero API cost)
2. **Query decomposition** -- Complex questions split into sub-queries
3. **3-way fusion retrieval** -- Vector + BM25 + Knowledge Graph
4. **Neural reranking** -- BGE -> FlashRank -> Cohere fallback chain
5. **Adaptive retrieval** -- Dynamic top_k escalation (5 -> 8 -> 12)
6. **Answer verification** -- Embedding + LLM grounding checks
7. **Self-RAG retry** -- Automatic retry if confidence < 70%
8. **Semantic caching** -- 92% similarity threshold, 7-day TTL

### Code Generation
- **RAG-enhanced** -- Uses SDK documentation for accurate code
- **Template engine** -- Full 12-file project scaffolding
- **Feature detection** -- ImGui, events, CVars, drawables
- **Self-improving loop** -- Validate -> Fix -> Compile (up to 5 iterations)
- **MSVC compilation** -- Optional compile verification
- **Feedback learning** -- Thumbs up/down improves future generations

### SDK Intelligence
- **C++ structural analysis** -- Tree-sitter parses all 177 SDK headers
- **Inheritance chains** -- Full hierarchy from ObjectWrapper to leaf classes
- **Method signatures** -- Return types, parameters, const qualifiers
- **Metadata injection** -- LLM sees C++ structure, not just text

### Resilience
- **LLM fallback chain** -- Anthropic -> OpenAI -> Gemini Pro -> \
OpenRouter -> Gemini Flash -> Ollama
- **Circuit breakers** -- Automatic provider failover
- **Rate limiting** -- Configurable requests/minute
- **Cost tracking** -- Per-query cost, daily budget alerts

### MCP Server
The system includes an MCP server for Claude Code IDE integration.
Run with: `python -m bakkesmod_rag.mcp_server`

### Tips
- Enable **cache** for faster, cheaper repeated queries
- Use the **SDK Explorer** tab to browse classes before asking questions
- **Higher confidence = more reliable** (green > yellow > red)
- Check the **Live Console** tab for verbose log output
"""


# ---------------------------------------------------------------------------
# Log handler that pushes to NiceGUI ui.log widget
# ---------------------------------------------------------------------------

class NiceGuiLogHandler(logging.Handler):
    """Logging handler that pushes records to a NiceGUI log widget."""

    def __init__(self, log_widget):
        super().__init__()
        self.log_widget = log_widget
        self.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))

    def emit(self, record):
        try:
            msg = self.format(record)
            self.log_widget.push(msg)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

@ui.page("/")
def main_page():
    """Build the 7-tab NiceGUI interface."""
    ui.dark_mode().enable()

    # -- Header -------------------------------------------------------------
    with ui.header().classes("items-center justify-between"):
        ui.label("BakkesMod RAG").classes("text-h5 text-bold")
        ui.label("SDK Documentation Assistant").classes(
            "text-subtitle2 text-grey-5"
        )

    # -- Tabs ---------------------------------------------------------------
    with ui.tabs().classes("w-full") as tabs:
        tab_console = ui.tab("Live Console")
        tab_query = ui.tab("Query Documentation")
        tab_generate = ui.tab("Generate Plugin Code")
        tab_sdk = ui.tab("SDK Explorer")
        tab_dashboard = ui.tab("Dashboard")
        tab_help = ui.tab("Help")
        tab_settings = ui.tab("Settings")

    with ui.tab_panels(tabs, value=tab_query).classes("w-full"):

        # ==============================================================
        # Tab 1: Live Console
        # ==============================================================
        with ui.tab_panel(tab_console):
            ui.label("Live Console").classes("text-h6")
            ui.label(
                "Real-time Python log output from all RAG subsystems."
            ).classes("text-caption text-grey-5")
            console_log = ui.log(max_lines=500).classes(
                "w-full h-96"
            )
            # Attach handler to bakkesmod_rag and adjacent subsystem loggers
            handler = NiceGuiLogHandler(console_log)
            handler.setLevel(logging.DEBUG)
            for _log_name in [
                "bakkesmod_rag",
                "llama_index",
                "llama_index.core",
                "llama_index.core.indices",
                "llama_index.core.query_engine",
                "openai",
                "httpx",
            ]:
                _lg = logging.getLogger(_log_name)
                _lg.addHandler(handler)
                if _log_name == "bakkesmod_rag":
                    _lg.setLevel(logging.DEBUG)
                elif _log_name.startswith("llama_index"):
                    _lg.setLevel(logging.INFO)
                else:
                    _lg.setLevel(logging.WARNING)

        # ==============================================================
        # Tab 2: Query Documentation
        # ==============================================================
        with ui.tab_panel(tab_query):
            ui.label("Query Documentation").classes("text-h6")
            query_input = ui.input(
                "Ask about the BakkesMod SDK..."
            ).classes("w-full").props("outlined")
            with ui.row().classes("items-center gap-4"):
                use_cache = ui.switch("Use cache", value=True)
                confidence_badge = ui.badge("").props(
                    "color=grey outline"
                )

            response_area = ui.markdown("").classes(
                "w-full border rounded p-4 min-h-[200px]"
            )
            sources_area = ui.markdown("").classes("w-full text-caption")

            async def run_query():
                if not query_input.value or not query_input.value.strip():
                    ui.notify(
                        "Please enter a question", type="warning"
                    )
                    return
                response_area.content = ""
                sources_area.content = ""
                confidence_badge.text = ""
                spinner = ui.spinner(size="lg")
                try:
                    eng = get_engine()
                    if eng is None:
                        response_area.content = (
                            "RAG engine not available. "
                            f"Error: {_engine_error or 'unknown'}"
                        )
                        return
                    result = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: eng.query(
                            query_input.value,
                            use_cache=use_cache.value,
                        ),
                    )
                    response_area.content = (
                        result.answer
                        if hasattr(result, "answer")
                        else str(result)
                    )
                    # Confidence badge
                    if hasattr(result, "confidence") and result.confidence:
                        pct = result.confidence
                        label = getattr(
                            result, "confidence_label", ""
                        )
                        color = "green"
                        if pct < 0.5:
                            color = "red"
                        elif pct < 0.8:
                            color = "orange"
                        confidence_badge.text = (
                            f"{label} {pct:.0%}"
                        )
                        confidence_badge.props(f"color={color}")
                    # Sources
                    if hasattr(result, "sources") and result.sources:
                        src_lines = ["**Sources:**"]
                        for s in result.sources:
                            fname = s.get("file_name", "unknown")
                            score = s.get("score")
                            if score is not None:
                                src_lines.append(
                                    f"- `{fname}` (score: {score:.2f})"
                                )
                            else:
                                src_lines.append(f"- `{fname}`")
                        sources_area.content = "\n".join(src_lines)
                    # Verification warning
                    if (
                        hasattr(result, "verification_warning")
                        and result.verification_warning
                    ):
                        ui.notify(
                            result.verification_warning,
                            type="warning",
                            timeout=8000,
                        )
                except Exception as exc:
                    response_area.content = f"Error: {exc}"
                    logger.exception("Query failed")
                finally:
                    spinner.delete()

            ui.button(
                "Ask", on_click=run_query
            ).props("color=orange")

        # ==============================================================
        # Tab 3: Generate Plugin Code
        # ==============================================================
        with ui.tab_panel(tab_generate):
            ui.label("Generate Plugin Code").classes("text-h6")
            codegen_input = ui.textarea(
                "Describe your plugin..."
            ).classes("w-full").props("outlined rows=4")

            header_code = ui.code("// Header will appear here", language="cpp")
            impl_code = ui.code(
                "// Implementation will appear here", language="cpp"
            )
            codegen_info = ui.markdown("")

            async def run_generate():
                if (
                    not codegen_input.value
                    or not codegen_input.value.strip()
                ):
                    ui.notify(
                        "Please describe your plugin", type="warning"
                    )
                    return
                header_code.content = "// Generating..."
                impl_code.content = "// Generating..."
                codegen_info.content = ""
                spinner = ui.spinner(size="lg")
                try:
                    eng = get_engine()
                    if eng is None:
                        header_code.content = (
                            "// RAG engine not available"
                        )
                        impl_code.content = (
                            f"// Error: {_engine_error or 'unknown'}"
                        )
                        return
                    result = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: eng.generate_code(
                            codegen_input.value
                        ),
                    )
                    header_code.content = (
                        result.header if result.header else "// No header"
                    )
                    impl_code.content = (
                        result.implementation
                        if result.implementation
                        else "// No implementation"
                    )
                    # Info section
                    info_parts = []
                    if result.features_used:
                        info_parts.append(
                            "**Features:** "
                            + ", ".join(result.features_used)
                        )
                    if result.fix_iterations:
                        info_parts.append(
                            f"**Fix iterations:** {result.fix_iterations}"
                        )
                    if result.validation:
                        valid = result.validation.get("valid", False)
                        errors = result.validation.get("errors", [])
                        warnings = result.validation.get("warnings", [])
                        info_parts.append(
                            f"**Valid:** {valid}"
                        )
                        if errors:
                            info_parts.append(
                                f"**Errors:** {len(errors)}"
                            )
                        if warnings:
                            info_parts.append(
                                f"**Warnings:** {len(warnings)}"
                            )
                    if result.compile_result:
                        success = result.compile_result.get(
                            "success", False
                        )
                        info_parts.append(
                            f"**Compiled:** {'Yes' if success else 'No'}"
                        )
                    codegen_info.content = "  \n".join(info_parts)
                except Exception as exc:
                    header_code.content = f"// Error: {exc}"
                    impl_code.content = ""
                    logger.exception("Code generation failed")
                finally:
                    spinner.delete()

            ui.button(
                "Generate", on_click=run_generate
            ).props("color=orange")

        # ==============================================================
        # Tab 4: SDK Explorer
        # ==============================================================
        with ui.tab_panel(tab_sdk):
            ui.label("SDK Explorer").classes("text-h6")
            ui.label(
                "Browse BakkesMod SDK classes, methods, and "
                "inheritance chains."
            ).classes("text-caption text-grey-5")

            sdk_search = ui.input("Search SDK classes...").classes(
                "w-full"
            ).props("outlined")
            sdk_list_container = ui.column().classes("w-full")
            sdk_detail_container = ui.column().classes(
                "w-full border rounded p-4 min-h-[200px]"
            )

            async def show_class_detail(class_name: str):
                """Display details for a selected class."""
                sdk_detail_container.clear()
                classes = await asyncio.get_running_loop().run_in_executor(
                    None, _get_sdk_classes
                )
                cls_info = classes.get(class_name)
                if cls_info is None:
                    with sdk_detail_container:
                        ui.label(f"Class '{class_name}' not found.")
                    return
                with sdk_detail_container:
                    ui.label(cls_info.name).classes("text-h6")
                    if cls_info.base_classes:
                        ui.label(
                            f"Inherits from: "
                            f"{', '.join(cls_info.base_classes)}"
                        ).classes("text-subtitle2 text-grey-5")
                    if cls_info.file:
                        ui.label(
                            f"File: {cls_info.file}"
                        ).classes("text-caption text-grey-6")
                    if cls_info.category:
                        ui.badge(cls_info.category).props(
                            "color=blue outline"
                        )
                    if cls_info.methods:
                        ui.label(
                            f"Methods ({len(cls_info.methods)}):"
                        ).classes("text-subtitle2 q-mt-md")
                        for m in cls_info.methods[:50]:
                            sig = (
                                f"{m.return_type} {m.name}"
                                f"({m.parameters})"
                            )
                            if m.is_const:
                                sig += " const"
                            if m.is_virtual:
                                sig = "virtual " + sig
                            ui.label(sig).classes(
                                "text-caption font-mono"
                            )
                    else:
                        ui.label("No public methods found.").classes(
                            "text-caption text-grey-6"
                        )

            async def refresh_sdk_list():
                """Refresh the SDK class list filtered by search."""
                sdk_list_container.clear()
                classes = await asyncio.get_running_loop().run_in_executor(
                    None, _get_sdk_classes
                )
                if not classes:
                    with sdk_list_container:
                        ui.label(
                            "No SDK classes found. Ensure "
                            "docs_bakkesmod_only/ directory exists."
                        ).classes("text-grey-5")
                    return
                search_val = (
                    sdk_search.value.lower()
                    if sdk_search.value
                    else ""
                )
                filtered = sorted(
                    (
                        name
                        for name in classes
                        if search_val in name.lower()
                    )
                )
                with sdk_list_container:
                    if not filtered:
                        ui.label("No matching classes.").classes(
                            "text-grey-5"
                        )
                    for name in filtered[:100]:
                        cls = classes[name]
                        method_count = len(cls.methods)
                        with ui.row().classes(
                            "items-center cursor-pointer hover:bg-grey-9 "
                            "p-1 rounded"
                        ):
                            btn = ui.button(
                                f"{name} ({method_count} methods)",
                                on_click=lambda _n=name: show_class_detail(
                                    _n
                                ),
                            ).props("flat dense color=white")

            sdk_search.on(
                "update:model-value",
                lambda: refresh_sdk_list(),
            )
            ui.button(
                "Load SDK Classes",
                on_click=refresh_sdk_list,
            ).props("color=orange")

        # ==============================================================
        # Tab 5: Dashboard
        # ==============================================================
        with ui.tab_panel(tab_dashboard):
            ui.label("Dashboard").classes("text-h6")
            dash_container = ui.column().classes("w-full")

            async def refresh_dashboard():
                """Refresh dashboard stats."""
                try:
                    dash_container.clear()
                except Exception:
                    return
                eng = get_engine()
                with dash_container:
                    if eng is None:
                        # ── Auto-repair status card ───────────────────────
                        if _repair_report is not None and not _repair_report.is_empty():
                            border = "#4caf50" if _repair_report.any_success else "#f44336"
                            with ui.card().classes("w-full q-mb-sm").style(
                                f"border-left: 3px solid {border};"
                            ):
                                with ui.row().classes("items-center gap-2 q-mb-xs"):
                                    icon = "build_circle" if _repair_report.any_success else "error_outline"
                                    color = "green" if _repair_report.any_success else "red"
                                    ui.icon(icon, color=color).classes("text-h5")
                                    ui.label("Auto-Repair Results").classes(
                                        "text-subtitle1 text-bold"
                                    )
                                for line in _repair_report.as_lines():
                                    ui.label(line).classes("text-caption text-mono q-ml-md")
                                if _repair_report.any_success and _engine_error is None:
                                    with ui.row().classes("q-mt-sm"):
                                        ui.label(
                                            "Repairs applied — click Refresh to retry."
                                        ).classes("text-caption text-positive")
                        elif not _repair_attempted:
                            with ui.card().classes("w-full q-mb-sm").style(
                                "border-left: 3px solid #2196f3;"
                            ):
                                with ui.row().classes("items-center gap-2"):
                                    ui.spinner(size="sm")
                                    ui.label(
                                        "Auto-repair will run shortly…"
                                    ).classes("text-caption text-grey-5")

                        _auto_pref = _load_settings_dict().get(
                            "auto_build_indexes", "prompt"
                        )
                        with ui.card().classes("w-full q-mb-sm").style(
                            "border-left: 3px solid #ff9800;"
                        ):
                            with ui.row().classes("items-center gap-2 q-mb-xs"):
                                ui.icon("warning_amber", color="orange").classes(
                                    "text-h5"
                                )
                                ui.label("Indexes Not Built").classes(
                                    "text-subtitle1 text-bold"
                                )
                            if _engine_error:
                                ui.label(f"Details: {_engine_error}").classes(
                                    "text-caption text-negative q-mb-xs"
                                )
                            ui.label(
                                "The RAG search indexes have not been built. "
                                "Click Build Indexes Now to index all BakkesMod "
                                "SDK documentation (~5–15 min)."
                            ).classes("text-caption text-grey-5 q-mb-sm")
                            ui.label(
                                "Startup behavior:"
                            ).classes("text-caption text-bold")
                            auto_radio = ui.radio(
                                {
                                    "auto": "Auto-build if missing (recommended)",
                                    "prompt": "Show this prompt on startup",
                                    "never": "Never auto-build",
                                },
                                value=_auto_pref,
                            ).classes("q-mb-sm")
                            build_log_widget = ui.log(
                                max_lines=200
                            ).classes("w-full h-32 q-mb-xs")

                            async def _start_build():
                                _s = _load_settings_dict()
                                _s["auto_build_indexes"] = auto_radio.value
                                try:
                                    SETTINGS_FILE.write_text(
                                        json.dumps(_s, indent=2),
                                        encoding="utf-8",
                                    )
                                except Exception:
                                    pass
                                ui.notify(
                                    "Building indexes — see Live Console for output.",
                                    type="info",
                                )
                                await _do_build_indexes_bg(build_log_widget)
                                ui.notify(
                                    "Build complete! Restart app to load indexes.",
                                    type="positive",
                                    timeout=8000,
                                )

                            ui.button(
                                "Build Indexes Now",
                                icon="build",
                                on_click=_start_build,
                            ).props("color=orange")
                        return

                    # ── Auto-repair success banner ───────────────────────
                    if _repair_report is not None and _repair_report.any_success:
                        with ui.card().classes("w-full q-mb-sm").style(
                            "border-left: 3px solid #4caf50;"
                        ):
                            with ui.row().classes("items-center gap-2 q-mb-xs"):
                                ui.icon("build_circle", color="green").classes("text-h5")
                                ui.label("Auto-Repair Applied").classes(
                                    "text-subtitle1 text-bold"
                                )
                            for line in _repair_report.as_lines():
                                ui.label(line).classes("text-caption text-mono q-ml-md")

                    # Provider status via get_llm
                    with ui.card().classes("w-full"):
                        ui.label("LLM Provider").classes("text-subtitle1")
                        try:
                            llm = await asyncio.get_running_loop().run_in_executor(
                                None,
                                lambda: get_llm(allow_null=True),
                            )
                            if isinstance(llm, NullLLM):
                                ui.label(
                                    "No LLM configured"
                                ).classes("text-warning")
                                ui.label(
                                    "Add an API key in Settings, or "
                                    "install Ollama for offline use."
                                ).classes("text-caption text-grey-5")
                            else:
                                provider_name = getattr(
                                    llm, "model", "unknown"
                                )
                                if hasattr(llm, "metadata"):
                                    provider_name = (
                                        llm.metadata.model_name
                                        or provider_name
                                    )
                                ui.label(
                                    f"Active: {provider_name}"
                                ).classes("text-positive")
                        except Exception as exc:
                            ui.label(
                                f"Provider detection failed: {exc}"
                            ).classes("text-negative")

                    # Basic stats
                    with ui.card().classes("w-full q-mt-md"):
                        ui.label("Session Stats").classes("text-subtitle1")
                        with ui.grid(columns=2).classes("w-full gap-4"):
                            ui.label(
                                f"Documents: {eng.num_documents}"
                            )
                            ui.label(f"Nodes: {eng.num_nodes}")

                    # Cost tracking
                    if (
                        hasattr(eng, "cost_tracker")
                        and eng.cost_tracker
                    ):
                        with ui.card().classes("w-full q-mt-md"):
                            ui.label("Cost Tracking").classes(
                                "text-subtitle1"
                            )
                            ct = eng.cost_tracker
                            try:
                                summary = ct.get_summary()
                                if isinstance(summary, dict):
                                    for k, v in summary.items():
                                        ui.label(f"{k}: {v}")
                                else:
                                    ui.label(str(summary))
                            except Exception:
                                ui.label(
                                    "Cost data not available"
                                ).classes("text-grey-5")

                    # Cache stats
                    if hasattr(eng, "cache") and eng.cache:
                        with ui.card().classes("w-full q-mt-md"):
                            ui.label("Cache").classes("text-subtitle1")
                            try:
                                stats = eng.cache.get_stats()
                                if isinstance(stats, dict):
                                    for k, v in stats.items():
                                        ui.label(f"{k}: {v}")
                                else:
                                    ui.label(str(stats))
                            except Exception:
                                ui.label(
                                    "Cache stats not available"
                                ).classes("text-grey-5")

                    # Metrics
                    if hasattr(eng, "metrics") and eng.metrics:
                        with ui.card().classes("w-full q-mt-md"):
                            ui.label("Metrics").classes(
                                "text-subtitle1"
                            )
                            try:
                                m = eng.metrics
                                if hasattr(m, "get_summary"):
                                    data = m.get_summary()
                                    if isinstance(data, dict):
                                        for k, v in data.items():
                                            ui.label(f"{k}: {v}")
                                    else:
                                        ui.label(str(data))
                                else:
                                    ui.label(
                                        "Metrics summary not available"
                                    ).classes("text-grey-5")
                            except Exception:
                                ui.label(
                                    "Metrics not available"
                                ).classes("text-grey-5")

            ui.button(
                "Refresh", on_click=refresh_dashboard
            ).props("color=orange")
            # NOTE: periodic timer intentionally omitted — it would raise
            # "parent slot deleted" errors when Dashboard is not the active tab.
            # Use the Refresh button or wait for auto-refresh on tab activation.

        # ==============================================================
        # Tab 6: Help
        # ==============================================================
        with ui.tab_panel(tab_help):
            ui.markdown(HELP_TEXT)

        # ==============================================================
        # Tab 7: Settings
        # ==============================================================
        with ui.tab_panel(tab_settings):
            # Load saved backend settings for initial widget values
            _saved = _load_settings_dict()
            _r = _saved.get("retriever", {})
            _c = _saved.get("cache", {})
            _s = _saved.get("self_rag", {})
            _g = _saved.get("codegen", {})
            _o = _saved.get("observability", {})
            _llm_s = _saved.get("llm", {})
            _emb_s = _saved.get("embedding", {})
            _chunk_s = _saved.get("chunking", {})
            _ver_s = _saved.get("verification", {})
            _cost_s = _saved.get("cost", {})
            _prod_s = _saved.get("production", {})
            _stor_s = _saved.get("storage", {})
            _cpp_s = _saved.get("cpp_intelligence", {})
            _ir_s = _saved.get("intent_router", {})
            _gr_s = _saved.get("guardrails", {})

            # ── 0. System Status Banner ────────────────────────────────
            with ui.card().classes("w-full q-mb-md").style(
                "border-left: 3px solid #ff9800;"
            ):
                with ui.row().classes("items-center justify-between w-full q-mb-sm"):
                    with ui.row().classes("items-center gap-2"):
                        ui.icon("monitor_heart", color="orange").classes("text-h5")
                        ui.label("System Status").classes("text-h6 text-bold")
                    diag_btn = ui.button(
                        "Run Diagnostics",
                        icon="play_arrow",
                    ).props("color=orange outline dense")

                diag_container = ui.column().classes("w-full gap-1")

                # Initial placeholder rows
                _DIAG_NAMES = [
                    "OpenAI Embeddings",
                    "Active LLM",
                    "Knowledge Graph Index",
                    "BM25 / Vector Index",
                    "Neural Reranker",
                    "Semantic Cache",
                    "Query Decomposition",
                    "Adaptive Retrieval",
                ]
                with diag_container:
                    for _dn in _DIAG_NAMES:
                        with ui.row().classes("items-center gap-3 q-py-xs"):
                            ui.icon("radio_button_unchecked", color="grey-6")
                            ui.label(_dn).classes("text-body2 text-bold").style(
                                "min-width: 180px;"
                            )
                            ui.label("Click Run Diagnostics").classes(
                                "text-caption text-grey-6"
                            )

                async def _on_run_diagnostics():
                    diag_container.clear()
                    with diag_container:
                        with ui.row().classes("items-center gap-2"):
                            ui.spinner(size="sm", color="orange")
                            ui.label("Running diagnostics...").classes(
                                "text-caption text-grey-5"
                            )
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None, _run_diagnostics_blocking
                    )
                    diag_container.clear()
                    with diag_container:
                        for _res in results:
                            with ui.row().classes("items-center gap-3 q-py-xs"):
                                if _res["grey"]:
                                    ui.icon("remove_circle_outline", color="grey-6")
                                elif _res["passing"]:
                                    ui.icon("check_circle", color="positive")
                                else:
                                    ui.icon("cancel", color="negative")
                                ui.label(_res["name"]).classes(
                                    "text-body2 text-bold"
                                ).style("min-width: 180px;")
                                ui.label(_res["detail"]).classes(
                                    "text-caption text-grey-5"
                                )

                diag_btn.on_click(_on_run_diagnostics)

            # ── 1. API Key Setup — Wizard Numbered Cards ───────────────
            ui.label("API Key Setup").classes(
                "text-subtitle1 text-bold text-orange q-mb-xs"
            )
            ui.label(
                "Keys are saved to .env and used on next restart."
            ).classes("text-caption text-grey-6 q-mb-sm")

            # Card 1 — Embeddings (REQUIRED)
            with ui.card().classes("w-full q-mb-sm").style(
                "border-left: 4px solid #f44336;"
            ):
                with ui.row().classes("items-start gap-3 w-full"):
                    with ui.element("div").style(
                        "background:#f44336;color:#fff;width:1.8rem;height:1.8rem;"
                        "border-radius:50%;display:flex;align-items:center;"
                        "justify-content:center;font-weight:bold;flex-shrink:0;"
                        "font-size:0.9rem;"
                    ):
                        ui.label("1")
                    with ui.column().classes("w-full gap-2"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label("Embeddings").classes(
                                "text-subtitle2 text-bold"
                            )
                            ui.badge("REQUIRED", color="negative").props("outline")
                        ui.label(
                            "Required for all indexing, search, and caching. "
                            "Nothing works without this."
                        ).classes("text-caption text-grey-5")
                        _oai_configured = bool(os.getenv("OPENAI_API_KEY"))
                        with ui.row().classes("items-center gap-2 w-full"):
                            ui.icon("hub", color="grey-5")
                            ui.label("OpenAI").classes(
                                "text-caption text-bold"
                            ).style("min-width:80px;")
                            openai_input = ui.input(
                                "OPENAI_API_KEY",
                                value=os.getenv("OPENAI_API_KEY", ""),
                                password=True,
                            ).classes("flex-grow")
                            ui.button(
                                "Test",
                                on_click=lambda: _test_api_key(
                                    "openai_embed", openai_input.value
                                ),
                            ).props("outline dense color=orange")
                            ui.badge(
                                "CONFIGURED" if _oai_configured else "MISSING",
                                color="positive" if _oai_configured else "negative",
                            )

            # Card 2 — LLM Provider (at least one required)
            with ui.card().classes("w-full q-mb-sm").style(
                "border-left: 4px solid #ff9800;"
            ):
                with ui.row().classes("items-start gap-3 w-full"):
                    with ui.element("div").style(
                        "background:#ff9800;color:#fff;width:1.8rem;height:1.8rem;"
                        "border-radius:50%;display:flex;align-items:center;"
                        "justify-content:center;font-weight:bold;flex-shrink:0;"
                        "font-size:0.9rem;"
                    ):
                        ui.label("2")
                    with ui.column().classes("w-full gap-2"):
                        with ui.row().classes("items-center gap-2"):
                            ui.label("LLM Provider").classes(
                                "text-subtitle2 text-bold"
                            )
                            ui.badge("AT LEAST ONE REQUIRED", color="warning").props(
                                "outline"
                            )
                        ui.label(
                            "Used to generate answers. System auto-picks the "
                            "best available provider from the fallback chain."
                        ).classes("text-caption text-grey-5")

                        _llm_provider_rows = [
                            ("anthropic", "ANTHROPIC_API_KEY",
                             "Anthropic Claude", "Best quality · Paid"),
                            ("google", "GOOGLE_API_KEY",
                             "Google Gemini", "High quality · Free tier"),
                            ("openrouter", "OPENROUTER_API_KEY",
                             "OpenRouter (DeepSeek)", "Free"),
                            ("openai", "OPENAI_API_KEY",
                             "OpenAI GPT-4o-mini", "Good fallback · Paid"),
                        ]
                        _llm_input_widgets = {}
                        for _pid, _ekey, _plabel, _clabel in _llm_provider_rows:
                            _configured = bool(os.getenv(_ekey))
                            with ui.row().classes("items-center gap-2 w-full"):
                                ui.icon(
                                    "check_circle" if _configured
                                    else "radio_button_unchecked",
                                    color="positive" if _configured else "grey-6",
                                )
                                ui.label(_plabel).classes(
                                    "text-caption text-bold"
                                ).style("min-width:150px;")
                                ui.label(_clabel).classes(
                                    "text-caption text-grey-5"
                                ).style("min-width:160px;")
                                _inp = ui.input(
                                    _ekey,
                                    value=os.getenv(_ekey, ""),
                                    password=True,
                                ).classes("flex-grow")
                                _llm_input_widgets[_pid] = _inp
                                _pid_cap = _pid
                                _inp_cap = _inp
                                ui.button(
                                    "Test",
                                    on_click=(
                                        lambda p=_pid_cap, i=_inp_cap:
                                        _test_api_key(p, i.value)
                                    ),
                                ).props("outline dense color=orange")

                        anthropic_input = _llm_input_widgets["anthropic"]
                        google_input = _llm_input_widgets["google"]
                        openrouter_input = _llm_input_widgets["openrouter"]
                        # openai_input already defined in Card 1

            # Card 3 — Optional Enhancements
            with ui.card().classes("w-full q-mb-md").style(
                "border-left: 4px solid #607d8b;"
            ):
                with ui.row().classes("items-start gap-3 w-full"):
                    with ui.element("div").style(
                        "background:#607d8b;color:#fff;width:1.8rem;height:1.8rem;"
                        "border-radius:50%;display:flex;align-items:center;"
                        "justify-content:center;font-weight:bold;flex-shrink:0;"
                        "font-size:0.9rem;"
                    ):
                        ui.label("3")
                    with ui.column().classes("w-full gap-2"):
                        ui.label("Optional Enhancements").classes(
                            "text-subtitle2 text-bold"
                        )
                        _cohere_configured = bool(os.getenv("COHERE_API_KEY"))
                        with ui.row().classes("items-center gap-2 w-full"):
                            ui.icon("hub", color="grey-5")
                            ui.label("Cohere").classes(
                                "text-caption text-bold"
                            ).style("min-width:80px;")
                            ui.label(
                                "Neural reranker (best quality). Falls back to "
                                "BGE/FlashRank if absent."
                            ).classes("text-caption text-grey-5").style(
                                "min-width:280px;"
                            )
                            cohere_input = ui.input(
                                "COHERE_API_KEY",
                                value=os.getenv("COHERE_API_KEY", ""),
                                password=True,
                            ).classes("flex-grow")
                            ui.button(
                                "Test",
                                on_click=lambda: _test_api_key(
                                    "cohere", cohere_input.value
                                ),
                            ).props("outline dense color=orange")
                            ui.badge(
                                "CONFIGURED" if _cohere_configured else "OPTIONAL",
                                color="positive" if _cohere_configured else "grey",
                            )
                        with ui.row().classes("items-center gap-2 w-full q-mt-xs"):
                            ui.icon("attach_money", color="grey-5")
                            ui.label("Daily Budget").classes(
                                "text-caption text-bold"
                            ).style("min-width:80px;")
                            ui.label("USD per day (0 = unlimited)").classes(
                                "text-caption text-grey-5"
                            ).style("min-width:280px;")
                            budget_input = ui.number(
                                "Daily Budget (USD)",
                                value=float(
                                    os.getenv("DAILY_BUDGET_USD", "0") or "0"
                                ),
                                min=0,
                                step=1,
                            ).classes("flex-grow")

            # ── 2.5. Feature Toggles ──────────────────────────────────
            # Authoritative on/off switches for every feature.  The same
            # switches appear again inside their respective expansion sections
            # below, kept in sync via bind_value().
            _ft_state = {
                # Retrieval
                "enable_kg": _r.get("enable_kg", True),
                "enable_reranker": _r.get("enable_reranker", True),
                "enable_llm_rewrite": _r.get("enable_llm_rewrite", True),
                "enable_decomp": _r.get("enable_query_decomposition", True),
                "adaptive_top_k": _r.get("adaptive_top_k", True),
                "hierarchical": _r.get("use_hierarchical_chunking", True),
                "use_mmr": _r.get("use_mmr", True),
                "use_colbert": _r.get("use_colbert", False),
                # Processing
                "semantic_chunking": _chunk_s.get("enable_semantic_chunking", True),
                "cache_enabled": _c.get("enabled", True),
                # Quality
                "srag_enabled": _s.get("enabled", True),
                "ver_enabled": _ver_s.get("enabled", True),
                # Code Intelligence
                "cg_enabled": _g.get("enabled", True),
                "cg_validate": _g.get("validate_output", True),
                "cg_self_improving": _g.get("self_improving", True),
                "cg_compile": _g.get("enable_compilation", True),
                "cg_feedback": _g.get("feedback_enabled", True),
                "cpp_enabled": _cpp_s.get("enabled", True),
                # System
                "ir_enabled": _ir_s.get("enabled", True),
                "gr_enabled": _gr_s.get("enabled", True),
                "cost_track": _cost_s.get("track_costs", True),
                "rate_limit": _prod_s.get("rate_limit_enabled", True),
                "circuit_breaker": _prod_s.get("circuit_breaker_enabled", True),
                # Observability
                "phoenix_enabled": _o.get("phoenix_enabled", False),
                "prometheus_enabled": _o.get("prometheus_enabled", False),
                "otel_enabled": _o.get("enable_otel", False),
            }

            with ui.card().classes("w-full q-mb-md").style(
                "border-left: 3px solid #4caf50;"
            ):
                with ui.row().classes("items-center gap-2 q-mb-xs"):
                    ui.icon("toggle_on", color="positive").classes("text-h5")
                    ui.label("Feature Toggles").classes("text-subtitle1 text-bold")
                    ui.label(
                        "Master on/off for every feature — saved with Settings."
                    ).classes("text-caption text-grey-5 q-ml-sm")

                # Retrieval
                ui.label("Retrieval").classes(
                    "text-caption text-bold text-orange q-mt-xs"
                )
                with ui.row().classes("flex-wrap gap-x-8 gap-y-1 q-mb-xs"):
                    ret_enable_kg = ui.switch(
                        "Knowledge Graph",
                        value=_ft_state["enable_kg"],
                    ).bind_value(_ft_state, "enable_kg")
                    ret_enable_reranker = ui.switch(
                        "Neural Reranker",
                        value=_ft_state["enable_reranker"],
                    ).bind_value(_ft_state, "enable_reranker")
                    ret_enable_llm_rewrite = ui.switch(
                        "LLM Query Rewrite",
                        value=_ft_state["enable_llm_rewrite"],
                    ).bind_value(_ft_state, "enable_llm_rewrite")
                    ret_enable_decomp = ui.switch(
                        "Query Decomposition",
                        value=_ft_state["enable_decomp"],
                    ).bind_value(_ft_state, "enable_decomp")
                    ret_adaptive_top_k = ui.switch(
                        "Adaptive top_k",
                        value=_ft_state["adaptive_top_k"],
                    ).bind_value(_ft_state, "adaptive_top_k")
                    ret_hierarchical = ui.switch(
                        "Hierarchical Chunking",
                        value=_ft_state["hierarchical"],
                    ).bind_value(_ft_state, "hierarchical")
                    ret_use_mmr = ui.switch(
                        "MMR Diversity",
                        value=_ft_state["use_mmr"],
                    ).bind_value(_ft_state, "use_mmr")
                    ret_use_colbert = ui.switch(
                        "ColBERT (experimental)",
                        value=_ft_state["use_colbert"],
                    ).bind_value(_ft_state, "use_colbert")

                # Processing
                ui.label("Processing").classes(
                    "text-caption text-bold text-orange q-mt-xs"
                )
                with ui.row().classes("flex-wrap gap-x-8 gap-y-1 q-mb-xs"):
                    chunk_semantic = ui.switch(
                        "Semantic Chunking",
                        value=_ft_state["semantic_chunking"],
                    ).bind_value(_ft_state, "semantic_chunking")
                    cache_enabled = ui.switch(
                        "Semantic Cache",
                        value=_ft_state["cache_enabled"],
                    ).bind_value(_ft_state, "cache_enabled")
                    srag_enabled = ui.switch(
                        "Self-RAG Retry",
                        value=_ft_state["srag_enabled"],
                    ).bind_value(_ft_state, "srag_enabled")
                    ver_enabled = ui.switch(
                        "Answer Verification",
                        value=_ft_state["ver_enabled"],
                    ).bind_value(_ft_state, "ver_enabled")

                # Code Intelligence
                ui.label("Code Intelligence").classes(
                    "text-caption text-bold text-orange q-mt-xs"
                )
                with ui.row().classes("flex-wrap gap-x-8 gap-y-1 q-mb-xs"):
                    cg_enabled = ui.switch(
                        "Code Generation",
                        value=_ft_state["cg_enabled"],
                    ).bind_value(_ft_state, "cg_enabled")
                    cg_validate = ui.switch(
                        "Validate Output",
                        value=_ft_state["cg_validate"],
                    ).bind_value(_ft_state, "cg_validate")
                    cg_self_improving = ui.switch(
                        "Self-Improving Loop",
                        value=_ft_state["cg_self_improving"],
                    ).bind_value(_ft_state, "cg_self_improving")
                    cg_compile = ui.switch(
                        "MSVC Compilation",
                        value=_ft_state["cg_compile"],
                    ).bind_value(_ft_state, "cg_compile")
                    cg_feedback = ui.switch(
                        "Feedback Learning",
                        value=_ft_state["cg_feedback"],
                    ).bind_value(_ft_state, "cg_feedback")
                    cpp_enabled = ui.switch(
                        "C++ Intelligence",
                        value=_ft_state["cpp_enabled"],
                    ).bind_value(_ft_state, "cpp_enabled")

                # System
                ui.label("System").classes(
                    "text-caption text-bold text-orange q-mt-xs"
                )
                with ui.row().classes("flex-wrap gap-x-8 gap-y-1 q-mb-xs"):
                    ir_enabled = ui.switch(
                        "Intent Router",
                        value=_ft_state["ir_enabled"],
                    ).bind_value(_ft_state, "ir_enabled")
                    gr_enabled = ui.switch(
                        "Guardrails",
                        value=_ft_state["gr_enabled"],
                    ).bind_value(_ft_state, "gr_enabled")
                    cost_track = ui.switch(
                        "Cost Tracking",
                        value=_ft_state["cost_track"],
                    ).bind_value(_ft_state, "cost_track")
                    prod_rate_limit = ui.switch(
                        "Rate Limiting",
                        value=_ft_state["rate_limit"],
                    ).bind_value(_ft_state, "rate_limit")
                    prod_circuit = ui.switch(
                        "Circuit Breaker",
                        value=_ft_state["circuit_breaker"],
                    ).bind_value(_ft_state, "circuit_breaker")

                # Observability
                ui.label("Observability").classes(
                    "text-caption text-bold text-orange q-mt-xs"
                )
                with ui.row().classes("flex-wrap gap-x-8 gap-y-1"):
                    obs_phoenix = ui.switch(
                        "Phoenix Tracing",
                        value=_ft_state["phoenix_enabled"],
                    ).bind_value(_ft_state, "phoenix_enabled")
                    obs_prometheus = ui.switch(
                        "Prometheus Metrics",
                        value=_ft_state["prometheus_enabled"],
                    ).bind_value(_ft_state, "prometheus_enabled")
                    obs_otel = ui.switch(
                        "OpenTelemetry",
                        value=_ft_state["otel_enabled"],
                    ).bind_value(_ft_state, "otel_enabled")

            # ── 2. Backend Configuration Sections ─────────────────────
            ui.label("Backend Configuration").classes(
                "text-subtitle1 text-bold text-orange q-mb-xs"
            )
            ui.label(
                "All changes take effect on next app restart."
            ).classes("text-caption text-grey-6 q-mb-sm")

            # ── LLM Configuration ──────────────────────────────────────
            with ui.expansion("LLM Configuration", icon="psychology").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Which LLM providers and models are used for answering "
                        "queries and building the knowledge graph."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    with ui.grid(columns=2).classes("w-full gap-4"):
                        _llm_primary_prov_init = _llm_s.get("primary_provider", "gemini")
                        llm_primary_provider = ui.select(
                            label="Primary Provider",
                            options=["anthropic", "openai", "gemini", "openrouter", "ollama"],
                            value=_llm_primary_prov_init,
                        )
                        _pm_opts = PROVIDER_MODELS.get(_llm_primary_prov_init, [])
                        _pm_saved = _llm_s.get("primary_model", "gemini-2.5-flash")
                        _pm_init = _pm_saved if _pm_saved in _pm_opts else (_pm_opts[0] if _pm_opts else _pm_saved)
                        llm_primary_model = ui.select(
                            label="Primary Model",
                            options=_pm_opts or [_pm_saved],
                            value=_pm_init,
                            with_input=True,
                        ).classes("w-full")

                        _llm_kg_prov_init = _llm_s.get("kg_provider", "openai")
                        llm_kg_provider = ui.select(
                            label="KG Provider",
                            options=["openai", "anthropic", "gemini"],
                            value=_llm_kg_prov_init,
                        )
                        _kgm_opts = PROVIDER_MODELS.get(_llm_kg_prov_init, [])
                        _kgm_saved = _llm_s.get("kg_model", "gpt-4o-mini")
                        _kgm_init = _kgm_saved if _kgm_saved in _kgm_opts else (_kgm_opts[0] if _kgm_opts else _kgm_saved)
                        llm_kg_model = ui.select(
                            label="KG Model",
                            options=_kgm_opts or [_kgm_saved],
                            value=_kgm_init,
                            with_input=True,
                        ).classes("w-full")

                    # Provider → model cascade + live refresh
                    def _on_primary_provider_change(e):
                        prov = e.value
                        models = PROVIDER_MODELS.get(prov, [])
                        llm_primary_model.options = models
                        if llm_primary_model.value not in models and models:
                            llm_primary_model.value = models[0]
                        llm_primary_model.update()

                    def _on_kg_provider_change(e):
                        prov = e.value
                        models = PROVIDER_MODELS.get(prov, [])
                        llm_kg_model.options = models
                        if llm_kg_model.value not in models and models:
                            llm_kg_model.value = models[0]
                        llm_kg_model.update()

                    llm_primary_provider.on_value_change(_on_primary_provider_change)
                    llm_kg_provider.on_value_change(_on_kg_provider_change)

                    async def _refresh_primary_models():
                        prov = llm_primary_provider.value
                        key = os.getenv(
                            {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
                             "gemini": "GOOGLE_API_KEY", "openrouter": "OPENROUTER_API_KEY"}.get(prov, ""),
                            ""
                        )
                        ui.notify(f"Fetching {prov} models…", type="info", timeout=3000)
                        models = await _fetch_models_live(prov, key)
                        llm_primary_model.options = models
                        if llm_primary_model.value not in models and models:
                            llm_primary_model.value = models[0]
                        llm_primary_model.update()
                        ui.notify(f"Loaded {len(models)} models for {prov}", type="positive")

                    async def _refresh_kg_models():
                        prov = llm_kg_provider.value
                        key = os.getenv(
                            {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
                             "gemini": "GOOGLE_API_KEY", "openrouter": "OPENROUTER_API_KEY"}.get(prov, ""),
                            ""
                        )
                        ui.notify(f"Fetching {prov} models…", type="info", timeout=3000)
                        models = await _fetch_models_live(prov, key)
                        llm_kg_model.options = models
                        if llm_kg_model.value not in models and models:
                            llm_kg_model.value = models[0]
                        llm_kg_model.update()
                        ui.notify(f"Loaded {len(models)} models for {prov}", type="positive")

                    with ui.row().classes("gap-2 q-mt-xs"):
                        ui.button(
                            "Refresh Primary Models", icon="refresh",
                            on_click=_refresh_primary_models,
                        ).props("dense flat color=grey-5")
                        ui.button(
                            "Refresh KG Models", icon="refresh",
                            on_click=_refresh_kg_models,
                        ).props("dense flat color=grey-5")

                    with ui.grid(columns=2).classes("w-full gap-4 q-mt-sm"):
                        llm_temperature = ui.number(
                            "Temperature (0–2)",
                            value=_llm_s.get("temperature", 0.0),
                            min=0.0, max=2.0, step=0.1,
                        )
                        llm_max_retries = ui.number(
                            "Max Retries",
                            value=_llm_s.get("max_retries", 5),
                            min=0, max=20, step=1,
                        )
                        llm_timeout = ui.number(
                            "Timeout (seconds)",
                            value=_llm_s.get("timeout", 60),
                            min=5, max=300, step=5,
                        )

            # ── Embedding ──────────────────────────────────────────────
            with ui.expansion("Embedding", icon="functions").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Embedding model used for vector search and semantic cache."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    with ui.grid(columns=2).classes("w-full gap-4"):
                        emb_provider = ui.select(
                            label="Provider",
                            options=["openai", "huggingface"],
                            value=_emb_s.get("provider", "openai"),
                        )
                        emb_model = ui.input(
                            "Model",
                            value=_emb_s.get("model", "text-embedding-3-small"),
                        )
                        emb_batch_size = ui.number(
                            "Batch Size",
                            value=_emb_s.get("batch_size", 100),
                            min=1, max=500, step=10,
                        )
                        emb_max_retries = ui.number(
                            "Max Retries",
                            value=_emb_s.get("max_retries", 10),
                            min=0, max=20, step=1,
                        )

            # ── Retrieval ──────────────────────────────────────────────
            with ui.expansion("Retrieval", icon="search").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "3-way fusion: Vector + BM25 + Knowledge Graph. "
                        "Controls how documents are fetched, fused, and reranked."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    with ui.grid(columns=3).classes("w-full gap-4"):
                        ret_vector_top_k = ui.number(
                            "Vector top_k",
                            value=_r.get("vector_top_k", 5),
                            min=1, max=50, step=1,
                        )
                        ret_bm25_top_k = ui.number(
                            "BM25 top_k",
                            value=_r.get("bm25_top_k", 5),
                            min=1, max=50, step=1,
                        )
                        ret_kg_similarity_top_k = ui.number(
                            "KG similarity top_k",
                            value=_r.get("kg_similarity_top_k", 3),
                            min=1, max=20, step=1,
                        )
                        ret_kg_max_triplets = ui.number(
                            "KG max triplets/chunk",
                            value=_r.get("kg_max_triplets_per_chunk", 2),
                            min=1, max=10, step=1,
                        )
                        ret_fusion_num_queries = ui.number(
                            "Fusion query variants",
                            value=_r.get("fusion_num_queries", 4),
                            min=1, max=8, step=1,
                        )
                        ret_rerank_top_n = ui.number(
                            "Rerank top_n",
                            value=_r.get("rerank_top_n", 5),
                            min=1, max=30, step=1,
                        )
                        ret_max_sub_queries = ui.number(
                            "Max sub-queries",
                            value=_r.get("max_sub_queries", 4),
                            min=1, max=10, step=1,
                        )
                        ret_decomp_complexity = ui.number(
                            "Decomp complexity threshold",
                            value=_r.get("decomposition_complexity_threshold", 80),
                            min=10, max=200, step=5,
                        )
                        ret_merge_threshold = ui.number(
                            "Hierarchical merge threshold",
                            value=_r.get("merge_threshold", 0.5),
                            min=0.0, max=1.0, step=0.05,
                        )
                        ret_mmr_threshold = ui.number(
                            "MMR diversity threshold",
                            value=_r.get("mmr_threshold", 0.7),
                            min=0.0, max=1.0, step=0.05,
                        )
                    ret_fusion_mode = ui.select(
                        label="Fusion Mode",
                        options=["reciprocal_rerank", "simple"],
                        value=_r.get("fusion_mode", "reciprocal_rerank"),
                    ).classes("q-mt-sm").style("max-width:300px;")
                    ui.label(
                        "Toggles (synced with Feature Toggles above):"
                    ).classes("text-caption text-bold q-mt-sm")
                    with ui.row().classes("flex-wrap gap-4 q-mt-xs"):
                        ui.switch(
                            "Knowledge Graph",
                            value=_ft_state["enable_kg"],
                        ).bind_value(_ft_state, "enable_kg")
                        ui.switch(
                            "Neural Reranker",
                            value=_ft_state["enable_reranker"],
                        ).bind_value(_ft_state, "enable_reranker")
                        ui.switch(
                            "LLM Query Rewrite",
                            value=_ft_state["enable_llm_rewrite"],
                        ).bind_value(_ft_state, "enable_llm_rewrite")
                        ui.switch(
                            "Query Decomposition",
                            value=_ft_state["enable_decomp"],
                        ).bind_value(_ft_state, "enable_decomp")
                        ui.switch(
                            "Adaptive top_k",
                            value=_ft_state["adaptive_top_k"],
                        ).bind_value(_ft_state, "adaptive_top_k")
                        ui.switch(
                            "Hierarchical Chunking",
                            value=_ft_state["hierarchical"],
                        ).bind_value(_ft_state, "hierarchical")
                        ui.switch(
                            "MMR Diversity",
                            value=_ft_state["use_mmr"],
                        ).bind_value(_ft_state, "use_mmr")
                        ui.switch(
                            "ColBERT (experimental)",
                            value=_ft_state["use_colbert"],
                        ).bind_value(_ft_state, "use_colbert")
                    ret_colbert_model = ui.input(
                        "ColBERT Model",
                        value=_r.get("colbert_model", "colbert-ir/colbertv2.0"),
                    ).classes("q-mt-sm").style("max-width:400px;")

            # ── Chunking ───────────────────────────────────────────────
            with ui.expansion("Chunking", icon="content_cut").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Controls how documents are split into chunks for indexing. "
                        "Markdown files use text chunking; .h/.cpp use code chunking."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    with ui.grid(columns=2).classes("w-full gap-4"):
                        chunk_size = ui.number(
                            "Chunk Size",
                            value=_chunk_s.get("chunk_size", 1024),
                            min=128, max=4096, step=64,
                        )
                        chunk_overlap = ui.number(
                            "Chunk Overlap",
                            value=_chunk_s.get("chunk_overlap", 128),
                            min=0, max=512, step=16,
                        )
                        chunk_semantic_pct = ui.number(
                            "Semantic Breakpoint %ile",
                            value=_chunk_s.get("semantic_breakpoint_percentile", 95),
                            min=50, max=99, step=1,
                        )
                        chunk_code_lines = ui.number(
                            "Code Chunk Lines",
                            value=_chunk_s.get("code_chunk_lines", 40),
                            min=10, max=200, step=5,
                        )
                        chunk_code_overlap = ui.number(
                            "Code Chunk Overlap Lines",
                            value=_chunk_s.get("code_chunk_lines_overlap", 15),
                            min=0, max=50, step=5,
                        )
                    with ui.row().classes("flex-wrap gap-4 q-mt-sm"):
                        ui.switch(
                            "Semantic Chunking",
                            value=_ft_state["semantic_chunking"],
                        ).bind_value(_ft_state, "semantic_chunking")
                        chunk_include_metadata = ui.switch(
                            "Include Metadata",
                            value=_chunk_s.get("include_metadata", True),
                        )
                        chunk_prev_next = ui.switch(
                            "Prev/Next Relations",
                            value=_chunk_s.get("include_prev_next_rel", True),
                        )

            # ── Semantic Cache ─────────────────────────────────────────
            with ui.expansion("Semantic Cache", icon="cached").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Caches query results by embedding similarity "
                        "(saves API cost for repeated questions)."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    ui.switch(
                        "Enable Cache",
                        value=_ft_state["cache_enabled"],
                    ).bind_value(_ft_state, "cache_enabled")
                    with ui.grid(columns=2).classes("w-full gap-4 q-mt-sm"):
                        cache_threshold = ui.number(
                            "Similarity Threshold (0–1)",
                            value=_c.get("similarity_threshold", 0.92),
                            min=0.5, max=1.0, step=0.01,
                        )
                        cache_ttl_days = ui.number(
                            "TTL (days)",
                            value=_c.get("ttl_seconds", 86400 * 7) // 86400,
                            min=1, max=90, step=1,
                        )
                        cache_redis_db = ui.number(
                            "Redis DB",
                            value=_c.get("redis_db", 0),
                            min=0, max=15, step=1,
                        )
                    cache_backend = ui.select(
                        label="Cache Backend",
                        options=["file", "redis"],
                        value=_c.get("backend", "file"),
                    ).classes("q-mt-sm").style("max-width:200px;")
                    cache_redis_url = ui.input(
                        "Redis URL",
                        value=_c.get("redis_url", "redis://localhost:6379"),
                    ).classes("w-full q-mt-sm")
                    cache_dir = ui.input(
                        "Cache Directory",
                        value=_c.get("cache_dir", ".cache/semantic"),
                    ).classes("w-full q-mt-sm")

            # ── Self-RAG ───────────────────────────────────────────────
            with ui.expansion("Self-RAG (Retry Loop)", icon="refresh").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Automatically retries low-confidence answers "
                        "with a broader retrieval pass."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    with ui.row().classes("items-center gap-6"):
                        ui.switch(
                            "Enable Self-RAG",
                            value=_ft_state["srag_enabled"],
                        ).bind_value(_ft_state, "srag_enabled")
                        srag_force_rewrite = ui.switch(
                            "Force LLM Rewrite on Retry",
                            value=_s.get("force_llm_rewrite_on_retry", True),
                        )
                    with ui.grid(columns=2).classes("w-full gap-4 q-mt-sm"):
                        srag_threshold = ui.number(
                            "Confidence Threshold (0–1)",
                            value=_s.get("confidence_threshold", 0.70),
                            min=0.0, max=1.0, step=0.05,
                        )
                        srag_retries = ui.number(
                            "Max Retries",
                            value=_s.get("max_retries", 2),
                            min=0, max=5, step=1,
                        )

            # ── Answer Verification ────────────────────────────────────
            with ui.expansion("Answer Verification", icon="fact_check").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Checks whether generated answers are grounded in "
                        "the retrieved source documents."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    ui.switch(
                        "Enable Verification",
                        value=_ft_state["ver_enabled"],
                    ).bind_value(_ft_state, "ver_enabled")
                    with ui.grid(columns=2).classes("w-full gap-4 q-mt-sm"):
                        ver_grounded = ui.number(
                            "Grounded Threshold (0–1)",
                            value=_ver_s.get("grounded_threshold", 0.75),
                            min=0.0, max=1.0, step=0.05,
                        )
                        ver_borderline = ui.number(
                            "Borderline Threshold (0–1)",
                            value=_ver_s.get("borderline_threshold", 0.55),
                            min=0.0, max=1.0, step=0.05,
                        )
                        ver_borderline_penalty = ui.number(
                            "Borderline Confidence Penalty",
                            value=_ver_s.get("borderline_confidence_penalty", 0.15),
                            min=0.0, max=1.0, step=0.05,
                        )
                        ver_ungrounded_penalty = ui.number(
                            "Ungrounded Confidence Penalty",
                            value=_ver_s.get("ungrounded_confidence_penalty", 0.30),
                            min=0.0, max=1.0, step=0.05,
                        )

            # ── Code Generation ────────────────────────────────────────
            with ui.expansion("Code Generation", icon="code").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Controls the self-improving code generation loop "
                        "and optional MSVC compilation check."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    with ui.row().classes("flex-wrap gap-6"):
                        ui.switch(
                            "Enabled",
                            value=_ft_state["cg_enabled"],
                        ).bind_value(_ft_state, "cg_enabled")
                        ui.switch(
                            "Validate Output",
                            value=_ft_state["cg_validate"],
                        ).bind_value(_ft_state, "cg_validate")
                        ui.switch(
                            "Self-Improving Loop",
                            value=_ft_state["cg_self_improving"],
                        ).bind_value(_ft_state, "cg_self_improving")
                        ui.switch(
                            "MSVC Compilation Check",
                            value=_ft_state["cg_compile"],
                        ).bind_value(_ft_state, "cg_compile")
                        ui.switch(
                            "Feedback Learning",
                            value=_ft_state["cg_feedback"],
                        ).bind_value(_ft_state, "cg_feedback")
                    with ui.grid(columns=2).classes("w-full gap-4 q-mt-sm"):
                        cg_max_iter = ui.number(
                            "Max Fix Iterations",
                            value=_g.get("max_fix_iterations", 5),
                            min=1, max=10, step=1,
                        )
                        cg_max_chunks = ui.number(
                            "Max Context Chunks",
                            value=_g.get("max_context_chunks", 5),
                            min=1, max=20, step=1,
                        )
                    cg_msvc_path = ui.input(
                        "MSVC Path (blank = auto-detect via vswhere)",
                        value=_g.get("msvc_path", "") or "",
                    ).classes("w-full q-mt-sm")

            # ── Observability ──────────────────────────────────────────
            with ui.expansion("Observability", icon="monitoring").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Structured logging, Phoenix tracing, "
                        "Prometheus metrics, and OpenTelemetry."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    with ui.grid(columns=2).classes("w-full gap-4"):
                        obs_log_level = ui.select(
                            label="Log Level",
                            options=["DEBUG", "INFO", "WARNING", "ERROR"],
                            value=_o.get("log_level", "INFO"),
                        )
                        obs_log_format = ui.input(
                            "Log Format",
                            value=_o.get("log_format", "json"),
                        )
                        obs_phoenix_host = ui.input(
                            "Phoenix Host",
                            value=_o.get("phoenix_host", "127.0.0.1"),
                        )
                        obs_phoenix_port = ui.number(
                            "Phoenix Port",
                            value=_o.get("phoenix_port", 6006),
                            min=1, max=65535, step=1,
                        )
                        obs_prometheus_port = ui.number(
                            "Prometheus Port",
                            value=_o.get("prometheus_port", 8000),
                            min=1, max=65535, step=1,
                        )
                        obs_otel_endpoint = ui.input(
                            "OTel Endpoint",
                            value=_o.get("otel_endpoint", "http://localhost:4317"),
                        )
                        obs_otel_service = ui.input(
                            "OTel Service Name",
                            value=_o.get("otel_service_name", "bakkesmod-rag"),
                        )
                    with ui.row().classes("flex-wrap gap-4 q-mt-sm"):
                        ui.switch(
                            "Phoenix Tracing",
                            value=_ft_state["phoenix_enabled"],
                        ).bind_value(_ft_state, "phoenix_enabled")
                        ui.switch(
                            "Prometheus Metrics",
                            value=_ft_state["prometheus_enabled"],
                        ).bind_value(_ft_state, "prometheus_enabled")
                        ui.switch(
                            "OpenTelemetry",
                            value=_ft_state["otel_enabled"],
                        ).bind_value(_ft_state, "otel_enabled")

            # ── Cost Tracking ──────────────────────────────────────────
            with ui.expansion("Cost Tracking", icon="attach_money").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Per-token cost tracking and budget alerts."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    with ui.row().classes("items-center gap-4"):
                        ui.switch(
                            "Track Costs",
                            value=_ft_state["cost_track"],
                        ).bind_value(_ft_state, "cost_track")
                        cost_alert_pct = ui.number(
                            "Alert Threshold (%)",
                            value=_cost_s.get("alert_threshold_pct", 80.0),
                            min=10, max=100, step=5,
                        ).style("max-width:200px;")
                    ui.label(
                        "Per-provider rates (USD per 1M tokens):"
                    ).classes("text-caption text-bold q-mt-sm")
                    with ui.grid(columns=3).classes("w-full gap-4 q-mt-xs"):
                        cost_embed = ui.number(
                            "OpenAI Embedding",
                            value=_cost_s.get("openai_embedding_cost", 0.02),
                            min=0, step=0.001,
                        )
                        cost_gpt4mini_in = ui.number(
                            "GPT-4o-mini Input",
                            value=_cost_s.get("openai_gpt4o_mini_input", 0.15),
                            min=0, step=0.01,
                        )
                        cost_gpt4mini_out = ui.number(
                            "GPT-4o-mini Output",
                            value=_cost_s.get("openai_gpt4o_mini_output", 0.60),
                            min=0, step=0.01,
                        )
                        cost_claude_in = ui.number(
                            "Claude Sonnet Input",
                            value=_cost_s.get("anthropic_claude_sonnet_input", 3.0),
                            min=0, step=0.1,
                        )
                        cost_claude_out = ui.number(
                            "Claude Sonnet Output",
                            value=_cost_s.get("anthropic_claude_sonnet_output", 15.0),
                            min=0, step=0.5,
                        )
                        cost_gemini_in = ui.number(
                            "Gemini Flash Input",
                            value=_cost_s.get("gemini_flash_input", 0.075),
                            min=0, step=0.01,
                        )
                        cost_gemini_out = ui.number(
                            "Gemini Flash Output",
                            value=_cost_s.get("gemini_flash_output", 0.30),
                            min=0, step=0.01,
                        )

            # ── Production Resilience ──────────────────────────────────
            with ui.expansion("Production Resilience", icon="shield").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Circuit breakers, rate limiting, and retry strategies."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    with ui.row().classes("flex-wrap gap-4"):
                        ui.switch(
                            "Rate Limiting",
                            value=_ft_state["rate_limit"],
                        ).bind_value(_ft_state, "rate_limit")
                        ui.switch(
                            "Circuit Breaker",
                            value=_ft_state["circuit_breaker"],
                        ).bind_value(_ft_state, "circuit_breaker")
                        prod_jitter = ui.switch(
                            "Retry Jitter",
                            value=_prod_s.get("retry_jitter", True),
                        )
                    with ui.grid(columns=3).classes("w-full gap-4 q-mt-sm"):
                        prod_rpm = ui.number(
                            "Requests/Minute",
                            value=_prod_s.get("requests_per_minute", 60),
                            min=1, max=1000, step=10,
                        )
                        prod_failure = ui.number(
                            "Circuit Failure Threshold",
                            value=_prod_s.get("failure_threshold", 5),
                            min=1, max=20, step=1,
                        )
                        prod_recovery = ui.number(
                            "Circuit Recovery Timeout (s)",
                            value=_prod_s.get("recovery_timeout", 60),
                            min=5, max=600, step=5,
                        )
                        prod_max_retries = ui.number(
                            "Max Retries",
                            value=_prod_s.get("max_retries", 3),
                            min=0, max=10, step=1,
                        )
                        prod_backoff = ui.number(
                            "Retry Backoff Factor",
                            value=_prod_s.get("retry_backoff_factor", 2.0),
                            min=1.0, max=10.0, step=0.5,
                        )

            # ── Storage Paths ──────────────────────────────────────────
            with ui.expansion("Storage Paths", icon="folder").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Filesystem paths for index storage, caches, and logs."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    stor_dir = ui.input(
                        "Storage Directory",
                        value=_stor_s.get("storage_dir", "rag_storage"),
                    ).classes("w-full")
                    stor_cache_dir = ui.input(
                        "Cache Directory",
                        value=_stor_s.get("cache_dir", ".cache"),
                    ).classes("w-full q-mt-sm")
                    stor_logs_dir = ui.input(
                        "Logs Directory",
                        value=_stor_s.get("logs_dir", "logs"),
                    ).classes("w-full q-mt-sm")

            # ── C++ Intelligence ───────────────────────────────────────
            with ui.expansion("C++ Intelligence", icon="memory").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Tree-sitter-based extraction of class hierarchies, "
                        "method signatures, and inheritance chains from SDK headers."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    ui.switch(
                        "Enabled",
                        value=_ft_state["cpp_enabled"],
                    ).bind_value(_ft_state, "cpp_enabled")
                    with ui.grid(columns=3).classes("w-full gap-4 q-mt-sm"):
                        cpp_max_methods = ui.number(
                            "Max Methods in Metadata",
                            value=_cpp_s.get("max_methods_in_metadata", 30),
                            min=5, max=100, step=5,
                        )
                        cpp_max_sigs = ui.number(
                            "Max Signatures in Metadata",
                            value=_cpp_s.get("max_signatures_in_metadata", 20),
                            min=5, max=100, step=5,
                        )
                        cpp_max_types = ui.number(
                            "Max Related Types",
                            value=_cpp_s.get("max_related_types", 15),
                            min=5, max=50, step=5,
                        )
                    with ui.row().classes("flex-wrap gap-4 q-mt-sm"):
                        cpp_inheritance = ui.switch(
                            "Include Inheritance Chain",
                            value=_cpp_s.get("include_inheritance_chain", True),
                        )
                        cpp_sigs = ui.switch(
                            "Include Method Signatures",
                            value=_cpp_s.get("include_method_signatures", True),
                        )
                        cpp_fwd_decl = ui.switch(
                            "Include Forward Declarations",
                            value=_cpp_s.get("include_forward_declarations", True),
                        )

            # ── Intent Router ──────────────────────────────────────────
            with ui.expansion("Intent Router", icon="alt_route").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Classifies query intent (SDK lookup, code generation, "
                        "how-to, etc.) to route to the best pipeline."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    with ui.row().classes("items-center gap-6"):
                        ui.switch(
                            "Enabled",
                            value=_ft_state["ir_enabled"],
                        ).bind_value(_ft_state, "ir_enabled")
                        ir_threshold = ui.number(
                            "LLM Confirmation Threshold (0–1)",
                            value=_ir_s.get("llm_confirmation_threshold", 0.6),
                            min=0.0, max=1.0, step=0.05,
                        ).style("max-width:280px;")

            # ── Guardrails ─────────────────────────────────────────────
            with ui.expansion("Guardrails", icon="security").classes(
                "w-full q-mt-xs"
            ):
                with ui.card().classes("w-full"):
                    ui.label(
                        "Input validation and safety checks applied to all queries."
                    ).classes("text-caption text-grey-5 q-mb-sm")
                    ui.switch(
                        "Enabled",
                        value=_ft_state["gr_enabled"],
                    ).bind_value(_ft_state, "gr_enabled")
                    with ui.grid(columns=2).classes("w-full gap-4 q-mt-sm"):
                        gr_min = ui.number(
                            "Min Query Length (chars)",
                            value=_gr_s.get("min_length", 3),
                            min=1, max=20, step=1,
                        )
                        gr_max = ui.number(
                            "Max Query Length (chars)",
                            value=_gr_s.get("max_length", 1000),
                            min=100, max=5000, step=100,
                        )

            # ── 3. Save Button ─────────────────────────────────────────
            def save_settings():
                """Save API keys to .env and all RAG backend settings to JSON."""
                # --- API keys → .env ---
                try:
                    from dotenv import set_key
                    env_path = Path(__file__).parent / ".env"
                    if not env_path.exists():
                        env_path.write_text("", encoding="utf-8")
                    for _ekey, _widget in [
                        ("ANTHROPIC_API_KEY", anthropic_input),
                        ("OPENAI_API_KEY", openai_input),
                        ("GOOGLE_API_KEY", google_input),
                        ("OPENROUTER_API_KEY", openrouter_input),
                        ("COHERE_API_KEY", cohere_input),
                    ]:
                        set_key(str(env_path), _ekey, _widget.value or "")
                    set_key(
                        str(env_path),
                        "DAILY_BUDGET_USD",
                        str(int(budget_input.value or 0)),
                    )
                except Exception as exc:
                    ui.notify(f"Failed to save API keys: {exc}", type="negative")
                    return

                # --- Backend config → nicegui_settings.json ---
                backend = {
                    "llm": {
                        "primary_provider": llm_primary_provider.value,
                        "primary_model": llm_primary_model.value,
                        "kg_provider": llm_kg_provider.value,
                        "kg_model": llm_kg_model.value,
                        "temperature": float(llm_temperature.value or 0.0),
                        "max_retries": int(llm_max_retries.value or 5),
                        "timeout": int(llm_timeout.value or 60),
                    },
                    "embedding": {
                        "provider": emb_provider.value,
                        "model": emb_model.value,
                        "batch_size": int(emb_batch_size.value or 100),
                        "max_retries": int(emb_max_retries.value or 10),
                    },
                    "retriever": {
                        "vector_top_k": int(ret_vector_top_k.value or 5),
                        "bm25_top_k": int(ret_bm25_top_k.value or 5),
                        "kg_similarity_top_k": int(ret_kg_similarity_top_k.value or 3),
                        "kg_max_triplets_per_chunk": int(ret_kg_max_triplets.value or 2),
                        "fusion_mode": ret_fusion_mode.value,
                        "fusion_num_queries": int(ret_fusion_num_queries.value or 4),
                        "enable_kg": ret_enable_kg.value,
                        "enable_reranker": ret_enable_reranker.value,
                        "enable_llm_rewrite": ret_enable_llm_rewrite.value,
                        "enable_query_decomposition": ret_enable_decomp.value,
                        "adaptive_top_k": ret_adaptive_top_k.value,
                        "use_hierarchical_chunking": ret_hierarchical.value,
                        "merge_threshold": float(ret_merge_threshold.value or 0.5),
                        "use_mmr": ret_use_mmr.value,
                        "mmr_threshold": float(ret_mmr_threshold.value or 0.7),
                        "rerank_top_n": int(ret_rerank_top_n.value or 5),
                        "use_colbert": ret_use_colbert.value,
                        "colbert_model": ret_colbert_model.value,
                        "max_sub_queries": int(ret_max_sub_queries.value or 4),
                        "decomposition_complexity_threshold": int(
                            ret_decomp_complexity.value or 80
                        ),
                    },
                    "chunking": {
                        "chunk_size": int(chunk_size.value or 1024),
                        "chunk_overlap": int(chunk_overlap.value or 128),
                        "enable_semantic_chunking": chunk_semantic.value,
                        "semantic_breakpoint_percentile": int(
                            chunk_semantic_pct.value or 95
                        ),
                        "code_chunk_lines": int(chunk_code_lines.value or 40),
                        "code_chunk_lines_overlap": int(chunk_code_overlap.value or 15),
                        "include_metadata": chunk_include_metadata.value,
                        "include_prev_next_rel": chunk_prev_next.value,
                    },
                    "cache": {
                        "enabled": cache_enabled.value,
                        "similarity_threshold": float(cache_threshold.value or 0.92),
                        "ttl_seconds": int((cache_ttl_days.value or 7) * 86400),
                        "backend": cache_backend.value,
                        "redis_url": cache_redis_url.value,
                        "redis_db": int(cache_redis_db.value or 0),
                        "cache_dir": cache_dir.value,
                    },
                    "self_rag": {
                        "enabled": srag_enabled.value,
                        "confidence_threshold": float(srag_threshold.value or 0.70),
                        "max_retries": int(srag_retries.value or 2),
                        "force_llm_rewrite_on_retry": srag_force_rewrite.value,
                    },
                    "verification": {
                        "enabled": ver_enabled.value,
                        "grounded_threshold": float(ver_grounded.value or 0.75),
                        "borderline_threshold": float(ver_borderline.value or 0.55),
                        "borderline_confidence_penalty": float(
                            ver_borderline_penalty.value or 0.15
                        ),
                        "ungrounded_confidence_penalty": float(
                            ver_ungrounded_penalty.value or 0.30
                        ),
                    },
                    "codegen": {
                        "enabled": cg_enabled.value,
                        "validate_output": cg_validate.value,
                        "self_improving": cg_self_improving.value,
                        "max_fix_iterations": int(cg_max_iter.value or 5),
                        "enable_compilation": cg_compile.value,
                        "msvc_path": cg_msvc_path.value or None,
                        "feedback_enabled": cg_feedback.value,
                        "max_context_chunks": int(cg_max_chunks.value or 5),
                    },
                    "observability": {
                        "log_level": obs_log_level.value,
                        "log_format": obs_log_format.value,
                        "phoenix_enabled": obs_phoenix.value,
                        "phoenix_host": obs_phoenix_host.value,
                        "phoenix_port": int(obs_phoenix_port.value or 6006),
                        "prometheus_enabled": obs_prometheus.value,
                        "prometheus_port": int(obs_prometheus_port.value or 8000),
                        "enable_otel": obs_otel.value,
                        "otel_endpoint": obs_otel_endpoint.value,
                        "otel_service_name": obs_otel_service.value,
                    },
                    "cost": {
                        "track_costs": cost_track.value,
                        "alert_threshold_pct": float(cost_alert_pct.value or 80.0),
                        "openai_embedding_cost": float(cost_embed.value or 0.02),
                        "openai_gpt4o_mini_input": float(cost_gpt4mini_in.value or 0.15),
                        "openai_gpt4o_mini_output": float(
                            cost_gpt4mini_out.value or 0.60
                        ),
                        "anthropic_claude_sonnet_input": float(
                            cost_claude_in.value or 3.0
                        ),
                        "anthropic_claude_sonnet_output": float(
                            cost_claude_out.value or 15.0
                        ),
                        "gemini_flash_input": float(cost_gemini_in.value or 0.075),
                        "gemini_flash_output": float(cost_gemini_out.value or 0.30),
                    },
                    "production": {
                        "rate_limit_enabled": prod_rate_limit.value,
                        "requests_per_minute": int(prod_rpm.value or 60),
                        "circuit_breaker_enabled": prod_circuit.value,
                        "failure_threshold": int(prod_failure.value or 5),
                        "recovery_timeout": int(prod_recovery.value or 60),
                        "max_retries": int(prod_max_retries.value or 3),
                        "retry_backoff_factor": float(prod_backoff.value or 2.0),
                        "retry_jitter": prod_jitter.value,
                    },
                    "storage": {
                        "storage_dir": stor_dir.value,
                        "cache_dir": stor_cache_dir.value,
                        "logs_dir": stor_logs_dir.value,
                    },
                    "cpp_intelligence": {
                        "enabled": cpp_enabled.value,
                        "max_methods_in_metadata": int(cpp_max_methods.value or 30),
                        "max_signatures_in_metadata": int(cpp_max_sigs.value or 20),
                        "max_related_types": int(cpp_max_types.value or 15),
                        "include_inheritance_chain": cpp_inheritance.value,
                        "include_method_signatures": cpp_sigs.value,
                        "include_forward_declarations": cpp_fwd_decl.value,
                    },
                    "intent_router": {
                        "enabled": ir_enabled.value,
                        "llm_confirmation_threshold": float(
                            ir_threshold.value or 0.6
                        ),
                    },
                    "guardrails": {
                        "enabled": gr_enabled.value,
                        "min_length": int(gr_min.value or 3),
                        "max_length": int(gr_max.value or 1000),
                    },
                }
                # Preserve top-level settings (e.g. auto_build_indexes) that
                # are managed outside this form.
                _existing = _load_settings_dict()
                for _k in ("auto_build_indexes",):
                    if _k in _existing:
                        backend[_k] = _existing[_k]
                try:
                    SETTINGS_FILE.write_text(
                        json.dumps(backend, indent=2), encoding="utf-8"
                    )
                except Exception as exc:
                    ui.notify(
                        f"Failed to save backend settings: {exc}", type="negative"
                    )
                    return

                return True

            async def _confirm_and_save():
                """Show confirmation dialog, save settings, restart app."""
                with ui.dialog() as dialog, ui.card().style("min-width: 340px;"):
                    with ui.row().classes("items-center gap-2 q-mb-xs"):
                        ui.icon("restart_alt", color="orange").classes("text-h5")
                        ui.label("Save & restart?").classes(
                            "text-subtitle1 text-bold"
                        )
                    ui.label(
                        "Settings will be saved and the app will restart. "
                        "Your browser will reconnect automatically."
                    ).classes("text-caption text-grey-5")
                    with ui.row().classes("justify-end gap-2 q-mt-md"):
                        ui.button(
                            "Cancel",
                            on_click=lambda: dialog.submit(False),
                        ).props("flat dense")
                        ui.button(
                            "Yes, restart",
                            icon="restart_alt",
                            on_click=lambda: dialog.submit(True),
                        ).props("color=orange dense")
                confirmed = await dialog
                if not confirmed:
                    return
                if not save_settings():
                    return
                ui.notify("Restarting...", type="positive", timeout=2000)
                await asyncio.sleep(1.5)
                os.execv(sys.executable, [sys.executable] + sys.argv)

            ui.button(
                "Save Settings", on_click=_confirm_and_save
            ).props("color=orange").classes("q-mt-md q-mb-xl")

    # ── Startup: initial dashboard load + auto-repair + auto-build ──────
    # All timers are at page scope (outside any tab_panel) to avoid
    # "parent slot deleted" errors when a tab is not active.
    ui.timer(0.5, refresh_dashboard, once=True)

    # Auto-repair: if the engine failed to start, try to fix it
    # (runs 3 s after page load so the UI is fully rendered first)
    async def _startup_repair_then_refresh():
        await _ensure_engine_running()
        # Refresh dashboard so repair status / recovered engine is shown
        await refresh_dashboard()
    ui.timer(3.0, _startup_repair_then_refresh, once=True)

    _startup_settings = _load_settings_dict()
    if _startup_settings.get("auto_build_indexes") == "auto" and not _indexes_exist():
        async def _auto_build_on_startup():
            try:
                ui.notify(
                    "Auto-building indexes (configured in Settings)…",
                    type="info",
                    timeout=6000,
                )
            except RuntimeError:
                pass  # Parent element deleted (tab switched), skip notification
            await _do_build_indexes_bg(console_log)
            try:
                ui.notify(
                    "Index build complete — restart the app to activate.",
                    type="positive",
                    timeout=8000,
                )
            except RuntimeError:
                pass  # Parent element deleted, skip notification
        ui.timer(5.0, _auto_build_on_startup, once=True)


# ---------------------------------------------------------------------------
# API key test helper
# ---------------------------------------------------------------------------

async def _test_api_key(provider: str, key: str):
    """Test an API key by attempting a quick LLM call."""
    if not key or not key.strip():
        ui.notify(f"No {provider} key provided", type="warning")
        return

    ui.notify(f"Testing {provider} key...", type="info", timeout=2000)

    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _do_test_key(provider, key)
        )
        if result:
            ui.notify(
                f"{provider} key is valid!", type="positive"
            )
        else:
            ui.notify(
                f"{provider} key test failed", type="negative"
            )
    except Exception as exc:
        ui.notify(
            f"{provider} key test failed: {exc}", type="negative"
        )


def _run_diagnostics_blocking() -> list:
    """Run all system diagnostic checks. Returns list of result dicts.

    Each dict has: name (str), detail (str), passing (bool), grey (bool).
    grey=True means the component is disabled (not a failure).
    """
    results = []

    # 1. OpenAI Embeddings
    try:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            results.append({
                "name": "OpenAI Embeddings", "passing": False, "grey": False,
                "detail": "OPENAI_API_KEY not set (required)",
            })
        else:
            from openai import OpenAI
            client = OpenAI(api_key=key)
            client.embeddings.create(input=["diagnostic test"], model="text-embedding-3-small")
            results.append({
                "name": "OpenAI Embeddings", "passing": True, "grey": False,
                "detail": "text-embedding-3-small responding",
            })
    except Exception as exc:
        results.append({
            "name": "OpenAI Embeddings", "passing": False, "grey": False,
            "detail": str(exc)[:80],
        })

    # 2. Active LLM
    try:
        from bakkesmod_rag.config import get_config as _get_cfg
        from bakkesmod_rag.llm_provider import get_llm as _get_llm
        _cfg = _get_cfg()
        _llm = _get_llm(_cfg)
        if isinstance(_llm, NullLLM):
            results.append({
                "name": "Active LLM", "passing": False, "grey": False,
                "detail": "No LLM provider available — set at least one API key",
            })
        else:
            model = getattr(_llm, "model", "unknown")
            results.append({
                "name": "Active LLM", "passing": True, "grey": False,
                "detail": str(model),
            })
    except Exception as exc:
        results.append({
            "name": "Active LLM", "passing": False, "grey": False,
            "detail": str(exc)[:80],
        })

    # 3. Knowledge Graph Index
    try:
        graph_file = Path("rag_storage") / "graph_store.json"
        if graph_file.exists():
            size_kb = graph_file.stat().st_size / 1024
            results.append({
                "name": "Knowledge Graph Index", "passing": True, "grey": False,
                "detail": f"graph_store.json ({size_kb:.0f} KB)",
            })
        else:
            results.append({
                "name": "Knowledge Graph Index", "passing": False, "grey": False,
                "detail": "Not built yet — run comprehensive builder",
            })
    except Exception as exc:
        results.append({
            "name": "Knowledge Graph Index", "passing": False, "grey": False,
            "detail": str(exc)[:80],
        })

    # 4. BM25 / Vector Index
    try:
        storage = Path("rag_storage")
        docstore = storage / "docstore.json"
        vec1 = storage / "default__vector_store.json"
        vec2 = storage / "vector_store.json"
        vector_ok = vec1.exists() or vec2.exists()
        if docstore.exists() and vector_ok:
            doc_kb = docstore.stat().st_size / 1024
            results.append({
                "name": "BM25 / Vector Index", "passing": True, "grey": False,
                "detail": f"docstore {doc_kb:.0f} KB, vector store present",
            })
        else:
            missing = []
            if not docstore.exists():
                missing.append("docstore.json")
            if not vector_ok:
                missing.append("vector_store")
            results.append({
                "name": "BM25 / Vector Index", "passing": False, "grey": False,
                "detail": f"Missing: {', '.join(missing)} — run comprehensive builder",
            })
    except Exception as exc:
        results.append({
            "name": "BM25 / Vector Index", "passing": False, "grey": False,
            "detail": str(exc)[:80],
        })

    # 5. Neural Reranker
    try:
        from bakkesmod_rag.config import get_config as _get_cfg2
        _cfg2 = _get_cfg2()
        if not _cfg2.retriever.enable_reranker:
            results.append({
                "name": "Neural Reranker", "passing": False, "grey": True,
                "detail": "Disabled in config",
            })
        else:
            found = None
            try:
                import FlagEmbedding  # noqa: F401
                found = f"BGE ({_cfg2.retriever.bge_reranker_model})"
            except ImportError:
                pass
            if not found:
                try:
                    from flashrank import Ranker  # noqa: F401
                    found = f"FlashRank ({_cfg2.retriever.flashrank_model})"
                except ImportError:
                    pass
            if not found and os.getenv("COHERE_API_KEY"):
                found = f"Cohere ({_cfg2.retriever.reranker_model})"
            if found:
                results.append({
                    "name": "Neural Reranker", "passing": True, "grey": False,
                    "detail": found,
                })
            else:
                results.append({
                    "name": "Neural Reranker", "passing": False, "grey": False,
                    "detail": "No reranker available (install FlagEmbedding/flashrank or set COHERE_API_KEY)",
                })
    except Exception as exc:
        results.append({
            "name": "Neural Reranker", "passing": False, "grey": False,
            "detail": str(exc)[:80],
        })

    # 6. Semantic Cache
    try:
        from bakkesmod_rag.config import get_config as _get_cfg3
        _cfg3 = _get_cfg3()
        if not _cfg3.cache.enabled:
            results.append({
                "name": "Semantic Cache", "passing": False, "grey": True,
                "detail": "Disabled in config",
            })
        else:
            cache_path = Path(_cfg3.cache.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            test_file = cache_path / ".diag_test"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            results.append({
                "name": "Semantic Cache", "passing": True, "grey": False,
                "detail": f"{_cfg3.cache.cache_dir} readable/writable",
            })
    except Exception as exc:
        results.append({
            "name": "Semantic Cache", "passing": False, "grey": False,
            "detail": str(exc)[:80],
        })

    # 7. Query Decomposition
    try:
        from bakkesmod_rag.config import get_config as _get_cfg4
        _cfg4 = _get_cfg4()
        enabled = _cfg4.retriever.enable_query_decomposition
        results.append({
            "name": "Query Decomposition", "passing": enabled, "grey": not enabled,
            "detail": (
                f"Enabled (max {_cfg4.retriever.max_sub_queries} sub-queries)"
                if enabled else "Disabled"
            ),
        })
    except Exception as exc:
        results.append({
            "name": "Query Decomposition", "passing": False, "grey": False,
            "detail": str(exc)[:80],
        })

    # 8. Adaptive Retrieval
    try:
        from bakkesmod_rag.config import get_config as _get_cfg5
        _cfg5 = _get_cfg5()
        enabled = _cfg5.retriever.adaptive_top_k
        results.append({
            "name": "Adaptive Retrieval", "passing": enabled, "grey": not enabled,
            "detail": (
                f"Enabled (escalation: {_cfg5.retriever.top_k_escalation})"
                if enabled else "Disabled"
            ),
        })
    except Exception as exc:
        results.append({
            "name": "Adaptive Retrieval", "passing": False, "grey": False,
            "detail": str(exc)[:80],
        })

    return results


def _do_test_key(provider: str, key: str) -> bool:
    """Blocking test of an API key. Returns True on success."""
    try:
        if provider == "anthropic":
            from llama_index.llms.anthropic import Anthropic
            llm = Anthropic(
                model="claude-sonnet-4-5",
                api_key=key,
                max_retries=1,
            )
            resp = llm.complete("Say OK")
            return bool(resp and resp.text)

        elif provider == "openai":
            from llama_index.llms.openai import OpenAI as OpenAILLM
            llm = OpenAILLM(model="gpt-4o-mini", api_key=key)
            resp = llm.complete("Say OK")
            return bool(resp and resp.text)

        elif provider == "google":
            from llama_index.llms.google_genai import GoogleGenAI
            llm = GoogleGenAI(model="gemini-2.5-flash", api_key=key)
            resp = llm.complete("Say OK")
            return bool(resp and resp.text)

        elif provider == "openrouter":
            from llama_index.llms.openrouter import OpenRouter
            llm = OpenRouter(
                model="deepseek/deepseek-chat-v3-0324",
                api_key=key,
            )
            resp = llm.complete("Say OK")
            return bool(resp and resp.text)

        elif provider == "cohere":
            import cohere
            client = cohere.Client(api_key=key)
            resp = client.chat(message="Say OK", model="command-r")
            return bool(resp)

        elif provider == "openai_embed":
            from openai import OpenAI
            client = OpenAI(api_key=key)
            client.embeddings.create(input=["test"], model="text-embedding-3-small")
            return True

        return False
    except Exception as exc:
        logger.warning("Key test for %s failed: %s", provider, exc)
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ui.run(
        title="BakkesMod RAG",
        dark=True,
        host="127.0.0.1" if IS_EXE else "0.0.0.0",
        port=8080,
        native=IS_EXE,
        reload=False,
        show=not IS_EXE,
    )
