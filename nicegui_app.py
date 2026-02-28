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

import asyncio
import io
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

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
# Lazy RAGEngine singleton
# ---------------------------------------------------------------------------

_engine = None  # type: ignore[assignment]
_engine_error: Optional[str] = None


def get_engine():
    """Return the RAGEngine singleton, creating it on first call."""
    global _engine, _engine_error
    if _engine is not None:
        return _engine
    if _engine_error is not None:
        return None
    try:
        from bakkesmod_rag.engine import RAGEngine
        _engine = RAGEngine()
        return _engine
    except Exception as exc:
        _engine_error = str(exc)
        logger.error("RAGEngine init failed: %s", exc)
        return None


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
            # Attach handler to root bakkesmod_rag logger
            handler = NiceGuiLogHandler(console_log)
            handler.setLevel(logging.DEBUG)
            root_rag_logger = logging.getLogger("bakkesmod_rag")
            root_rag_logger.addHandler(handler)
            root_rag_logger.setLevel(logging.DEBUG)

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
                dash_container.clear()
                eng = get_engine()
                with dash_container:
                    if eng is None:
                        ui.label(
                            "RAG engine not available. "
                            "Configure API keys in Settings."
                        ).classes("text-grey-5")
                        return

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
            ui.timer(30.0, refresh_dashboard)

        # ==============================================================
        # Tab 6: Help
        # ==============================================================
        with ui.tab_panel(tab_help):
            ui.markdown(HELP_TEXT)

        # ==============================================================
        # Tab 7: Settings
        # ==============================================================
        with ui.tab_panel(tab_settings):
            ui.label("Settings").classes("text-h6")

            # API Keys
            with ui.card().classes("w-full"):
                ui.label("API Keys").classes("text-subtitle1")
                ui.label(
                    "Keys are saved to .env and used on next restart."
                ).classes("text-caption text-grey-5")

                with ui.grid(columns=3).classes("w-full gap-2"):
                    anthropic_input = ui.input(
                        "ANTHROPIC_API_KEY",
                        value=os.getenv("ANTHROPIC_API_KEY", ""),
                        password=True,
                    ).classes("col-span-2")
                    ui.button(
                        "Test",
                        on_click=lambda: _test_api_key(
                            "anthropic", anthropic_input.value
                        ),
                    ).props("outline dense")

                    openai_input = ui.input(
                        "OPENAI_API_KEY",
                        value=os.getenv("OPENAI_API_KEY", ""),
                        password=True,
                    ).classes("col-span-2")
                    ui.button(
                        "Test",
                        on_click=lambda: _test_api_key(
                            "openai", openai_input.value
                        ),
                    ).props("outline dense")

                    google_input = ui.input(
                        "GOOGLE_API_KEY",
                        value=os.getenv("GOOGLE_API_KEY", ""),
                        password=True,
                    ).classes("col-span-2")
                    ui.button(
                        "Test",
                        on_click=lambda: _test_api_key(
                            "google", google_input.value
                        ),
                    ).props("outline dense")

                    openrouter_input = ui.input(
                        "OPENROUTER_API_KEY",
                        value=os.getenv("OPENROUTER_API_KEY", ""),
                        password=True,
                    ).classes("col-span-2")
                    ui.button(
                        "Test",
                        on_click=lambda: _test_api_key(
                            "openrouter", openrouter_input.value
                        ),
                    ).props("outline dense")

                    cohere_input = ui.input(
                        "COHERE_API_KEY",
                        value=os.getenv("COHERE_API_KEY", ""),
                        password=True,
                    ).classes("col-span-2")
                    ui.button(
                        "Test",
                        on_click=lambda: _test_api_key(
                            "cohere", cohere_input.value
                        ),
                    ).props("outline dense")

            # Budget
            with ui.card().classes("w-full q-mt-md"):
                ui.label("Budget").classes("text-subtitle1")
                budget_input = ui.number(
                    "Daily Budget (USD)",
                    value=float(
                        os.getenv("DAILY_BUDGET_USD", "10")
                    ),
                    min=0,
                )

            # Save button
            def save_settings():
                """Save all settings to .env file."""
                try:
                    from dotenv import set_key
                    env_path = Path(__file__).parent / ".env"
                    if not env_path.exists():
                        env_path.write_text("", encoding="utf-8")
                    set_key(
                        str(env_path),
                        "ANTHROPIC_API_KEY",
                        anthropic_input.value or "",
                    )
                    set_key(
                        str(env_path),
                        "OPENAI_API_KEY",
                        openai_input.value or "",
                    )
                    set_key(
                        str(env_path),
                        "GOOGLE_API_KEY",
                        google_input.value or "",
                    )
                    set_key(
                        str(env_path),
                        "OPENROUTER_API_KEY",
                        openrouter_input.value or "",
                    )
                    set_key(
                        str(env_path),
                        "COHERE_API_KEY",
                        cohere_input.value or "",
                    )
                    set_key(
                        str(env_path),
                        "DAILY_BUDGET_USD",
                        str(budget_input.value or "10"),
                    )
                    ui.notify(
                        "Settings saved. Restart the app to apply "
                        "API key changes.",
                        type="positive",
                        timeout=5000,
                    )
                except Exception as exc:
                    ui.notify(
                        f"Failed to save settings: {exc}",
                        type="negative",
                    )

            ui.button(
                "Save Settings", on_click=save_settings
            ).props("color=orange").classes("q-mt-md")


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
