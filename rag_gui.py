"""BakkesMod RAG -- Gradio GUI entry point.

Modernised 5-tab interface with dark Rocket League theme:
  - Tab 1: Query Documentation (streaming, diagnostics, verification)
  - Tab 2: Generate Plugin Code (self-improving loop, compile status, feedback)
  - Tab 3: SDK Explorer (C++ class browser, inheritance chains)
  - Tab 4: Dashboard (stats, cost tracking, provider health)
  - Tab 5: Help (full feature reference)
  + Global collapsible debug panel with verbose logging

All RAG logic is delegated to ``bakkesmod_rag.RAGEngine``.
"""

import os
import sys
import time
import logging
import threading
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Generator, Optional
from dotenv import load_dotenv

load_dotenv()

try:
    import gradio as gr
except ImportError:
    print("ERROR: Gradio not installed.  Run:  pip install gradio")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Debug log handler — captures all bakkesmod_rag.* logs for the GUI
# ---------------------------------------------------------------------------

class GUILogHandler(logging.Handler):
    """Thread-safe logging handler that buffers log records for the GUI."""

    def __init__(self, max_entries: int = 500) -> None:
        super().__init__()
        self._buffer: deque = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._version = 0  # incremented on each new log line

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with self._lock:
                self._buffer.append(msg)
                self._version += 1
        except Exception:
            pass

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def get_logs(self, min_level: str = "DEBUG") -> str:
        """Return buffered logs filtered by minimum level."""
        level_num = getattr(logging, min_level.upper(), logging.DEBUG)
        with self._lock:
            lines = list(self._buffer)
        # Filter by level (level is embedded in the formatted string)
        if min_level.upper() != "DEBUG":
            level_names = {
                "INFO": {"INFO", "WARNING", "ERROR", "CRITICAL"},
                "WARNING": {"WARNING", "ERROR", "CRITICAL"},
                "ERROR": {"ERROR", "CRITICAL"},
            }
            allowed = level_names.get(min_level.upper(), set())
            if allowed:
                lines = [
                    ln for ln in lines
                    if any(f" {lvl} " in ln or ln.startswith(f"{lvl} ") for lvl in allowed)
                ]
        return "\n".join(lines) if lines else "(no logs yet)"

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
            self._version = 0


# ---------------------------------------------------------------------------
# Stdout/stderr capture — routes print() output into the log handler too
# ---------------------------------------------------------------------------

import io

class _TeeWriter(io.TextIOBase):
    """Writes to the original stream AND appends to the GUI log buffer."""

    def __init__(self, original, handler: GUILogHandler, prefix: str = ""):
        self._original = original
        self._handler = handler
        self._prefix = prefix

    def write(self, s: str):
        if s and s.strip():
            with self._handler._lock:
                for line in s.rstrip("\n").split("\n"):
                    self._handler._buffer.append(
                        f"{datetime.now().strftime('%H:%M:%S')}  "
                        f"{self._prefix}{line}"
                    )
                    self._handler._version += 1
        if self._original and not self._original.closed:
            try:
                return self._original.write(s)
            except Exception:
                return len(s) if s else 0
        return len(s) if s else 0

    def flush(self):
        if self._original and not self._original.closed:
            try:
                self._original.flush()
            except Exception:
                pass

    def isatty(self):
        return False


# Install the handler on the root bakkesmod_rag logger
_gui_log_handler = GUILogHandler(max_entries=500)
_gui_log_handler.setLevel(logging.DEBUG)
_gui_log_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
                      datefmt="%H:%M:%S")
)
_root_logger = logging.getLogger("bakkesmod_rag")
_root_logger.setLevel(logging.DEBUG)
_root_logger.addHandler(_gui_log_handler)
_root_logger.propagate = False  # prevent duplicate logs reaching root logger


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

engine = None
query_count = 0
successful_queries = 0
total_query_time = 0.0
cache_hits = 0
last_generated_code = None
_active_provider = "Unknown"


# ---------------------------------------------------------------------------
# Helper: detect LLM provider name from LLM instance
# ---------------------------------------------------------------------------

def _detect_provider_name(llm) -> str:
    """Extract a human-readable provider name from a LlamaIndex LLM."""
    cls = type(llm).__name__.lower()
    if "anthropic" in cls:
        model = getattr(llm, "model", "claude")
        return f"Anthropic ({model})"
    if "gemini" in cls:
        model = getattr(llm, "model", "gemini")
        return f"Google ({model})"
    if "openrouter" in cls or "openai" in cls:
        model = getattr(llm, "model", "")
        if "deepseek" in model.lower():
            return f"OpenRouter ({model})"
        return f"OpenAI ({model})"
    return type(llm).__name__


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def initialize():
    """Build the RAG engine.  Called once at startup."""
    global engine, _active_provider
    import nest_asyncio
    nest_asyncio.apply()
    from bakkesmod_rag import RAGEngine

    try:
        engine = RAGEngine()
        _active_provider = _detect_provider_name(engine.llm)
        return (
            f"✅ System initialized  |  "
            f"{engine.num_documents} docs  |  "
            f"{engine.num_nodes} chunks  |  "
            f"LLM: {_active_provider}"
        )
    except Exception as e:
        return f"❌ Initialization failed: {e}"


# ---------------------------------------------------------------------------
# Tab 1: Query Documentation
# ---------------------------------------------------------------------------

def query_rag(query: str, use_cache: bool = True) -> Generator[str, None, None]:
    """Stream an answer with full diagnostic metadata."""
    global query_count, successful_queries, total_query_time, cache_hits

    if not query or not query.strip():
        yield "Please enter a question."
        return

    if engine is None:
        yield "❌ System not initialized. Please restart the application."
        return

    query_count += 1

    try:
        gen, get_meta = engine.query_streaming(query, use_cache=use_cache)

        # Stream the answer tokens
        yield ""
        for token in gen:
            yield token

        # Metadata after generation
        meta = get_meta()
        successful_queries += 1
        total_query_time += meta.query_time

        yield "\n\n---\n\n"

        # Confidence badge
        if meta.cached:
            cache_hits += 1
            yield (
                f"🔵 **CACHED** (similarity: {meta.confidence:.1%})  ·  "
                f"⏱ {meta.query_time:.2f}s  ·  "
                f"💰 ~$0.00\n\n"
            )
        else:
            conf = meta.confidence
            if conf >= 0.80:
                badge = "🟢 **HIGH**"
            elif conf >= 0.55:
                badge = "🟡 **MEDIUM**"
            else:
                badge = "🔴 **LOW**"

            yield (
                f"{badge} Confidence: {conf:.0%}  ·  "
                f"⏱ {meta.query_time:.2f}s  ·  "
                f"🤖 {_active_provider}\n\n"
            )

            # Verification status
            if meta.verification_warning:
                yield f"⚠️ **Verification:** {meta.verification_warning}\n\n"
            else:
                yield "✅ **Verified:** Response grounded in source documents\n\n"

            # Sources
            if meta.sources:
                yield "**📄 Sources:**\n"
                seen = set()
                for src in meta.sources:
                    name = src.get("file_name", "unknown") if isinstance(src, dict) else src
                    score = src.get("score") if isinstance(src, dict) else None
                    if name not in seen:
                        seen.add(name)
                        score_str = f" ({score:.3f})" if score else ""
                        yield f"- `{name}`{score_str}\n"
                yield "\n"

            # Self-RAG retry info
            if meta.retry_count > 0:
                yield (
                    f"🔄 **Self-RAG:** {meta.retry_count} "
                    f"{'retry' if meta.retry_count == 1 else 'retries'} performed\n"
                )
                if meta.all_attempts:
                    for i, attempt in enumerate(meta.all_attempts):
                        yield (
                            f"  - Attempt {i+1}: "
                            f"{attempt.get('confidence_label', '?')} "
                            f"({attempt.get('confidence', 0):.0%})\n"
                        )
                yield "\n"

        # Query intelligence accordion content
        yield "<details><summary>🔍 <b>Query Intelligence</b></summary>\n\n"
        yield f"**Original query:** {query}\n\n"
        if meta.expanded_query and meta.expanded_query != query:
            yield f"**Expanded query:** {meta.expanded_query}\n\n"
        yield f"**Confidence explanation:** {meta.confidence_explanation}\n\n"
        yield "</details>\n"

    except Exception as e:
        yield f"\n\n❌ **Error:** {e}\n"


# ---------------------------------------------------------------------------
# Tab 2: Code Generation
# ---------------------------------------------------------------------------

def generate_code(requirements: str):
    """Generate plugin code with self-improving loop visibility."""
    global last_generated_code

    if not requirements or not requirements.strip():
        return "", "", "Please enter plugin requirements."

    if engine is None:
        return "", "", "❌ System not initialized."

    try:
        result = engine.generate_code(requirements)

        last_generated_code = {
            "header": result.header,
            "implementation": result.implementation,
            "project_files": result.project_files,
            "requirements": requirements,
            "timestamp": datetime.now().isoformat(),
            "generation_id": result.generation_id,
        }

        status = ""

        # Features detected
        if result.features_used:
            status += f"**🔌 Features:** {', '.join(result.features_used)}\n\n"

        # Self-improving loop info
        if result.fix_iterations > 0:
            status += (
                f"🔄 **Self-Improving:** {result.fix_iterations} fix "
                f"{'iteration' if result.fix_iterations == 1 else 'iterations'}\n\n"
            )

        # Compile status
        if result.compile_result:
            cr = result.compile_result
            if cr.get("success"):
                status += "✅ **Compiled** successfully with MSVC\n\n"
            else:
                errors = cr.get("errors", [])
                status += f"❌ **Compile errors:** {len(errors)} issues\n\n"
        else:
            status += "⏭ **Compilation:** Skipped (MSVC not available or disabled)\n\n"

        # Fix history
        if result.fix_history:
            status += "<details><summary>🔧 <b>Fix History</b></summary>\n\n"
            for i, fix in enumerate(result.fix_history):
                status += f"**Iteration {i+1}:**\n"
                if isinstance(fix, dict):
                    for k, v in fix.items():
                        status += f"- {k}: {v}\n"
                else:
                    status += f"- {fix}\n"
                status += "\n"
            status += "</details>\n\n"

        # Project files
        if result.project_files:
            status += f"**📁 Project files:** {len(result.project_files)} files\n"
            for fname in sorted(result.project_files.keys()):
                size = len(result.project_files[fname])
                status += f"- `{fname}` ({size:,} bytes)\n"
            status += "\n"

        # Validation
        validation = result.validation
        if validation and not validation.get("valid", True):
            status += "⚠️ **Validation warnings:**\n"
            for err in validation.get("errors", []):
                status += f"- ❌ {err}\n"
            for warn in validation.get("warnings", []):
                status += f"- ⚠️ {warn}\n"
        else:
            status += "✅ **All validation checks passed**\n"

        # Generation ID
        if result.generation_id:
            status += f"\n📋 Generation ID: `{result.generation_id}`"

        return result.header, result.implementation, status

    except Exception as e:
        return "", "", f"❌ **Error:** {e}"


def submit_feedback(is_positive: bool) -> str:
    """Record thumbs up/down feedback for the last code generation."""
    if not last_generated_code or not last_generated_code.get("generation_id"):
        return "No code generation to give feedback on."

    if engine is None:
        return "Engine not initialized."

    try:
        gen_id = last_generated_code["generation_id"]
        success = engine.record_code_feedback(gen_id, accepted=is_positive)
        if success:
            emoji = "👍" if is_positive else "👎"
            return f"{emoji} Feedback recorded for `{gen_id}`. Thank you!"
        return "Failed to record feedback."
    except Exception as e:
        return f"Error: {e}"


def feedback_positive():
    return submit_feedback(True)


def feedback_negative():
    return submit_feedback(False)


def export_code(plugin_name: str = "MyPlugin") -> str:
    """Export the last generated code to files on disk."""
    if not last_generated_code:
        return "No code to export. Generate code first."

    if not plugin_name or not plugin_name.strip():
        plugin_name = "MyPlugin"

    plugin_name = "".join(c for c in plugin_name if c.isalnum() or c in "_")

    try:
        output_dir = Path("generated_plugins") / plugin_name
        output_dir.mkdir(parents=True, exist_ok=True)

        project_files = last_generated_code.get("project_files", {})
        files_written = []

        if project_files:
            for fname, content in project_files.items():
                file_path = output_dir / fname
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
                files_written.append(fname)
        else:
            header_path = output_dir / f"{plugin_name}.h"
            with open(header_path, "w", encoding="utf-8") as f:
                f.write(last_generated_code["header"])
            files_written.append(f"{plugin_name}.h")

            impl_path = output_dir / f"{plugin_name}.cpp"
            with open(impl_path, "w", encoding="utf-8") as f:
                f.write(last_generated_code["implementation"])
            files_written.append(f"{plugin_name}.cpp")

        file_list = "\n".join(f"- `{f}`" for f in sorted(files_written))
        return (
            f"✅ **Exported to:** `{output_dir.absolute()}`\n\n"
            f"**Files ({len(files_written)}):**\n{file_list}"
        )

    except Exception as e:
        return f"❌ Export error: {e}"


# ---------------------------------------------------------------------------
# Tab 3: SDK Explorer
# ---------------------------------------------------------------------------

def _get_sdk_classes() -> dict:
    """Analyze SDK headers and return class info dict."""
    try:
        from bakkesmod_rag.cpp_analyzer import CppAnalyzer
        analyzer = CppAnalyzer()
        sdk_dir = "docs_bakkesmod_only"
        if os.path.isdir(sdk_dir):
            return analyzer.analyze_directory(sdk_dir)
    except Exception as e:
        logging.warning("SDK analysis failed: %s", e)
    return {}


# Cache SDK analysis (it's expensive to re-run)
_sdk_classes_cache: Optional[dict] = None


def _ensure_sdk_classes() -> dict:
    global _sdk_classes_cache
    if _sdk_classes_cache is None:
        _sdk_classes_cache = _get_sdk_classes()
    return _sdk_classes_cache


def get_class_list(search_term: str = "") -> str:
    """Return formatted list of SDK classes, optionally filtered."""
    all_classes = _ensure_sdk_classes()
    if not all_classes:
        return "No SDK classes found. Ensure `docs_bakkesmod_only/` exists."

    if search_term and search_term.strip():
        term = search_term.strip().lower()
        filtered = {
            k: v for k, v in all_classes.items()
            if term in k.lower() or term in v.category
        }
    else:
        filtered = all_classes

    if not filtered:
        return f"No classes matching '{search_term}'."

    # Group by category
    categories: dict = {}
    for name, cls in sorted(filtered.items()):
        cat = cls.category or "other"
        categories.setdefault(cat, []).append(cls)

    lines = [f"**{len(filtered)} classes found**\n"]
    for cat in sorted(categories.keys()):
        lines.append(f"\n### {cat.title()}")
        for cls in categories[cat]:
            base = f" → {cls.base_classes[0]}" if cls.base_classes else ""
            methods = len(cls.methods)
            lines.append(f"- **{cls.name}**{base} ({methods} methods)")

    return "\n".join(lines)


def get_class_detail(class_name: str) -> str:
    """Return detailed info for a specific class."""
    all_classes = _ensure_sdk_classes()

    if not class_name or not class_name.strip():
        return "Enter a class name to view details."

    name = class_name.strip()
    if name not in all_classes:
        # Try case-insensitive match
        matches = [k for k in all_classes if k.lower() == name.lower()]
        if matches:
            name = matches[0]
        else:
            return f"Class `{name}` not found. Try searching in the class list."

    cls = all_classes[name]

    try:
        from bakkesmod_rag.cpp_analyzer import CppAnalyzer
        analyzer = CppAnalyzer()
    except ImportError:
        return "CppAnalyzer not available."

    chain = analyzer.build_inheritance_chain(name, all_classes)

    lines = [f"# {cls.name}\n"]
    lines.append(f"**Category:** {cls.category.title()}")
    lines.append(f"**File:** `{cls.file}`")

    if cls.base_classes:
        lines.append(f"**Base class:** `{cls.base_classes[0]}`")

    if chain:
        chain_str = " → ".join([cls.name] + chain)
        lines.append(f"**Inheritance chain:** `{chain_str}`")

    if cls.forward_declarations:
        lines.append(f"\n**Related types:** {', '.join(f'`{t}`' for t in cls.forward_declarations[:15])}")

    lines.append(f"\n**Methods ({len(cls.methods)}):**\n")

    # Separate getters, setters, other
    getters = [m for m in cls.methods if m.name.startswith("Get")]
    setters = [m for m in cls.methods if m.name.startswith("Set")]
    other = [m for m in cls.methods if not m.name.startswith("Get") and not m.name.startswith("Set")]

    if other:
        lines.append("### Other Methods")
        for m in other:
            sig = f"`{m.return_type} {m.name}({m.parameters})`"
            if m.is_const:
                sig += " const"
            lines.append(f"- {sig}")

    if getters:
        lines.append(f"\n### Getters ({len(getters)})")
        for m in getters:
            lines.append(f"- `{m.return_type} {m.name}()`")

    if setters:
        lines.append(f"\n### Setters ({len(setters)})")
        for m in setters:
            lines.append(f"- `void {m.name}({m.parameters})`")

    return "\n".join(lines)


def get_inheritance_tree() -> str:
    """Build a text-based inheritance tree of all SDK classes."""
    all_classes = _ensure_sdk_classes()
    if not all_classes:
        return "No SDK classes found."

    # Find root classes (no base class or base class not in dict)
    roots = []
    children: dict = {}
    for name, cls in all_classes.items():
        if not cls.base_classes or cls.base_classes[0] not in all_classes:
            roots.append(name)
        else:
            parent = cls.base_classes[0]
            children.setdefault(parent, []).append(name)

    def _build_tree(name: str, indent: int = 0) -> list:
        prefix = "  " * indent + ("├─ " if indent > 0 else "")
        methods = len(all_classes[name].methods) if name in all_classes else 0
        lines = [f"{prefix}**{name}** ({methods} methods)"]
        for child in sorted(children.get(name, [])):
            lines.extend(_build_tree(child, indent + 1))
        return lines

    lines = ["# SDK Class Hierarchy\n"]
    for root in sorted(roots):
        lines.extend(_build_tree(root))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab 4: Dashboard
# ---------------------------------------------------------------------------

def get_dashboard() -> str:
    """Return comprehensive dashboard with stats, cost, and health."""
    lines = ["# 📊 Dashboard\n"]

    # Session stats
    avg_time = (total_query_time / query_count) if query_count > 0 else 0
    success_rate = (successful_queries / query_count * 100) if query_count > 0 else 0
    cache_rate = (cache_hits / query_count * 100) if query_count > 0 else 0

    lines.append("## Session Statistics\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total queries | {query_count} |")
    lines.append(f"| Successful | {successful_queries} ({success_rate:.0f}%) |")
    lines.append(f"| Cache hits | {cache_hits} ({cache_rate:.0f}%) |")
    lines.append(f"| Avg response time | {avg_time:.2f}s |")
    lines.append("")

    # Cost tracking
    lines.append("## 💰 Cost Tracking\n")
    try:
        from bakkesmod_rag.cost_tracker import get_tracker
        tracker = get_tracker()
        daily_cost = tracker.get_daily_cost()
        budget = engine.config.cost.daily_budget_usd if engine else None

        lines.append(f"**Today's cost:** ${daily_cost:.4f}")
        if budget:
            pct = (daily_cost / budget * 100) if budget > 0 else 0
            bar_filled = int(min(pct, 100) / 5)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            lines.append(f"\n**Budget:** ${daily_cost:.4f} / ${budget:.2f}")
            lines.append(f"`[{bar}]` {pct:.1f}%")
        lines.append(f"\n**Cache savings:** ~${cache_hits * 0.03:.2f}")
    except Exception:
        lines.append("Cost tracking unavailable.")
    lines.append("")

    # LLM Provider health
    lines.append("## 🤖 LLM Provider\n")
    lines.append(f"**Active:** {_active_provider}")
    if engine:
        lines.append(f"**Model class:** `{type(engine.llm).__name__}`")
    lines.append("")

    # System info
    lines.append("## 🖥️ System\n")
    if engine:
        lines.append(f"| Component | Status |")
        lines.append(f"|-----------|--------|")
        lines.append(f"| Documents | {engine.num_documents} loaded |")
        lines.append(f"| Nodes | {engine.num_nodes} indexed |")
        lines.append(f"| Retrieval | Vector + BM25 + KG |")
        lines.append(f"| Cache | {'✅ Enabled' if engine.config.cache.enabled else '❌ Disabled'} |")
        lines.append(f"| Verification | {'✅ Enabled' if engine.config.verification.enabled else '❌ Disabled'} |")
        lines.append(f"| Self-RAG | {'✅ Enabled' if engine.config.self_rag.enabled else '❌ Disabled'} |")
        lines.append(f"| Code gen | {'✅ Enabled' if engine.config.codegen.enabled else '❌ Disabled'} |")
        lines.append(f"| C++ intelligence | {'✅ Enabled' if engine.config.cpp_intelligence.enabled else '❌ Disabled'} |")
    else:
        lines.append("Engine not initialized.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Debug panel
# ---------------------------------------------------------------------------

def get_debug_logs(log_level: str = "DEBUG") -> str:
    """Return formatted debug logs at the specified level."""
    logs = _gui_log_handler.get_logs(min_level=log_level)
    return f"```\n{logs}\n```"


def clear_debug_logs() -> str:
    """Clear the debug log buffer."""
    _gui_log_handler.clear()
    return "```\n(logs cleared)\n```"


# ---------------------------------------------------------------------------
# Live Console — auto-refreshing log viewer
# ---------------------------------------------------------------------------

_last_console_version = 0


def get_live_console() -> str:
    """Return the full log buffer as a live console view."""
    global _last_console_version
    _last_console_version = _gui_log_handler.version
    logs = _gui_log_handler.get_logs(min_level="DEBUG")
    line_count = len(_gui_log_handler._buffer)
    return (
        f"**Live Console** — {line_count} lines  |  "
        f"Auto-refreshes every 2s\n\n"
        f"```\n{logs}\n```"
    )


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

# Rocket League dark theme
_RL_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.orange,
    neutral_hue=gr.themes.colors.gray,
    font=("Segoe UI", "system-ui", "sans-serif"),
).set(
    body_background_fill="#0f1117",
    body_background_fill_dark="#0f1117",
    body_text_color="#e0e0e0",
    body_text_color_dark="#e0e0e0",
    block_background_fill="#1a1c23",
    block_background_fill_dark="#1a1c23",
    block_border_color="#2a2d35",
    block_border_color_dark="#2a2d35",
    block_label_text_color="#a0a0b0",
    block_label_text_color_dark="#a0a0b0",
    block_title_text_color="#ffffff",
    block_title_text_color_dark="#ffffff",
    input_background_fill="#22252e",
    input_background_fill_dark="#22252e",
    input_border_color="#3a3d45",
    input_border_color_dark="#3a3d45",
    button_primary_background_fill="#1a6fff",
    button_primary_background_fill_dark="#1a6fff",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    button_secondary_background_fill="#ff6a00",
    button_secondary_background_fill_dark="#ff6a00",
    button_secondary_text_color="#ffffff",
    button_secondary_text_color_dark="#ffffff",
)

_CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; }
.tab-nav button { font-size: 1.05em !important; }
.tab-nav button.selected {
    border-bottom: 3px solid #1a6fff !important;
    color: #1a6fff !important;
}
.debug-panel pre { font-size: 0.8em; line-height: 1.4; max-height: 600px; overflow-y: auto; }
.debug-panel { font-family: 'Cascadia Code', 'Consolas', monospace !important; }
footer { display: none !important; }
"""


def create_gui():
    """Create and return the Gradio Blocks application."""

    init_status = initialize()
    print(init_status)

    with gr.Blocks(
        title="BakkesMod RAG System",
    ) as demo:

        # Auto-refresh timer for live console (2 second interval)
        console_timer = gr.Timer(2, active=True)

        # ---- Header ---------------------------------------------------
        gr.Markdown(
            "# 🚀 BakkesMod RAG System\n"
            "### Documentation Assistant & Plugin Code Generator"
        )
        status_bar = gr.Markdown(value=init_status)

        with gr.Tabs():

            # ============================================================
            # Tab 0: Live Console (auto-refreshing)
            # ============================================================
            with gr.Tab("🖥️ Live Console", id="console_tab"):
                gr.Markdown(
                    "Real-time system output. Auto-refreshes every 2 seconds."
                )
                with gr.Row():
                    console_clear_btn = gr.Button(
                        "🗑️ Clear", variant="secondary", scale=1,
                    )
                console_output = gr.Markdown(
                    value=get_live_console(),
                    elem_classes=["debug-panel"],
                )
                # Wire the auto-refresh timer
                console_timer.tick(
                    fn=get_live_console,
                    outputs=console_output,
                )
                console_clear_btn.click(
                    fn=lambda: (
                        _gui_log_handler.clear(),
                        get_live_console(),
                    )[-1],
                    outputs=console_output,
                )

            # ============================================================
            # Tab 1: Query Documentation
            # ============================================================
            with gr.Tab("📖 Query", id="query_tab"):
                gr.Markdown(
                    "Ask questions about BakkesMod SDK, plugin development, "
                    "ImGui, events, wrappers, and more."
                )

                with gr.Row():
                    with gr.Column(scale=3):
                        query_input = gr.Textbox(
                            label="Your Question",
                            placeholder="How do I hook the goal scored event?",
                            lines=2,
                        )
                        with gr.Row():
                            query_btn = gr.Button(
                                "🔍 Search", variant="primary", scale=3,
                            )
                            use_cache_cb = gr.Checkbox(
                                label="Use cache",
                                value=True,
                                scale=1,
                            )

                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**Quick questions:**\n"
                            "- What is CarWrapper?\n"
                            "- How do I get ball velocity?\n"
                            "- How do I create an ImGui window?\n"
                            "- What events can I hook?\n"
                            "- How do I spawn a bot?"
                        )

                query_output = gr.Markdown(label="Response")

                query_btn.click(
                    fn=query_rag,
                    inputs=[query_input, use_cache_cb],
                    outputs=query_output,
                )
                query_input.submit(
                    fn=query_rag,
                    inputs=[query_input, use_cache_cb],
                    outputs=query_output,
                )

            # ============================================================
            # Tab 2: Generate Plugin Code
            # ============================================================
            with gr.Tab("⚡ Code Gen", id="codegen_tab"):
                gr.Markdown(
                    "Describe your plugin and get complete, validated C++ code. "
                    "The self-improving loop will fix errors automatically."
                )

                code_requirements = gr.Textbox(
                    label="Plugin Requirements",
                    placeholder=(
                        "Create a plugin that draws a speed display "
                        "on screen and logs goals with scorer info"
                    ),
                    lines=4,
                )
                generate_btn = gr.Button(
                    "⚡ Generate Code", variant="primary",
                )

                with gr.Row():
                    with gr.Column():
                        header_output = gr.Code(
                            label="Header (.h)",
                            language="cpp",
                            lines=20,
                        )
                    with gr.Column():
                        impl_output = gr.Code(
                            label="Implementation (.cpp)",
                            language="cpp",
                            lines=20,
                        )

                code_status = gr.Markdown(label="Generation Status")

                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            plugin_name_input = gr.Textbox(
                                label="Plugin Name",
                                placeholder="MyPlugin",
                                value="MyPlugin",
                                scale=2,
                            )
                            export_btn = gr.Button(
                                "📁 Export", variant="secondary", scale=1,
                            )
                    with gr.Column(scale=1):
                        with gr.Row():
                            thumbs_up_btn = gr.Button("👍", scale=1)
                            thumbs_down_btn = gr.Button("👎", scale=1)

                export_status = gr.Markdown()
                feedback_status = gr.Markdown()

                generate_btn.click(
                    fn=generate_code,
                    inputs=code_requirements,
                    outputs=[header_output, impl_output, code_status],
                )

                export_btn.click(
                    fn=export_code,
                    inputs=plugin_name_input,
                    outputs=export_status,
                )

                thumbs_up_btn.click(fn=feedback_positive, outputs=feedback_status)
                thumbs_down_btn.click(fn=feedback_negative, outputs=feedback_status)

            # ============================================================
            # Tab 3: SDK Explorer
            # ============================================================
            with gr.Tab("🔬 SDK Explorer", id="sdk_tab"):
                gr.Markdown(
                    "Browse BakkesMod SDK classes, inheritance chains, "
                    "and method signatures."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        sdk_search = gr.Textbox(
                            label="Search classes",
                            placeholder="car, ball, game, player...",
                        )
                        sdk_search_btn = gr.Button("Search", variant="primary")
                        class_list_output = gr.Markdown(
                            label="Classes",
                            value="Click Search or enter a term to browse SDK classes.",
                        )

                    with gr.Column(scale=2):
                        class_name_input = gr.Textbox(
                            label="Class name",
                            placeholder="CarWrapper",
                        )
                        class_detail_btn = gr.Button(
                            "View Details", variant="primary",
                        )
                        class_detail_output = gr.Markdown(
                            label="Class Details",
                            value="Enter a class name to view its methods and inheritance.",
                        )

                with gr.Accordion("🌳 Full Inheritance Tree", open=False):
                    tree_btn = gr.Button("Build Tree")
                    tree_output = gr.Markdown()

                sdk_search_btn.click(
                    fn=get_class_list,
                    inputs=sdk_search,
                    outputs=class_list_output,
                )
                sdk_search.submit(
                    fn=get_class_list,
                    inputs=sdk_search,
                    outputs=class_list_output,
                )
                class_detail_btn.click(
                    fn=get_class_detail,
                    inputs=class_name_input,
                    outputs=class_detail_output,
                )
                class_name_input.submit(
                    fn=get_class_detail,
                    inputs=class_name_input,
                    outputs=class_detail_output,
                )
                tree_btn.click(fn=get_inheritance_tree, outputs=tree_output)

            # ============================================================
            # Tab 4: Dashboard
            # ============================================================
            with gr.Tab("📊 Dashboard", id="dashboard_tab"):
                dashboard_output = gr.Markdown()
                dashboard_btn = gr.Button("🔄 Refresh", variant="primary")

                dashboard_btn.click(fn=get_dashboard, outputs=dashboard_output)
                demo.load(fn=get_dashboard, outputs=dashboard_output)

            # ============================================================
            # Tab 5: Help
            # ============================================================
            with gr.Tab("❓ Help", id="help_tab"):
                gr.Markdown(
                    "## BakkesMod RAG System — Feature Reference\n\n"
                    "### 🔍 Query Pipeline\n"
                    "1. **Synonym expansion** — 60+ BakkesMod domain mappings (zero API cost)\n"
                    "2. **Query decomposition** — Complex questions split into sub-queries\n"
                    "3. **3-way fusion retrieval** — Vector + BM25 + Knowledge Graph\n"
                    "4. **Neural reranking** — BGE → FlashRank → Cohere fallback chain\n"
                    "5. **Adaptive retrieval** — Dynamic top_k escalation (5→8→12)\n"
                    "6. **Answer verification** — Embedding + LLM grounding checks\n"
                    "7. **Self-RAG retry** — Automatic retry if confidence < 70%\n"
                    "8. **Semantic caching** — 92% similarity threshold, 7-day TTL\n\n"
                    "### ⚡ Code Generation\n"
                    "- **RAG-enhanced** — Uses SDK documentation for accurate code\n"
                    "- **Template engine** — Full 12-file project scaffolding\n"
                    "- **Feature detection** — ImGui, events, CVars, drawables\n"
                    "- **Self-improving loop** — Validate → Fix → Compile (up to 5 iterations)\n"
                    "- **MSVC compilation** — Optional compile verification\n"
                    "- **Feedback learning** — 👍/👎 improves future generations\n\n"
                    "### 🔬 SDK Intelligence\n"
                    "- **C++ structural analysis** — Tree-sitter parses all 177 SDK headers\n"
                    "- **Inheritance chains** — Full hierarchy from ObjectWrapper to leaf classes\n"
                    "- **Method signatures** — Return types, parameters, const qualifiers\n"
                    "- **Metadata injection** — LLM sees C++ structure, not just text\n\n"
                    "### 🛡️ Resilience\n"
                    "- **LLM fallback chain** — Anthropic → OpenRouter → Gemini → OpenAI\n"
                    "- **Circuit breakers** — Automatic provider failover\n"
                    "- **Rate limiting** — Configurable requests/minute\n"
                    "- **Cost tracking** — Per-query cost, daily budget alerts\n\n"
                    "### ⌨️ Tips\n"
                    "- Press **Enter** to submit queries (no need to click Search)\n"
                    "- Enable **cache** for faster, cheaper repeated queries\n"
                    "- Check the **Debug Panel** at the bottom for verbose logs\n"
                    "- Use **SDK Explorer** to browse class methods before asking questions\n"
                    "- **Higher confidence = more reliable** (green > yellow > red)\n"
                )

        # ================================================================
        # Global Debug Panel
        # ================================================================
        with gr.Accordion("🐛 Debug Panel", open=False, elem_classes=["debug-panel"]):
            with gr.Row():
                log_level_dropdown = gr.Dropdown(
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    value="DEBUG",
                    label="Log Level",
                    scale=1,
                )
                refresh_logs_btn = gr.Button(
                    "🔄 Refresh Logs", variant="primary", scale=1,
                )
                clear_logs_btn = gr.Button(
                    "🗑️ Clear", variant="secondary", scale=1,
                )

            debug_output = gr.Markdown(
                value="```\n(click Refresh to load logs)\n```",
                elem_classes=["debug-panel"],
            )

            refresh_logs_btn.click(
                fn=get_debug_logs,
                inputs=log_level_dropdown,
                outputs=debug_output,
            )
            clear_logs_btn.click(
                fn=clear_debug_logs,
                outputs=debug_output,
            )

        # ---- Footer ---------------------------------------------------
        gr.Markdown(
            "---\n"
            "**BakkesMod RAG System** v2.0  ·  "
            "LlamaIndex + Anthropic/Gemini/OpenAI  ·  "
            "434 tests passing"
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Launch the Gradio GUI."""
    print("=" * 70)
    print("  BakkesMod RAG — GUI Application")
    print("=" * 70)

    demo = create_gui()

    print("\n" + "=" * 70)
    print("  GUI Ready!  Opening browser...")
    print("=" * 70)
    print("\nPress Ctrl+C to stop.\n")

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        share=False,
        show_error=True,
        theme=_RL_THEME,
        css=_CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
