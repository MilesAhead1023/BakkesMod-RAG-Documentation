"""
Self-Repair System
==================
Detects and automatically fixes common dependency issues that prevent the
RAG engine from initialising.  Called when ``RAGEngine()`` raises an
exception so that the NiceGUI app can apply repairs and retry — without
any manual intervention from the user.

Repairs are executed as child ``pip`` subprocesses so they cannot
interfere with the running process's import state.  After a successful
repair the caller should retry ``RAGEngine()`` to pick up the changes.

Usage::

    from bakkesmod_rag.self_repair import repair_for_error, quick_environment_check

    report = repair_for_error(str(exc))
    if report.any_success:
        engine = RAGEngine()   # retry after repair
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import logging
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger("bakkesmod_rag.self_repair")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RepairAction:
    """One attempted repair action.

    Attributes:
        issue: Human-readable description of the detected problem.
        action_taken: What the repair system did (or tried to do).
        success: Whether the repair completed without errors.
        detail: Optional extra detail (pip output, suggestions, etc.).
    """

    issue: str
    action_taken: str
    success: bool
    detail: str = ""


@dataclass
class RepairReport:
    """Collection of all repair actions attempted in one session.

    Attributes:
        actions: Ordered list of ``RepairAction`` instances.
    """

    actions: List[RepairAction] = field(default_factory=list)

    @property
    def any_success(self) -> bool:
        """True if at least one repair succeeded."""
        return any(a.success for a in self.actions)

    @property
    def any_failed(self) -> bool:
        """True if at least one repair failed."""
        return any(not a.success for a in self.actions)

    def is_empty(self) -> bool:
        """True if no repairs were attempted."""
        return not self.actions

    def as_lines(self) -> List[str]:
        """Return a human-readable list of repair action lines."""
        lines: List[str] = []
        for a in self.actions:
            icon = "✅" if a.success else "❌"
            lines.append(f"{icon} {a.issue}")
            lines.append(f"   → {a.action_taken}")
            if a.detail:
                # Truncate long pip output to keep logs readable
                lines.append(f"   {a.detail[:200]}")
        return lines


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pip(*args: str) -> tuple[bool, str]:
    """Run a pip sub-command in a child process.

    Args:
        *args: Arguments forwarded to ``python -m pip``.

    Returns:
        Tuple of (success, combined_stdout_stderr).
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", *args],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = (result.stdout + result.stderr).strip()
        # Truncate very long output (pip install can be verbose)
        return result.returncode == 0, output[-500:] if len(output) > 500 else output
    except subprocess.TimeoutExpired:
        return False, "pip command timed out after 120 s"
    except Exception as exc:
        return False, f"pip failed to run: {exc}"


# ---------------------------------------------------------------------------
# Repair handlers
# ---------------------------------------------------------------------------

def _repair_torch_dll(report: RepairReport) -> None:
    """Reinstall torch 2.8.0 (last version before PyQt import-order regression).

    PyTorch 2.9.0+ introduced a Windows regression where c10.dll fails to
    initialize if imported after PyQt. Torch 2.8.0 is the last confirmed-working
    version and is fully capable for HuggingFace embeddings and other features.
    The c10.dll is pre-loaded in nicegui_app.py to ensure it also works at
    runtime in the exe.
    """
    ok, _detail = _pip("show", "torch")
    if ok:
        logger.info("[auto-repair] Reinstalling torch 2.8.0…")
        reinstall_ok, reinstall_detail = _pip(
            "uninstall", "-y", "torch", "torchvision", "torchaudio"
        )
        if reinstall_ok:
            ok, detail = _pip(
                "install",
                "torch==2.8.0", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cpu",
            )
            report.actions.append(RepairAction(
                issue="Broken torch installation (c10.dll DLL conflict with PyQt/NiceGUI on Windows)",
                action_taken=(
                    "Reinstalled torch==2.8.0 from pytorch.org/whl/cpu "
                    "(last version before the PyQt import-order regression in 2.9.0)"
                    if ok else "Reinstall failed — see detail"
                ),
                success=ok,
                detail=detail[:300],
            ))
        else:
            report.actions.append(RepairAction(
                issue="Broken torch installation (c10.dll DLL conflict with PyQt/NiceGUI on Windows)",
                action_taken="Attempted to uninstall broken version — uninstall failed",
                success=False,
                detail=reinstall_detail[:300],
            ))
    else:
        # torch is not installed, but we still got a DLL error —
        # likely a missing MSVC Redistributable.
        report.actions.append(RepairAction(
            issue="DLL load failure — torch is not installed, likely a missing runtime DLL",
            action_taken="No package change possible; providing guidance",
            success=False,
            detail=(
                "Install Microsoft Visual C++ Redistributable (x64) from: "
                "https://aka.ms/vs/17/release/vc_redist.x64.exe"
            ),
        ))


# Map Python import names to pip package install specs
_MODULE_TO_PKG: dict[str, str] = {
    "llama_index": "llama-index-core>=0.14.6",
    "openai": "openai>=1.0.0",
    "anthropic": "anthropic>=0.25.0",
    "google": "google-genai>=1.33.0",
    "dotenv": "python-dotenv>=1.0.0",
    "pydantic": "pydantic>=2.0.0",
    "numpy": "numpy>=1.24.0",
    "faiss": "faiss-cpu>=1.7.4",
    "nicegui": "nicegui>=2.0.0",
    "tiktoken": "tiktoken>=0.5.0",
    "tenacity": "tenacity>=8.0.0",
    "nest_asyncio": "nest-asyncio>=1.5.0",
    "pygments": "pygments>=2.15.0",
    "colorama": "colorama>=0.4.6",
}


def _repair_missing_module(module_name: str, report: RepairReport) -> None:
    """Try to install the pip package that provides ``module_name``.

    Args:
        module_name: Top-level Python module that could not be imported.
        report: Report to append the ``RepairAction`` to.
    """
    pkg = _MODULE_TO_PKG.get(module_name)
    if not pkg:
        report.actions.append(RepairAction(
            issue=f"Missing module: {module_name}",
            action_taken="No known pip package mapping — skipped",
            success=False,
            detail=f"Add '{module_name}' to requirements.txt and run pip install -r requirements.txt",
        ))
        return

    logger.info("[auto-repair] Installing missing package '%s' for module '%s'…", pkg, module_name)
    ok, detail = _pip("install", pkg)
    report.actions.append(RepairAction(
        issue=f"Missing package providing '{module_name}'",
        action_taken=f"pip install {pkg}",
        success=ok,
        detail=detail[:300],
    ))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def repair_for_error(error_msg: str) -> RepairReport:
    """Analyse an error message and apply all applicable auto-repairs.

    This is the main entry-point called by ``nicegui_app.py`` when
    ``RAGEngine()`` raises an exception.

    Args:
        error_msg: The string representation of the caught exception.

    Returns:
        A ``RepairReport`` describing every attempted repair and its outcome.
        The report may be empty if no applicable repair was found.
    """
    report = RepairReport()
    lower = error_msg.lower()

    # ── 1. Torch DLL load failure on Windows ────────────────────────────────
    torch_signals = [
        "dll initialization",
        "c10.dll",
        "winerror 1114",
        "[winerror 1114]",
        "error loading torch",
        "torch\\lib\\",
        "torch/lib/",
    ]
    if any(s in lower for s in torch_signals):
        _repair_torch_dll(report)

    # ── 2. Missing Python module ─────────────────────────────────────────────
    m = re.search(r"No module named '([^']+)'", error_msg)
    if m:
        module = m.group(1).split(".")[0]
        _repair_missing_module(module, report)

    if report.is_empty():
        logger.debug(
            "[auto-repair] No applicable repair found for error: %.100s", error_msg
        )

    return report


def quick_environment_check() -> List[str]:
    """Return a list of warning strings about the current environment.

    This function makes **no changes** — it is safe to call at any time and
    is used by the NiceGUI status banner to show environment warnings without
    triggering any repairs.

    Returns:
        List of warning message strings.  Empty list means no warnings.
    """
    warnings: List[str] = []

    if not os.getenv("OPENAI_API_KEY"):
        warnings.append("OPENAI_API_KEY is not set — embeddings will not work")

    llm_keys = [
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
    ]
    if not any(os.getenv(k) for k in llm_keys):
        warnings.append(
            "No LLM API key is configured — the system cannot generate answers. "
            "Open Settings to add at least one key."
        )

    return warnings
