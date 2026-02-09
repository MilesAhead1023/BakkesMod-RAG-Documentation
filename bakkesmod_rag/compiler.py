"""
Plugin Compiler
===============
Compiles generated BakkesMod plugin C++ code using MSVC (cl.exe).

Uses ``vswhere.exe`` to discover the Visual Studio installation, then
invokes ``vcvarsall.bat`` to set up the build environment before
compiling with ``cl.exe``.

The compiler runs in a temporary directory and cleans up afterwards.
If MSVC is not found, all operations degrade gracefully.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("bakkesmod_rag.compiler")

# Default vswhere location (ships with VS2017+ installer)
_VSWHERE_DEFAULT = (
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
)

# MSVC error/warning regex:
#   file.cpp(123,45) : error C2065: 'foo' : undeclared identifier
_MSVC_DIAG_RE = re.compile(
    r"^(.*?)\((\d+)(?:,(\d+))?\)\s*:\s*(error|warning)\s+(C\d+)\s*:\s*(.*)$",
    re.MULTILINE,
)


@dataclass
class CompilerError:
    """A single compiler diagnostic (error or warning).

    Attributes:
        file: Source file that produced the diagnostic.
        line: Line number.
        column: Column number (0 if not reported).
        level: ``"error"`` or ``"warning"``.
        code: MSVC error code (e.g. ``"C2065"``).
        message: Human-readable description.
    """

    file: str
    line: int
    column: int
    level: str   # "error" | "warning"
    code: str    # e.g. "C2065"
    message: str

    def __str__(self) -> str:
        loc = f"{self.file}({self.line}"
        if self.column:
            loc += f",{self.column}"
        loc += ")"
        return f"{loc}: {self.level} {self.code}: {self.message}"


@dataclass
class CompileResult:
    """Result of a compilation attempt.

    Attributes:
        success: Whether compilation succeeded (zero errors).
        errors: List of compiler errors.
        warnings: List of compiler warnings.
        output: Raw compiler stdout+stderr.
        return_code: Process return code.
    """

    success: bool = False
    errors: List[CompilerError] = field(default_factory=list)
    warnings: List[CompilerError] = field(default_factory=list)
    output: str = ""
    return_code: int = -1


class PluginCompiler:
    """Compiles BakkesMod plugin C++ code using MSVC.

    Automatically discovers the Visual Studio installation via
    ``vswhere.exe`` and locates ``vcvarsall.bat``.  If MSVC is not
    found, :pyattr:`available` is ``False`` and :pymeth:`compile_project`
    returns a failed :class:`CompileResult` immediately.

    Args:
        msvc_path: Explicit path to the VS installation root.
            If ``None``, auto-detected via ``vswhere``.
        sdk_include_dirs: Additional include directories (e.g. BakkesMod
            SDK headers).
        compile_timeout: Maximum seconds for the compiler subprocess.
    """

    def __init__(
        self,
        msvc_path: Optional[str] = None,
        sdk_include_dirs: Optional[List[str]] = None,
        compile_timeout: int = 30,
    ) -> None:
        self.compile_timeout = compile_timeout
        self.sdk_include_dirs = sdk_include_dirs or []
        self._temp_dir: Optional[Path] = None

        # Discover MSVC
        if msvc_path:
            self._vs_root = Path(msvc_path)
        else:
            self._vs_root = self._find_vs_installation()

        self._vcvarsall = self._find_vcvarsall()
        self.available = self._vcvarsall is not None

        if self.available:
            logger.info("MSVC compiler available: %s", self._vcvarsall)
        else:
            logger.warning(
                "MSVC compiler not found — compilation will be skipped"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile_project(self, files: Dict[str, str]) -> CompileResult:
        """Compile a set of C++ source files.

        Writes files to a temporary directory, invokes ``cl.exe`` via
        ``vcvarsall.bat``, parses the output for errors/warnings, and
        cleans up the temporary directory.

        Args:
            files: Dict mapping filename to file content (e.g.
                ``{"plugin.h": "...", "plugin.cpp": "..."}``).

        Returns:
            :class:`CompileResult` with parsed diagnostics.
        """
        if not self.available:
            return CompileResult(
                success=False,
                output="MSVC compiler not available",
                return_code=-1,
            )

        try:
            project_dir = self._write_temp_project(files)
            result = self._run_compiler(project_dir, files)
            return result
        except Exception as e:
            logger.error("Compilation failed with exception: %s", e)
            return CompileResult(
                success=False,
                output=str(e),
                return_code=-1,
            )
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Remove the temporary project directory if it exists."""
        if self._temp_dir and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
                self._temp_dir = None
            except Exception as e:
                logger.warning("Failed to clean up temp dir: %s", e)

    # ------------------------------------------------------------------
    # MSVC Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def _find_vs_installation() -> Optional[Path]:
        """Find the Visual Studio installation root via vswhere.exe.

        Returns:
            Path to the VS installation root, or ``None`` if not found.
        """
        if not _VSWHERE_DEFAULT.exists():
            logger.debug("vswhere.exe not found at %s", _VSWHERE_DEFAULT)
            return None

        try:
            result = subprocess.run(
                [
                    str(_VSWHERE_DEFAULT),
                    "-products", "*",
                    "-latest",
                    "-prerelease",
                    "-property", "installationPath",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            path = result.stdout.strip()
            if path and Path(path).exists():
                return Path(path)
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.debug("vswhere failed: %s", e)

        return None

    def _find_vcvarsall(self) -> Optional[Path]:
        """Locate vcvarsall.bat within the VS installation.

        Returns:
            Path to vcvarsall.bat, or ``None`` if not found.
        """
        if not self._vs_root:
            return None

        candidate = self._vs_root / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        if candidate.exists():
            return candidate

        return None

    # ------------------------------------------------------------------
    # Compilation
    # ------------------------------------------------------------------

    def _write_temp_project(self, files: Dict[str, str]) -> Path:
        """Write project files to a temporary directory.

        Args:
            files: Dict mapping filename to content.

        Returns:
            Path to the temporary directory.
        """
        self._temp_dir = Path(tempfile.mkdtemp(prefix="bakkesmod_compile_"))

        for filename, content in files.items():
            filepath = self._temp_dir / filename
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding="utf-8")

        return self._temp_dir

    def _run_compiler(
        self,
        project_dir: Path,
        files: Dict[str, str],
    ) -> CompileResult:
        """Run cl.exe on the project files.

        Invokes ``vcvarsall.bat x64`` to set up the environment, then
        compiles all ``.cpp`` files with ``/EHsc /c`` (compile-only,
        no linking — we just want to check for errors).

        Args:
            project_dir: Path to temporary project directory.
            files: Original files dict (to identify .cpp files).

        Returns:
            :class:`CompileResult` with parsed diagnostics.
        """
        cpp_files = [f for f in files if f.endswith(".cpp")]
        if not cpp_files:
            return CompileResult(
                success=True,
                output="No .cpp files to compile",
                return_code=0,
            )

        # Build include path arguments
        include_args = [f'/I"{project_dir}"']
        for inc_dir in self.sdk_include_dirs:
            if Path(inc_dir).exists():
                include_args.append(f'/I"{inc_dir}"')

        cpp_paths = " ".join(
            f'"{project_dir / f}"' for f in cpp_files
        )
        includes = " ".join(include_args)

        # Construct the command:
        #   vcvarsall.bat x64 && cl.exe /EHsc /c /W3 /nologo <includes> <files>
        cmd = (
            f'"{self._vcvarsall}" x64 && '
            f"cl.exe /EHsc /c /W3 /nologo {includes} {cpp_paths}"
        )

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                shell=True,
                timeout=self.compile_timeout,
                cwd=str(project_dir),
            )

            errors, warnings = self._parse_diagnostics(proc.stdout + proc.stderr)

            return CompileResult(
                success=proc.returncode == 0 and len(errors) == 0,
                errors=errors,
                warnings=warnings,
                output=(proc.stdout + proc.stderr).strip(),
                return_code=proc.returncode,
            )

        except subprocess.TimeoutExpired:
            return CompileResult(
                success=False,
                output=f"Compilation timed out after {self.compile_timeout}s",
                return_code=-1,
            )

    @staticmethod
    def _parse_diagnostics(
        output: str,
    ) -> tuple[List[CompilerError], List[CompilerError]]:
        """Parse MSVC compiler output for errors and warnings.

        Args:
            output: Combined stdout+stderr from cl.exe.

        Returns:
            Tuple of (errors, warnings) as lists of CompilerError.
        """
        errors: List[CompilerError] = []
        warnings: List[CompilerError] = []

        for match in _MSVC_DIAG_RE.finditer(output):
            diag = CompilerError(
                file=match.group(1).strip(),
                line=int(match.group(2)),
                column=int(match.group(3)) if match.group(3) else 0,
                level=match.group(4),
                code=match.group(5),
                message=match.group(6).strip(),
            )

            if diag.level == "error":
                errors.append(diag)
            else:
                warnings.append(diag)

        return errors, warnings

    def format_errors_for_llm(self, result: CompileResult) -> str:
        """Format compiler errors into a prompt-friendly string.

        Args:
            result: CompileResult from a failed compilation.

        Returns:
            Multi-line string describing each error for LLM consumption.
        """
        if not result.errors:
            return "No compiler errors."

        lines = ["The following compiler errors were found:\n"]
        for i, err in enumerate(result.errors, 1):
            lines.append(
                f"{i}. {err.file} line {err.line}: "
                f"{err.code} — {err.message}"
            )

        if result.warnings:
            lines.append(f"\nAdditionally, {len(result.warnings)} warnings.")

        return "\n".join(lines)
