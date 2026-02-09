"""
Tests for PluginCompiler
========================
Validates MSVC detection, error parsing, temp file management,
timeout handling, and graceful degradation.
"""

import os
import re
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from bakkesmod_rag.compiler import (
    PluginCompiler,
    CompileResult,
    CompilerError,
    _MSVC_DIAG_RE,
)


# ---------------------------------------------------------------------------
# MSVC error regex tests
# ---------------------------------------------------------------------------

class TestMSVCErrorRegex:
    """Verify the MSVC diagnostic regex parses real compiler output."""

    def test_parse_error_with_column(self):
        line = r"plugin.cpp(42,10) : error C2065: 'foo' : undeclared identifier"
        match = _MSVC_DIAG_RE.search(line)
        assert match is not None
        assert match.group(1).strip() == "plugin.cpp"
        assert match.group(2) == "42"
        assert match.group(3) == "10"
        assert match.group(4) == "error"
        assert match.group(5) == "C2065"
        assert "'foo'" in match.group(6)

    def test_parse_error_without_column(self):
        line = r"plugin.cpp(15) : error C2143: syntax error"
        match = _MSVC_DIAG_RE.search(line)
        assert match is not None
        assert match.group(2) == "15"
        assert match.group(3) is None
        assert match.group(4) == "error"

    def test_parse_warning(self):
        line = r"plugin.h(7) : warning C4100: 'params' : unreferenced formal parameter"
        match = _MSVC_DIAG_RE.search(line)
        assert match is not None
        assert match.group(4) == "warning"
        assert match.group(5) == "C4100"

    def test_no_match_on_normal_text(self):
        line = "Compiling... plugin.cpp"
        match = _MSVC_DIAG_RE.search(line)
        assert match is None

    def test_parse_full_path_error(self):
        line = r"C:\Users\test\plugin.cpp(100,5) : error C2039: 'GetVelocity' is not a member"
        match = _MSVC_DIAG_RE.search(line)
        assert match is not None
        assert "plugin.cpp" in match.group(1)
        assert match.group(2) == "100"
        assert match.group(3) == "5"


# ---------------------------------------------------------------------------
# CompilerError dataclass tests
# ---------------------------------------------------------------------------

class TestCompilerError:
    """Test CompilerError string representation."""

    def test_str_with_column(self):
        err = CompilerError(
            file="plugin.cpp", line=42, column=10,
            level="error", code="C2065", message="undeclared identifier",
        )
        s = str(err)
        assert "plugin.cpp(42,10)" in s
        assert "error C2065" in s

    def test_str_without_column(self):
        err = CompilerError(
            file="plugin.h", line=7, column=0,
            level="warning", code="C4100", message="unreferenced",
        )
        s = str(err)
        assert "plugin.h(7)" in s
        assert ",0" not in s.split(":")[0]  # column=0 means no column shown


# ---------------------------------------------------------------------------
# PluginCompiler._parse_diagnostics tests
# ---------------------------------------------------------------------------

class TestParseDiagnostics:
    """Test compiler output parsing."""

    def test_parse_mixed_output(self):
        output = """Microsoft (R) C/C++ Optimizing Compiler
plugin.cpp(10) : error C2065: 'gameWrapper' : undeclared identifier
plugin.cpp(20,5) : warning C4100: 'params' : unreferenced
plugin.h(3) : error C2143: syntax error : missing ';'
"""
        errors, warnings = PluginCompiler._parse_diagnostics(output)
        assert len(errors) == 2
        assert len(warnings) == 1
        assert errors[0].line == 10
        assert errors[0].code == "C2065"
        assert warnings[0].code == "C4100"

    def test_parse_clean_output(self):
        output = "Microsoft (R) C/C++ Optimizing Compiler\nplugin.cpp\n"
        errors, warnings = PluginCompiler._parse_diagnostics(output)
        assert len(errors) == 0
        assert len(warnings) == 0

    def test_parse_multiple_errors_same_file(self):
        output = """plugin.cpp(1) : error C1000: err1
plugin.cpp(2) : error C1001: err2
plugin.cpp(3) : error C1002: err3
"""
        errors, warnings = PluginCompiler._parse_diagnostics(output)
        assert len(errors) == 3
        assert errors[0].line == 1
        assert errors[2].line == 3


# ---------------------------------------------------------------------------
# PluginCompiler initialization tests
# ---------------------------------------------------------------------------

class TestCompilerInit:
    """Test compiler initialization and MSVC discovery."""

    @patch.object(PluginCompiler, "_find_vs_installation", return_value=None)
    def test_not_available_when_no_vs(self, mock_find):
        compiler = PluginCompiler()
        assert compiler.available is False

    @patch.object(PluginCompiler, "_find_vs_installation")
    def test_available_with_valid_vcvarsall(self, mock_find, tmp_path):
        # Create a fake vcvarsall.bat
        vs_root = tmp_path / "VS"
        vcvars = vs_root / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        vcvars.parent.mkdir(parents=True)
        vcvars.write_text("@echo off")
        mock_find.return_value = vs_root

        compiler = PluginCompiler()
        assert compiler.available is True

    def test_explicit_msvc_path(self, tmp_path):
        vs_root = tmp_path / "VS"
        vcvars = vs_root / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        vcvars.parent.mkdir(parents=True)
        vcvars.write_text("@echo off")

        compiler = PluginCompiler(msvc_path=str(vs_root))
        assert compiler.available is True


# ---------------------------------------------------------------------------
# Compile project tests (mocked subprocess)
# ---------------------------------------------------------------------------

class TestCompileProject:
    """Test the compile_project method with mocked subprocesses."""

    @patch.object(PluginCompiler, "_find_vs_installation", return_value=None)
    def test_compile_when_unavailable(self, mock_find):
        compiler = PluginCompiler()
        result = compiler.compile_project({"plugin.cpp": "int main() {}"})
        assert result.success is False
        assert "not available" in result.output

    def test_compile_no_cpp_files(self, tmp_path):
        vs_root = tmp_path / "VS"
        vcvars = vs_root / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        vcvars.parent.mkdir(parents=True)
        vcvars.write_text("@echo off")

        compiler = PluginCompiler(msvc_path=str(vs_root))
        result = compiler.compile_project({"plugin.h": "#pragma once"})
        assert result.success is True
        assert "No .cpp files" in result.output

    @patch("subprocess.run")
    def test_compile_success(self, mock_run, tmp_path):
        vs_root = tmp_path / "VS"
        vcvars = vs_root / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        vcvars.parent.mkdir(parents=True)
        vcvars.write_text("@echo off")

        mock_run.return_value = MagicMock(
            returncode=0, stdout="Compiling...\n", stderr=""
        )

        compiler = PluginCompiler(msvc_path=str(vs_root))
        result = compiler.compile_project({
            "plugin.h": "#pragma once",
            "plugin.cpp": '#include "plugin.h"\nint main() {}',
        })
        assert result.success is True
        assert len(result.errors) == 0

    @patch("subprocess.run")
    def test_compile_with_errors(self, mock_run, tmp_path):
        vs_root = tmp_path / "VS"
        vcvars = vs_root / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        vcvars.parent.mkdir(parents=True)
        vcvars.write_text("@echo off")

        mock_run.return_value = MagicMock(
            returncode=2,
            stdout="plugin.cpp(5) : error C2065: 'x' : undeclared\n",
            stderr="",
        )

        compiler = PluginCompiler(msvc_path=str(vs_root))
        result = compiler.compile_project({
            "plugin.cpp": "void f() { x = 1; }",
        })
        assert result.success is False
        assert len(result.errors) == 1
        assert result.errors[0].code == "C2065"

    @patch("subprocess.run", side_effect=Exception("subprocess crashed"))
    def test_compile_exception(self, mock_run, tmp_path):
        vs_root = tmp_path / "VS"
        vcvars = vs_root / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        vcvars.parent.mkdir(parents=True)
        vcvars.write_text("@echo off")

        compiler = PluginCompiler(msvc_path=str(vs_root))
        result = compiler.compile_project({"plugin.cpp": "int main() {}"})
        assert result.success is False


# ---------------------------------------------------------------------------
# Temp file cleanup tests
# ---------------------------------------------------------------------------

class TestTempFileCleanup:
    """Test that temp directories are cleaned up properly."""

    @patch("subprocess.run")
    def test_cleanup_after_compile(self, mock_run, tmp_path):
        vs_root = tmp_path / "VS"
        vcvars = vs_root / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        vcvars.parent.mkdir(parents=True)
        vcvars.write_text("@echo off")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        compiler = PluginCompiler(msvc_path=str(vs_root))
        compiler.compile_project({"plugin.cpp": "int main() {}"})

        # After compile_project, temp dir should be cleaned up
        assert compiler._temp_dir is None


# ---------------------------------------------------------------------------
# Format errors for LLM tests
# ---------------------------------------------------------------------------

class TestFormatErrorsForLLM:
    """Test error formatting for LLM feedback."""

    @patch.object(PluginCompiler, "_find_vs_installation", return_value=None)
    def test_format_errors(self, mock_find):
        compiler = PluginCompiler()
        result = CompileResult(
            success=False,
            errors=[
                CompilerError("plugin.cpp", 10, 0, "error", "C2065", "undeclared"),
                CompilerError("plugin.cpp", 20, 5, "error", "C2143", "syntax error"),
            ],
            warnings=[
                CompilerError("plugin.h", 7, 0, "warning", "C4100", "unreferenced"),
            ],
        )
        formatted = compiler.format_errors_for_llm(result)
        assert "1." in formatted
        assert "2." in formatted
        assert "C2065" in formatted
        assert "1 warnings" in formatted

    @patch.object(PluginCompiler, "_find_vs_installation", return_value=None)
    def test_format_no_errors(self, mock_find):
        compiler = PluginCompiler()
        result = CompileResult(success=True)
        formatted = compiler.format_errors_for_llm(result)
        assert "No compiler errors" in formatted


# ---------------------------------------------------------------------------
# Timeout tests
# ---------------------------------------------------------------------------

class TestTimeout:
    """Test compilation timeout handling."""

    @patch("subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("cl.exe", 30))
    def test_timeout_returns_failure(self, mock_run, tmp_path):
        vs_root = tmp_path / "VS"
        vcvars = vs_root / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        vcvars.parent.mkdir(parents=True)
        vcvars.write_text("@echo off")

        compiler = PluginCompiler(msvc_path=str(vs_root), compile_timeout=30)
        result = compiler.compile_project({"plugin.cpp": "int main() {}"})
        assert result.success is False
        assert "timed out" in result.output.lower()
