"""Tests for servers.lint — unified code quality and linting tools."""

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.servers.lint import (
    LintIssue,
    _check_tool_installed,
    _detect_language,
    _run_command,
    lint,
)


class TestLintIssue:
    def test_to_string(self):
        issue = LintIssue(
            file="foo.py",
            line=10,
            column=5,
            severity="error",
            code="E501",
            message="line too long",
            source="ruff",
        )
        result = issue.to_string()
        assert "foo.py:10:5" in result
        assert "[error]" in result
        assert "E501" in result
        assert "line too long" in result


class TestRunCommand:
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("subprocess.run")
    def test_success(self, mock_run, mock_validate):
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        code, out, err = _run_command(["echo", "hi"], cwd="/tmp")
        assert code == 0
        assert out == "ok"

    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    def test_command_not_found(self, mock_validate):
        code, out, err = _run_command(["nonexistent_cmd_xyz"])
        assert code == -1
        assert "not found" in err

    @patch("src.servers.lint.validate_path")
    def test_invalid_path(self, mock_validate):
        mock_validate.side_effect = PermissionError("denied")
        code, out, err = _run_command(["echo", "hi"], cwd="/bad")
        assert code == -1
        assert "Invalid path" in err


class TestCheckToolInstalled:
    @patch("subprocess.run")
    def test_installed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        assert _check_tool_installed("python3") is True

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_not_installed(self, mock_run):
        assert _check_tool_installed("nonexistent_tool_xyz") is False

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="x", timeout=5))
    def test_timeout(self, mock_run):
        assert _check_tool_installed("bad_tool") is False


class TestDetectLanguage:
    def test_python_file(self, tmp_path):
        py = tmp_path / "test.py"
        py.write_text("x = 1")
        with (
            patch("src.servers.lint.validate_path", side_effect=lambda x: x),
            patch("os.path.isfile", return_value=True),
        ):
            assert _detect_language(str(py)) == "python"

    def test_javascript_file(self, tmp_path):
        js = tmp_path / "test.js"
        js.write_text("x = 1")
        with (
            patch("src.servers.lint.validate_path", side_effect=lambda x: x),
            patch("os.path.isfile", return_value=True),
        ):
            assert _detect_language(str(js)) == "javascript"

    def test_typescript_file(self, tmp_path):
        ts = tmp_path / "test.ts"
        ts.write_text("x = 1")
        with (
            patch("src.servers.lint.validate_path", side_effect=lambda x: x),
            patch("os.path.isfile", return_value=True),
        ):
            assert _detect_language(str(ts)) == "javascript"

    def test_unknown_extension(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("x")
        with (
            patch("src.servers.lint.validate_path", side_effect=lambda x: x),
            patch("os.path.isfile", return_value=True),
        ):
            assert _detect_language(str(txt)) is None

    def test_invalid_path(self):
        with patch("src.servers.lint.validate_path", side_effect=ValueError("bad")):
            assert _detect_language("bad") is None

    def test_directory_with_python(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1")
        with (
            patch("src.servers.lint.validate_path", side_effect=lambda x: x),
            patch("os.path.isfile", return_value=False),
            patch("os.walk", return_value=[(str(tmp_path), [], ["main.py"])]),
        ):
            assert _detect_language(str(tmp_path)) == "python"

    def test_directory_with_js(self, tmp_path):
        (tmp_path / "index.js").write_text("x = 1")
        with (
            patch("src.servers.lint.validate_path", side_effect=lambda x: x),
            patch("os.path.isfile", return_value=False),
            patch("os.walk", return_value=[(str(tmp_path), [], ["index.js"])]),
        ):
            assert _detect_language(str(tmp_path)) == "javascript"

    def test_directory_prefers_python(self, tmp_path):
        (tmp_path / "main.py").write_text("x = 1")
        (tmp_path / "index.js").write_text("x = 1")
        with (
            patch("src.servers.lint.validate_path", side_effect=lambda x: x),
            patch("os.path.isfile", return_value=False),
            patch("os.walk", return_value=[(str(tmp_path), [], ["main.py", "index.js"])]),
        ):
            assert _detect_language(str(tmp_path)) == "python"

    def test_empty_directory(self, tmp_path):
        with (
            patch("src.servers.lint.validate_path", side_effect=lambda x: x),
            patch("os.path.isfile", return_value=False),
            patch("os.walk", return_value=[(str(tmp_path), [], [])]),
        ):
            assert _detect_language(str(tmp_path)) is None


class TestLint:
    def test_invalid_operation(self):
        result = lint("src/", operation="invalid_op")
        assert "Invalid operation" in result

    @patch("src.servers.lint._detect_language", return_value="python")
    @patch("src.servers.lint._check_tool_installed", return_value=False)
    def test_check_ruff_not_installed(self, mock_tool, mock_detect):
        result = lint("src/", operation="check")
        assert "not installed" in result

    @patch("src.servers.lint._detect_language", return_value="javascript")
    @patch("src.servers.lint._check_tool_installed", return_value=False)
    def test_check_npx_not_installed(self, mock_tool, mock_detect):
        result = lint("src/", operation="check")
        assert "not available" in result

    @patch("src.servers.lint._detect_language", return_value="javascript")
    def test_format_javascript_not_supported(self, mock_detect):
        result = lint("src/", operation="format")
        assert "not yet supported" in result

    @patch("src.servers.lint._detect_language", return_value="javascript")
    def test_typecheck_javascript_not_supported(self, mock_detect):
        result = lint("src/", operation="typecheck")
        assert "not yet supported" in result

    @patch("src.servers.lint._detect_language", return_value=None)
    @patch("src.servers.lint._check_tool_installed", return_value=False)
    def test_check_no_linters_available(self, mock_tool, mock_detect):
        result = lint("src/", operation="check")
        assert "no linters available" in result

    @patch("src.servers.lint._detect_language", return_value=None)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    @patch("src.servers.lint._lint_check_python", return_value="py ok")
    @patch("src.servers.lint._lint_check_javascript", return_value="js ok")
    def test_check_undetected_tries_both(self, mock_js, mock_py, mock_tool, mock_detect):
        result = lint("src/", operation="check")
        assert "Python" in result
        assert "JavaScript" in result

    def test_format_defaults_to_python(self):
        with (
            patch("src.servers.lint._detect_language", return_value=None),
            patch("src.servers.lint._lint_format_python", return_value="formatted"),
        ):
            result = lint("src/", operation="format")
            assert "formatted" in result

    @patch("src.servers.lint._detect_language", return_value="python")
    @patch("src.servers.lint._lint_format_python", return_value="format check done")
    def test_format_check_only(self, mock_fmt, mock_detect):
        result = lint("src/", operation="format", fix=False)
        assert "format check done" in result

    @patch("src.servers.lint._detect_language", return_value="python")
    @patch("src.servers.lint._check_tool_installed", return_value=False)
    def test_security_ruff_not_installed(self, mock_tool, mock_detect):
        result = lint("src/", operation="security")
        assert "not installed" in result

    @patch("src.servers.lint._detect_language", return_value="python")
    @patch("src.servers.lint._check_tool_installed", return_value=False)
    def test_complexity_ruff_not_installed(self, mock_tool, mock_detect):
        result = lint("src/", operation="complexity")
        assert "not installed" in result

    @patch("src.servers.lint._detect_language", return_value="python")
    @patch("src.servers.lint._check_tool_installed", return_value=False)
    def test_summary_ruff_not_installed(self, mock_tool, mock_detect):
        result = lint("src/", operation="summary")
        assert "not installed" in result

    @patch("src.servers.lint._detect_language", return_value="python")
    @patch("src.servers.lint._lint_check_python", return_value="check done")
    @patch("src.servers.lint._lint_format_python", return_value="fmt done")
    @patch("src.servers.lint._lint_typecheck_python", return_value="type done")
    @patch("src.servers.lint._lint_security", return_value="sec done")
    @patch("src.servers.lint._lint_complexity", return_value="complex done")
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_all_operation(
        self, mock_tool, mock_complexity, mock_sec, mock_type, mock_fmt, mock_check, mock_detect
    ):
        result = lint("src/", operation="all")
        assert "Lint Check" in result
        assert "Format Check" in result
        assert "Type Check" in result
        assert "Security Check" in result
        assert "Complexity Check" in result


class TestLintCheckPython:
    @patch("src.servers.lint._check_tool_installed", return_value=False)
    def test_ruff_not_installed(self, mock_tool):
        from src.servers.lint import _lint_check_python

        result = _lint_check_python("src/")
        assert "not installed" in result

    @patch("src.servers.lint.validate_path", side_effect=PermissionError("denied"))
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_invalid_path(self, mock_tool, mock_val):
        from src.servers.lint import _lint_check_python

        result = _lint_check_python("/bad")
        assert "Error" in result

    @patch("src.servers.lint._run_command", return_value=(0, "", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_no_issues_empty_stdout(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_python

        result = _lint_check_python("src/")
        assert "No issues" in result

    @patch(
        "src.servers.lint._run_command",
        return_value=(
            0,
            '[{"code":"E501","filename":"a.py","location":{"row":1,"column":1},"message":"line too long"}]',
            "",
        ),
    )
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_issues_found(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_python

        result = _lint_check_python("src/")
        assert "1 issue" in result
        assert "E501" in result

    @patch(
        "src.servers.lint._run_command",
        return_value=(
            0,
            '[{"code":"E501","filename":"a.py","location":{"row":1,"column":1},"message":"line too long","fix":{"message":"fix it"}}]',
            "",
        ),
    )
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_auto_fixable_flag(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_python

        result = _lint_check_python("src/")
        assert "Auto-fixable" in result

    @patch("src.servers.lint._run_command", return_value=(1, "", "error: something"))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_stderr_with_error(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_python

        result = _lint_check_python("src/")
        assert "Error running Ruff" in result

    @patch("src.servers.lint._run_command", return_value=(0, "not json", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_json_decode_error_fallback(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_python

        result = _lint_check_python("src/")
        assert "not json" in result

    @patch("src.servers.lint._run_command", return_value=(0, "[]", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_empty_issues_list(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_python

        result = _lint_check_python("src/")
        assert "No issues found" in result

    @patch("src.servers.lint._run_command", return_value=(0, "", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_format_check_only(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_format_python

        result = _lint_format_python("src/", check_only=True)
        assert "properly formatted" in result

    @patch("src.servers.lint._run_command", return_value=(1, "needs fixing", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_format_check_issues(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_format_python

        result = _lint_format_python("src/", check_only=True)
        assert "Formatting issues" in result

    @patch("src.servers.lint._run_command", return_value=(0, "", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_format_complete(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_format_python

        result = _lint_format_python("src/")
        assert "Formatting complete" in result


class TestLintTypecheckPython:
    @patch("src.servers.lint._check_tool_installed", return_value=False)
    def test_mypy_not_installed(self, mock_tool):
        from src.servers.lint import _lint_typecheck_python

        result = _lint_typecheck_python("src/")
        assert "not installed" in result

    @patch("src.servers.lint._run_command", return_value=(0, "", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_no_type_errors(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_typecheck_python

        result = _lint_typecheck_python("src/")
        assert "No type errors" in result

    @patch(
        "src.servers.lint._run_command",
        return_value=(1, "a.py:1: error: bad type\na.py:2: warning: x", ""),
    )
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_type_errors_found(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_typecheck_python

        result = _lint_typecheck_python("src/")
        assert "1 error" in result


class TestLintSecurity:
    @patch("src.servers.lint._run_command", return_value=(0, "[]", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_no_security_issues(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_security

        result = _lint_security("src/")
        assert "No security issues" in result

    @patch("src.servers.lint._run_command", return_value=(1, "", "error: fail"))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_security_stderr_error(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_security

        result = _lint_security("src/")
        assert "Error" in result


class TestLintSummary:
    @patch("src.servers.lint._run_command", return_value=(0, "[]", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_no_issues(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_summary

        result = _lint_summary("src/")
        assert "No issues found" in result

    @patch(
        "src.servers.lint._run_command",
        return_value=(0, '[{"code":"E501"},{"code":"F401"},{"code":"W291"}]', ""),
    )
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_summary_with_categories(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_summary

        result = _lint_summary("src/")
        assert "Lint Summary" in result


class TestLintCheckJavaScript:
    @patch("src.servers.lint._check_tool_installed", return_value=False)
    def test_npx_not_installed(self, mock_tool):
        from src.servers.lint import _lint_check_javascript

        result = _lint_check_javascript("src/")
        assert "not available" in result

    @patch("src.servers.lint.validate_path", side_effect=PermissionError("denied"))
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_invalid_path(self, mock_tool, mock_val):
        from src.servers.lint import _lint_check_javascript

        result = _lint_check_javascript("/bad")
        assert "Error" in result

    @patch("src.servers.lint._run_command", return_value=(0, "", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_no_issues(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_javascript

        result = _lint_check_javascript("src/")
        assert "No issues found" in result

    @patch("src.servers.lint._run_command", return_value=(1, "", "eslint: command not found"))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_eslint_not_installed(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_javascript

        result = _lint_check_javascript("src/")
        assert "ESLint is not installed" in result

    @patch("src.servers.lint._run_command", return_value=(1, "issues found", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_issues_found(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_javascript

        result = _lint_check_javascript("src/")
        assert "issues found" in result

    @patch("src.servers.lint._run_command", return_value=(1, "", "Cannot find module eslint"))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_cannot_find_module(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_javascript

        result = _lint_check_javascript("src/")
        assert "ESLint is not installed" in result


class TestLintCheckPythonWithFix:
    @patch("src.servers.lint._run_command", return_value=(0, "", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_fix_applied(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_python

        result = _lint_check_python("src/", fix=True)
        assert "No issues" in result


class TestLintCheckPythonWithSelect:
    @patch("src.servers.lint._run_command", return_value=(0, "", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_select_filter(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_python

        result = _lint_check_python("src/", select=["E501"])
        assert "No issues" in result


class TestLintCheckPythonWithIgnore:
    @patch("src.servers.lint._run_command", return_value=(0, "", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_ignore_filter(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_python

        result = _lint_check_python("src/", ignore=["E501"])
        assert "No issues" in result


class TestLintFormatPythonError:
    @patch("src.servers.lint._run_command", return_value=(1, "", "format error"))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_format_error(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_format_python

        result = _lint_format_python("src/")
        assert "Error" in result


class TestLintTypecheckPythonWithStrict:
    @patch("src.servers.lint._run_command", return_value=(0, "", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_strict_mode(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_typecheck_python

        result = _lint_typecheck_python("src/", strict=True)
        assert "No type errors" in result


class TestLintTypecheckPythonInvalidPath:
    @patch("src.servers.lint.validate_path", side_effect=PermissionError("denied"))
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_invalid_path(self, mock_tool, mock_val):
        from src.servers.lint import _lint_typecheck_python

        result = _lint_typecheck_python("/bad")
        assert "Error" in result


class TestLintTypecheckNoOutput:
    @patch("src.servers.lint._run_command", return_value=(1, "", "type error"))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_stderr_only(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_typecheck_python

        result = _lint_typecheck_python("src/")
        assert "type error" in result


class TestLintTypecheckNoOutputNoStderr:
    @patch("src.servers.lint._run_command", return_value=(1, "", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_no_output_no_stderr(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_typecheck_python

        result = _lint_typecheck_python("src/")
        assert "Type checking complete" in result


class TestLintSecurityWithIssues:
    @patch(
        "src.servers.lint._run_command",
        return_value=(
            0,
            '[{"filename":"a.py","location":{"row":1},"code":"S101","message":"assert used"}]',
            "",
        ),
    )
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_security_issues_found(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_security

        result = _lint_security("src/")
        assert "1 potential security" in result


class TestLintSecurityInvalidPath:
    @patch("src.servers.lint.validate_path", side_effect=PermissionError("denied"))
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_invalid_path(self, mock_tool, mock_val):
        from src.servers.lint import _lint_security

        result = _lint_security("/bad")
        assert "Error" in result


class TestLintSecurityJsonDecodeError:
    @patch("src.servers.lint._run_command", return_value=(0, "bad json", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_json_decode_error(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_security

        result = _lint_security("src/")
        assert "bad json" in result


class TestLintComplexityWithIssues:
    @patch(
        "src.servers.lint._run_command",
        return_value=(
            0,
            '[{"filename":"a.py","location":{"row":1},"code":"C901","message":"`func` is too complex (15 > 10)"}]',
            "",
        ),
    )
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_complexity_issues_found(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_complexity

        result = _lint_complexity("src/", max_complexity=10)
        assert "func" in result


class TestLintComplexityNoMatchFilter:
    @patch(
        "src.servers.lint._run_command",
        return_value=(
            0,
            '[{"filename":"a.py","location":{"row":1},"code":"C901","message":"`func` is too complex (8 > 10)"}]',
            "",
        ),
    )
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_complexity_below_threshold(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_complexity

        result = _lint_complexity("src/", max_complexity=10)
        assert "complexity <= 10" in result


class TestLintComplexityInvalidPath:
    @patch("src.servers.lint.validate_path", side_effect=PermissionError("denied"))
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_invalid_path(self, mock_tool, mock_val):
        from src.servers.lint import _lint_complexity

        result = _lint_complexity("/bad")
        assert "Error" in result


class TestLintComplexityJsonDecodeError:
    @patch("src.servers.lint._run_command", return_value=(0, "bad json", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_json_decode_error(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_complexity

        result = _lint_complexity("src/")
        assert "bad json" in result


class TestLintComplexityStderrError:
    @patch("src.servers.lint._run_command", return_value=(1, "", "error: fail"))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_stderr_error(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_complexity

        result = _lint_complexity("src/")
        assert "Error" in result


class TestLintSummaryInvalidPath:
    @patch("src.servers.lint.validate_path", side_effect=PermissionError("denied"))
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_invalid_path(self, mock_tool, mock_val):
        from src.servers.lint import _lint_summary

        result = _lint_summary("/bad")
        assert "Error" in result


class TestLintSummaryJsonDecodeError:
    @patch("src.servers.lint._run_command", return_value=(0, "bad json", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_json_decode_error(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_summary

        result = _lint_summary("src/")
        assert "bad json" in result


class TestLintSummaryStderrError:
    @patch("src.servers.lint._run_command", return_value=(1, "", "error: fail"))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_stderr_error(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_summary

        result = _lint_summary("src/")
        assert "Error" in result


class TestRunCommandGeneralException:
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("subprocess.run", side_effect=RuntimeError("weird"))
    def test_general_exception(self, mock_run, mock_validate):
        code, out, err = _run_command(["echo", "hi"])
        assert code == -1
        assert "Error" in err


class TestLintCheckJsWithExtAndFix:
    @patch("src.servers.lint._run_command", return_value=(0, "", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_fix_and_ext(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_check_javascript

        result = _lint_check_javascript("src/", fix=True, ext=[".ts"])
        assert "No issues found" in result


class TestLintFormatCheckOnly:
    @patch("src.servers.lint._run_command", return_value=(0, "", ""))
    @patch("src.servers.lint.validate_path", side_effect=lambda x: x)
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_format_check_only_no_issues(self, mock_tool, mock_val, mock_run):
        from src.servers.lint import _lint_format_python

        result = _lint_format_python("src/", check_only=True)
        assert "properly formatted" in result


class TestLintFormatInvalidPath:
    @patch("src.servers.lint.validate_path", side_effect=PermissionError("denied"))
    @patch("src.servers.lint._check_tool_installed", return_value=True)
    def test_invalid_path(self, mock_tool, mock_val):
        from src.servers.lint import _lint_format_python

        result = _lint_format_python("/bad")
        assert "Error" in result
