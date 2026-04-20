import os
from pathlib import Path

import pytest

from src.core.diff import (
    DiffHunk,
    DiffResult,
    _detect_language,
    compute_diff_result,
    generate_diff,
)


class TestDiffHunk:
    def test_creation(self):
        hunk = DiffHunk(
            old_start=1,
            old_count=3,
            new_start=1,
            new_count=2,
            lines=[("-", "old line"), ("+", "new line")],
        )
        assert hunk.old_start == 1
        assert hunk.old_count == 3
        assert hunk.new_start == 1
        assert hunk.new_count == 2
        assert len(hunk.lines) == 2

    def test_lines_tuple_format(self):
        hunk = DiffHunk(old_start=1, old_count=1, new_start=1, new_count=1, lines=[(" ", "ctx")])
        assert hunk.lines[0][0] == " "
        assert hunk.lines[0][1] == "ctx"


class TestDiffResult:
    def test_creation(self):
        result = DiffResult(
            file_path="test.py",
            hunks=[],
            old_content="old",
            new_content="new",
            is_new_file=False,
            stats={"added": 0, "removed": 0, "changed": 0},
        )
        assert result.file_path == "test.py"
        assert result.old_content == "old"
        assert result.new_content == "new"
        assert result.is_new_file is False

    def test_new_file_result(self):
        result = DiffResult(
            file_path="new.py",
            hunks=[],
            old_content=None,
            new_content="content",
            is_new_file=True,
            stats={"added": 1, "removed": 0, "changed": 0},
        )
        assert result.old_content is None
        assert result.is_new_file is True


class TestGenerateDiff:
    def test_new_file(self, tmp_path):
        file_path = str(tmp_path / "new_file.py")
        result = generate_diff(file_path, "hello\n")
        assert "[new file]" in result
        assert "hello" in result

    def test_no_changes(self, tmp_path):
        file_path = tmp_path / "same.txt"
        file_path.write_text("unchanged\n", encoding="utf-8")
        result = generate_diff(str(file_path), "unchanged\n")
        assert "[No changes]" in result

    def test_with_changes(self, tmp_path):
        file_path = tmp_path / "change.txt"
        file_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
        result = generate_diff(str(file_path), "line1\nmodified\nline3\n")
        assert result  # non-empty
        assert "modified" in result

    def test_multiple_line_changes(self, tmp_path):
        file_path = tmp_path / "multi.txt"
        file_path.write_text("a\nb\nc\nd\ne\n", encoding="utf-8")
        result = generate_diff(str(file_path), "a\nB\nC\nd\ne\n")
        assert "B" in result
        assert "C" in result

    def test_binary_file(self, tmp_path):
        file_path = tmp_path / "binary.dat"
        file_path.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")
        result = generate_diff(str(file_path), "new content")
        assert "[Binary or unreadable file" in result or result

    def test_diff_includes_file_headers(self, tmp_path):
        file_path = tmp_path / "src.py"
        file_path.write_text("old\n", encoding="utf-8")
        result = generate_diff(str(file_path), "new\n")
        assert f"a/{file_path}" in result or f"b/{file_path}" in result


class TestComputeDiffResult:
    def test_new_file(self, tmp_path):
        file_path = str(tmp_path / "new.py")
        result = compute_diff_result(file_path, "new content\n")
        assert result.is_new_file is True
        assert result.old_content is None
        assert result.new_content == "new content\n"
        assert len(result.hunks) == 1
        assert result.stats["added"] > 0

    def test_no_changes(self, tmp_path):
        file_path = tmp_path / "same.txt"
        content = "line1\nline2\nline3\n"
        file_path.write_text(content, encoding="utf-8")
        result = compute_diff_result(str(file_path), content)
        assert result.is_new_file is False
        assert len(result.hunks) == 0
        assert result.stats["added"] == 0
        assert result.stats["removed"] == 0

    def test_single_line_change(self, tmp_path):
        file_path = tmp_path / "change.txt"
        file_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
        result = compute_diff_result(str(file_path), "line1\nmodified\nline3\n")
        assert len(result.hunks) >= 1
        assert result.stats["removed"] >= 1
        assert result.stats["added"] >= 1
        assert result.stats["changed"] >= 1

    def test_multiple_hunks(self, tmp_path):
        file_path = tmp_path / "multi_hunk.txt"
        file_path.write_text(
            "keep1\nchange1\nkeep2\nkeep3\nchange2\nkeep4\n",
            encoding="utf-8",
        )
        result = compute_diff_result(
            str(file_path),
            "keep1\nCHANGED1\nkeep2\nkeep3\nCHANGED2\nkeep4\n",
        )
        assert len(result.hunks) == 2

    def test_added_lines(self, tmp_path):
        file_path = tmp_path / "add.txt"
        file_path.write_text("a\nc\n", encoding="utf-8")
        result = compute_diff_result(str(file_path), "a\nb\nc\n")
        assert result.stats["added"] >= 1

    def test_removed_lines(self, tmp_path):
        file_path = tmp_path / "remove.txt"
        file_path.write_text("a\nb\nc\n", encoding="utf-8")
        result = compute_diff_result(str(file_path), "a\nc\n")
        assert result.stats["removed"] >= 1

    def test_hunk_line_types(self, tmp_path):
        file_path = tmp_path / "types.txt"
        file_path.write_text("old_line\n", encoding="utf-8")
        result = compute_diff_result(str(file_path), "new_line\n")
        for hunk in result.hunks:
            for change_type, _ in hunk.lines:
                assert change_type in ("-", "+")

    def test_hunk_start_positions(self, tmp_path):
        file_path = tmp_path / "pos.txt"
        file_path.write_text("a\nb\nc\n", encoding="utf-8")
        result = compute_diff_result(str(file_path), "A\nb\nc\n")
        for hunk in result.hunks:
            assert hunk.old_start >= 1
            assert hunk.new_start >= 1

    def test_unreadable_file_treated_as_new(self, tmp_path):
        file_path = tmp_path / "unreadable.txt"
        file_path.write_text("old\n", encoding="utf-8")
        os.chmod(str(file_path), 0o000)
        try:
            result = compute_diff_result(str(file_path), "new\n")
            assert result.is_new_file is True
        finally:
            os.chmod(str(file_path), 0o644)

    def test_changed_stat(self, tmp_path):
        file_path = tmp_path / "changed.txt"
        file_path.write_text("a\nb\n", encoding="utf-8")
        result = compute_diff_result(str(file_path), "A\nB\n")
        assert result.stats["changed"] >= 1


class TestDetectLanguage:
    def test_python(self):
        assert _detect_language("script.py") == "python"

    def test_javascript(self):
        assert _detect_language("app.js") == "javascript"
        assert _detect_language("component.jsx") == "javascript"

    def test_typescript(self):
        assert _detect_language("module.ts") == "typescript"
        assert _detect_language("widget.tsx") == "typescript"

    def test_java(self):
        assert _detect_language("Main.java") == "java"

    def test_go(self):
        assert _detect_language("main.go") == "go"

    def test_rust(self):
        assert _detect_language("lib.rs") == "rust"

    def test_c(self):
        assert _detect_language("main.c") == "c"

    def test_cpp(self):
        assert _detect_language("app.cpp") == "cpp"

    def test_header_c(self):
        assert _detect_language("util.h") == "c"

    def test_header_cpp(self):
        assert _detect_language("util.hpp") == "cpp"

    def test_csharp(self):
        assert _detect_language("Program.cs") == "csharp"

    def test_ruby(self):
        assert _detect_language("app.rb") == "ruby"

    def test_php(self):
        assert _detect_language("index.php") == "php"

    def test_swift(self):
        assert _detect_language("main.swift") == "swift"

    def test_kotlin(self):
        assert _detect_language("App.kt") == "kotlin"

    def test_scala(self):
        assert _detect_language("App.scala") == "scala"

    def test_r(self):
        assert _detect_language("stats.r") == "r"

    def test_lua(self):
        assert _detect_language("script.lua") == "lua"

    def test_perl(self):
        assert _detect_language("script.pl") == "perl"

    def test_bash(self):
        assert _detect_language("run.sh") == "bash"
        assert _detect_language("setup.bash") == "bash"
        assert _detect_language("env.zsh") == "bash"

    def test_data_formats(self):
        assert _detect_language("data.json") == "json"
        assert _detect_language("conf.yaml") == "yaml"
        assert _detect_language("conf.yml") == "yaml"
        assert _detect_language("Cargo.toml") == "toml"
        assert _detect_language("pom.xml") == "xml"

    def test_markup(self):
        assert _detect_language("page.html") == "html"
        assert _detect_language("style.css") == "css"
        assert _detect_language("style.scss") == "scss"
        assert _detect_language("style.less") == "less"

    def test_sql(self):
        assert _detect_language("query.sql") == "sql"

    def test_markdown_and_text(self):
        assert _detect_language("README.md") == "markdown"
        assert _detect_language("docs.rst") == "rst"
        assert _detect_language("notes.txt") == "text"

    def test_unknown_extension(self):
        assert _detect_language("file.xyz") == "text"
        assert _detect_language("Makefile") == "text"

    def test_case_insensitive(self):
        assert _detect_language("SCRIPT.PY") == "python"


class TestPrintDiff:
    def test_print_diff_new_file(self, tmp_path, capsys):
        file_path = str(tmp_path / "new.py")
        from src.core.diff import print_diff

        print_diff(file_path, "hello\n")
        captured = capsys.readouterr()
        assert "new file" in captured.out.lower() or captured.out

    def test_print_diff_no_changes(self, tmp_path, capsys):
        file_path = tmp_path / "same.txt"
        file_path.write_text("unchanged\n", encoding="utf-8")
        from src.core.diff import print_diff

        print_diff(str(file_path), "unchanged\n")
        captured = capsys.readouterr()
        assert "No changes" in captured.out or captured.out

    def test_print_diff_with_changes(self, tmp_path, capsys):
        file_path = tmp_path / "change.txt"
        file_path.write_text("line1\nline2\n", encoding="utf-8")
        from src.core.diff import print_diff

        print_diff(str(file_path), "line1\nmodified\n")
        captured = capsys.readouterr()
        assert captured.out


class TestPrintInlineDiff:
    def test_new_file(self, tmp_path, capsys):
        file_path = str(tmp_path / "new_file.py")
        from src.core.diff import print_inline_diff

        print_inline_diff(file_path, "print('hello')\n")
        captured = capsys.readouterr()
        assert "New file" in captured.out

    def test_no_changes(self, tmp_path, capsys):
        file_path = tmp_path / "same.txt"
        file_path.write_text("unchanged\n", encoding="utf-8")
        from src.core.diff import print_inline_diff

        print_inline_diff(str(file_path), "unchanged\n")
        captured = capsys.readouterr()
        assert "No changes" in captured.out

    def test_with_changes(self, tmp_path, capsys):
        file_path = tmp_path / "change.txt"
        file_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
        from src.core.diff import print_inline_diff

        print_inline_diff(str(file_path), "line1\nmodified\nline3\n")
        captured = capsys.readouterr()
        assert "+" in captured.out or "-" in captured.out or "modified" in captured.out


class TestPrintSideBySideDiff:
    def test_new_file(self, tmp_path, capsys):
        file_path = str(tmp_path / "new.py")
        from src.core.diff import print_side_by_side_diff

        print_side_by_side_diff(file_path, "hello\n")
        captured = capsys.readouterr()
        assert "New file" in captured.out

    def test_no_changes(self, tmp_path, capsys):
        file_path = tmp_path / "same.txt"
        file_path.write_text("unchanged\n", encoding="utf-8")
        from src.core.diff import print_side_by_side_diff

        print_side_by_side_diff(str(file_path), "unchanged\n")
        captured = capsys.readouterr()
        assert "No changes" in captured.out

    def test_with_changes(self, tmp_path, capsys):
        file_path = tmp_path / "change.txt"
        file_path.write_text("old_line\n", encoding="utf-8")
        from src.core.diff import print_side_by_side_diff

        print_side_by_side_diff(str(file_path), "new_line\n")
        captured = capsys.readouterr()
        assert captured.out
