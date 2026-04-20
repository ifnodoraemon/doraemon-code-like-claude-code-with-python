"""Tests for servers._services.code_nav — find_definition."""

import os

import pytest

from src.servers._services.code_nav import _check_javascript, _check_python, find_definition


class TestCheckPython:
    def test_find_function(self, tmp_path):
        code = "def hello():\n    pass\n"
        path = str(tmp_path / "test.py")
        with open(path, "w") as f:
            f.write(code)
        result = _check_python(path, code, "hello")
        assert len(result) == 1
        assert "[def]" in result[0]
        assert "test.py:1" in result[0]

    def test_find_class(self, tmp_path):
        code = "class MyClass:\n    pass\n"
        path = str(tmp_path / "test.py")
        with open(path, "w") as f:
            f.write(code)
        result = _check_python(path, code, "MyClass")
        assert len(result) == 1
        assert "[class]" in result[0]

    def test_find_async_function(self, tmp_path):
        code = "async def fetch():\n    pass\n"
        path = str(tmp_path / "test.py")
        with open(path, "w") as f:
            f.write(code)
        result = _check_python(path, code, "fetch")
        assert len(result) == 1
        assert "[def]" in result[0]

    def test_not_found(self, tmp_path):
        code = "def hello():\n    pass\n"
        path = str(tmp_path / "test.py")
        with open(path, "w") as f:
            f.write(code)
        result = _check_python(path, code, "nonexistent")
        assert result == []

    def test_syntax_error(self, tmp_path):
        code = "def broken(\n"
        path = str(tmp_path / "test.py")
        with open(path, "w") as f:
            f.write(code)
        result = _check_python(path, code, "broken")
        assert result == []

    def test_multiple_definitions(self, tmp_path):
        code = "class Foo:\n    pass\n\ndef foo():\n    pass\n"
        path = str(tmp_path / "test.py")
        with open(path, "w") as f:
            f.write(code)
        result = _check_python(path, code, "Foo")
        assert len(result) == 1
        assert "[class]" in result[0]


class TestCheckJavascript:
    def test_find_class(self, tmp_path):
        code = "class MyClass {\n  constructor() {}\n}\n"
        result = _check_javascript("test.js", code, "MyClass")
        assert len(result) == 1
        assert "[class]" in result[0]

    def test_find_function(self, tmp_path):
        code = "function handleClick() {\n  return true;\n}\n"
        result = _check_javascript("test.js", code, "handleClick")
        assert len(result) == 1
        assert "[function]" in result[0]

    def test_find_const_variable(self, tmp_path):
        code = "const API_URL = 'http://example.com';\n"
        result = _check_javascript("test.js", code, "API_URL")
        assert len(result) == 1
        assert "[variable]" in result[0]

    def test_not_found(self):
        code = "function foo() {}\n"
        result = _check_javascript("test.js", code, "bar")
        assert result == []

    def test_multiple_matches(self):
        code = "class Foo {}\nfunction Foo() {}\nconst Foo = 1;\n"
        result = _check_javascript("test.js", code, "Foo")
        assert len(result) == 3


class TestFindDefinition:
    def test_find_python_symbol(self, tmp_path):
        py_file = tmp_path / "mod.py"
        py_file.write_text("def target():\n    pass\n")
        result = find_definition(str(tmp_path), "target")
        assert "[def]" in result
        assert "mod.py" in result

    def test_find_javascript_symbol(self, tmp_path):
        js_file = tmp_path / "mod.js"
        js_file.write_text("function target() { return 1; }\n")
        result = find_definition(str(tmp_path), "target")
        assert "[function]" in result
        assert "mod.js" in result

    def test_not_found(self, tmp_path):
        py_file = tmp_path / "mod.py"
        py_file.write_text("def other():\n    pass\n")
        result = find_definition(str(tmp_path), "nonexistent")
        assert "No definition found" in result

    def test_skips_excluded_dirs(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        (git_dir / "hooks.py").write_text("def target():\n    pass\n")
        result = find_definition(str(tmp_path), "target")
        assert "No definition found" in result

    def test_skips_non_matching_files(self, tmp_path):
        (tmp_path / "other.py").write_text("def foo():\n    pass\n")
        result = find_definition(str(tmp_path), "bar")
        assert "No definition found" in result

    def test_nonexistent_root(self, tmp_path):
        result = find_definition(str(tmp_path / "nonexistent"), "something")
        assert "No definition found" in result
