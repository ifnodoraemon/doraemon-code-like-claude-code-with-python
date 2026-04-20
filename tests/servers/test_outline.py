"""Tests for servers._services.outline — file structure outline parsing."""

import pytest

from src.servers._services.outline import _parse_javascript, _parse_python, parse_outline


class TestParsePython:
    def test_class_with_methods(self):
        code = 'class Foo:\n    """Doc."""\n    def bar(self, x):\n        pass\n'
        result = _parse_python(code)
        assert "class Foo:" in result
        assert "def bar(self)" in result or "def bar(self, x)" in result

    def test_top_level_function(self):
        code = "def hello(name):\n    pass\n"
        result = _parse_python(code)
        assert "def hello(name)" in result

    def test_class_with_bases(self):
        code = "class Foo(Bar, Baz):\n    pass\n"
        result = _parse_python(code)
        assert "class Foo(Bar, Baz):" in result

    def test_no_definitions(self):
        code = "x = 1\ny = 2\n"
        result = _parse_python(code)
        assert "No top-level" in result

    def test_syntax_error(self):
        code = "def broken(\n"
        result = _parse_python(code)
        assert "SyntaxError" in result

    def test_async_function_not_parsed(self):
        code = "async def fetch(url):\n    pass\n"
        result = _parse_python(code)
        assert "No top-level" in result

    def test_class_with_docstring(self):
        code = 'class Foo:\n    """A class."""\n    pass\n'
        result = _parse_python(code)
        assert "A class." in result

    def test_varargs(self):
        code = "def fn(a, *args, **kwargs):\n    pass\n"
        result = _parse_python(code)
        assert "*args" in result
        assert "**kwargs" in result


class TestParseJavascript:
    def test_class(self):
        code = "export class MyComponent {\n  render() {}\n}\n"
        result = _parse_javascript(code)
        assert "MyComponent" in result

    def test_function(self):
        code = "function handleClick() {\n  return true;\n}\n"
        result = _parse_javascript(code)
        assert "handleClick" in result

    def test_const_arrow(self):
        code = "const fetcher = () => {\n  return fetch(url);\n}\n"
        result = _parse_javascript(code)
        assert "fetcher" in result

    def test_export_async_not_matched(self):
        code = "export async function getData() {\n  return await fetch(url);\n}\n"
        result = _parse_javascript(code)
        assert "No obvious" in result

    def test_no_definitions(self):
        code = "const x = 1;\n"
        result = _parse_javascript(code)
        assert "No obvious" in result


class TestParseOutline:
    def test_python_file(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("class Foo:\n    def bar(self):\n        pass\n")
        result = parse_outline(str(f))
        assert "class Foo:" in result

    def test_javascript_file(self, tmp_path):
        f = tmp_path / "test.js"
        f.write_text("class App {\n  render() {}\n}\n")
        result = parse_outline(str(f))
        assert "App" in result

    def test_unknown_file_fallback(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("line1\nline2\n")
        result = parse_outline(str(f))
        assert "First 20 lines" in result
        assert "line1" in result

    def test_nonexistent_file(self, tmp_path):
        result = parse_outline(str(tmp_path / "nope.py"))
        assert "Error" in result

    def test_file_with_many_lines(self, tmp_path):
        f = tmp_path / "big.txt"
        lines = [f"line {i}" for i in range(30)]
        f.write_text("\n".join(lines))
        result = parse_outline(str(f))
        assert "..." in result
