import ast
import os
import re


def find_definition(root_path: str, symbol: str) -> str:
    """
    Find the definition of a symbol (class or function) in the codebase.
    Returns a list of locations "file:line: type"
    """
    matches = []

    # Walk through the directory
    for root, dirs, files in os.walk(root_path):
        # Exclude common noise
        dirs[:] = [
            d for d in dirs if d not in {".git", "__pycache__", "node_modules", "venv", "env"}
        ]

        for file in files:
            ext = os.path.splitext(file)[1].lower()
            full_path = os.path.join(root, file)

            # Optimization: Quick check if symbol exists in file content before parsing
            try:
                with open(full_path, encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except OSError:
                continue

            if symbol not in content:
                continue

            # If symbol exists, do deep check
            if ext == ".py":
                result = _check_python(full_path, content, symbol)
                if result:
                    matches.extend(result)
            elif ext in [".js", ".ts", ".jsx", ".tsx"]:
                result = _check_javascript(full_path, content, symbol)
                if result:
                    matches.extend(result)

    if not matches:
        return f"No definition found for symbol '{symbol}'."

    return "\n".join(matches)


def _check_python(path: str, content: str, symbol: str) -> list[str]:
    results = []
    try:
        tree = ast.parse(content)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                if node.name == symbol:
                    kind = "class" if isinstance(node, ast.ClassDef) else "def"
                    results.append(f"{path}:{node.lineno} [{kind}]")
    except SyntaxError:
        pass
    return results


def _check_javascript(path: str, content: str, symbol: str) -> list[str]:
    results = []
    lines = content.splitlines()

    # Regex for definitions
    # class Symbol, function Symbol, const Symbol = ..., let Symbol = ...
    patterns = [
        (r"class\s+" + re.escape(symbol) + r"\b", "class"),
        (r"function\s+" + re.escape(symbol) + r"\b", "function"),
        (r"(const|let|var)\s+" + re.escape(symbol) + r"\s*=", "variable"),
    ]

    for i, line in enumerate(lines, 1):
        for pat, kind in patterns:
            if re.search(pat, line):
                results.append(f"{path}:{i} [{kind}]")

    return results
