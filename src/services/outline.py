import ast
import os
import re


def parse_outline(file_path: str) -> str:
    """
    Parse a file and return its structure outline.
    Supports: Python (AST), JS/TS (Regex), others (Preview).
    """
    ext = os.path.splitext(file_path)[1].lower()

    try:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        if ext == ".py":
            return _parse_python(content)
        elif ext in [".js", ".ts", ".jsx", ".tsx"]:
            return _parse_javascript(content)
        else:
            # Fallback: Just return the first 20 lines as a "Head" outline
            lines = content.splitlines()[:20]
            return (
                "[Non-code file outline (First 20 lines)]\n"
                + "\n".join(lines)
                + ("\n..." if len(content.splitlines()) > 20 else "")
            )

    except Exception as e:
        return f"Error parsing outline: {e}"


def _parse_python(content: str) -> str:
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return "[SyntaxError: Could not parse Python file]"

    lines = []

    def get_args(args):
        arg_list = [a.arg for a in args.args]
        if args.vararg:
            arg_list.append(f"*{args.vararg.arg}")
        if args.kwarg:
            arg_list.append(f"**{args.kwarg.arg}")
        return ", ".join(arg_list)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            base_str = f"({', '.join(bases)})" if bases else ""
            lines.append(f"class {node.name}{base_str}:")

            if ast.get_docstring(node):
                doc = ast.get_docstring(node).splitlines()[0]
                lines.append(f"    # {doc}")

            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    args = get_args(item.args)
                    lines.append(f"    def {item.name}({args}):")

        elif isinstance(node, ast.FunctionDef):
            args = get_args(node.args)
            lines.append(f"def {node.name}({args}):")
            if ast.get_docstring(node):
                doc = ast.get_docstring(node).splitlines()[0]
                lines.append(f"    # {doc}")

    if not lines:
        return "[No top-level classes or functions found]"

    return "\n".join(lines)


def _parse_javascript(content: str) -> str:
    """Simple regex based parser for JS/TS"""
    lines = []

    # Matches: function name(), class Name, const name = () or async
    patterns = [
        r"^\s*(export\s+)?class\s+(\w+)",
        r"^\s*(export\s+)?function\s+(\w+)",
        r"^\s*(export\s+)?const\s+(\w+)\s*=\s*(\(|async)",
    ]

    for line in content.splitlines():
        for p in patterns:
            if re.search(p, line):
                lines.append(line.strip().rstrip("{"))

    if not lines:
        return "[No obvious class/function definitions found]"

    return "\n".join(lines)
