"""
Codebase Map Generator

Generates a high-level map of the codebase structure to help AI models
understand the overall architecture. Similar to Aider's RepoMap.
"""

import ast
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SymbolInfo:
    """Information about a symbol (class, function, etc.)."""

    name: str
    kind: str  # 'class', 'function', 'async_function', 'variable'
    line_start: int
    line_end: int
    parent: str | None = None
    signature: str | None = None
    docstring: str | None = None


@dataclass
class FileInfo:
    """Information about a file."""

    path: str
    language: str
    symbols: list[SymbolInfo] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    exports: list[str] = field(default_factory=list)


@dataclass
class CodebaseMap:
    """A map of the codebase structure."""

    root_path: str
    files: list[FileInfo] = field(default_factory=list)
    total_symbols: int = 0
    total_lines: int = 0

    def to_prompt(self, max_tokens: int = 8000) -> str:
        """
        Generate a prompt-friendly representation of the codebase map.

        This is optimized for token efficiency while preserving structure.
        """
        lines = [f"# Codebase: {self.root_path}\n"]
        lines.append(f"# Files: {len(self.files)} | Symbols: {self.total_symbols}\n\n")

        # Group files by directory
        by_dir: dict[str, list[FileInfo]] = defaultdict(list)
        for f in self.files:
            dir_name = str(Path(f.path).parent)
            by_dir[dir_name].append(f)

        for dir_name in sorted(by_dir.keys()):
            files = by_dir[dir_name]
            if dir_name == ".":
                lines.append("## /\n")
            else:
                lines.append(f"## {dir_name}/\n")

            for file_info in files:
                file_name = Path(file_info.path).name
                lines.append(f"### {file_name}\n")

                # Show imports (limited)
                if file_info.imports:
                    imp_str = ", ".join(file_info.imports[:10])
                    if len(file_info.imports) > 10:
                        imp_str += f" (+{len(file_info.imports) - 10})"
                    lines.append(f"  imports: {imp_str}\n")

                # Show symbols
                for sym in file_info.symbols[:20]:  # Limit symbols per file
                    prefix = f"  {sym.parent}." if sym.parent else "  "
                    kind_icon = {"class": "🏗️", "function": "⚡", "async_function": "⏳"}.get(
                        sym.kind, "📌"
                    )
                    sig = f" {sym.signature}" if sym.signature else ""
                    lines.append(f"{prefix}{kind_icon} {sym.name}{sig}\n")

                if len(file_info.symbols) > 20:
                    lines.append(f"  ... (+{len(file_info.symbols) - 20} more)\n")

                lines.append("\n")

        result = "".join(lines)

        # Truncate if too long
        if len(result) > max_tokens * 4:  # Rough char-to-token ratio
            result = result[: max_tokens * 4] + "\n... (truncated)"

        return result


class CodebaseMapper:
    """Generate maps of codebase structure."""

    # Files to include
    INCLUDE_EXTENSIONS = {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".go",
        ".rs",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".scala",
        ".lua",
        ".sh",
    }

    # Directories to skip
    SKIP_DIRS = {
        "__pycache__",
        "node_modules",
        ".git",
        ".venv",
        "venv",
        "env",
        ".env",
        "dist",
        "build",
        "target",
        "vendor",
        ".idea",
        ".vscode",
        "coverage",
        ".pytest_cache",
        "migrations",
        "docs",
        "tests",
        "test",
        "spec",
    }

    # Skip patterns
    SKIP_PATTERNS = {
        "*.test.*",
        "*.spec.*",
        "*_test.*",
        "*_spec.*",
        "*.min.js",
        "*.min.css",
        "*.generated.*",
    }

    def __init__(self, root_path: str):
        self.root_path = Path(root_path).resolve()

    def generate_map(
        self,
        max_files: int = 100,
        include_tests: bool = False,
        include_private: bool = False,
    ) -> CodebaseMap:
        """
        Generate a codebase map.

        Args:
            max_files: Maximum number of files to analyze
            include_tests: Whether to include test files
            include_private: Whether to include private symbols

        Returns:
            CodebaseMap with structure information
        """
        codebase_map = CodebaseMap(root_path=str(self.root_path))
        file_count = 0

        for root, dirs, files in os.walk(self.root_path):
            # Filter directories
            dirs[:] = [d for d in dirs if d not in self.SKIP_DIRS and not d.startswith(".")]

            for file in files:
                if file_count >= max_files:
                    break

                ext = os.path.splitext(file)[1].lower()
                if ext not in self.INCLUDE_EXTENSIONS:
                    continue

                # Skip test files
                if not include_tests and self._is_test_file(file):
                    continue

                # Skip patterns
                if any(self._matches_pattern(file, p) for p in self.SKIP_PATTERNS):
                    continue

                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, self.root_path)

                try:
                    file_info = self._analyze_file(file_path, rel_path, include_private)
                    if file_info:
                        codebase_map.files.append(file_info)
                        codebase_map.total_symbols += len(file_info.symbols)
                        file_count += 1
                except Exception as e:
                    logger.debug("Failed to analyze %s: %s", file_path, e)

            if file_count >= max_files:
                break

        # Sort files by path
        codebase_map.files.sort(key=lambda f: f.path)

        return codebase_map

    def _is_test_file(self, filename: str) -> bool:
        """Check if a file is a test file."""
        lower = filename.lower()
        return (
            "test" in lower
            or "spec" in lower
            or lower.startswith("test_")
            or lower.endswith("_test.py")
            or lower.endswith(".test.js")
            or lower.endswith(".spec.ts")
        )

    def _matches_pattern(self, filename: str, pattern: str) -> bool:
        """Check if filename matches a pattern."""
        import fnmatch

        return fnmatch.fnmatch(filename, pattern)

    def _analyze_file(
        self, file_path: str, rel_path: str, include_private: bool
    ) -> FileInfo | None:
        """Analyze a single file."""
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".py":
            return self._analyze_python(file_path, rel_path, include_private)
        elif ext in {".js", ".ts", ".tsx", ".jsx"}:
            return self._analyze_javascript(file_path, rel_path)
        else:
            # Generic analysis
            return self._analyze_generic(file_path, rel_path)

    def _analyze_python(
        self, file_path: str, rel_path: str, include_private: bool
    ) -> FileInfo | None:
        """Analyze a Python file."""
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                source = f.read()
            tree = ast.parse(source, filename=file_path)
        except Exception:
            return None

        file_info = FileInfo(path=rel_path, language="python")

        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    file_info.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    file_info.imports.append(f"{module}.{alias.name}" if module else alias.name)

        # Extract symbols
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                if not include_private and node.name.startswith("_"):
                    continue
                sym = SymbolInfo(
                    name=node.name,
                    kind="class",
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                )
                # Get docstring
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                ):
                    sym.docstring = (
                        node.body[0].value.value[:100]
                        if isinstance(node.body[0].value.value, str)
                        else None
                    )
                file_info.symbols.append(sym)

                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if not include_private and item.name.startswith("_"):
                            continue
                        method_sym = SymbolInfo(
                            name=item.name,
                            kind="function",
                            line_start=item.lineno,
                            line_end=item.end_lineno or item.lineno,
                            parent=node.name,
                            signature=self._get_function_signature(item),
                        )
                        file_info.symbols.append(method_sym)
                    elif isinstance(item, ast.AsyncFunctionDef):
                        if not include_private and item.name.startswith("_"):
                            continue
                        method_sym = SymbolInfo(
                            name=item.name,
                            kind="async_function",
                            line_start=item.lineno,
                            line_end=item.end_lineno or item.lineno,
                            parent=node.name,
                            signature=self._get_function_signature(item),
                        )
                        file_info.symbols.append(method_sym)

            elif isinstance(node, ast.FunctionDef):
                if not include_private and node.name.startswith("_"):
                    continue
                sym = SymbolInfo(
                    name=node.name,
                    kind="function",
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    signature=self._get_function_signature(node),
                )
                file_info.symbols.append(sym)

            elif isinstance(node, ast.AsyncFunctionDef):
                if not include_private and node.name.startswith("_"):
                    continue
                sym = SymbolInfo(
                    name=node.name,
                    kind="async_function",
                    line_start=node.lineno,
                    line_end=node.end_lineno or node.lineno,
                    signature=self._get_function_signature(node),
                )
                file_info.symbols.append(sym)

        return file_info

    def _get_function_signature(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
        """Extract function signature as string."""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        sig = f"({', '.join(args)})"
        if node.returns:
            sig += f" -> {ast.unparse(node.returns)}"
        return sig

    def _analyze_javascript(self, file_path: str, rel_path: str) -> FileInfo | None:
        """Analyze a JavaScript/TypeScript file."""
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            return None

        file_info = FileInfo(
            path=rel_path, language="typescript" if "ts" in file_path else "javascript"
        )

        # Simple regex-based extraction (not as accurate as AST but works)
        import re

        # Extract imports
        import_pattern = r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]'
        for match in re.finditer(import_pattern, content):
            file_info.imports.append(match.group(1))

        # Extract exports
        export_pattern = r"export\s+(?:default\s+)?(?:function|class|const|let|var)\s+(\w+)"
        for match in re.finditer(export_pattern, content):
            file_info.exports.append(match.group(1))

        # Extract functions
        func_pattern = r"(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)"
        for match in re.finditer(func_pattern, content):
            sym = SymbolInfo(
                name=match.group(1),
                kind="function",
                line_start=content[: match.start()].count("\n") + 1,
                line_end=content[: match.end()].count("\n") + 1,
                signature=f"({match.group(2)})",
            )
            file_info.symbols.append(sym)

        # Extract arrow functions
        arrow_pattern = r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:\([^)]*\)|[^=])\s*=>"
        for match in re.finditer(arrow_pattern, content):
            sym = SymbolInfo(
                name=match.group(1),
                kind="function",
                line_start=content[: match.start()].count("\n") + 1,
                line_end=content[: match.end()].count("\n") + 1,
            )
            file_info.symbols.append(sym)

        # Extract classes
        class_pattern = r"class\s+(\w+)(?:\s+extends\s+(\w+))?"
        for match in re.finditer(class_pattern, content):
            sym = SymbolInfo(
                name=match.group(1),
                kind="class",
                line_start=content[: match.start()].count("\n") + 1,
                line_end=content[: match.end()].count("\n") + 1,
            )
            file_info.symbols.append(sym)

        return file_info

    def _analyze_generic(self, file_path: str, rel_path: str) -> FileInfo | None:
        """Generic file analysis."""
        try:
            with open(file_path, encoding="utf-8", errors="replace") as f:
                f.read()
        except Exception:
            return None

        ext = os.path.splitext(file_path)[1].lower()
        lang_map = {
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".c": "c",
            ".cpp": "cpp",
            ".rb": "ruby",
            ".php": "php",
        }

        return FileInfo(
            path=rel_path,
            language=lang_map.get(ext, "text"),
        )


def generate_codebase_map(
    root_path: str = ".",
    max_files: int = 100,
    include_tests: bool = False,
    include_private: bool = False,
) -> str:
    """
    Generate a codebase map as a string.

    Args:
        root_path: Root directory of the codebase
        max_files: Maximum number of files to analyze
        include_tests: Whether to include test files
        include_private: Whether to include private symbols

    Returns:
        String representation of the codebase map
    """
    mapper = CodebaseMapper(root_path)
    codebase_map = mapper.generate_map(
        max_files=max_files,
        include_tests=include_tests,
        include_private=include_private,
    )
    return codebase_map.to_prompt()
