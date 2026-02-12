"""
Language Server Protocol (LSP) MCP Server

Provides IDE-level code intelligence for Doraemon:
- Code diagnostics (syntax errors, linting)
- Code completions
- Hover information
- Symbol references
- Rename refactoring

This is a simplified LSP client that communicates with language servers
like pylsp (Python), typescript-language-server (TypeScript), etc.
"""

import asyncio
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

from src.core.security import validate_path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("DoraemonLSP")


@dataclass
class Position:
    """Line and column position in a file."""
    line: int  # 0-indexed
    character: int  # 0-indexed


@dataclass
class Range:
    """Range in a file."""
    start: Position
    end: Position


@dataclass
class Diagnostic:
    """A diagnostic message (error, warning, etc.)."""
    message: str
    severity: str  # "error", "warning", "info", "hint"
    range: Range
    source: str  # e.g., "pylsp", "ruff", "mypy"


@dataclass
class CompletionItem:
    """A code completion suggestion."""
    label: str
    kind: str  # "function", "class", "variable", etc.
    detail: str | None = None
    documentation: str | None = None


class PylspClient:
    """
    Simple LSP client for Python Language Server (pylsp).

    Uses subprocess and stdio to communicate with pylsp.
    """

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._initialized = False

    async def start(self) -> bool:
        """Start the pylsp server."""
        try:
            self._process = subprocess.Popen(
                ["pylsp"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.info("pylsp server started")

            # Send initialize request
            await self._initialize()
            return True
        except FileNotFoundError:
            logger.warning("pylsp not found. Install with: pip install python-lsp-server")
            return False
        except Exception as e:
            logger.error(f"Failed to start pylsp: {e}")
            return False

    async def _initialize(self):
        """Send LSP initialize request."""
        init_params = {
            "processId": os.getpid(),
            "rootUri": f"file://{os.getcwd()}",
            "capabilities": {
                "textDocument": {
                    "completion": {"completionItem": {"snippetSupport": False}},
                    "hover": {"contentFormat": ["plaintext"]},
                    "publishDiagnostics": {"relatedInformation": True},
                }
            },
        }

        response = await self._send_request("initialize", init_params)
        if response:
            await self._send_notification("initialized", {})
            self._initialized = True
            logger.info("pylsp initialized")

    async def _send_request(self, method: str, params: dict) -> dict | None:
        """Send an LSP request and wait for response."""
        if not self._process or not self._process.stdin or not self._process.stdout:
            return None

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params,
        }

        content = json.dumps(request)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"

        try:
            self._process.stdin.write(message.encode())
            self._process.stdin.flush()

            # Read response
            headers = {}
            while True:
                line = self._process.stdout.readline().decode()
                if line == "\r\n":
                    break
                if ":" in line:
                    key, value = line.strip().split(":", 1)
                    headers[key.strip()] = value.strip()

            content_length = int(headers.get("Content-Length", 0))
            if content_length:
                content = self._process.stdout.read(content_length).decode()
                return json.loads(content)

        except Exception as e:
            logger.error(f"LSP request failed: {e}")

        return None

    async def _send_notification(self, method: str, params: dict):
        """Send an LSP notification (no response expected)."""
        if not self._process or not self._process.stdin:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }

        content = json.dumps(notification)
        message = f"Content-Length: {len(content)}\r\n\r\n{content}"

        try:
            self._process.stdin.write(message.encode())
            self._process.stdin.flush()
        except Exception as e:
            logger.error(f"LSP notification failed: {e}")

    async def get_diagnostics(self, file_path: str) -> list[Diagnostic]:
        """Get diagnostics for a file."""
        if not self._initialized:
            return []

        uri = f"file://{os.path.abspath(file_path)}"

        # Read file content in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()

        def _read_file():
            with open(file_path, encoding="utf-8", errors="replace") as f:
                return f.read()

        content = await loop.run_in_executor(None, _read_file)

        await self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": "python",
                "version": 1,
                "text": content,
            }
        })

        # Wait briefly for diagnostics
        await asyncio.sleep(0.5)

        # Note: In a full implementation, we'd listen for publishDiagnostics
        # For now, we use external tools like ruff for diagnostics
        return []

    async def get_completions(
        self, file_path: str, line: int, character: int
    ) -> list[CompletionItem]:
        """Get code completions at a position."""
        if not self._initialized:
            return []

        uri = f"file://{os.path.abspath(file_path)}"

        # Open the document first
        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        await self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": "python",
                "version": 1,
                "text": content,
            }
        })

        response = await self._send_request("textDocument/completion", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        })

        if not response or "result" not in response:
            return []

        result = response["result"]
        items = result.get("items", []) if isinstance(result, dict) else result

        completions = []
        for item in items[:20]:  # Limit results
            kind_map = {
                1: "text", 2: "method", 3: "function", 4: "constructor",
                5: "field", 6: "variable", 7: "class", 8: "interface",
                9: "module", 10: "property",
            }
            completions.append(CompletionItem(
                label=item.get("label", ""),
                kind=kind_map.get(item.get("kind", 1), "text"),
                detail=item.get("detail"),
                documentation=item.get("documentation"),
            ))

        return completions

    async def get_hover(self, file_path: str, line: int, character: int) -> str | None:
        """Get hover information at a position."""
        if not self._initialized:
            return None

        uri = f"file://{os.path.abspath(file_path)}"

        # Open the document first
        with open(file_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        await self._send_notification("textDocument/didOpen", {
            "textDocument": {
                "uri": uri,
                "languageId": "python",
                "version": 1,
                "text": content,
            }
        })

        response = await self._send_request("textDocument/hover", {
            "textDocument": {"uri": uri},
            "position": {"line": line, "character": character},
        })

        if not response or "result" not in response:
            return None

        result = response.get("result")
        if not result:
            return None

        contents = result.get("contents", "")
        if isinstance(contents, dict):
            return contents.get("value", str(contents))
        elif isinstance(contents, list):
            return "\n".join(
                c.get("value", str(c)) if isinstance(c, dict) else str(c)
                for c in contents
            )
        return str(contents)

    def stop(self):
        """Stop the pylsp server."""
        if self._process:
            self._process.terminate()
            self._process = None
            self._initialized = False


# Global LSP client (lazy initialization)
_lsp_client: PylspClient | None = None


async def _get_lsp_client() -> PylspClient | None:
    """Get or create the LSP client."""
    global _lsp_client
    if _lsp_client is None:
        _lsp_client = PylspClient()
        if not await _lsp_client.start():
            _lsp_client = None
    return _lsp_client


def _run_ruff(file_path: str) -> list[dict]:
    """Run ruff for quick Python diagnostics."""
    try:
        result = subprocess.run(
            ["ruff", "check", "--output-format=json", file_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.stdout:
            return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return []


def _run_mypy(file_path: str) -> list[dict]:
    """Run mypy for type checking."""
    try:
        result = subprocess.run(
            ["mypy", "--show-error-codes", "--no-error-summary", file_path],
            capture_output=True,
            text=True,
            timeout=60,
        )
        diagnostics = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            # Parse mypy output: file:line: severity: message
            match = re.match(r"(.+):(\d+): (error|warning|note): (.+)", line)
            if match:
                diagnostics.append({
                    "file": match.group(1),
                    "line": int(match.group(2)),
                    "severity": match.group(3),
                    "message": match.group(4),
                })
        return diagnostics
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return []


# ========================================
# Internal Implementation Functions
# ========================================


async def _lsp_diagnostics_impl(path: str, include_mypy: bool = False) -> str:
    """
    Get code diagnostics (errors, warnings) for a file.

    Uses ruff for fast linting and optionally mypy for type checking.
    This is useful for finding issues before running code.

    Args:
        path: Path to the file to analyze
        include_mypy: Include mypy type checking (slower but more thorough)

    Returns:
        Formatted list of diagnostics with line numbers and messages
    """
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    if not os.path.exists(valid_path):
        return f"Error: File not found: {path}"

    ext = os.path.splitext(valid_path)[1].lower()
    if ext not in [".py"]:
        return f"LSP diagnostics not supported for {ext} files (Python only for now)"

    results = []

    # Run ruff (fast)
    ruff_results = _run_ruff(valid_path)
    for diag in ruff_results:
        line = diag.get("location", {}).get("row", 0)
        code = diag.get("code", "")
        message = diag.get("message", "")
        results.append(f"L{line} [{code}]: {message}")

    # Optionally run mypy (slower)
    if include_mypy:
        mypy_results = _run_mypy(valid_path)
        for diag in mypy_results:
            line = diag.get("line", 0)
            severity = diag.get("severity", "error")
            message = diag.get("message", "")
            results.append(f"L{line} [mypy:{severity}]: {message}")

    if not results:
        return f"No issues found in {path}"

    return f"Diagnostics for {path}:\n\n" + "\n".join(results)


async def _lsp_completions_impl(path: str, line: int, column: int) -> str:
    """
    Get code completion suggestions at a specific position.

    Useful for understanding what methods/properties are available.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        column: Column number (1-indexed)

    Returns:
        List of completion suggestions with their types
    """
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    if not os.path.exists(valid_path):
        return f"Error: File not found: {path}"

    ext = os.path.splitext(valid_path)[1].lower()
    if ext not in [".py"]:
        return f"LSP completions not supported for {ext} files"

    client = await _get_lsp_client()
    if not client:
        return "LSP server not available. Install pylsp: pip install python-lsp-server"

    # Convert to 0-indexed
    completions = await client.get_completions(valid_path, line - 1, column - 1)

    if not completions:
        return f"No completions available at {path}:{line}:{column}"

    result = f"Completions at {path}:{line}:{column}:\n\n"
    for c in completions[:15]:
        detail = f" - {c.detail}" if c.detail else ""
        result += f"  [{c.kind}] {c.label}{detail}\n"

    if len(completions) > 15:
        result += f"\n  ... and {len(completions) - 15} more"

    return result


async def _lsp_hover_impl(path: str, line: int, column: int) -> str:
    """
    Get hover information (documentation, type info) at a position.

    Useful for understanding what a symbol means without leaving context.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        column: Column number (1-indexed)

    Returns:
        Documentation or type information for the symbol
    """
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    if not os.path.exists(valid_path):
        return f"Error: File not found: {path}"

    ext = os.path.splitext(valid_path)[1].lower()
    if ext not in [".py"]:
        return f"LSP hover not supported for {ext} files"

    client = await _get_lsp_client()
    if not client:
        return "LSP server not available. Install pylsp: pip install python-lsp-server"

    hover_info = await client.get_hover(valid_path, line - 1, column - 1)

    if not hover_info:
        return f"No information available at {path}:{line}:{column}"

    return f"Info at {path}:{line}:{column}:\n\n{hover_info}"


async def _lsp_references_impl(path: str, symbol: str) -> str:
    """
    Find all references to a symbol in the codebase.

    Uses simple grep-based search for now. Full LSP implementation
    would use textDocument/references.

    Args:
        path: Starting directory to search
        symbol: The symbol name to find references for

    Returns:
        List of files and lines that reference the symbol
    """
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    # Use grep to find references
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", symbol, valid_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        lines = result.stdout.strip().split("\n")
        if not lines or lines == [""]:
            return f"No references found for '{symbol}'"

        # Limit results
        if len(lines) > 50:
            lines = lines[:50]
            lines.append("... (showing first 50 of many)")

        return f"References to '{symbol}':\n\n" + "\n".join(lines)

    except subprocess.TimeoutExpired:
        return "Search timed out"
    except Exception as e:
        return f"Error searching: {e}"


async def _lsp_rename_impl(path: str, old_name: str, new_name: str, preview: bool = True) -> str:
    """
    Rename a symbol across files.

    Currently uses simple find-replace. Full LSP implementation
    would use textDocument/rename for safe refactoring.

    Args:
        path: File or directory to apply rename
        old_name: Current name of the symbol
        new_name: New name for the symbol
        preview: If True, show changes without applying (default)

    Returns:
        Preview of changes or confirmation of applied changes
    """
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    if not os.path.exists(valid_path):
        return f"Error: Path not found: {path}"

    # Find all occurrences
    try:
        if os.path.isdir(valid_path):
            result = subprocess.run(
                ["grep", "-rln", "--include=*.py", old_name, valid_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            files = [f for f in result.stdout.strip().split("\n") if f]
        else:
            files = [valid_path] if old_name in open(valid_path).read() else []

        if not files:
            return f"No occurrences of '{old_name}' found"

        changes = []
        for file_path in files[:20]:  # Limit files
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            count = content.count(old_name)
            changes.append(f"  {file_path}: {count} occurrence(s)")

        if preview:
            result = f"Preview: Rename '{old_name}' -> '{new_name}'\n\n"
            result += "\n".join(changes)
            result += f"\n\nTotal: {len(files)} file(s) affected"
            result += "\n\nUse preview=False to apply changes."
            return result

        # Apply changes
        applied = 0
        for file_path in files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
                new_content = content.replace(old_name, new_name)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(new_content)
                applied += 1
            except Exception as e:
                logger.warning(f"Failed to rename in {file_path}: {e}")

        return f"Renamed '{old_name}' -> '{new_name}' in {applied} file(s)"

    except subprocess.TimeoutExpired:
        return "Search timed out"
    except Exception as e:
        return f"Error: {e}"


async def _lsp_definition_impl(path: str, symbol: str) -> str:
    """
    Find the definition of a symbol.

    Searches for class, function, or variable definitions.

    Args:
        path: Directory to search in
        symbol: The symbol to find the definition for

    Returns:
        File and line where the symbol is defined
    """
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    # Patterns for definitions
    patterns = [
        f"class {symbol}",
        f"def {symbol}",
        f"{symbol} = ",
        f"{symbol}:",  # Type annotation
    ]

    for pattern in patterns:
        try:
            result = subprocess.run(
                ["grep", "-rn", "--include=*.py", pattern, valid_path],
                capture_output=True,
                text=True,
                timeout=30,
            )

            lines = [line for line in result.stdout.strip().split("\n") if line]
            if lines:
                # Return first match (most likely the definition)
                return f"Definition of '{symbol}':\n\n{lines[0]}"

        except Exception:
            continue

    return f"Definition not found for '{symbol}'"


# ========================================
# MCP Tools
# ========================================


@mcp.tool()
async def lsp_diagnostics(path: str, include_mypy: bool = False) -> str:
    """
    Get code diagnostics (errors, warnings) for a file.

    Uses ruff for fast linting and optionally mypy for type checking.
    This is useful for finding issues before running code.

    Args:
        path: Path to the file to analyze
        include_mypy: Include mypy type checking (slower but more thorough)

    Returns:
        Formatted list of diagnostics with line numbers and messages

    """
    return await _lsp_diagnostics_impl(path, include_mypy)


@mcp.tool()
async def lsp_completions(path: str, line: int, column: int) -> str:
    """
    Get code completion suggestions at a specific position.

    Useful for understanding what methods/properties are available.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        column: Column number (1-indexed)

    Returns:
        List of completion suggestions with their types

    """
    return await _lsp_completions_impl(path, line, column)


@mcp.tool()
async def lsp_hover(path: str, line: int, column: int) -> str:
    """
    Get hover information (documentation, type info) at a position.

    Useful for understanding what a symbol means without leaving context.

    Args:
        path: Path to the file
        line: Line number (1-indexed)
        column: Column number (1-indexed)

    Returns:
        Documentation or type information for the symbol

    """
    return await _lsp_hover_impl(path, line, column)


@mcp.tool()
async def lsp_references(path: str, symbol: str) -> str:
    """
    Find all references to a symbol in the codebase.

    Uses simple grep-based search for now. Full LSP implementation
    would use textDocument/references.

    Args:
        path: Starting directory to search
        symbol: The symbol name to find references for

    Returns:
        List of files and lines that reference the symbol

    """
    return await _lsp_references_impl(path, symbol)


@mcp.tool()
async def lsp_rename(path: str, old_name: str, new_name: str, preview: bool = True) -> str:
    """
    Rename a symbol across files.

    Currently uses simple find-replace. Full LSP implementation
    would use textDocument/rename for safe refactoring.

    Args:
        path: File or directory to apply rename
        old_name: Current name of the symbol
        new_name: New name for the symbol
        preview: If True, show changes without applying (default)

    Returns:
        Preview of changes or confirmation of applied changes

    """
    return await _lsp_rename_impl(path, old_name, new_name, preview)


@mcp.tool()
async def lsp_definition(path: str, symbol: str) -> str:
    """
    Find the definition of a symbol.

    Searches for class, function, or variable definitions.

    Args:
        path: Directory to search in
        symbol: The symbol to find the definition for

    Returns:
        File and line where the symbol is defined

    """
    return await _lsp_definition_impl(path, symbol)


if __name__ == "__main__":
    mcp.run()
