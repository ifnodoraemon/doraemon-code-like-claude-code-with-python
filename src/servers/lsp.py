"""
Language Server Protocol (LSP) Tools

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

from src.core.logger import configure_root_logger
from src.core.security.security import validate_path

# Setup logging
configure_root_logger()
logger = logging.getLogger(__name__)


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

        loop = asyncio.get_running_loop()

        def _do_io():
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

        return await loop.run_in_executor(None, _do_io)

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

        def _read_path_content():
            with open(file_path, encoding="utf-8", errors="replace") as f:
                return f.read()

        content = await loop.run_in_executor(None, _read_path_content)

        await self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "python",
                    "version": 1,
                    "text": content,
                }
            },
        )

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

        # Open the document first (in executor to avoid blocking)
        loop = asyncio.get_running_loop()

        def _read_path_content():
            with open(file_path, encoding="utf-8", errors="replace") as f:
                return f.read()

        content = await loop.run_in_executor(None, _read_path_content)

        await self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "python",
                    "version": 1,
                    "text": content,
                }
            },
        )

        response = await self._send_request(
            "textDocument/completion",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
        )

        if not response or "result" not in response:
            return []

        result = response["result"]
        items = result.get("items", []) if isinstance(result, dict) else result

        completions = []
        for item in items[:20]:  # Limit results
            kind_map = {
                1: "text",
                2: "method",
                3: "function",
                4: "constructor",
                5: "field",
                6: "variable",
                7: "class",
                8: "interface",
                9: "module",
                10: "property",
            }
            completions.append(
                CompletionItem(
                    label=item.get("label", ""),
                    kind=kind_map.get(item.get("kind", 1), "text"),
                    detail=item.get("detail"),
                    documentation=item.get("documentation"),
                )
            )

        return completions

    async def get_hover(self, file_path: str, line: int, character: int) -> str | None:
        """Get hover information at a position."""
        if not self._initialized:
            return None

        uri = f"file://{os.path.abspath(file_path)}"

        # Open the document first (in executor to avoid blocking)
        loop = asyncio.get_running_loop()

        def _read_path_content():
            with open(file_path, encoding="utf-8", errors="replace") as f:
                return f.read()

        content = await loop.run_in_executor(None, _read_path_content)

        await self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": "python",
                    "version": 1,
                    "text": content,
                }
            },
        )

        response = await self._send_request(
            "textDocument/hover",
            {
                "textDocument": {"uri": uri},
                "position": {"line": line, "character": character},
            },
        )

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
                c.get("value", str(c)) if isinstance(c, dict) else str(c) for c in contents
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
    """Run ruff for quick Python diagnostics, delegating to lint module."""
    try:
        from src.servers.lint import _run_command

        exit_code, stdout, stderr = _run_command(
            ["ruff", "check", "--output-format=json", file_path]
        )
        if stdout:
            return json.loads(stdout)
    except (ImportError, json.JSONDecodeError):
        pass
    return []


def _run_mypy(file_path: str) -> list[dict]:
    """Run mypy for type checking, delegating to lint module."""
    try:
        from src.servers.lint import _run_command

        exit_code, stdout, stderr = _run_command(
            ["mypy", "--show-error-codes", "--no-error-summary", file_path],
            timeout=60,
        )
        diagnostics = []
        for line in stdout.strip().split("\n"):
            if not line:
                continue
            match = re.match(r"(.+):(\d+): (error|warning|note): (.+)", line)
            if match:
                diagnostics.append(
                    {
                        "file": match.group(1),
                        "line": int(match.group(2)),
                        "severity": match.group(3),
                        "message": match.group(4),
                    }
                )
        return diagnostics
    except ImportError:
        pass
    return []


# ========================================
# Helpers
# ========================================


def _validate_py_file(path: str) -> tuple[str, str | None]:
    """Validate path is an existing Python file. Returns (resolved_path, error)."""
    try:
        valid = validate_path(path)
    except (PermissionError, ValueError) as e:
        return "", f"Error: {e}"
    if not os.path.exists(valid):
        return "", f"Error: File not found: {path}"
    if not valid.endswith(".py"):
        return "", f"Error: Only .py files supported, got {os.path.splitext(valid)[1]}"
    return valid, None


async def _require_lsp() -> tuple[PylspClient | None, str | None]:
    """Get LSP client or return error message."""
    client = await _get_lsp_client()
    if not client:
        return None, "LSP server not available. Install pylsp: pip install python-lsp-server"
    return client, None


# ========================================
# Tool entry points
# ========================================


async def lsp_diagnostics(path: str, include_mypy: bool = False) -> str:
    """Get code diagnostics (errors, warnings) for a Python file."""
    valid_path, err = _validate_py_file(path)
    if err:
        return err

    results = []
    for diag in _run_ruff(valid_path):
        line = diag.get("location", {}).get("row", 0)
        code = diag.get("code", "")
        message = diag.get("message", "")
        results.append(f"L{line} [{code}]: {message}")

    if include_mypy:
        for diag in _run_mypy(valid_path):
            results.append(
                f"L{diag.get('line', 0)} [mypy:{diag.get('severity', 'error')}]: "
                f"{diag.get('message', '')}"
            )

    if not results:
        return f"No issues found in {path}"
    return f"Diagnostics for {path}:\n\n" + "\n".join(results)


async def lsp_completions(path: str, line: int, column: int) -> str:
    """Get code completion suggestions at a position (1-indexed)."""
    valid_path, err = _validate_py_file(path)
    if err:
        return err
    client, err = await _require_lsp()
    if err:
        return err

    completions = await client.get_completions(valid_path, line - 1, column - 1)
    if not completions:
        return f"No completions at {path}:{line}:{column}"

    lines = [f"Completions at {path}:{line}:{column}:\n"]
    for c in completions[:15]:
        detail = f" - {c.detail}" if c.detail else ""
        lines.append(f"  [{c.kind}] {c.label}{detail}")
    if len(completions) > 15:
        lines.append(f"\n  ... and {len(completions) - 15} more")
    return "\n".join(lines)


async def lsp_hover(path: str, line: int, column: int) -> str:
    """Get type/documentation info at a position (1-indexed)."""
    valid_path, err = _validate_py_file(path)
    if err:
        return err
    client, err = await _require_lsp()
    if err:
        return err

    hover_info = await client.get_hover(valid_path, line - 1, column - 1)
    if not hover_info:
        return f"No information at {path}:{line}:{column}"
    return f"Info at {path}:{line}:{column}:\n\n{hover_info}"


async def lsp_references(path: str, symbol: str) -> str:
    """Find all references to a symbol via grep."""
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    try:
        from src.servers.filesystem import grep_search

        result = grep_search(pattern=symbol, include="*.py", path=valid_path)
        if not result or result.startswith("No matches"):
            return f"No references found for '{symbol}'"
        return f"References to '{symbol}':\n\n{result}"
    except ImportError:
        return "Error: filesystem module not available"
    except Exception as e:
        return f"Error searching: {e}"


async def lsp_rename(
    path: str,
    old_name: str,
    new_name: str,
    preview: bool = True,
) -> str:
    """Rename a symbol across Python files (find-replace)."""
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"
    if not os.path.exists(valid_path):
        return f"Error: Path not found: {path}"

    try:
        # Find affected files
        if os.path.isdir(valid_path):
            result = subprocess.run(
                ["grep", "-rln", "--include=*.py", old_name, valid_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            files = [f for f in result.stdout.strip().split("\n") if f]
        else:
            with open(valid_path, encoding="utf-8") as f:
                files = [valid_path] if old_name in f.read() else []

        if not files:
            return f"No occurrences of '{old_name}' found"

        if preview:
            changes = []
            for fp in files[:20]:
                with open(fp, encoding="utf-8") as f:
                    count = f.read().count(old_name)
                changes.append(f"  {fp}: {count} occurrence(s)")
            return (
                f"Preview: '{old_name}' -> '{new_name}'\n\n"
                + "\n".join(changes)
                + f"\n\nTotal: {len(files)} file(s). Use preview=False to apply."
            )

        # Apply
        applied = 0
        for fp in files:
            try:
                with open(fp, encoding="utf-8") as f:
                    content = f.read()
                with open(fp, "w", encoding="utf-8") as f:
                    f.write(re.sub(r"\b" + re.escape(old_name) + r"\b", new_name, content))
                applied += 1
            except Exception as e:
                logger.warning(f"Failed to rename in {fp}: {e}")
        return f"Renamed '{old_name}' -> '{new_name}' in {applied} file(s)"

    except subprocess.TimeoutExpired:
        return "Search timed out"
    except Exception as e:
        return f"Error: {e}"


async def lsp_definition(path: str, symbol: str) -> str:
    """Find where a symbol is defined (class/def/assignment)."""
    try:
        valid_path = validate_path(path)
    except (PermissionError, ValueError) as e:
        return f"Error: {e}"

    for pattern in [f"class {symbol}", f"def {symbol}", f"{symbol} = ", f"{symbol}:"]:
        try:
            from src.servers.filesystem import grep_search

            result = grep_search(pattern=pattern, include="*.py", path=valid_path)
            if result and not result.startswith("No matches"):
                first_line = result.strip().split("\n")[0]
                return f"Definition of '{symbol}':\n\n{first_line}"
        except ImportError:
            return "Error: filesystem module not available"
        except Exception:
            continue

    return f"Definition not found for '{symbol}'"
