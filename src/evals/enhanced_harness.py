"""
Enhanced Evaluation Harness with Extended Assertion Types

Supports:
- file_contains / file_not_contains
- pattern_exists / pattern_not_exists
- tool_not_used
- error_raised
- syntax_valid
- performance metrics
"""

import ast
import os
import re
from pathlib import Path


class EnhancedAssertion:
    """Extended assertion checker with more types."""

    @staticmethod
    def check_file_contains(sandbox_dir: str, path: str, pattern: str) -> tuple[bool, str]:
        """Check if file contains pattern."""
        full_path = os.path.join(sandbox_dir, path)
        if not os.path.exists(full_path):
            return False, f"File {path} not found"

        with open(full_path) as f:
            content = f.read()

        if pattern.lower() in content.lower():
            return True, f"✅ File {path} contains '{pattern}'"
        return False, f"❌ File {path} missing '{pattern}'"

    @staticmethod
    def check_file_not_contains(sandbox_dir: str, path: str, pattern: str) -> tuple[bool, str]:
        """Check if file does NOT contain pattern."""
        full_path = os.path.join(sandbox_dir, path)
        if not os.path.exists(full_path):
            return True, f"✅ File {path} not found (as expected)"

        with open(full_path) as f:
            content = f.read()

        if pattern.lower() not in content.lower():
            return True, f"✅ File {path} does not contain '{pattern}'"
        return False, f"❌ File {path} unexpectedly contains '{pattern}'"

    @staticmethod
    def check_pattern_exists(sandbox_dir: str, pattern: str) -> tuple[bool, str]:
        """Check if regex pattern exists in any Python file."""
        for py_file in Path(sandbox_dir).rglob("*.py"):
            with open(py_file) as f:
                content = f.read()
            if re.search(pattern, content):
                return True, f"✅ Pattern '{pattern}' found in {py_file.name}"

        return False, f"❌ Pattern '{pattern}' not found in any file"

    @staticmethod
    def check_pattern_not_exists(sandbox_dir: str, pattern: str) -> tuple[bool, str]:
        """Check if regex pattern does NOT exist."""
        for py_file in Path(sandbox_dir).rglob("*.py"):
            with open(py_file) as f:
                content = f.read()
            if re.search(pattern, content):
                return False, f"❌ Pattern '{pattern}' unexpectedly found in {py_file.name}"

        return True, f"✅ Pattern '{pattern}' not found (as expected)"

    @staticmethod
    def check_tool_not_used(trace: list[dict], tool: str) -> tuple[bool, str]:
        """Check that a tool was NOT used."""
        used = any(e["type"] == "tool_call" and e["name"] == tool for e in trace)
        if not used:
            return True, f"✅ Tool {tool} not used (as expected)"
        return False, f"❌ Tool {tool} was used unexpectedly"

    @staticmethod
    def check_syntax_valid(sandbox_dir: str, language: str) -> tuple[bool, str]:
        """Check if code has valid syntax."""
        if language == "python":
            for py_file in Path(sandbox_dir).rglob("*.py"):
                try:
                    with open(py_file) as f:
                        ast.parse(f.read())
                except SyntaxError as e:
                    return False, f"❌ Syntax error in {py_file.name}: {e}"

            return True, "✅ All Python files have valid syntax"

        return True, f"✅ Syntax check for {language} not implemented"

    @staticmethod
    def check_error_raised(trace: list[dict], error_type: str) -> tuple[bool, str]:
        """Check if specific error was raised."""
        for event in trace:
            if event.get("type") == "error" and error_type in str(event.get("error", "")):
                return True, f"✅ {error_type} raised as expected"

        return False, f"❌ {error_type} not raised"


def check_assertions_enhanced(
    assertions: list[dict], trace: list[dict], final_output: str, sandbox_dir: str
) -> dict:
    """Enhanced assertion checking with more types."""
    passed = True
    reasons = []
    checker = EnhancedAssertion()

    for asm in assertions:
        atype = asm.get("type")

        # Original types
        if atype == "file_exists":
            rel_path = asm["path"]
            # Support wildcards
            if "*" in rel_path:
                matches = list(Path(sandbox_dir).glob(rel_path))
                if matches:
                    reasons.append(f"✅ File matching {rel_path} exists")
                else:
                    passed = False
                    reasons.append(f"❌ No file matching {rel_path}")
            else:
                full_path = os.path.join(sandbox_dir, rel_path)
                if os.path.exists(full_path):
                    reasons.append(f"✅ File {rel_path} exists")
                else:
                    passed = False
                    reasons.append(f"❌ File {rel_path} not found")

        elif atype == "tool_used":
            tool_name = asm["tool"]
            called = any(e["type"] == "tool_call" and e["name"] == tool_name for e in trace)
            if called:
                reasons.append(f"✅ Tool {tool_name} called")
            else:
                passed = False
                reasons.append(f"❌ Tool {tool_name} NOT called")

        elif atype == "output_contains":
            pattern = asm["pattern"]
            if pattern.lower() in final_output.lower():
                reasons.append(f"✅ Output matches '{pattern}'")
            else:
                passed = False
                reasons.append(f"❌ Output missing '{pattern}'")

        # New types
        elif atype == "file_contains":
            success, msg = checker.check_file_contains(sandbox_dir, asm["path"], asm["pattern"])
            reasons.append(msg)
            if not success:
                passed = False

        elif atype == "file_not_contains":
            success, msg = checker.check_file_not_contains(sandbox_dir, asm["path"], asm["pattern"])
            reasons.append(msg)
            if not success:
                passed = False

        elif atype == "pattern_exists":
            success, msg = checker.check_pattern_exists(sandbox_dir, asm["pattern"])
            reasons.append(msg)
            if not success:
                passed = False

        elif atype == "pattern_not_exists":
            success, msg = checker.check_pattern_not_exists(sandbox_dir, asm["pattern"])
            reasons.append(msg)
            if not success:
                passed = False

        elif atype == "tool_not_used":
            success, msg = checker.check_tool_not_used(trace, asm["tool"])
            reasons.append(msg)
            if not success:
                passed = False

        elif atype == "syntax_valid":
            success, msg = checker.check_syntax_valid(sandbox_dir, asm["language"])
            reasons.append(msg)
            if not success:
                passed = False

        elif atype == "error_raised":
            success, msg = checker.check_error_raised(trace, asm["error_type"])
            reasons.append(msg)
            if not success:
                passed = False

    return {"pass": passed, "reasons": reasons}
