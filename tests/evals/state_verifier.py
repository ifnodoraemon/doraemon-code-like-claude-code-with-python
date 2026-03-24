"""
环境状态验证器

验证环境的实际状态而非表面输出，这是 Anthropic 评估最佳实践的核心要求。
"""

import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class VerificationResult:
    """验证结果"""

    success: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.success


class StateVerifier:
    """
    环境状态验证器

    用于验证环境的实际状态而非表面输出。
    基于 Anthropic 最佳实践：验证数据库记录，而非 UI 确认。
    """

    def __init__(self, base_dir: str | None = None):
        """
        初始化验证器

        Args:
            base_dir: 基础目录，默认为当前工作目录
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    # ==================== 文件系统验证 ====================

    def verify_file_exists(self, path: str) -> VerificationResult:
        """验证文件是否存在"""
        full_path = self._resolve_path(path)
        exists = full_path.exists() and full_path.is_file()
        return VerificationResult(
            success=exists,
            message=f"文件 {'存在' if exists else '不存在'}: {path}",
            details={"path": str(full_path), "exists": exists},
        )

    def verify_file_content(
        self, path: str, expected_content: str, exact: bool = False
    ) -> VerificationResult:
        """
        验证文件内容

        Args:
            path: 文件路径
            expected_content: 期望的内容
            exact: 是否精确匹配，False 则检查是否包含
        """
        full_path = self._resolve_path(path)
        if not full_path.exists():
            return VerificationResult(
                success=False,
                message=f"文件不存在: {path}",
                details={"path": str(full_path)},
            )

        try:
            actual_content = full_path.read_text(encoding="utf-8")
            if exact:
                match = actual_content == expected_content
            else:
                match = expected_content in actual_content

            return VerificationResult(
                success=match,
                message=f"文件内容 {'匹配' if match else '不匹配'}",
                details={
                    "path": str(full_path),
                    "match_type": "exact" if exact else "contains",
                    "actual_length": len(actual_content),
                    "expected_length": len(expected_content),
                },
            )
        except Exception as e:
            return VerificationResult(
                success=False, message=f"读取文件失败: {e}", details={"error": str(e)}
            )

    def verify_file_contains_pattern(self, path: str, pattern: str) -> VerificationResult:
        """验证文件是否包含指定模式（正则表达式）"""
        full_path = self._resolve_path(path)
        if not full_path.exists():
            return VerificationResult(
                success=False,
                message=f"文件不存在: {path}",
                details={"path": str(full_path)},
            )

        try:
            content = full_path.read_text(encoding="utf-8")
            match = re.search(pattern, content)
            return VerificationResult(
                success=match is not None,
                message=f"模式 {'找到' if match else '未找到'}: {pattern}",
                details={
                    "path": str(full_path),
                    "pattern": pattern,
                    "match": match.group() if match else None,
                },
            )
        except Exception as e:
            return VerificationResult(
                success=False, message=f"验证失败: {e}", details={"error": str(e)}
            )

    def verify_file_permissions(self, path: str, expected_mode: int) -> VerificationResult:
        """验证文件权限"""
        full_path = self._resolve_path(path)
        if not full_path.exists():
            return VerificationResult(
                success=False,
                message=f"文件不存在: {path}",
                details={"path": str(full_path)},
            )

        actual_mode = full_path.stat().st_mode & 0o777
        match = actual_mode == expected_mode
        return VerificationResult(
            success=match,
            message=f"权限 {'匹配' if match else '不匹配'}: {oct(actual_mode)} vs {oct(expected_mode)}",
            details={
                "path": str(full_path),
                "actual_mode": oct(actual_mode),
                "expected_mode": oct(expected_mode),
            },
        )

    def verify_directory_exists(self, path: str) -> VerificationResult:
        """验证目录是否存在"""
        full_path = self._resolve_path(path)
        exists = full_path.exists() and full_path.is_dir()
        return VerificationResult(
            success=exists,
            message=f"目录 {'存在' if exists else '不存在'}: {path}",
            details={"path": str(full_path), "exists": exists},
        )

    def verify_directory_structure(
        self, path: str, expected_structure: list[str]
    ) -> VerificationResult:
        """
        验证目录结构

        Args:
            path: 目录路径
            expected_structure: 期望的文件/目录列表（相对路径）
        """
        full_path = self._resolve_path(path)
        if not full_path.exists():
            return VerificationResult(
                success=False,
                message=f"目录不存在: {path}",
                details={"path": str(full_path)},
            )

        missing = []
        found = []
        for item in expected_structure:
            item_path = full_path / item
            if item_path.exists():
                found.append(item)
            else:
                missing.append(item)

        success = len(missing) == 0
        return VerificationResult(
            success=success,
            message=f"目录结构 {'完整' if success else '不完整'}，缺少: {missing}",
            details={"path": str(full_path), "found": found, "missing": missing},
        )

    # ==================== 代码验证 ====================

    def verify_syntax(self, file_path: str) -> VerificationResult:
        """验证 Python 语法"""
        full_path = self._resolve_path(file_path)
        if not full_path.exists():
            return VerificationResult(
                success=False,
                message=f"文件不存在: {file_path}",
                details={"path": str(full_path)},
            )

        try:
            content = full_path.read_text(encoding="utf-8")
            ast.parse(content)
            return VerificationResult(
                success=True,
                message="语法正确",
                details={"path": str(full_path)},
            )
        except SyntaxError as e:
            return VerificationResult(
                success=False,
                message=f"语法错误: {e.msg} (行 {e.lineno})",
                details={
                    "path": str(full_path),
                    "error": str(e),
                    "line": e.lineno,
                    "offset": e.offset,
                },
            )

    def verify_tests_pass(self, test_path: str, timeout: int = 60) -> VerificationResult:
        """
        运行测试并验证通过

        Args:
            test_path: 测试文件或目录路径
            timeout: 超时时间（秒）
        """
        full_path = self._resolve_path(test_path)
        if not full_path.exists():
            return VerificationResult(
                success=False,
                message=f"测试路径不存在: {test_path}",
                details={"path": str(full_path)},
            )

        try:
            result = subprocess.run(
                ["python", "-m", "pytest", str(full_path), "-v", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.base_dir),
            )

            success = result.returncode == 0
            # 解析测试结果
            passed = len(re.findall(r"PASSED", result.stdout))
            failed = len(re.findall(r"FAILED", result.stdout))
            errors = len(re.findall(r"ERROR", result.stdout))

            return VerificationResult(
                success=success,
                message=f"测试 {'通过' if success else '失败'}: {passed} passed, {failed} failed, {errors} errors",
                details={
                    "path": str(full_path),
                    "returncode": result.returncode,
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
                    "stderr": result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr,
                },
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                success=False,
                message=f"测试超时 ({timeout}s)",
                details={"path": str(full_path), "timeout": timeout},
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                message=f"运行测试失败: {e}",
                details={"error": str(e)},
            )

    def verify_coverage(self, file_path: str, min_coverage: float = 0.8) -> VerificationResult:
        """
        验证测试覆盖率

        Args:
            file_path: 要检查覆盖率的文件
            min_coverage: 最低覆盖率要求 (0-1)
        """
        full_path = self._resolve_path(file_path)
        if not full_path.exists():
            return VerificationResult(
                success=False,
                message=f"文件不存在: {file_path}",
                details={"path": str(full_path)},
            )

        try:
            # 运行 coverage
            subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    "--cov=" + str(full_path.parent),
                    "--cov-report=json",
                    str(self.base_dir / "tests"),
                ],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.base_dir),
            )

            # 读取覆盖率报告
            coverage_file = self.base_dir / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file) as f:
                    coverage_data = json.load(f)
                total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0) / 100
            else:
                total_coverage = 0

            success = total_coverage >= min_coverage
            return VerificationResult(
                success=success,
                message=f"覆盖率 {total_coverage:.1%} {'达标' if success else '未达标'} (要求 {min_coverage:.1%})",
                details={
                    "path": str(full_path),
                    "coverage": total_coverage,
                    "min_coverage": min_coverage,
                },
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                message=f"检查覆盖率失败: {e}",
                details={"error": str(e)},
            )

    def verify_lint_score(self, file_path: str, min_score: float = 8.0) -> VerificationResult:
        """
        验证代码质量（使用 ruff）

        Args:
            file_path: 要检查的文件
            min_score: 最低分数要求 (0-10)
        """
        full_path = self._resolve_path(file_path)
        if not full_path.exists():
            return VerificationResult(
                success=False,
                message=f"文件不存在: {file_path}",
                details={"path": str(full_path)},
            )

        try:
            result = subprocess.run(
                ["ruff", "check", str(full_path), "--output-format=json"],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # 解析 ruff 输出
            issues = []
            if result.stdout.strip():
                try:
                    issues = json.loads(result.stdout)
                except json.JSONDecodeError:
                    pass

            # 计算分数（每个问题扣 0.5 分，最低 0 分）
            score = max(0, 10 - len(issues) * 0.5)
            success = score >= min_score

            return VerificationResult(
                success=success,
                message=f"Lint 分数 {score:.1f} {'达标' if success else '未达标'} (要求 {min_score})",
                details={
                    "path": str(full_path),
                    "score": score,
                    "min_score": min_score,
                    "issues_count": len(issues),
                    "issues": issues[:10],  # 只返回前 10 个问题
                },
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                message=f"Lint 检查失败: {e}",
                details={"error": str(e)},
            )

    # ==================== Git 验证 ====================

    def verify_commit_exists(self, message_pattern: str) -> VerificationResult:
        """验证是否存在匹配的提交"""
        try:
            result = subprocess.run(
                ["git", "log", "--oneline", "-20"],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
            )

            commits = result.stdout.strip().split("\n")
            matching = [c for c in commits if re.search(message_pattern, c)]

            success = len(matching) > 0
            return VerificationResult(
                success=success,
                message=f"{'找到' if success else '未找到'}匹配的提交: {message_pattern}",
                details={
                    "pattern": message_pattern,
                    "matching_commits": matching,
                    "recent_commits": commits[:5],
                },
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                message=f"Git 操作失败: {e}",
                details={"error": str(e)},
            )

    def verify_branch_exists(self, branch_name: str) -> VerificationResult:
        """验证分支是否存在"""
        try:
            result = subprocess.run(
                ["git", "branch", "-a"],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
            )

            branches = [b.strip().replace("* ", "") for b in result.stdout.strip().split("\n")]
            exists = branch_name in branches or f"remotes/origin/{branch_name}" in branches

            return VerificationResult(
                success=exists,
                message=f"分支 {branch_name} {'存在' if exists else '不存在'}",
                details={"branch": branch_name, "all_branches": branches},
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                message=f"Git 操作失败: {e}",
                details={"error": str(e)},
            )

    def verify_no_uncommitted_changes(self) -> VerificationResult:
        """验证没有未提交的更改"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
            )

            changes = [line for line in result.stdout.strip().split("\n") if line]
            success = len(changes) == 0

            return VerificationResult(
                success=success,
                message=f"{'没有' if success else '有'}未提交的更改",
                details={"uncommitted_changes": changes},
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                message=f"Git 操作失败: {e}",
                details={"error": str(e)},
            )

    def verify_file_in_commit(self, commit_hash: str, file_path: str) -> VerificationResult:
        """验证文件是否在指定提交中被修改"""
        try:
            result = subprocess.run(
                ["git", "show", "--name-only", "--format=", commit_hash],
                capture_output=True,
                text=True,
                cwd=str(self.base_dir),
            )

            files = [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
            in_commit = file_path in files

            return VerificationResult(
                success=in_commit,
                message=f"文件 {file_path} {'在' if in_commit else '不在'}提交 {commit_hash[:7]} 中",
                details={
                    "commit": commit_hash,
                    "file": file_path,
                    "files_in_commit": files,
                },
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                message=f"Git 操作失败: {e}",
                details={"error": str(e)},
            )

    # ==================== 通用验证 ====================

    def verify_json_schema(self, data: dict | str, schema: dict) -> VerificationResult:
        """
        验证 JSON 数据结构

        Args:
            data: JSON 数据或文件路径
            schema: 期望的结构（简化版，只检查键是否存在）
        """
        try:
            if isinstance(data, str):
                # 如果是文件路径
                full_path = self._resolve_path(data)
                if full_path.exists():
                    with open(full_path) as f:
                        data = json.load(f)
                else:
                    data = json.loads(data)

            missing_keys = []
            for key in schema.get("required", []):
                if key not in data:
                    missing_keys.append(key)

            success = len(missing_keys) == 0
            return VerificationResult(
                success=success,
                message=f"JSON 结构 {'有效' if success else '无效'}，缺少: {missing_keys}",
                details={"missing_keys": missing_keys, "data_keys": list(data.keys())},
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                message=f"JSON 验证失败: {e}",
                details={"error": str(e)},
            )

    def verify_command_output(
        self,
        command: list[str],
        expected_output: str | None = None,
        expected_returncode: int = 0,
    ) -> VerificationResult:
        """
        验证命令输出

        Args:
            command: 要执行的命令
            expected_output: 期望的输出（包含检查）
            expected_returncode: 期望的返回码
        """
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.base_dir),
            )

            returncode_match = result.returncode == expected_returncode
            output_match = True
            if expected_output:
                output_match = expected_output in result.stdout

            success = returncode_match and output_match
            return VerificationResult(
                success=success,
                message=f"命令执行 {'成功' if success else '失败'}",
                details={
                    "command": " ".join(command),
                    "returncode": result.returncode,
                    "expected_returncode": expected_returncode,
                    "stdout": result.stdout[-1000:],
                    "stderr": result.stderr[-500:],
                },
            )
        except Exception as e:
            return VerificationResult(
                success=False,
                message=f"命令执行失败: {e}",
                details={"error": str(e)},
            )

    # ==================== 批量验证 ====================

    def verify_assertions(self, assertions: list[dict]) -> dict:
        """
        批量验证断言

        Args:
            assertions: 断言列表，每个断言包含 type 和相关参数

        Returns:
            验证结果汇总
        """
        results = []
        for assertion in assertions:
            assertion_type = assertion.get("type")
            result = self._verify_single_assertion(assertion)
            results.append(
                {
                    "type": assertion_type,
                    "assertion": assertion,
                    "result": result,
                }
            )

        passed = sum(1 for r in results if r["result"].success)
        total = len(results)

        return {
            "passed": passed,
            "total": total,
            "success": passed == total,
            "pass_rate": passed / total if total > 0 else 0,
            "results": results,
        }

    def _verify_single_assertion(self, assertion: dict) -> VerificationResult:
        """验证单个断言"""
        assertion_type = assertion.get("type")

        handlers = {
            "file_exists": lambda a: self.verify_file_exists(a.get("path")),
            "file_contains": lambda a: self.verify_file_contains_pattern(
                a.get("path"), a.get("pattern")
            ),
            "file_content": lambda a: self.verify_file_content(
                a.get("path"), a.get("content"), a.get("exact", False)
            ),
            "directory_exists": lambda a: self.verify_directory_exists(a.get("path")),
            "directory_structure": lambda a: self.verify_directory_structure(
                a.get("path"), a.get("structure")
            ),
            "syntax_valid": lambda a: self.verify_syntax(a.get("path")),
            "tests_pass": lambda a: self.verify_tests_pass(a.get("path"), a.get("timeout", 60)),
            "coverage": lambda a: self.verify_coverage(a.get("path"), a.get("min", 0.8)),
            "lint_score": lambda a: self.verify_lint_score(a.get("path"), a.get("min", 8.0)),
            "commit_exists": lambda a: self.verify_commit_exists(a.get("pattern")),
            "branch_exists": lambda a: self.verify_branch_exists(a.get("branch")),
            "no_uncommitted": lambda a: self.verify_no_uncommitted_changes(),
            "json_schema": lambda a: self.verify_json_schema(a.get("data"), a.get("schema")),
            "command_output": lambda a: self.verify_command_output(
                a.get("command"),
                a.get("expected_output"),
                a.get("expected_returncode", 0),
            ),
        }

        handler = handlers.get(assertion_type)
        if handler:
            return handler(assertion)
        else:
            return VerificationResult(
                success=False,
                message=f"未知的断言类型: {assertion_type}",
                details={"assertion": assertion},
            )

    def _resolve_path(self, path: str) -> Path:
        """解析路径（相对于 base_dir）"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.base_dir / p


# 便捷函数
def verify_task_result(task: dict, result: dict, base_dir: str | None = None) -> dict:
    """
    验证任务结果

    Args:
        task: 任务定义（包含 assertions）
        result: 任务执行结果
        base_dir: 基础目录

    Returns:
        验证结果汇总
    """
    verifier = StateVerifier(base_dir)
    assertions = task.get("assertions", [])
    return verifier.verify_assertions(assertions)


if __name__ == "__main__":
    # 示例用法
    verifier = StateVerifier()

    # 文件验证
    print("=== 文件验证 ===")
    print(verifier.verify_file_exists("README.md"))
    print(verifier.verify_file_contains_pattern("README.md", r"Doraemon"))

    # 代码验证
    print("\n=== 代码验证 ===")
    print(verifier.verify_syntax("tests/evals/state_verifier.py"))

    # Git 验证
    print("\n=== Git 验证 ===")
    print(verifier.verify_branch_exists("master"))
    print(verifier.verify_no_uncommitted_changes())

    # 批量验证
    print("\n=== 批量验证 ===")
    assertions = [
        {"type": "file_exists", "path": "README.md"},
        {"type": "file_contains", "path": "README.md", "pattern": "Doraemon"},
        {"type": "branch_exists", "branch": "master"},
    ]
    result = verifier.verify_assertions(assertions)
    print(f"通过: {result['passed']}/{result['total']}")
