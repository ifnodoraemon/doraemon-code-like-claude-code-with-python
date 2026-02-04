"""
评估指标收集器

收集和分析评估过程中的各种指标。
基于 Anthropic 最佳实践，提供业界标准的评估指标。

主要功能:
- pass@k 和 pass^k 指标计算
- 延迟分布分析 (p50, p95, p99)
- 成本跟踪和分析
- 错误分类和分布
- 综合评估报告生成
"""

import json
import math
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Token 成本配置 (每 1K tokens 的美元价格)
DEFAULT_TOKEN_COSTS = {
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-opus": {"input": 0.015, "output": 0.075},
    "claude-3-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "claude-opus-4-5": {"input": 0.015, "output": 0.075},
    "gemini-pro": {"input": 0.00025, "output": 0.0005},
    "gemini-3-pro": {"input": 0.0005, "output": 0.001},
    "default": {"input": 0.001, "output": 0.002},
}

# 错误类型映射
ERROR_TYPE_PATTERNS = {
    "timeout": ["timeout", "timed out", "time limit"],
    "assertion_failed": ["assertion", "assert", "expected", "actual"],
    "tool_error": ["tool error", "tool failed", "tool_call"],
    "syntax_error": ["syntax", "syntaxerror", "parse error"],
    "runtime_error": ["runtime", "runtimeerror", "exception"],
    "permission": ["permission", "access denied", "forbidden"],
    "not_found": ["not found", "notfound", "does not exist", "missing"],
    "validation": ["validation", "invalid", "schema"],
    "network": ["network", "connection", "http", "request failed"],
    "memory": ["memory", "oom", "out of memory"],
}


class MetricsCollector:
    """
    评估指标收集器

    基于 Anthropic 最佳实践，提供业界标准的评估指标收集和分析功能。

    主要功能:
    - pass@k: 计算 k 次尝试中至少 1 次成功的概率
    - pass^k: 计算 k 次尝试全部成功的概率（生产可靠性指标）
    - 延迟分布: p50, p95, p99 等百分位数
    - 成本跟踪: token 使用量和成本分析
    - 错误分类: 按类型、类别、难度分析错误

    使用示例:
        collector = MetricsCollector()
        collector.start_collection()
        for result in evaluation_results:
            collector.record_task_result(result)
        collector.end_collection()
        report = collector.generate_anthropic_style_report(results)
    """

    def __init__(
        self,
        output_dir: str = "eval_results/metrics",
        model_name: str = "default",
        token_costs: dict[str, dict[str, float]] | None = None,
    ):
        """
        初始化指标收集器

        Args:
            output_dir: 输出目录路径
            model_name: 模型名称，用于成本计算
            token_costs: 自定义 token 成本配置，格式为 {model: {input: cost, output: cost}}
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: dict[str, list[Any]] = defaultdict(list)
        self.start_time: float | None = None
        self.end_time: float | None = None
        self.model_name = model_name
        self.token_costs = token_costs or DEFAULT_TOKEN_COSTS

    def start_collection(self) -> None:
        """开始收集指标"""
        self.start_time = time.time()
        self.metrics.clear()

    def end_collection(self) -> None:
        """结束收集指标"""
        self.end_time = time.time()

    def record_task_result(self, result: dict[str, Any]) -> None:
        """
        记录任务结果

        Args:
            result: 任务结果字典，应包含以下字段:
                - task_id: 任务 ID
                - success: 是否成功
                - execution_time: 执行时间（秒）
                - tool_calls: 工具调用列表
                - difficulty: 难度等级
                - category: 任务类别
                - errors: 错误列表
                - input_tokens: 输入 token 数
                - output_tokens: 输出 token 数
                - llm_evaluation: LLM 评估结果
        """
        # 基础指标
        self.metrics["task_ids"].append(result.get("task_id"))
        self.metrics["success"].append(result.get("success", False))
        self.metrics["execution_time"].append(result.get("execution_time", 0))

        # 工具使用指标
        tool_calls = result.get("tool_calls", [])
        self.metrics["tool_call_count"].append(len(tool_calls))
        self.metrics["tool_calls"].extend(tool_calls)

        # 难度和类别
        self.metrics["difficulty"].append(result.get("difficulty", 0))
        self.metrics["category"].append(result.get("category", "unknown"))

        # 错误信息
        if result.get("errors"):
            self.metrics["errors"].extend(result.get("errors", []))

        # Token 使用量
        self.metrics["input_tokens"].append(result.get("input_tokens", 0))
        self.metrics["output_tokens"].append(result.get("output_tokens", 0))

        # LLM 评估分数（如果有）
        if "llm_evaluation" in result:
            llm_eval = result["llm_evaluation"]
            self.metrics["llm_overall_score"].append(llm_eval.get("overall_score", 0))
            self.metrics["llm_task_completion"].append(
                llm_eval.get("task_completion", {}).get("score", 0)
            )
            self.metrics["llm_tool_usage"].append(
                llm_eval.get("tool_usage", {}).get("score", 0)
            )
            self.metrics["llm_code_quality"].append(
                llm_eval.get("code_quality", {}).get("score", 0)
            )

    def calculate_metrics(self) -> dict:
        """计算汇总指标"""
        if not self.metrics["success"]:
            return {"error": "没有收集到数据"}

        total_tasks = len(self.metrics["success"])
        successful_tasks = sum(self.metrics["success"])

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "collection_time": self.end_time - self.start_time
            if self.end_time and self.start_time
            else 0,
            # 核心指标
            "core_metrics": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": total_tasks - successful_tasks,
                "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
                "avg_execution_time": sum(self.metrics["execution_time"]) / total_tasks
                if total_tasks > 0
                else 0,
                "total_execution_time": sum(self.metrics["execution_time"]),
            },
            # 工具使用指标
            "tool_metrics": self._calculate_tool_metrics(),
            # 难度分析
            "difficulty_metrics": self._calculate_difficulty_metrics(),
            # 类别分析
            "category_metrics": self._calculate_category_metrics(),
            # LLM 评估指标
            "llm_metrics": self._calculate_llm_metrics(),
            # 性能指标
            "performance_metrics": self._calculate_performance_metrics(),
            # 错误分析
            "error_metrics": self._calculate_error_metrics(),
        }

        return metrics

    def _calculate_tool_metrics(self) -> dict:
        """计算工具使用指标"""
        if not self.metrics["tool_call_count"]:
            return {}

        # 工具调用统计
        tool_counts = defaultdict(int)
        for tool in self.metrics["tool_calls"]:
            tool_counts[tool] += 1

        total_calls = sum(self.metrics["tool_call_count"])
        total_tasks = len(self.metrics["tool_call_count"])

        return {
            "total_tool_calls": total_calls,
            "avg_tool_calls_per_task": total_calls / total_tasks
            if total_tasks > 0
            else 0,
            "max_tool_calls": max(self.metrics["tool_call_count"])
            if self.metrics["tool_call_count"]
            else 0,
            "min_tool_calls": min(self.metrics["tool_call_count"])
            if self.metrics["tool_call_count"]
            else 0,
            "tool_usage_distribution": dict(tool_counts),
            "most_used_tools": sorted(
                tool_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }

    def _calculate_difficulty_metrics(self) -> dict:
        """计算难度相关指标"""
        if not self.metrics["difficulty"]:
            return {}

        # 按难度分组
        by_difficulty = defaultdict(lambda: {"total": 0, "success": 0, "times": []})

        for i, difficulty in enumerate(self.metrics["difficulty"]):
            by_difficulty[difficulty]["total"] += 1
            if self.metrics["success"][i]:
                by_difficulty[difficulty]["success"] += 1
            by_difficulty[difficulty]["times"].append(self.metrics["execution_time"][i])

        # 计算每个难度的指标
        difficulty_stats = {}
        for difficulty, stats in by_difficulty.items():
            difficulty_stats[difficulty] = {
                "total_tasks": stats["total"],
                "successful_tasks": stats["success"],
                "success_rate": stats["success"] / stats["total"]
                if stats["total"] > 0
                else 0,
                "avg_execution_time": sum(stats["times"]) / len(stats["times"])
                if stats["times"]
                else 0,
            }

        # 将难度字符串转换为数字
        difficulty_map = {"easy": 1, "medium": 2, "hard": 3, "expert": 4}
        numeric_difficulties = [
            difficulty_map.get(d, 2) if isinstance(d, str) else d
            for d in self.metrics["difficulty"]
        ]

        return {
            "avg_difficulty": sum(numeric_difficulties)
            / len(numeric_difficulties)
            if numeric_difficulties
            else 0,
            "by_difficulty": difficulty_stats,
        }

    def _calculate_category_metrics(self) -> dict:
        """计算类别相关指标"""
        if not self.metrics["category"]:
            return {}

        # 按类别分组
        by_category = defaultdict(lambda: {"total": 0, "success": 0, "times": []})

        for i, category in enumerate(self.metrics["category"]):
            by_category[category]["total"] += 1
            if self.metrics["success"][i]:
                by_category[category]["success"] += 1
            by_category[category]["times"].append(self.metrics["execution_time"][i])

        # 计算每个类别的指标
        category_stats = {}
        for category, stats in by_category.items():
            category_stats[category] = {
                "total_tasks": stats["total"],
                "successful_tasks": stats["success"],
                "success_rate": stats["success"] / stats["total"]
                if stats["total"] > 0
                else 0,
                "avg_execution_time": sum(stats["times"]) / len(stats["times"])
                if stats["times"]
                else 0,
            }

        return {
            "total_categories": len(by_category),
            "by_category": category_stats,
        }

    def _calculate_llm_metrics(self) -> dict:
        """计算 LLM 评估指标"""
        if not self.metrics["llm_overall_score"]:
            return {}

        def safe_avg(lst):
            return sum(lst) / len(lst) if lst else 0

        return {
            "avg_overall_score": safe_avg(self.metrics["llm_overall_score"]),
            "avg_task_completion": safe_avg(self.metrics["llm_task_completion"]),
            "avg_tool_usage": safe_avg(self.metrics["llm_tool_usage"]),
            "avg_code_quality": safe_avg(self.metrics["llm_code_quality"]),
            "score_distribution": self._calculate_score_distribution(
                self.metrics["llm_overall_score"]
            ),
        }

    def _calculate_score_distribution(self, scores: list[float]) -> dict:
        """计算分数分布"""
        if not scores:
            return {}

        distribution = {
            "excellent (9-10)": 0,
            "good (7-8)": 0,
            "average (5-6)": 0,
            "poor (3-4)": 0,
            "failing (0-2)": 0,
        }

        for score in scores:
            if score >= 9:
                distribution["excellent (9-10)"] += 1
            elif score >= 7:
                distribution["good (7-8)"] += 1
            elif score >= 5:
                distribution["average (5-6)"] += 1
            elif score >= 3:
                distribution["poor (3-4)"] += 1
            else:
                distribution["failing (0-2)"] += 1

        return distribution

    def _calculate_performance_metrics(self) -> dict:
        """计算性能指标"""
        if not self.metrics["execution_time"]:
            return {}

        times = self.metrics["execution_time"]
        sorted_times = sorted(times)
        n = len(times)

        return {
            "min_time": min(times),
            "max_time": max(times),
            "avg_time": sum(times) / n,
            "median_time": sorted_times[n // 2] if n > 0 else 0,
            "p95_time": sorted_times[int(n * 0.95)] if n > 0 else 0,
            "p99_time": sorted_times[int(n * 0.99)] if n > 0 else 0,
            "total_time": sum(times),
        }

    def _calculate_error_metrics(self) -> dict:
        """计算错误指标"""
        if not self.metrics["errors"]:
            return {"total_errors": 0, "error_types": {}}

        # 错误类型统计
        error_types = defaultdict(int)
        for error in self.metrics["errors"]:
            # 简单的错误分类
            error_str = str(error).lower()
            if "timeout" in error_str:
                error_types["timeout"] += 1
            elif "permission" in error_str:
                error_types["permission"] += 1
            elif "not found" in error_str:
                error_types["not_found"] += 1
            elif "syntax" in error_str:
                error_types["syntax"] += 1
            else:
                error_types["other"] += 1

        return {
            "total_errors": len(self.metrics["errors"]),
            "error_types": dict(error_types),
            "error_rate": len(self.metrics["errors"]) / len(self.metrics["success"])
            if self.metrics["success"]
            else 0,
        }

    # ========================================================================
    # Anthropic 最佳实践指标 (Industry Standard Metrics)
    # ========================================================================

    def calculate_pass_at_k(self, results: list[dict[str, Any]], k: int) -> float:
        """
        计算 pass@k 指标：k 次尝试中至少 1 次成功的概率

        pass@k 是评估代码生成模型的标准指标，表示在 k 次独立尝试中
        至少有一次成功的概率。

        公式: pass@k = 1 - (1 - success_rate)^k

        Args:
            results: 评估结果列表，每个结果应包含 'success' 字段
            k: 尝试次数

        Returns:
            pass@k 概率值 (0.0 - 1.0)

        Example:
            >>> collector = MetricsCollector()
            >>> results = [{"success": True}, {"success": False}, {"success": True}]
            >>> collector.calculate_pass_at_k(results, k=3)
            0.963  # 约 96.3% 的概率至少成功一次
        """
        if not results or k <= 0:
            return 0.0

        success_count = sum(1 for r in results if r.get("success", False))
        n = len(results)
        success_rate = success_count / n if n > 0 else 0.0

        # pass@k = 1 - (1 - success_rate)^k
        return 1.0 - math.pow(1.0 - success_rate, k)

    def calculate_pass_power_k(self, results: list[dict[str, Any]], k: int) -> float:
        """
        计算 pass^k 指标：k 次尝试全部成功的概率（生产可靠性指标）

        pass^k 是衡量生产环境可靠性的关键指标，表示连续 k 次尝试
        全部成功的概率。这对于需要高可靠性的生产系统尤为重要。

        公式: pass^k = success_rate^k

        Args:
            results: 评估结果列表，每个结果应包含 'success' 字段
            k: 连续成功次数

        Returns:
            pass^k 概率值 (0.0 - 1.0)

        Example:
            >>> collector = MetricsCollector()
            >>> results = [{"success": True}, {"success": True}, {"success": False}]
            >>> collector.calculate_pass_power_k(results, k=3)
            0.296  # 约 29.6% 的概率连续 3 次成功
        """
        if not results or k <= 0:
            return 0.0

        success_count = sum(1 for r in results if r.get("success", False))
        n = len(results)
        success_rate = success_count / n if n > 0 else 0.0

        # pass^k = success_rate^k
        return math.pow(success_rate, k)

    def calculate_latency_distribution(
        self, results: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        计算延迟分布指标

        提供全面的延迟统计信息，包括百分位数、均值、标准差等。
        这些指标对于理解系统性能和识别异常值非常重要。

        Args:
            results: 评估结果列表，每个结果应包含 'execution_time' 字段

        Returns:
            延迟分布字典，包含:
            - p50: 中位数（50th 百分位）
            - p95: 95th 百分位
            - p99: 99th 百分位
            - mean: 平均值
            - std: 标准差
            - min: 最小值
            - max: 最大值

        Example:
            >>> collector = MetricsCollector()
            >>> results = [{"execution_time": 1.0}, {"execution_time": 2.0}, ...]
            >>> dist = collector.calculate_latency_distribution(results)
            >>> print(f"P95 延迟: {dist['p95']:.2f}s")
        """
        if not results:
            return {
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        latencies = [r.get("execution_time", 0.0) for r in results]
        latencies = [lat for lat in latencies if lat > 0]  # 过滤无效值

        if not latencies:
            return {
                "p50": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

        sorted_latencies = sorted(latencies)

        def percentile(data: list[float], p: float) -> float:
            """计算百分位数"""
            if not data:
                return 0.0
            k = (len(data) - 1) * p / 100.0
            f = math.floor(k)
            c = math.ceil(k)
            if f == c:
                return data[int(k)]
            return data[int(f)] * (c - k) + data[int(c)] * (k - f)

        return {
            "p50": percentile(sorted_latencies, 50),
            "p95": percentile(sorted_latencies, 95),
            "p99": percentile(sorted_latencies, 99),
            "mean": statistics.mean(latencies),
            "std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            "min": min(latencies),
            "max": max(latencies),
        }

    def calculate_cost_metrics(self, results: list[dict[str, Any]]) -> dict[str, float]:
        """
        计算成本指标

        跟踪 token 使用量和相关成本，帮助优化资源使用和预算控制。

        Args:
            results: 评估结果列表，每个结果应包含:
                - input_tokens: 输入 token 数
                - output_tokens: 输出 token 数

        Returns:
            成本指标字典，包含:
            - total_tokens: 总 token 数
            - input_tokens: 输入 token 总数
            - output_tokens: 输出 token 总数
            - total_cost: 总成本（美元）
            - cost_per_task: 每任务平均成本
            - cost_per_success: 每成功任务成本

        Example:
            >>> collector = MetricsCollector(model_name="claude-3-opus")
            >>> results = [{"input_tokens": 1000, "output_tokens": 500, "success": True}]
            >>> cost = collector.calculate_cost_metrics(results)
            >>> print(f"总成本: ${cost['total_cost']:.4f}")
        """
        if not results:
            return {
                "total_tokens": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_cost": 0.0,
                "cost_per_task": 0.0,
                "cost_per_success": 0.0,
            }

        input_tokens = sum(r.get("input_tokens", 0) for r in results)
        output_tokens = sum(r.get("output_tokens", 0) for r in results)
        total_tokens = input_tokens + output_tokens

        # 获取模型成本配置
        model_costs = self.token_costs.get(
            self.model_name, self.token_costs.get("default", {"input": 0.001, "output": 0.002})
        )

        # 计算成本 (每 1K tokens)
        input_cost = (input_tokens / 1000.0) * model_costs["input"]
        output_cost = (output_tokens / 1000.0) * model_costs["output"]
        total_cost = input_cost + output_cost

        # 计算每任务和每成功任务成本
        n_tasks = len(results)
        n_success = sum(1 for r in results if r.get("success", False))

        return {
            "total_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": total_cost,
            "cost_per_task": total_cost / n_tasks if n_tasks > 0 else 0.0,
            "cost_per_success": total_cost / n_success if n_success > 0 else 0.0,
        }

    def analyze_errors(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """
        分析错误类型和分布

        提供详细的错误分析，包括错误类型分布、按类别和难度的错误统计。
        这有助于识别系统的薄弱环节和改进方向。

        Args:
            results: 评估结果列表，每个结果应包含:
                - success: 是否成功
                - errors: 错误列表
                - category: 任务类别
                - difficulty: 难度等级

        Returns:
            错误分析字典，包含:
            - error_rate: 总体错误率
            - error_types: 按类型分类的错误计数
            - error_by_category: 按类别分类的错误统计
            - error_by_difficulty: 按难度分类的错误统计

        Example:
            >>> collector = MetricsCollector()
            >>> results = [{"success": False, "errors": ["timeout"], "category": "code"}]
            >>> errors = collector.analyze_errors(results)
            >>> print(f"错误率: {errors['error_rate']*100:.1f}%")
        """
        if not results:
            return {
                "error_rate": 0.0,
                "error_types": {},
                "error_by_category": {},
                "error_by_difficulty": {},
            }

        # 统计失败任务
        failed_results = [r for r in results if not r.get("success", False)]
        error_rate = len(failed_results) / len(results) if results else 0.0

        # 错误类型分类
        error_types: dict[str, int] = defaultdict(int)
        for result in failed_results:
            errors = result.get("errors", [])
            if not errors:
                error_types["unknown"] += 1
                continue

            for error in errors:
                error_str = str(error).lower()
                classified = False

                for error_type, patterns in ERROR_TYPE_PATTERNS.items():
                    if any(pattern in error_str for pattern in patterns):
                        error_types[error_type] += 1
                        classified = True
                        break

                if not classified:
                    error_types["other"] += 1

        # 按类别统计错误
        error_by_category: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "failed": 0, "error_rate": 0.0}
        )
        for result in results:
            category = result.get("category", "unknown")
            error_by_category[category]["total"] += 1
            if not result.get("success", False):
                error_by_category[category]["failed"] += 1

        for _category, stats in error_by_category.items():
            stats["error_rate"] = (
                stats["failed"] / stats["total"] if stats["total"] > 0 else 0.0
            )

        # 按难度统计错误
        error_by_difficulty: dict[int | str, dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "failed": 0, "error_rate": 0.0}
        )
        for result in results:
            difficulty = result.get("difficulty", 0)
            error_by_difficulty[difficulty]["total"] += 1
            if not result.get("success", False):
                error_by_difficulty[difficulty]["failed"] += 1

        for _difficulty, stats in error_by_difficulty.items():
            stats["error_rate"] = (
                stats["failed"] / stats["total"] if stats["total"] > 0 else 0.0
            )

        return {
            "error_rate": error_rate,
            "error_types": dict(error_types),
            "error_by_category": dict(error_by_category),
            "error_by_difficulty": dict(error_by_difficulty),
        }

    def generate_anthropic_style_report(
        self, results: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """
        生成符合 Anthropic 标准的评估报告

        整合所有指标，生成一份全面的评估报告，包括:
        - 摘要统计
        - pass@k 和 pass^k 指标
        - 延迟分布
        - 成本分析
        - 错误分析
        - 按类别和难度的详细统计
        - 改进建议

        Args:
            results: 评估结果列表

        Returns:
            完整的评估报告字典

        Example:
            >>> collector = MetricsCollector(model_name="claude-3-opus")
            >>> results = [...]  # 评估结果
            >>> report = collector.generate_anthropic_style_report(results)
            >>> print(json.dumps(report, indent=2))
        """
        if not results:
            return {"error": "没有评估结果"}

        # 基础统计
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.get("success", False))
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0

        # 计算各项指标
        latency_dist = self.calculate_latency_distribution(results)
        cost_metrics = self.calculate_cost_metrics(results)
        error_analysis = self.analyze_errors(results)

        # 按类别统计
        by_category = self._aggregate_by_field(results, "category")

        # 按难度统计
        by_difficulty = self._aggregate_by_field(results, "difficulty")

        # 生成改进建议
        recommendations = self._generate_recommendations(
            success_rate, latency_dist, cost_metrics, error_analysis, by_category, by_difficulty
        )

        return {
            "summary": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "failed_tasks": total_tasks - successful_tasks,
                "success_rate": success_rate,
                "model_name": self.model_name,
                "timestamp": datetime.now().isoformat(),
            },
            "pass_metrics": {
                "pass_at_1": self.calculate_pass_at_k(results, k=1),
                "pass_at_3": self.calculate_pass_at_k(results, k=3),
                "pass_at_5": self.calculate_pass_at_k(results, k=5),
                "pass_power_3": self.calculate_pass_power_k(results, k=3),
                "pass_power_5": self.calculate_pass_power_k(results, k=5),
            },
            "latency": latency_dist,
            "cost": cost_metrics,
            "errors": error_analysis,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "recommendations": recommendations,
        }

    def _aggregate_by_field(
        self, results: list[dict[str, Any]], field: str
    ) -> dict[str, dict[str, Any]]:
        """
        按指定字段聚合结果

        Args:
            results: 评估结果列表
            field: 聚合字段名

        Returns:
            按字段值分组的统计信息
        """
        aggregated: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "total": 0,
                "success": 0,
                "success_rate": 0.0,
                "avg_latency": 0.0,
                "latencies": [],
            }
        )

        for result in results:
            key = str(result.get(field, "unknown"))
            aggregated[key]["total"] += 1
            if result.get("success", False):
                aggregated[key]["success"] += 1
            latency = result.get("execution_time", 0.0)
            if latency > 0:
                aggregated[key]["latencies"].append(latency)

        # 计算统计值
        for _key, stats in aggregated.items():
            stats["success_rate"] = (
                stats["success"] / stats["total"] if stats["total"] > 0 else 0.0
            )
            stats["avg_latency"] = (
                statistics.mean(stats["latencies"]) if stats["latencies"] else 0.0
            )
            # 移除临时数据
            del stats["latencies"]

        return dict(aggregated)

    def _generate_recommendations(
        self,
        success_rate: float,
        latency_dist: dict[str, float],
        cost_metrics: dict[str, float],
        error_analysis: dict[str, Any],
        by_category: dict[str, dict[str, Any]],
        by_difficulty: dict[str, dict[str, Any]],
    ) -> list[str]:
        """
        基于评估结果生成改进建议

        Args:
            success_rate: 成功率
            latency_dist: 延迟分布
            cost_metrics: 成本指标
            error_analysis: 错误分析
            by_category: 按类别统计
            by_difficulty: 按难度统计

        Returns:
            改进建议列表
        """
        recommendations = []

        # 成功率建议
        if success_rate < 0.5:
            recommendations.append(
                f"[Critical] 成功率过低 ({success_rate*100:.1f}%)，建议检查任务定义和模型能力匹配度"
            )
        elif success_rate < 0.7:
            recommendations.append(
                f"[Warning] 成功率偏低 ({success_rate*100:.1f}%)，建议优化 prompt 或增加示例"
            )
        elif success_rate < 0.9:
            recommendations.append(
                f"[Info] 成功率良好 ({success_rate*100:.1f}%)，可考虑针对失败案例进行优化"
            )

        # 延迟建议
        if latency_dist.get("p95", 0) > 60:
            recommendations.append(
                f"[Warning] P95 延迟过高 ({latency_dist['p95']:.1f}s)，建议优化任务复杂度或增加超时处理"
            )
        if latency_dist.get("std", 0) > latency_dist.get("mean", 1) * 0.5:
            recommendations.append(
                "[Info] 延迟波动较大，建议检查是否存在异常任务"
            )

        # 成本建议
        if cost_metrics.get("cost_per_success", 0) > 0.1:
            recommendations.append(
                f"[Warning] 每成功任务成本较高 (${cost_metrics['cost_per_success']:.4f})，"
                "建议优化 token 使用或考虑更经济的模型"
            )

        # 错误类型建议
        error_types = error_analysis.get("error_types", {})
        if error_types.get("timeout", 0) > 0:
            recommendations.append(
                f"[Warning] 存在 {error_types['timeout']} 个超时错误，建议增加超时时间或优化任务"
            )
        if error_types.get("tool_error", 0) > 0:
            recommendations.append(
                f"[Warning] 存在 {error_types['tool_error']} 个工具错误，建议检查工具实现"
            )

        # 类别建议
        weak_categories = [
            cat for cat, stats in by_category.items()
            if stats.get("success_rate", 1.0) < 0.5 and stats.get("total", 0) >= 3
        ]
        if weak_categories:
            recommendations.append(
                f"[Warning] 以下类别表现较差: {', '.join(weak_categories)}，建议针对性优化"
            )

        # 难度建议
        for diff, stats in by_difficulty.items():
            if stats.get("success_rate", 1.0) < 0.3 and stats.get("total", 0) >= 3:
                recommendations.append(
                    f"[Warning] 难度 {diff} 的任务成功率过低 ({stats['success_rate']*100:.1f}%)，"
                    "建议降低难度或增强模型能力"
                )

        if not recommendations:
            recommendations.append("[Success] 评估结果良好，未发现明显问题")

        return recommendations

    def save_anthropic_report(
        self, results: list[dict[str, Any]], filename: str | None = None
    ) -> Path:
        """
        保存 Anthropic 风格的评估报告

        Args:
            results: 评估结果列表
            filename: 输出文件名（可选）

        Returns:
            保存的文件路径
        """
        report = self.generate_anthropic_style_report(results)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"anthropic_report_{timestamp}.json"

        output_file = self.output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"\nAnthropic 风格报告已保存到: {output_file}")
        return output_file

    def print_anthropic_summary(self, results: list[dict[str, Any]]) -> None:
        """
        打印 Anthropic 风格的评估摘要

        Args:
            results: 评估结果列表
        """
        report = self.generate_anthropic_style_report(results)

        if "error" in report:
            print(f"错误: {report['error']}")
            return

        print("\n" + "=" * 70)
        print("Anthropic Style Evaluation Report")
        print("=" * 70)

        # Summary
        summary = report["summary"]
        print("\n[Summary]")
        print(f"  Model: {summary['model_name']}")
        print(f"  Total Tasks: {summary['total_tasks']}")
        print(f"  Success Rate: {summary['success_rate']*100:.1f}%")
        print(f"  Timestamp: {summary['timestamp']}")

        # Pass Metrics
        pass_metrics = report["pass_metrics"]
        print("\n[Pass Metrics]")
        print(f"  pass@1: {pass_metrics['pass_at_1']*100:.1f}%")
        print(f"  pass@3: {pass_metrics['pass_at_3']*100:.1f}%")
        print(f"  pass@5: {pass_metrics['pass_at_5']*100:.1f}%")
        print(f"  pass^3 (reliability): {pass_metrics['pass_power_3']*100:.1f}%")
        print(f"  pass^5 (reliability): {pass_metrics['pass_power_5']*100:.1f}%")

        # Latency
        latency = report["latency"]
        print("\n[Latency Distribution]")
        print(f"  P50 (median): {latency['p50']:.2f}s")
        print(f"  P95: {latency['p95']:.2f}s")
        print(f"  P99: {latency['p99']:.2f}s")
        print(f"  Mean: {latency['mean']:.2f}s")
        print(f"  Std: {latency['std']:.2f}s")

        # Cost
        cost = report["cost"]
        print("\n[Cost Analysis]")
        print(f"  Total Tokens: {cost['total_tokens']:,}")
        print(f"  Input Tokens: {cost['input_tokens']:,}")
        print(f"  Output Tokens: {cost['output_tokens']:,}")
        print(f"  Total Cost: ${cost['total_cost']:.4f}")
        print(f"  Cost per Task: ${cost['cost_per_task']:.4f}")
        print(f"  Cost per Success: ${cost['cost_per_success']:.4f}")

        # Errors
        errors = report["errors"]
        print("\n[Error Analysis]")
        print(f"  Error Rate: {errors['error_rate']*100:.1f}%")
        if errors["error_types"]:
            print("  Error Types:")
            for error_type, count in sorted(
                errors["error_types"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"    - {error_type}: {count}")

        # By Category
        if report["by_category"]:
            print("\n[By Category]")
            for category, stats in sorted(
                report["by_category"].items(),
                key=lambda x: x[1]["success_rate"],
                reverse=True,
            ):
                print(
                    f"  {category}: {stats['success']}/{stats['total']} "
                    f"({stats['success_rate']*100:.1f}%) - {stats['avg_latency']:.2f}s avg"
                )

        # By Difficulty
        if report["by_difficulty"]:
            print("\n[By Difficulty]")
            for difficulty, stats in sorted(report["by_difficulty"].items()):
                print(
                    f"  Level {difficulty}: {stats['success']}/{stats['total']} "
                    f"({stats['success_rate']*100:.1f}%) - {stats['avg_latency']:.2f}s avg"
                )

        # Recommendations
        print("\n[Recommendations]")
        for rec in report["recommendations"]:
            print(f"  {rec}")

        print("\n" + "=" * 70)

    def save_metrics(self, filename: str | None = None) -> Path:
        """
        保存指标到文件

        Args:
            filename: 输出文件名（可选）

        Returns:
            保存的文件路径
        """
        metrics = self.calculate_metrics()

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"

        output_file = self.output_dir / filename

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

        print(f"\n指标已保存到: {output_file}")
        return output_file

    def print_summary(self) -> None:
        """打印指标摘要"""
        metrics = self.calculate_metrics()

        if "error" in metrics:
            print(f"错误: {metrics['error']}")
            return

        print("\n" + "=" * 60)
        print("评估指标摘要")
        print("=" * 60)

        # 核心指标
        core = metrics["core_metrics"]
        print("\n核心指标:")
        print(f"  总任务数: {core['total_tasks']}")
        print(f"  成功任务: {core['successful_tasks']}")
        print(f"  失败任务: {core['failed_tasks']}")
        print(f"  成功率: {core['success_rate']*100:.1f}%")
        print(f"  平均耗时: {core['avg_execution_time']:.2f}s")
        print(f"  总耗时: {core['total_execution_time']:.2f}s")

        # 工具指标
        if metrics.get("tool_metrics"):
            tool = metrics["tool_metrics"]
            print("\n工具使用:")
            print(f"  总调用次数: {tool['total_tool_calls']}")
            print(f"  平均调用/任务: {tool['avg_tool_calls_per_task']:.1f}")
            print("  最多使用的工具:")
            for tool_name, count in tool["most_used_tools"]:
                print(f"    - {tool_name}: {count}")

        # 难度指标
        if metrics.get("difficulty_metrics"):
            print("\n按难度统计:")
            for difficulty, stats in sorted(
                metrics["difficulty_metrics"]["by_difficulty"].items()
            ):
                print(
                    f"  难度 {difficulty}: {stats['successful_tasks']}/{stats['total_tasks']} "
                    f"({stats['success_rate']*100:.1f}%) - {stats['avg_execution_time']:.2f}s"
                )

        # 类别指标
        if metrics.get("category_metrics"):
            print("\n按类别统计:")
            for category, stats in metrics["category_metrics"]["by_category"].items():
                print(
                    f"  {category}: {stats['successful_tasks']}/{stats['total_tasks']} "
                    f"({stats['success_rate']*100:.1f}%) - {stats['avg_execution_time']:.2f}s"
                )

        # LLM 指标
        if metrics.get("llm_metrics"):
            llm = metrics["llm_metrics"]
            print("\nLLM 评估分数:")
            print(f"  总体评分: {llm['avg_overall_score']:.2f}/10")
            print(f"  任务完成: {llm['avg_task_completion']:.2f}/10")
            print(f"  工具使用: {llm['avg_tool_usage']:.2f}/10")
            print(f"  代码质量: {llm['avg_code_quality']:.2f}/10")

        # 性能指标
        if metrics.get("performance_metrics"):
            perf = metrics["performance_metrics"]
            print("\n性能指标:")
            print(f"  最小耗时: {perf['min_time']:.2f}s")
            print(f"  最大耗时: {perf['max_time']:.2f}s")
            print(f"  平均耗时: {perf['avg_time']:.2f}s")
            print(f"  中位数: {perf['median_time']:.2f}s")
            print(f"  P95: {perf['p95_time']:.2f}s")
            print(f"  P99: {perf['p99_time']:.2f}s")

        # 错误指标
        if metrics.get("error_metrics") and metrics["error_metrics"]["total_errors"] > 0:
            err = metrics["error_metrics"]
            print("\n错误统计:")
            print(f"  总错误数: {err['total_errors']}")
            print(f"  错误率: {err['error_rate']*100:.1f}%")
            print("  错误类型:")
            for error_type, count in err["error_types"].items():
                print(f"    - {error_type}: {count}")


def main():
    """主函数"""
    print("评估指标收集器已准备就绪")
    print("\n使用方法:")
    print("  collector = MetricsCollector()")
    print("  collector.start_collection()")
    print("  # ... 运行评估 ...")
    print("  collector.record_task_result(result)")
    print("  collector.end_collection()")
    print("  collector.print_summary()")
    print("  collector.save_metrics()")


if __name__ == "__main__":
    main()
