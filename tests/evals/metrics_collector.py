"""
评估指标收集器

收集和分析评估过程中的各种指标
"""

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MetricsCollector:
    """评估指标收集器"""

    def __init__(self, output_dir: str = "eval_results/metrics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = defaultdict(list)
        self.start_time = None
        self.end_time = None

    def start_collection(self):
        """开始收集指标"""
        self.start_time = time.time()
        self.metrics.clear()

    def end_collection(self):
        """结束收集指标"""
        self.end_time = time.time()

    def record_task_result(self, result: Dict):
        """记录任务结果"""
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

    def calculate_metrics(self) -> Dict:
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

    def _calculate_tool_metrics(self) -> Dict:
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

    def _calculate_difficulty_metrics(self) -> Dict:
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

        return {
            "avg_difficulty": sum(self.metrics["difficulty"])
            / len(self.metrics["difficulty"])
            if self.metrics["difficulty"]
            else 0,
            "by_difficulty": difficulty_stats,
        }

    def _calculate_category_metrics(self) -> Dict:
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

    def _calculate_llm_metrics(self) -> Dict:
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

    def _calculate_score_distribution(self, scores: List[float]) -> Dict:
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

    def _calculate_performance_metrics(self) -> Dict:
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

    def _calculate_error_metrics(self) -> Dict:
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

    def save_metrics(self, filename: Optional[str] = None):
        """保存指标到文件"""
        metrics = self.calculate_metrics()

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"

        output_file = self.output_dir / filename

        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n指标已保存到: {output_file}")
        return output_file

    def print_summary(self):
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
        print(f"\n核心指标:")
        print(f"  总任务数: {core['total_tasks']}")
        print(f"  成功任务: {core['successful_tasks']}")
        print(f"  失败任务: {core['failed_tasks']}")
        print(f"  成功率: {core['success_rate']*100:.1f}%")
        print(f"  平均耗时: {core['avg_execution_time']:.2f}s")
        print(f"  总耗时: {core['total_execution_time']:.2f}s")

        # 工具指标
        if metrics.get("tool_metrics"):
            tool = metrics["tool_metrics"]
            print(f"\n工具使用:")
            print(f"  总调用次数: {tool['total_tool_calls']}")
            print(f"  平均调用/任务: {tool['avg_tool_calls_per_task']:.1f}")
            print(f"  最多使用的工具:")
            for tool_name, count in tool["most_used_tools"]:
                print(f"    - {tool_name}: {count}")

        # 难度指标
        if metrics.get("difficulty_metrics"):
            print(f"\n按难度统计:")
            for difficulty, stats in sorted(
                metrics["difficulty_metrics"]["by_difficulty"].items()
            ):
                print(
                    f"  难度 {difficulty}: {stats['successful_tasks']}/{stats['total_tasks']} "
                    f"({stats['success_rate']*100:.1f}%) - {stats['avg_execution_time']:.2f}s"
                )

        # 类别指标
        if metrics.get("category_metrics"):
            print(f"\n按类别统计:")
            for category, stats in metrics["category_metrics"]["by_category"].items():
                print(
                    f"  {category}: {stats['successful_tasks']}/{stats['total_tasks']} "
                    f"({stats['success_rate']*100:.1f}%) - {stats['avg_execution_time']:.2f}s"
                )

        # LLM 指标
        if metrics.get("llm_metrics"):
            llm = metrics["llm_metrics"]
            print(f"\nLLM 评估分数:")
            print(f"  总体评分: {llm['avg_overall_score']:.2f}/10")
            print(f"  任务完成: {llm['avg_task_completion']:.2f}/10")
            print(f"  工具使用: {llm['avg_tool_usage']:.2f}/10")
            print(f"  代码质量: {llm['avg_code_quality']:.2f}/10")

        # 性能指标
        if metrics.get("performance_metrics"):
            perf = metrics["performance_metrics"]
            print(f"\n性能指标:")
            print(f"  最小耗时: {perf['min_time']:.2f}s")
            print(f"  最大耗时: {perf['max_time']:.2f}s")
            print(f"  平均耗时: {perf['avg_time']:.2f}s")
            print(f"  中位数: {perf['median_time']:.2f}s")
            print(f"  P95: {perf['p95_time']:.2f}s")
            print(f"  P99: {perf['p99_time']:.2f}s")

        # 错误指标
        if metrics.get("error_metrics") and metrics["error_metrics"]["total_errors"] > 0:
            err = metrics["error_metrics"]
            print(f"\n错误统计:")
            print(f"  总错误数: {err['total_errors']}")
            print(f"  错误率: {err['error_rate']*100:.1f}%")
            print(f"  错误类型:")
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
