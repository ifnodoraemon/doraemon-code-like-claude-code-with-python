"""
基线评估运行器

运行基线评估并保存结果，用于后续对比
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class BaselineRunner:
    """基线评估运行器"""

    def __init__(self, output_dir: str = "eval_results/baseline"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.baseline_file = self.output_dir / "baseline.json"

    def run_baseline_evaluation(self, agent, task_files: list) -> Dict:
        """运行基线评估"""
        from tests.evals.agent_evaluator import AgentEvaluator

        print("=" * 60)
        print("开始基线评估")
        print("=" * 60)

        all_results = []
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tasks": 0,
            "successful_tasks": 0,
            "by_category": {},
            "by_difficulty": {},
        }

        for task_file in task_files:
            print(f"\n评估任务集: {task_file}")
            evaluator = AgentEvaluator(task_file)
            report = evaluator.run_evaluation(agent)

            all_results.extend(evaluator.results)

            # 更新汇总
            summary["total_tasks"] += report["summary"]["total_tasks"]
            summary["successful_tasks"] += report["summary"]["successful_tasks"]

        # 计算总体指标
        summary["success_rate"] = (
            summary["successful_tasks"] / summary["total_tasks"]
            if summary["total_tasks"] > 0
            else 0
        )

        # 按类别和难度统计
        for result in all_results:
            category = result["category"]
            difficulty = result["difficulty"]

            if category not in summary["by_category"]:
                summary["by_category"][category] = {"total": 0, "success": 0}
            summary["by_category"][category]["total"] += 1
            if result["success"]:
                summary["by_category"][category]["success"] += 1

            if difficulty not in summary["by_difficulty"]:
                summary["by_difficulty"][difficulty] = {"total": 0, "success": 0}
            summary["by_difficulty"][difficulty]["total"] += 1
            if result["success"]:
                summary["by_difficulty"][difficulty]["success"] += 1

        # 保存基线
        self.save_baseline(summary, all_results)

        return summary

    def save_baseline(self, summary: Dict, results: list):
        """保存基线结果"""
        baseline_data = {
            "summary": summary,
            "results": results,
            "metrics": self.calculate_metrics(results),
        }

        with open(self.baseline_file, "w") as f:
            json.dump(baseline_data, f, indent=2)

        print(f"\n基线已保存到: {self.baseline_file}")

    def calculate_metrics(self, results: list) -> Dict:
        """计算关键指标"""
        if not results:
            return {}

        successful = [r for r in results if r["success"]]

        metrics = {
            "success_rate": len(successful) / len(results),
            "avg_execution_time": sum(r["execution_time"] for r in results)
            / len(results),
            "avg_tool_calls": sum(len(r.get("tool_calls", [])) for r in results)
            / len(results),
        }

        # 按难度的成功率
        by_difficulty = {}
        for result in results:
            diff = result["difficulty"]
            if diff not in by_difficulty:
                by_difficulty[diff] = {"total": 0, "success": 0}
            by_difficulty[diff]["total"] += 1
            if result["success"]:
                by_difficulty[diff]["success"] += 1

        metrics["success_by_difficulty"] = {
            diff: stats["success"] / stats["total"]
            for diff, stats in by_difficulty.items()
        }

        return metrics

    def load_baseline(self) -> Dict:
        """加载基线数据"""
        if not self.baseline_file.exists():
            return None

        with open(self.baseline_file) as f:
            return json.load(f)

    def compare_with_baseline(self, current_results: Dict) -> Dict:
        """与基线对比"""
        baseline = self.load_baseline()
        if not baseline:
            return {"error": "没有基线数据"}

        comparison = {
            "baseline_date": baseline["summary"]["timestamp"],
            "current_date": current_results["timestamp"],
            "improvements": [],
            "regressions": [],
        }

        # 对比成功率
        baseline_rate = baseline["summary"]["success_rate"]
        current_rate = current_results["success_rate"]
        rate_diff = current_rate - baseline_rate

        if rate_diff > 0.05:  # 提升超过 5%
            comparison["improvements"].append(
                f"成功率提升: {baseline_rate:.1%} → {current_rate:.1%} (+{rate_diff:.1%})"
            )
        elif rate_diff < -0.05:  # 下降超过 5%
            comparison["regressions"].append(
                f"成功率下降: {baseline_rate:.1%} → {current_rate:.1%} ({rate_diff:.1%})"
            )

        return comparison

    def print_baseline_report(self, summary: Dict):
        """打印基线报告"""
        print("\n" + "=" * 60)
        print("基线评估报告")
        print("=" * 60)

        print(f"\n评估时间: {summary['timestamp']}")
        print(f"总任务数: {summary['total_tasks']}")
        print(f"成功任务: {summary['successful_tasks']}")
        print(f"成功率: {summary['success_rate']*100:.1f}%")

        print(f"\n按类别统计:")
        for category, stats in summary["by_category"].items():
            rate = stats["success"] / stats["total"] * 100
            print(f"  {category}: {stats['success']}/{stats['total']} ({rate:.1f}%)")

        print(f"\n按难度统计:")
        for difficulty, stats in sorted(summary["by_difficulty"].items()):
            rate = stats["success"] / stats["total"] * 100
            print(f"  难度 {difficulty}: {stats['success']}/{stats['total']} ({rate:.1f}%)")


def main():
    """主函数"""
    print("基线评估运行器已准备就绪")
    print("\n使用方法:")
    print("  runner = BaselineRunner()")
    print("  summary = runner.run_baseline_evaluation(agent, task_files)")
    print("  runner.print_baseline_report(summary)")


if __name__ == "__main__":
    main()
