"""
多模型对比评估器

对比不同模型在相同任务上的表现
"""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MultiModelComparator:
    """多模型对比器"""

    MODELS = [
        "gemini-2.0-flash-exp",
        "gpt-4-turbo",
        "claude-3-5-sonnet-20241022",
    ]

    def __init__(self, output_dir: str = "eval_results/model_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compare_models(self, task_files: list[str]) -> dict:
        """对比多个模型"""
        from tests.evals.agent_evaluator import AgentEvaluator

        print("=" * 60)
        print("多模型对比评估")
        print("=" * 60)

        comparison_results = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
        }

        for model in self.MODELS:
            print(f"\n评估模型: {model}")
            print("-" * 60)

            # 创建使用该模型的 Agent
            # agent = create_agent(model=model)  # 需要实现

            model_results = {
                "model": model,
                "total_tasks": 0,
                "successful_tasks": 0,
                "by_category": {},
                "by_difficulty": {},
                "avg_execution_time": 0,
            }

            # 运行评估
            for task_file in task_files:
                AgentEvaluator(task_file)
                # report = evaluator.run_evaluation(agent)  # 需要实现

                # 更新结果
                # model_results["total_tasks"] += report["summary"]["total_tasks"]
                # model_results["successful_tasks"] += report["summary"]["successful_tasks"]

            comparison_results["models"][model] = model_results

        # 生成对比报告
        self.save_comparison(comparison_results)
        return comparison_results

    def save_comparison(self, results: dict):
        """保存对比结果"""
        output_file = (
            self.output_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n对比结果已保存到: {output_file}")

    def generate_comparison_report(self, results: dict) -> str:
        """生成对比报告"""
        report = []
        report.append("# 多模型对比报告\n")
        report.append(f"评估时间: {results['timestamp']}\n")
        report.append("\n## 总体对比\n")

        # 创建对比表格
        report.append("| 模型 | 成功率 | 平均耗时 | 总任务数 |")
        report.append("|------|--------|----------|----------|")

        for model, data in results["models"].items():
            success_rate = (
                data["successful_tasks"] / data["total_tasks"] * 100
                if data["total_tasks"] > 0
                else 0
            )
            report.append(
                f"| {model} | {success_rate:.1f}% | {data['avg_execution_time']:.2f}s | {data['total_tasks']} |"
            )

        # 按类别对比
        report.append("\n## 按类别对比\n")

        all_categories = set()
        for data in results["models"].values():
            all_categories.update(data["by_category"].keys())

        for category in sorted(all_categories):
            report.append(f"\n### {category}\n")
            report.append("| 模型 | 成功率 |")
            report.append("|------|--------|")

            for model, data in results["models"].items():
                if category in data["by_category"]:
                    stats = data["by_category"][category]
                    rate = stats["success"] / stats["total"] * 100
                    report.append(f"| {model} | {rate:.1f}% |")

        # 最佳模型推荐
        report.append("\n## 最佳模型推荐\n")

        best_overall = max(
            results["models"].items(),
            key=lambda x: x[1]["successful_tasks"] / x[1]["total_tasks"]
            if x[1]["total_tasks"] > 0
            else 0,
        )
        report.append(f"- **总体最佳**: {best_overall[0]}\n")

        # 按类别推荐
        for category in sorted(all_categories):
            best_for_category = None
            best_rate = 0

            for model, data in results["models"].items():
                if category in data["by_category"]:
                    stats = data["by_category"][category]
                    rate = stats["success"] / stats["total"]
                    if rate > best_rate:
                        best_rate = rate
                        best_for_category = model

            if best_for_category:
                report.append(f"- **{category}**: {best_for_category}\n")

        return "\n".join(report)

    def print_comparison(self, results: dict):
        """打印对比结果"""
        print("\n" + "=" * 60)
        print("多模型对比结果")
        print("=" * 60)

        for model, data in results["models"].items():
            success_rate = (
                data["successful_tasks"] / data["total_tasks"] * 100
                if data["total_tasks"] > 0
                else 0
            )
            print(f"\n{model}:")
            print(f"  成功率: {success_rate:.1f}%")
            print(f"  成功任务: {data['successful_tasks']}/{data['total_tasks']}")
            print(f"  平均耗时: {data['avg_execution_time']:.2f}s")


def main():
    """主函数"""
    print("多模型对比评估器已准备就绪")
    print("\n支持的模型:")
    for model in MultiModelComparator.MODELS:
        print(f"  - {model}")
    print("\n使用方法:")
    print("  comparator = MultiModelComparator()")
    print("  results = comparator.compare_models(task_files)")
    print("  comparator.print_comparison(results)")


if __name__ == "__main__":
    main()
