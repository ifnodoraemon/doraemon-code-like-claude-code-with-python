"""
Agent 效果评估运行器

自动化评估 Agent 在各种任务上的表现
"""

import json
import time
from pathlib import Path
from typing import Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class AgentEvaluator:
    """Agent 评估器"""

    def __init__(self, task_file: str):
        self.tasks = self.load_tasks(task_file)
        self.results = []

    def load_tasks(self, task_file: str) -> List[Dict]:
        """加载评估任务"""
        with open(task_file) as f:
            return json.load(f)

    def evaluate_task(self, agent, task: Dict) -> Dict:
        """评估单个任务"""
        result = {
            "task_id": task.get("id", "unknown"),
            "task_name": task.get("name", task.get("id", "unknown")),
            "difficulty": task.get("difficulty", "medium"),
            "category": task.get("category", "general"),
            "success": False,
            "execution_time": 0,
            "tool_calls": [],
            "errors": [],
        }

        try:
            start_time = time.time()

            # 执行任务
            response = agent.execute(task["prompt"])

            result["execution_time"] = time.time() - start_time

            # 提取工具调用
            result["tool_calls"] = self.extract_tool_calls(response)

            # 检查成功标准
            result["success"] = self.check_success_criteria(
                response, task.get("success_criteria", [])
            )

            # 评估工具使用
            result["tool_usage"] = self.evaluate_tool_usage(
                result["tool_calls"], task.get("expected_tools", [])
            )

        except Exception as e:
            result["errors"].append(str(e))

        return result

    def extract_tool_calls(self, response) -> List[str]:
        """提取工具调用"""
        # 实现工具调用提取逻辑
        if hasattr(response, "tool_calls"):
            return [tc.name for tc in response.tool_calls]
        return []

    def check_success_criteria(self, response, criteria: List[str]) -> bool:
        """检查成功标准"""
        # 实现成功标准检查逻辑
        # 这里需要根据实际情况实现
        return True  # Placeholder

    def evaluate_tool_usage(
        self, actual_tools: List[str], expected_tools: List[str]
    ) -> Dict:
        """评估工具使用"""
        return {
            "expected_tools": expected_tools,
            "actual_tools": actual_tools,
            "tool_selection_accuracy": self.calculate_tool_accuracy(
                actual_tools, expected_tools
            ),
            "tool_count": len(actual_tools),
        }

    def calculate_tool_accuracy(
        self, actual: List[str], expected: List[str]
    ) -> float:
        """计算工具选择准确率"""
        if not expected:
            return 1.0

        correct = sum(1 for tool in expected if tool in actual)
        return correct / len(expected)

    def run_evaluation(self, agent) -> Dict:
        """运行完整评估"""
        print(f"开始评估 {len(self.tasks)} 个任务...")

        for task in self.tasks:
            task_name = task.get('name', task.get('id', 'Unknown'))
            print(f"\n评估任务: {task_name} (难度: {task['difficulty']})")
            result = self.evaluate_task(agent, task)
            self.results.append(result)

            status = "✅ 成功" if result["success"] else "❌ 失败"
            print(f"  {status} - 耗时: {result['execution_time']:.2f}s")

        return self.generate_report()

    def generate_report(self) -> Dict:
        """生成评估报告"""
        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results if r["success"])

        # 按类别统计
        by_category = {}
        for result in self.results:
            category = result["category"]
            if category not in by_category:
                by_category[category] = {"total": 0, "success": 0}
            by_category[category]["total"] += 1
            if result["success"]:
                by_category[category]["success"] += 1

        # 按难度统计
        by_difficulty = {}
        for result in self.results:
            difficulty = result["difficulty"]
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = {"total": 0, "success": 0}
            by_difficulty[difficulty]["total"] += 1
            if result["success"]:
                by_difficulty[difficulty]["success"] += 1

        report = {
            "summary": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
                "avg_execution_time": sum(r["execution_time"] for r in self.results)
                / total_tasks
                if total_tasks > 0
                else 0,
            },
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "failed_tasks": [r for r in self.results if not r["success"]],
        }

        return report

    def print_report(self, report: Dict):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("Agent 效果评估报告")
        print("=" * 60)

        summary = report["summary"]
        print(f"\n总体表现:")
        print(f"  总任务数: {summary['total_tasks']}")
        print(f"  成功任务: {summary['successful_tasks']}")
        print(f"  成功率: {summary['success_rate']*100:.1f}%")
        print(f"  平均耗时: {summary['avg_execution_time']:.2f}s")

        print(f"\n按类别统计:")
        for category, stats in report["by_category"].items():
            success_rate = stats["success"] / stats["total"] * 100
            print(f"  {category}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

        print(f"\n按难度统计:")
        for difficulty, stats in sorted(report["by_difficulty"].items()):
            success_rate = stats["success"] / stats["total"] * 100
            print(
                f"  难度 {difficulty}: {stats['success']}/{stats['total']} ({success_rate:.1f}%)"
            )

        if report["failed_tasks"]:
            print(f"\n失败任务:")
            for task in report["failed_tasks"]:
                print(f"  - {task['task_name']} (ID: {task['task_id']})")
                if task["errors"]:
                    print(f"    错误: {task['errors'][0]}")


def main():
    """主函数"""
    # 加载评估任务
    evaluator = AgentEvaluator("tests/evals/tasks/agent_eval_tasks.json")

    # 创建 Agent 实例
    # agent = create_agent()  # 需要实现

    # 运行评估
    # report = evaluator.run_evaluation(agent)

    # 打印报告
    # evaluator.print_report(report)

    # 保存报告
    # with open("eval_results/agent_evaluation_report.json", "w") as f:
    #     json.dump(report, f, indent=2)

    print("Agent 评估器已准备就绪")
    print("使用方法:")
    print("  evaluator = AgentEvaluator('tasks.json')")
    print("  report = evaluator.run_evaluation(agent)")
    print("  evaluator.print_report(report)")


if __name__ == "__main__":
    main()
