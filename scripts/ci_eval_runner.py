#!/usr/bin/env python3
"""
CI Evaluation Runner

专门用于 CI/CD 环境的评估脚本，支持：
- 不同评估范围（basic/full/quick）
- 回归检测
- JSON 输出格式
- 退出码表示成功/失败
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_task_files(scope: str) -> list:
    """根据评估范围获取任务文件"""
    base_dir = Path(__file__).parent.parent / "tests" / "evals" / "tasks"

    scope_mapping = {
        "quick": ["basic"],
        "basic": ["basic", "intermediate"],
        "full": ["basic", "intermediate", "advanced", "expert", "realistic", "complex", "negative"],
    }

    dirs = scope_mapping.get(scope, ["basic"])
    task_files = []

    for dir_name in dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            for f in dir_path.glob("*.json"):
                task_files.append(str(f))

    return task_files


def run_evaluation(scope: str, output_dir: str) -> dict:
    """运行评估"""
    from tests.evals.comprehensive_evaluator import ComprehensiveEvaluator
    from tests.evals.agent_evaluator import AgentEvaluator

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 获取任务文件
    task_files = get_task_files(scope)

    if not task_files:
        return {
            "success": False,
            "error": f"No task files found for scope: {scope}",
        }

    # 创建 Mock Agent（CI 环境中使用）
    class MockAgent:
        def execute(self, prompt: str):
            import time
            import random

            time.sleep(random.uniform(0.1, 0.3))

            class Response:
                def __init__(self):
                    self.content = f"Executed: {prompt[:50]}..."
                    self.tool_calls = []

            return Response()

    def create_mock_agent():
        return MockAgent()

    # 运行评估
    evaluator = ComprehensiveEvaluator(
        parallel=False,  # CI 中使用串行以获得稳定结果
        num_trials=1,
    )

    try:
        report = evaluator.run_full_evaluation(create_mock_agent, task_files)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }

    # 提取关键指标
    summary = {
        "timestamp": datetime.now().isoformat(),
        "scope": scope,
        "success_rate": report.get("overall_success_rate", 0),
        "total_tasks": report.get("total_tasks", 0),
        "avg_latency": report.get("avg_execution_time", 0),
        "pass_at_1": report.get("pass_at_1", 0),
        "by_difficulty": report.get("by_difficulty", {}),
        "by_category": report.get("by_category", {}),
    }

    # 保存结果
    summary_file = output_path / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    report_file = output_path / "full_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    return summary


def check_regression(baseline_branch: str) -> dict:
    """检查性能回归"""
    # 读取当前评估结果
    current_file = Path("eval_results/summary.json")
    if not current_file.exists():
        return {"regression": False, "message": "No current results to compare"}

    with open(current_file) as f:
        current = json.load(f)

    # 尝试读取基线结果（从 artifacts 或历史记录）
    baseline_file = Path(f"eval_results/baseline/{baseline_branch}.json")
    if not baseline_file.exists():
        return {"regression": False, "message": f"No baseline found for {baseline_branch}"}

    with open(baseline_file) as f:
        baseline = json.load(f)

    # 比较关键指标
    regression_threshold = 0.05  # 5% 下降视为回归

    current_rate = current.get("success_rate", 0)
    baseline_rate = baseline.get("success_rate", 0)

    if baseline_rate > 0:
        change = (current_rate - baseline_rate) / baseline_rate
        if change < -regression_threshold:
            return {
                "regression": True,
                "message": f"Performance regression detected: {change:.1%} change in success rate",
                "current": current_rate,
                "baseline": baseline_rate,
            }

    return {
        "regression": False,
        "message": "No regression detected",
        "current": current_rate,
        "baseline": baseline_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="CI Evaluation Runner")
    parser.add_argument(
        "--scope",
        choices=["quick", "basic", "full"],
        default="basic",
        help="Evaluation scope",
    )
    parser.add_argument(
        "--output",
        default="eval_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Check for performance regression",
    )
    parser.add_argument(
        "--baseline",
        default="main",
        help="Baseline branch for regression check",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    args = parser.parse_args()

    if args.check_regression:
        result = check_regression(args.baseline)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(result["message"])
        sys.exit(1 if result.get("regression") else 0)

    # Run evaluation
    print(f"Running {args.scope} evaluation...")
    result = run_evaluation(args.scope, args.output)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result.get("success") is False:
            print(f"Evaluation failed: {result.get('error')}")
            sys.exit(1)

        print(f"\n{'=' * 50}")
        print("Evaluation Complete")
        print(f"{'=' * 50}")
        print(f"Scope: {args.scope}")
        print(f"Success Rate: {result.get('success_rate', 0):.1%}")
        print(f"Total Tasks: {result.get('total_tasks', 0)}")
        print(f"Avg Latency: {result.get('avg_latency', 0):.2f}s")
        print(f"Results saved to: {args.output}/")

    # Exit with appropriate code
    success_rate = result.get("success_rate", 0)
    if success_rate < 0.5:  # Fail if success rate below 50%
        sys.exit(1)


if __name__ == "__main__":
    main()
