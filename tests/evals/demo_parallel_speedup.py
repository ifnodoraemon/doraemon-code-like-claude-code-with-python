#!/usr/bin/env python3
"""
并行加速演示脚本

对比串行 vs 并行评估的性能差异
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.evals.parallel_evaluator import ParallelEvaluator


class MockAgent:
    """模拟 Agent，用于演示"""

    def __init__(self, agent_id: int = 0):
        self.agent_id = agent_id
        self.call_count = 0

    async def execute_task(self, task: Dict) -> Dict:
        """模拟执行任务"""
        self.call_count += 1

        # 模拟不同难度的任务耗时
        difficulty = task.get("difficulty", "easy")
        if difficulty == "easy":
            await asyncio.sleep(0.5)  # 0.5秒
        elif difficulty == "medium":
            await asyncio.sleep(1.0)  # 1秒
        elif difficulty == "hard":
            await asyncio.sleep(2.0)  # 2秒
        else:
            await asyncio.sleep(1.5)  # 1.5秒

        # 模拟成功率（90%）
        import random

        success = random.random() < 0.9

        return {
            "task_id": task.get("id", "unknown"),
            "success": success,
            "execution_time": 0.5 if difficulty == "easy" else 1.0,
            "agent_id": self.agent_id,
        }


def create_mock_agent(agent_id: int = 0):
    """创建模拟 Agent"""
    return MockAgent(agent_id)


def load_tasks(task_file: str) -> List[Dict]:
    """加载任务"""
    with open(task_file, "r", encoding="utf-8") as f:
        return json.load(f)


def run_serial_evaluation(tasks: List[Dict]) -> Dict:
    """串行评估"""
    print("\n" + "=" * 80)
    print("🐢 串行评估 (Serial Evaluation)")
    print("=" * 80)

    agent = create_mock_agent()
    results = []

    start_time = time.time()

    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] 执行任务: {task.get('id', 'unknown')}")
        result = asyncio.run(agent.execute_task(task))
        results.append(result)

    total_time = time.time() - start_time

    success_count = sum(1 for r in results if r["success"])
    print(f"\n✅ 完成: {success_count}/{len(tasks)} 成功")
    print(f"⏱️  总耗时: {total_time:.2f}秒")

    return {
        "mode": "serial",
        "total_time": total_time,
        "tasks": len(tasks),
        "success_count": success_count,
        "throughput": len(tasks) / total_time,
    }


def run_parallel_evaluation(tasks: List[Dict], max_workers: int = 4) -> Dict:
    """并行评估"""
    print("\n" + "=" * 80)
    print(f"🚀 并行评估 (Parallel Evaluation) - {max_workers} workers")
    print("=" * 80)

    evaluator = ParallelEvaluator(
        max_workers=max_workers, use_processes=False, output_dir="eval_results/demo"
    )

    # 创建临时任务文件
    temp_dir = Path("eval_results/demo/temp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = temp_dir / "tasks.json"

    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    start_time = time.time()

    # 运行并行评估
    result = evaluator.run_parallel_evaluation(
        agent_factory=create_mock_agent, task_files=[str(temp_file)], n_trials=1
    )

    total_time = time.time() - start_time

    success_count = result.get("summary", {}).get("successful_tasks", 0)
    print(f"\n✅ 完成: {success_count}/{len(tasks)} 成功")
    print(f"⏱️  总耗时: {total_time:.2f}秒")

    # 清理临时文件
    temp_file.unlink()

    return {
        "mode": f"parallel_{max_workers}",
        "total_time": total_time,
        "tasks": len(tasks),
        "success_count": success_count,
        "throughput": len(tasks) / total_time,
    }


def print_comparison(results: List[Dict]):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("📊 性能对比 (Performance Comparison)")
    print("=" * 80)

    # 找到基准（串行）
    baseline = next((r for r in results if r["mode"] == "serial"), None)
    if not baseline:
        print("❌ 未找到串行评估结果")
        return

    baseline_time = baseline["total_time"]

    print(f"\n{'模式':<20} {'耗时(秒)':<12} {'吞吐量(任务/秒)':<18} {'加速比':<10}")
    print("-" * 80)

    for result in results:
        mode = result["mode"]
        total_time = result["total_time"]
        throughput = result["throughput"]
        speedup = baseline_time / total_time

        print(
            f"{mode:<20} {total_time:>10.2f}s  {throughput:>15.2f}  {speedup:>8.2f}x"
        )

    # 计算最佳加速比
    best_result = max(results, key=lambda r: baseline_time / r["total_time"])
    best_speedup = baseline_time / best_result["total_time"]

    print("\n" + "=" * 80)
    print(f"🏆 最佳加速: {best_result['mode']} - {best_speedup:.2f}x 加速")
    print(
        f"⚡ 时间节省: {baseline_time - best_result['total_time']:.2f}秒 "
        f"({(1 - best_result['total_time']/baseline_time)*100:.1f}%)"
    )
    print("=" * 80)


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🎯 Doraemon Code 并行评估加速演示")
    print("=" * 80)

    # 加载任务
    task_file = "tests/evals/tasks/basic/file_and_code_tasks.json"
    if not Path(task_file).exists():
        print(f"❌ 任务文件不存在: {task_file}")
        return

    tasks = load_tasks(task_file)
    print(f"\n📋 加载任务: {len(tasks)} 个")

    # 运行评估
    results = []

    # 1. 串行评估
    serial_result = run_serial_evaluation(tasks)
    results.append(serial_result)

    # 2. 并行评估 (2 workers)
    parallel_2_result = run_parallel_evaluation(tasks, max_workers=2)
    results.append(parallel_2_result)

    # 3. 并行评估 (4 workers)
    parallel_4_result = run_parallel_evaluation(tasks, max_workers=4)
    results.append(parallel_4_result)

    # 4. 并行评估 (8 workers)
    parallel_8_result = run_parallel_evaluation(tasks, max_workers=8)
    results.append(parallel_8_result)

    # 打印对比
    print_comparison(results)

    # 保存结果
    output_file = Path("eval_results/demo/speedup_comparison.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "task_count": len(tasks),
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n💾 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
