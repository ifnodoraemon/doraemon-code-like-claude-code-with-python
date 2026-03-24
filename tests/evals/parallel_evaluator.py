"""
并行评估运行器

支持并行执行多个评估任务，提高评估效率
"""

import asyncio
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ParallelEvaluator:
    """并行评估器 - 支持多进程和多线程并行执行"""

    def __init__(
        self,
        max_workers: int = 4,
        use_processes: bool = False,
        output_dir: str = "eval_results/parallel",
    ):
        """
        初始化并行评估器

        Args:
            max_workers: 最大并行工作数
            use_processes: 是否使用多进程（默认使用多线程）
            output_dir: 输出目录
        """
        self.max_workers = max_workers
        self.use_processes = use_processes
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_parallel_evaluation(
        self, agent_factory, task_files: list[str], n_trials: int = 1
    ) -> dict:
        """
        并行运行评估

        Args:
            agent_factory: Agent 工厂函数，用于创建 Agent 实例
            task_files: 任务文件列表
            n_trials: 每个任务运行的次数

        Returns:
            评估结果汇总
        """
        from tests.evals.agent_evaluator import AgentEvaluator

        print("=" * 60)
        print(f"并行评估 (workers={self.max_workers}, trials={n_trials})")
        print("=" * 60)

        start_time = time.time()

        # 加载所有任务
        all_tasks = []
        for task_file in task_files:
            evaluator = AgentEvaluator(task_file)
            for task in evaluator.tasks:
                for trial in range(n_trials):
                    all_tasks.append(
                        {
                            "task": task,
                            "trial": trial,
                            "task_file": task_file,
                        }
                    )

        print(f"\n总任务数: {len(all_tasks)}")
        print(f"并行度: {self.max_workers}")

        # 选择执行器
        executor_class = ProcessPoolExecutor if self.use_processes else ThreadPoolExecutor

        # 并行执行
        results = []
        with executor_class(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._evaluate_single_task, agent_factory, task_data)
                for task_data in all_tasks
            ]

            # 收集结果
            completed = 0
            for future in futures:
                try:
                    result = future.result(timeout=300)  # 5分钟超时
                    results.append(result)
                    completed += 1

                    if completed % 10 == 0:
                        print(f"进度: {completed}/{len(all_tasks)}")

                except Exception as e:
                    print(f"任务执行失败: {e}")
                    results.append(
                        {
                            "success": False,
                            "error": str(e),
                        }
                    )

        total_time = time.time() - start_time

        # 生成汇总报告
        summary = self._generate_summary(results, total_time, n_trials)

        # 保存结果
        self._save_results(summary, results)

        return summary

    async def run_async_evaluation(
        self, agent_factory, task_files: list[str], n_trials: int = 1
    ) -> dict:
        """
        异步并行运行评估（使用 asyncio）

        Args:
            agent_factory: Agent 工厂函数
            task_files: 任务文件列表
            n_trials: 每个任务运行的次数

        Returns:
            评估结果汇总
        """
        from tests.evals.agent_evaluator import AgentEvaluator

        print("=" * 60)
        print(f"异步并行评估 (concurrency={self.max_workers}, trials={n_trials})")
        print("=" * 60)

        start_time = time.time()

        # 加载所有任务
        all_tasks = []
        for task_file in task_files:
            evaluator = AgentEvaluator(task_file)
            for task in evaluator.tasks:
                for trial in range(n_trials):
                    all_tasks.append(
                        {
                            "task": task,
                            "trial": trial,
                            "task_file": task_file,
                        }
                    )

        print(f"\n总任务数: {len(all_tasks)}")
        print(f"并发度: {self.max_workers}")

        # 使用信号量限制并发
        semaphore = asyncio.Semaphore(self.max_workers)

        async def evaluate_with_semaphore(task_data):
            async with semaphore:
                return await self._evaluate_single_task_async(agent_factory, task_data)

        # 并发执行
        tasks = [evaluate_with_semaphore(task_data) for task_data in all_tasks]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(
                    {
                        "success": False,
                        "error": str(result),
                    }
                )
            else:
                processed_results.append(result)

        total_time = time.time() - start_time

        # 生成汇总报告
        summary = self._generate_summary(processed_results, total_time, n_trials)

        # 保存结果
        self._save_results(summary, processed_results)

        return summary

    def _evaluate_single_task(self, agent_factory, task_data: dict) -> dict:
        """评估单个任务（同步版本）"""
        from tests.evals.agent_evaluator import AgentEvaluator

        task = task_data["task"]
        trial = task_data["trial"]

        try:
            # 创建 Agent 实例
            agent = agent_factory()

            # 创建评估器
            evaluator = AgentEvaluator(task_data["task_file"])

            # 评估任务
            result = evaluator.evaluate_task(agent, task)
            result["trial"] = trial

            return result

        except Exception as e:
            return {
                "task_id": task.get("id", "unknown"),
                "trial": trial,
                "success": False,
                "error": str(e),
            }

    async def _evaluate_single_task_async(self, agent_factory, task_data: dict) -> dict:
        """评估单个任务（异步版本）"""
        # 在异步环境中运行同步代码
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._evaluate_single_task, agent_factory, task_data
        )

    def _generate_summary(self, results: list[dict], total_time: float, n_trials: int) -> dict:
        """生成评估汇总"""
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.get("success", False))

        # 按任务 ID 分组（用于多次试验）
        by_task_id = {}
        for result in results:
            task_id = result.get("task_id", "unknown")
            if task_id not in by_task_id:
                by_task_id[task_id] = []
            by_task_id[task_id].append(result)

        # 计算每个任务的成功率
        task_success_rates = {}
        for task_id, task_results in by_task_id.items():
            successes = sum(1 for r in task_results if r.get("success", False))
            task_success_rates[task_id] = successes / len(task_results)

        # 按类别统计
        by_category = {}
        for result in results:
            category = result.get("category", "unknown")
            if category not in by_category:
                by_category[category] = {"total": 0, "success": 0}
            by_category[category]["total"] += 1
            if result.get("success", False):
                by_category[category]["success"] += 1

        # 按难度统计
        by_difficulty = {}
        for result in results:
            difficulty = result.get("difficulty", 0)
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = {"total": 0, "success": 0}
            by_difficulty[difficulty]["total"] += 1
            if result.get("success", False):
                by_difficulty[difficulty]["success"] += 1

        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_time": total_time,
            "total_tasks": total_tasks,
            "unique_tasks": len(by_task_id),
            "n_trials": n_trials,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "avg_time_per_task": total_time / total_tasks if total_tasks > 0 else 0,
            "task_success_rates": task_success_rates,
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "parallelization": {
                "max_workers": self.max_workers,
                "use_processes": self.use_processes,
                "speedup": self._estimate_speedup(results, total_time),
            },
        }

        return summary

    def _estimate_speedup(self, results: list[dict], total_time: float) -> float:
        """估算并行加速比"""
        # 计算串行执行时间（所有任务执行时间之和）
        serial_time = sum(r.get("execution_time", 0) for r in results)

        if total_time > 0:
            return serial_time / total_time
        return 1.0

    def _save_results(self, summary: dict, results: list[dict]):
        """保存评估结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存汇总
        summary_file = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # 保存详细结果
        results_file = self.output_dir / f"results_{timestamp}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print("\n结果已保存:")
        print(f"  汇总: {summary_file}")
        print(f"  详细: {results_file}")

    def print_summary(self, summary: dict):
        """打印评估汇总"""
        print("\n" + "=" * 60)
        print("并行评估汇总")
        print("=" * 60)

        print("\n执行信息:")
        print(f"  评估时间: {summary['timestamp']}")
        print(f"  总耗时: {summary['total_time']:.2f}s")
        print(f"  并行度: {summary['parallelization']['max_workers']}")
        print(f"  加速比: {summary['parallelization']['speedup']:.2f}x")

        print("\n任务统计:")
        print(f"  总任务数: {summary['total_tasks']}")
        print(f"  独立任务: {summary['unique_tasks']}")
        print(f"  试验次数: {summary['n_trials']}")
        print(f"  成功任务: {summary['successful_tasks']}")
        print(f"  成功率: {summary['success_rate'] * 100:.1f}%")
        print(f"  平均耗时: {summary['avg_time_per_task']:.2f}s/任务")

        print("\n按类别统计:")
        for category, stats in summary["by_category"].items():
            rate = stats["success"] / stats["total"] * 100
            print(f"  {category}: {stats['success']}/{stats['total']} ({rate:.1f}%)")

        print("\n按难度统计:")
        for difficulty, stats in sorted(summary["by_difficulty"].items()):
            rate = stats["success"] / stats["total"] * 100
            print(f"  难度 {difficulty}: {stats['success']}/{stats['total']} ({rate:.1f}%)")


def main():
    """主函数"""
    print("并行评估器已准备就绪")
    print("\n使用方法:")
    print("\n1. 多线程并行:")
    print("  evaluator = ParallelEvaluator(max_workers=4)")
    print("  summary = evaluator.run_parallel_evaluation(agent_factory, task_files)")
    print("\n2. 多进程并行:")
    print("  evaluator = ParallelEvaluator(max_workers=4, use_processes=True)")
    print("  summary = evaluator.run_parallel_evaluation(agent_factory, task_files)")
    print("\n3. 异步并行:")
    print("  evaluator = ParallelEvaluator(max_workers=10)")
    print("  summary = await evaluator.run_async_evaluation(agent_factory, task_files)")


if __name__ == "__main__":
    main()
