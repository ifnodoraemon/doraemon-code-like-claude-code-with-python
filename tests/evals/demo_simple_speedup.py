#!/usr/bin/env python3
"""
简单的并行加速演示

直接对比串行 vs 并行执行的性能差异
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor


def simulate_task(task_id: int, duration: float = 1.0) -> dict:
    """模拟一个耗时任务"""
    time.sleep(duration)
    return {
        "task_id": task_id,
        "duration": duration,
        "success": True,
    }


async def simulate_task_async(task_id: int, duration: float = 1.0) -> dict:
    """模拟一个异步耗时任务"""
    await asyncio.sleep(duration)
    return {
        "task_id": task_id,
        "duration": duration,
        "success": True,
    }


def run_serial(tasks: list[dict]) -> dict:
    """串行执行"""
    print("\n" + "=" * 80)
    print("🐢 串行执行 (Serial Execution)")
    print("=" * 80)

    start_time = time.time()
    results = []

    for i, task in enumerate(tasks, 1):
        print(f"  [{i}/{len(tasks)}] 执行任务 {task['id']} (耗时: {task['duration']}s)")
        result = simulate_task(task["id"], task["duration"])
        results.append(result)

    total_time = time.time() - start_time

    print(f"\n✅ 完成: {len(results)}/{len(tasks)} 任务")
    print(f"⏱️  总耗时: {total_time:.2f}秒")

    return {
        "mode": "serial",
        "total_time": total_time,
        "task_count": len(tasks),
        "throughput": len(tasks) / total_time,
    }


def run_parallel_threads(tasks: list[dict], max_workers: int = 4) -> dict:
    """多线程并行执行"""
    print("\n" + "=" * 80)
    print(f"🚀 多线程并行执行 (ThreadPool) - {max_workers} workers")
    print("=" * 80)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(simulate_task, task["id"], task["duration"]) for task in tasks]

        results = []
        for i, future in enumerate(futures, 1):
            result = future.result()
            results.append(result)
            if i % 5 == 0 or i == len(tasks):
                print(f"  进度: {i}/{len(tasks)}")

    total_time = time.time() - start_time

    print(f"\n✅ 完成: {len(results)}/{len(tasks)} 任务")
    print(f"⏱️  总耗时: {total_time:.2f}秒")

    return {
        "mode": f"parallel_threads_{max_workers}",
        "total_time": total_time,
        "task_count": len(tasks),
        "throughput": len(tasks) / total_time,
    }


async def run_parallel_async(tasks: list[dict], max_concurrent: int = 4) -> dict:
    """异步并行执行"""
    print("\n" + "=" * 80)
    print(f"⚡ 异步并行执行 (AsyncIO) - {max_concurrent} concurrent")
    print("=" * 80)

    start_time = time.time()

    # 使用信号量限制并发数
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_with_semaphore(task):
        async with semaphore:
            return await simulate_task_async(task["id"], task["duration"])

    # 创建所有任务
    coroutines = [run_with_semaphore(task) for task in tasks]

    # 并行执行
    results = await asyncio.gather(*coroutines)

    total_time = time.time() - start_time

    print(f"\n✅ 完成: {len(results)}/{len(tasks)} 任务")
    print(f"⏱️  总耗时: {total_time:.2f}秒")

    return {
        "mode": f"parallel_async_{max_concurrent}",
        "total_time": total_time,
        "task_count": len(tasks),
        "throughput": len(tasks) / total_time,
    }


def print_comparison(results: list[dict]):
    """打印对比结果"""
    print("\n" + "=" * 80)
    print("📊 性能对比 (Performance Comparison)")
    print("=" * 80)

    # 找到基准（串行）
    baseline = next((r for r in results if r["mode"] == "serial"), None)
    if not baseline:
        print("❌ 未找到串行执行结果")
        return

    baseline_time = baseline["total_time"]

    print(f"\n{'模式':<30} {'耗时(秒)':<12} {'吞吐量':<15} {'加速比':<10}")
    print("-" * 80)

    for result in results:
        mode = result["mode"]
        total_time = result["total_time"]
        throughput = result["throughput"]
        speedup = baseline_time / total_time

        print(f"{mode:<30} {total_time:>10.2f}s  {throughput:>12.2f}/s  {speedup:>8.2f}x")

    # 计算最佳加速比
    best_result = max(results, key=lambda r: baseline_time / r["total_time"])
    best_speedup = baseline_time / best_result["total_time"]

    print("\n" + "=" * 80)
    print(f"🏆 最佳加速: {best_result['mode']} - {best_speedup:.2f}x")
    print(
        f"⚡ 时间节省: {baseline_time - best_result['total_time']:.2f}秒 "
        f"({(1 - best_result['total_time'] / baseline_time) * 100:.1f}%)"
    )
    print("=" * 80)


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🎯 Doraemon Code 并行加速演示")
    print("=" * 80)

    # 创建测试任务（20个任务，每个耗时0.5秒）
    num_tasks = 20
    task_duration = 0.5

    tasks = [{"id": i, "duration": task_duration} for i in range(1, num_tasks + 1)]

    print("\n📋 测试配置:")
    print(f"  任务数量: {num_tasks}")
    print(f"  单任务耗时: {task_duration}秒")
    print(f"  理论总耗时(串行): {num_tasks * task_duration}秒")

    results = []

    # 1. 串行执行
    serial_result = run_serial(tasks)
    results.append(serial_result)

    # 2. 多线程并行 (2 workers)
    parallel_2_result = run_parallel_threads(tasks, max_workers=2)
    results.append(parallel_2_result)

    # 3. 多线程并行 (4 workers)
    parallel_4_result = run_parallel_threads(tasks, max_workers=4)
    results.append(parallel_4_result)

    # 4. 多线程并行 (8 workers)
    parallel_8_result = run_parallel_threads(tasks, max_workers=8)
    results.append(parallel_8_result)

    # 5. 异步并行 (4 concurrent)
    async_4_result = asyncio.run(run_parallel_async(tasks, max_concurrent=4))
    results.append(async_4_result)

    # 6. 异步并行 (8 concurrent)
    async_8_result = asyncio.run(run_parallel_async(tasks, max_concurrent=8))
    results.append(async_8_result)

    # 打印对比
    print_comparison(results)


if __name__ == "__main__":
    main()
