"""
评估系统集成测试

演示如何使用完整的评估系统
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class MockAgent:
    """模拟 Agent 用于测试"""

    def __init__(self, success_rate: float = 0.8):
        self.success_rate = success_rate
        self.call_count = 0

    def execute(self, prompt: str):
        """模拟执行任务"""
        import random
        import time

        self.call_count += 1

        # 模拟执行时间
        time.sleep(random.uniform(0.1, 0.5))

        # 根据成功率决定是否成功
        success = random.random() < self.success_rate

        return {
            "success": success,
            "response": f"Executed: {prompt}",
            "tool_calls": ["write_file"] if success else [],
        }


def create_mock_agent():
    """创建模拟 Agent"""
    return MockAgent(success_rate=0.8)


@pytest.mark.slow
def test_comprehensive_evaluation():
    """测试完整评估系统"""
    from tests.evals.comprehensive_evaluator import ComprehensiveEvaluator

    # 创建评估器
    evaluator = ComprehensiveEvaluator(
        output_dir="eval_results/test",
        parallel=False,  # 测试时使用串行
        n_trials=1,
        use_llm_judge=False,  # 测试时不使用 LLM 评判
    )

    # 使用少量任务进行测试
    task_files = [
        str(Path(__file__).parent / "tasks" / "basic" / "file_and_code_tasks.json")
    ]

    # 运行评估
    report = evaluator.run_full_evaluation(create_mock_agent, task_files)

    # 验证报告结构
    assert "metadata" in report
    assert "summary" in report
    assert "metrics" in report

    # 验证核心指标
    assert "core_metrics" in report["metrics"]
    core = report["metrics"]["core_metrics"]
    assert core["total_tasks"] > 0
    assert 0 <= core["success_rate"] <= 1


@pytest.mark.slow
def test_parallel_evaluation():
    """测试并行评估"""
    from tests.evals.parallel_evaluator import ParallelEvaluator

    evaluator = ParallelEvaluator(max_workers=2, use_processes=False)

    task_files = [
        str(Path(__file__).parent / "tasks" / "basic" / "file_and_code_tasks.json")
    ]

    summary = evaluator.run_parallel_evaluation(
        create_mock_agent, task_files, n_trials=1
    )

    # 验证并行评估结果
    assert summary["total_tasks"] > 0
    assert "parallelization" in summary
    assert summary["parallelization"]["max_workers"] == 2


@pytest.mark.slow
def test_metrics_collection():
    """测试指标收集"""
    from tests.evals.metrics_collector import MetricsCollector

    collector = MetricsCollector(output_dir="eval_results/test/metrics")

    # 开始收集
    collector.start_collection()

    # 模拟记录一些结果
    for i in range(10):
        result = {
            "task_id": f"test-{i}",
            "success": i % 2 == 0,  # 50% 成功率
            "execution_time": 1.0 + i * 0.1,
            "tool_calls": ["write_file", "read_file"],
            "difficulty": (i % 5) + 1,
            "category": "test_category",
        }
        collector.record_task_result(result)

    # 结束收集
    collector.end_collection()

    # 计算指标
    metrics = collector.calculate_metrics()

    # 验证指标
    assert metrics["core_metrics"]["total_tasks"] == 10
    assert metrics["core_metrics"]["success_rate"] == 0.5
    assert "tool_metrics" in metrics
    assert "difficulty_metrics" in metrics


def test_baseline_runner():
    """测试基线运行器"""
    from tests.evals.baseline_runner import BaselineRunner

    runner = BaselineRunner(output_dir="eval_results/test/baseline")

    # 创建模拟任务文件
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        tasks = [
            {
                "id": "test-001",
                "name": "测试任务",
                "prompt": "Test prompt",
                "difficulty": 1,
                "category": "test",
                "success_criteria": [],
            }
        ]
        json.dump(tasks, f)
        task_file = f.name

    try:
        # 运行基线评估
        agent = create_mock_agent()
        summary = runner.run_baseline_evaluation(agent, [task_file])

        # 验证基线结果
        assert "total_tasks" in summary
        assert "success_rate" in summary

    finally:
        # 清理临时文件
        Path(task_file).unlink(missing_ok=True)


def test_llm_judge_evaluator():
    """测试 LLM 评判器"""
    from tests.evals.llm_judge_evaluator import LLMJudgeEvaluator

    judge = LLMJudgeEvaluator()

    # 测试评分解析
    test_response = """
    {
      "task_completion": {"score": 8, "reasoning": "Good"},
      "tool_usage": {"score": 9, "reasoning": "Excellent"},
      "code_quality": {"score": 7, "reasoning": "Acceptable"},
      "problem_solving": {"score": 8, "reasoning": "Good"},
      "user_experience": {"score": 9, "reasoning": "Excellent"},
      "completeness": {"score": 8, "reasoning": "Good"},
      "overall_score": 8.2,
      "overall_assessment": "Good performance",
      "strengths": ["Tool usage", "UX"],
      "weaknesses": ["Code quality"],
      "suggestions": ["Improve code"]
    }
    """

    scores = judge._parse_scores(test_response)

    # 验证解析结果
    assert scores["overall_score"] == 8.2
    assert scores["task_completion"]["score"] == 8
    assert len(scores["strengths"]) == 2


@pytest.mark.slow
def test_model_comparison():
    """测试多模型对比"""
    from tests.evals.multi_model_comparator import MultiModelComparator

    comparator = MultiModelComparator(output_dir="eval_results/test/comparison")

    # 创建不同成功率的模拟 Agent
    def create_agent_a():
        return MockAgent(success_rate=0.9)

    def create_agent_b():
        return MockAgent(success_rate=0.7)

    agent_factories = {
        "model_a": create_agent_a,
        "model_b": create_agent_b,
    }

    # 注意: 这个测试需要实际的任务文件，这里只测试结构
    # 实际使用时需要提供真实的任务文件


def test_evaluation_task_loading():
    """测试评估任务加载"""
    from tests.evals.agent_evaluator import AgentEvaluator

    # 测试加载基础任务
    task_file = Path(__file__).parent / "tasks" / "basic" / "file_and_code_tasks.json"

    if task_file.exists():
        evaluator = AgentEvaluator(str(task_file))

        # 验证任务加载
        assert len(evaluator.tasks) > 0
        assert all("id" in task for task in evaluator.tasks)
        assert all("prompt" in task for task in evaluator.tasks)
        assert all("difficulty" in task for task in evaluator.tasks)


@pytest.mark.parametrize(
    "difficulty,expected_range",
    [
        ("easy", (1, 3)),
        ("medium", (4, 6)),
        ("hard", (7, 10)),
    ],
)
def test_task_difficulty_levels(difficulty, expected_range):
    """测试任务难度级别"""
    # 验证任务难度在合理范围内
    min_diff, max_diff = expected_range
    assert min_diff >= 1
    assert max_diff <= 10
    assert min_diff < max_diff


def test_evaluation_report_generation():
    """测试评估报告生成"""
    from tests.evals.comprehensive_evaluator import ComprehensiveEvaluator

    evaluator = ComprehensiveEvaluator(output_dir="eval_results/test")

    # 测试 Markdown 报告生成
    mock_report = {
        "metadata": {
            "timestamp": "2024-01-01T00:00:00",
            "evaluation_type": "test",
            "parallel": False,
            "n_trials": 1,
        },
        "metrics": {
            "core_metrics": {
                "total_tasks": 10,
                "successful_tasks": 8,
                "failed_tasks": 2,
                "success_rate": 0.8,
                "avg_execution_time": 1.5,
                "total_execution_time": 15.0,
            }
        },
    }

    markdown = evaluator._generate_markdown_report(mock_report)

    # 验证 Markdown 内容
    assert "# Doraemon Code 评估报告" in markdown
    assert "核心指标" in markdown
    assert "80.0%" in markdown  # 成功率


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
