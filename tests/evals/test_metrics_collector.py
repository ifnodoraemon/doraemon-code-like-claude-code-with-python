"""
MetricsCollector 单元测试

测试评估指标收集器的各项功能，包括:
- pass@k 和 pass^k 指标计算
- 延迟分布计算
- 成本跟踪
- 错误分类
- 综合报告生成
"""

import json
import math
import tempfile
from typing import Any

import pytest
from tests.evals.metrics_collector import MetricsCollector

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def collector() -> MetricsCollector:
    """创建一个基本的 MetricsCollector 实例"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield MetricsCollector(output_dir=tmpdir, model_name="claude-3-opus")


@pytest.fixture
def sample_results() -> list[dict[str, Any]]:
    """创建示例评估结果"""
    return [
        {
            "task_id": "task_1",
            "success": True,
            "execution_time": 1.5,
            "tool_calls": ["read", "write"],
            "difficulty": 1,
            "category": "code_generation",
            "input_tokens": 1000,
            "output_tokens": 500,
            "errors": [],
        },
        {
            "task_id": "task_2",
            "success": True,
            "execution_time": 2.0,
            "tool_calls": ["read", "search"],
            "difficulty": 2,
            "category": "code_generation",
            "input_tokens": 1200,
            "output_tokens": 600,
            "errors": [],
        },
        {
            "task_id": "task_3",
            "success": False,
            "execution_time": 5.0,
            "tool_calls": ["read"],
            "difficulty": 3,
            "category": "debugging",
            "input_tokens": 800,
            "output_tokens": 400,
            "errors": ["timeout: operation timed out"],
        },
        {
            "task_id": "task_4",
            "success": True,
            "execution_time": 1.0,
            "tool_calls": ["read", "write", "search"],
            "difficulty": 1,
            "category": "refactoring",
            "input_tokens": 1500,
            "output_tokens": 800,
            "errors": [],
        },
        {
            "task_id": "task_5",
            "success": False,
            "execution_time": 3.0,
            "tool_calls": ["read"],
            "difficulty": 2,
            "category": "debugging",
            "input_tokens": 900,
            "output_tokens": 450,
            "errors": ["assertion failed: expected 5, got 3"],
        },
    ]


@pytest.fixture
def mixed_error_results() -> list[dict[str, Any]]:
    """创建包含多种错误类型的结果"""
    return [
        {
            "success": False,
            "errors": ["timeout: operation timed out"],
            "category": "a",
            "difficulty": 1,
        },
        {"success": False, "errors": ["assertion failed"], "category": "a", "difficulty": 1},
        {
            "success": False,
            "errors": ["tool error: tool_call failed"],
            "category": "b",
            "difficulty": 2,
        },
        {
            "success": False,
            "errors": ["SyntaxError: invalid syntax"],
            "category": "b",
            "difficulty": 2,
        },
        {
            "success": False,
            "errors": ["RuntimeError: division by zero"],
            "category": "c",
            "difficulty": 3,
        },
        {"success": True, "errors": [], "category": "a", "difficulty": 1},
        {"success": True, "errors": [], "category": "b", "difficulty": 2},
    ]


# ============================================================================
# pass@k Tests
# ============================================================================


class TestPassAtK:
    """pass@k 指标测试"""

    def test_pass_at_k_all_success(self, collector: MetricsCollector):
        """测试全部成功的情况"""
        results = [{"success": True} for _ in range(10)]

        assert collector.calculate_pass_at_k(results, k=1) == pytest.approx(1.0)
        assert collector.calculate_pass_at_k(results, k=3) == pytest.approx(1.0)
        assert collector.calculate_pass_at_k(results, k=5) == pytest.approx(1.0)

    def test_pass_at_k_all_failure(self, collector: MetricsCollector):
        """测试全部失败的情况"""
        results = [{"success": False} for _ in range(10)]

        assert collector.calculate_pass_at_k(results, k=1) == pytest.approx(0.0)
        assert collector.calculate_pass_at_k(results, k=3) == pytest.approx(0.0)
        assert collector.calculate_pass_at_k(results, k=5) == pytest.approx(0.0)

    def test_pass_at_k_mixed(self, collector: MetricsCollector):
        """测试混合结果的情况"""
        # 60% 成功率
        results = [{"success": True}] * 6 + [{"success": False}] * 4
        success_rate = 0.6

        # pass@1 = 1 - (1 - 0.6)^1 = 0.6
        assert collector.calculate_pass_at_k(results, k=1) == pytest.approx(success_rate)

        # pass@3 = 1 - (1 - 0.6)^3 = 1 - 0.064 = 0.936
        expected_pass_3 = 1 - math.pow(1 - success_rate, 3)
        assert collector.calculate_pass_at_k(results, k=3) == pytest.approx(expected_pass_3)

        # pass@5 = 1 - (1 - 0.6)^5 = 1 - 0.01024 = 0.98976
        expected_pass_5 = 1 - math.pow(1 - success_rate, 5)
        assert collector.calculate_pass_at_k(results, k=5) == pytest.approx(expected_pass_5)

    def test_pass_at_k_empty_results(self, collector: MetricsCollector):
        """测试空结果的情况"""
        assert collector.calculate_pass_at_k([], k=1) == 0.0
        assert collector.calculate_pass_at_k([], k=3) == 0.0

    def test_pass_at_k_invalid_k(self, collector: MetricsCollector):
        """测试无效 k 值的情况"""
        results = [{"success": True}]
        assert collector.calculate_pass_at_k(results, k=0) == 0.0
        assert collector.calculate_pass_at_k(results, k=-1) == 0.0


# ============================================================================
# pass^k Tests
# ============================================================================


class TestPassPowerK:
    """pass^k 指标测试"""

    def test_pass_power_k_all_success(self, collector: MetricsCollector):
        """测试全部成功的情况"""
        results = [{"success": True} for _ in range(10)]

        assert collector.calculate_pass_power_k(results, k=1) == pytest.approx(1.0)
        assert collector.calculate_pass_power_k(results, k=3) == pytest.approx(1.0)
        assert collector.calculate_pass_power_k(results, k=5) == pytest.approx(1.0)

    def test_pass_power_k_all_failure(self, collector: MetricsCollector):
        """测试全部失败的情况"""
        results = [{"success": False} for _ in range(10)]

        assert collector.calculate_pass_power_k(results, k=1) == pytest.approx(0.0)
        assert collector.calculate_pass_power_k(results, k=3) == pytest.approx(0.0)

    def test_pass_power_k_mixed(self, collector: MetricsCollector):
        """测试混合结果的情况"""
        # 80% 成功率
        results = [{"success": True}] * 8 + [{"success": False}] * 2
        success_rate = 0.8

        # pass^1 = 0.8^1 = 0.8
        assert collector.calculate_pass_power_k(results, k=1) == pytest.approx(success_rate)

        # pass^3 = 0.8^3 = 0.512
        expected_power_3 = math.pow(success_rate, 3)
        assert collector.calculate_pass_power_k(results, k=3) == pytest.approx(expected_power_3)

        # pass^5 = 0.8^5 = 0.32768
        expected_power_5 = math.pow(success_rate, 5)
        assert collector.calculate_pass_power_k(results, k=5) == pytest.approx(expected_power_5)

    def test_pass_power_k_empty_results(self, collector: MetricsCollector):
        """测试空结果的情况"""
        assert collector.calculate_pass_power_k([], k=1) == 0.0

    def test_pass_power_k_reliability_interpretation(self, collector: MetricsCollector):
        """测试 pass^k 作为可靠性指标的解释"""
        # 90% 成功率
        results = [{"success": True}] * 9 + [{"success": False}] * 1

        # pass^3 表示连续 3 次成功的概率
        # 0.9^3 = 0.729，即约 73% 的可靠性
        reliability = collector.calculate_pass_power_k(results, k=3)
        assert reliability == pytest.approx(0.729)


# ============================================================================
# Latency Distribution Tests
# ============================================================================


class TestLatencyDistribution:
    """延迟分布测试"""

    def test_latency_distribution_basic(
        self, collector: MetricsCollector, sample_results: list[dict]
    ):
        """测试基本延迟分布计算"""
        dist = collector.calculate_latency_distribution(sample_results)

        assert "p50" in dist
        assert "p95" in dist
        assert "p99" in dist
        assert "mean" in dist
        assert "std" in dist
        assert "min" in dist
        assert "max" in dist

    def test_latency_distribution_values(self, collector: MetricsCollector):
        """测试延迟分布值的正确性"""
        results = [
            {"execution_time": 1.0},
            {"execution_time": 2.0},
            {"execution_time": 3.0},
            {"execution_time": 4.0},
            {"execution_time": 5.0},
        ]

        dist = collector.calculate_latency_distribution(results)

        assert dist["min"] == 1.0
        assert dist["max"] == 5.0
        assert dist["mean"] == pytest.approx(3.0)
        assert dist["p50"] == pytest.approx(3.0)  # 中位数

    def test_latency_distribution_empty(self, collector: MetricsCollector):
        """测试空结果的延迟分布"""
        dist = collector.calculate_latency_distribution([])

        assert dist["p50"] == 0.0
        assert dist["p95"] == 0.0
        assert dist["mean"] == 0.0

    def test_latency_distribution_single_value(self, collector: MetricsCollector):
        """测试单个值的延迟分布"""
        results = [{"execution_time": 5.0}]
        dist = collector.calculate_latency_distribution(results)

        assert dist["p50"] == 5.0
        assert dist["mean"] == 5.0
        assert dist["std"] == 0.0  # 单个值标准差为 0

    def test_latency_distribution_filters_invalid(self, collector: MetricsCollector):
        """测试过滤无效延迟值"""
        results = [
            {"execution_time": 1.0},
            {"execution_time": 0},  # 无效
            {"execution_time": 2.0},
            {"execution_time": -1.0},  # 无效（负数会被过滤）
        ]

        dist = collector.calculate_latency_distribution(results)
        # 只有 1.0 和 2.0 是有效的
        assert dist["mean"] == pytest.approx(1.5)


# ============================================================================
# Cost Metrics Tests
# ============================================================================


class TestCostMetrics:
    """成本指标测试"""

    def test_cost_metrics_basic(self, collector: MetricsCollector, sample_results: list[dict]):
        """测试基本成本计算"""
        cost = collector.calculate_cost_metrics(sample_results)

        assert "total_tokens" in cost
        assert "input_tokens" in cost
        assert "output_tokens" in cost
        assert "total_cost" in cost
        assert "cost_per_task" in cost
        assert "cost_per_success" in cost

    def test_cost_metrics_token_sum(self, collector: MetricsCollector, sample_results: list[dict]):
        """测试 token 总数计算"""
        cost = collector.calculate_cost_metrics(sample_results)

        expected_input = sum(r.get("input_tokens", 0) for r in sample_results)
        expected_output = sum(r.get("output_tokens", 0) for r in sample_results)

        assert cost["input_tokens"] == expected_input
        assert cost["output_tokens"] == expected_output
        assert cost["total_tokens"] == expected_input + expected_output

    def test_cost_metrics_calculation(self, collector: MetricsCollector):
        """测试成本计算的正确性"""
        results = [
            {"input_tokens": 1000, "output_tokens": 500, "success": True},
        ]

        # claude-3-opus: input=0.015/1K, output=0.075/1K
        cost = collector.calculate_cost_metrics(results)

        expected_input_cost = (1000 / 1000) * 0.015  # $0.015
        expected_output_cost = (500 / 1000) * 0.075  # $0.0375
        expected_total = expected_input_cost + expected_output_cost  # $0.0525

        assert cost["total_cost"] == pytest.approx(expected_total)

    def test_cost_metrics_empty(self, collector: MetricsCollector):
        """测试空结果的成本计算"""
        cost = collector.calculate_cost_metrics([])

        assert cost["total_tokens"] == 0
        assert cost["total_cost"] == 0.0
        assert cost["cost_per_task"] == 0.0

    def test_cost_metrics_per_success(self, collector: MetricsCollector):
        """测试每成功任务成本计算"""
        results = [
            {"input_tokens": 1000, "output_tokens": 500, "success": True},
            {"input_tokens": 1000, "output_tokens": 500, "success": False},
        ]

        cost = collector.calculate_cost_metrics(results)

        # 总成本除以成功任务数（1）
        assert cost["cost_per_success"] == cost["total_cost"]
        # 总成本除以总任务数（2）
        assert cost["cost_per_task"] == cost["total_cost"] / 2

    def test_cost_metrics_different_model(self):
        """测试不同模型的成本计算"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(output_dir=tmpdir, model_name="gpt-3.5-turbo")

            results = [{"input_tokens": 1000, "output_tokens": 500, "success": True}]
            cost = collector.calculate_cost_metrics(results)

            # gpt-3.5-turbo: input=0.0005/1K, output=0.0015/1K
            expected_input_cost = (1000 / 1000) * 0.0005
            expected_output_cost = (500 / 1000) * 0.0015
            expected_total = expected_input_cost + expected_output_cost

            assert cost["total_cost"] == pytest.approx(expected_total)


# ============================================================================
# Error Analysis Tests
# ============================================================================


class TestErrorAnalysis:
    """错误分析测试"""

    def test_analyze_errors_basic(
        self, collector: MetricsCollector, mixed_error_results: list[dict]
    ):
        """测试基本错误分析"""
        errors = collector.analyze_errors(mixed_error_results)

        assert "error_rate" in errors
        assert "error_types" in errors
        assert "error_by_category" in errors
        assert "error_by_difficulty" in errors

    def test_analyze_errors_rate(
        self, collector: MetricsCollector, mixed_error_results: list[dict]
    ):
        """测试错误率计算"""
        errors = collector.analyze_errors(mixed_error_results)

        # 7 个结果中 5 个失败
        assert errors["error_rate"] == pytest.approx(5 / 7)

    def test_analyze_errors_types(
        self, collector: MetricsCollector, mixed_error_results: list[dict]
    ):
        """测试错误类型分类"""
        errors = collector.analyze_errors(mixed_error_results)

        error_types = errors["error_types"]
        assert error_types.get("timeout", 0) == 1
        assert error_types.get("assertion_failed", 0) == 1
        assert error_types.get("tool_error", 0) == 1
        assert error_types.get("syntax_error", 0) == 1
        assert error_types.get("runtime_error", 0) == 1

    def test_analyze_errors_by_category(
        self, collector: MetricsCollector, mixed_error_results: list[dict]
    ):
        """测试按类别统计错误"""
        errors = collector.analyze_errors(mixed_error_results)

        by_category = errors["error_by_category"]

        # 类别 a: 3 个任务，2 个失败
        assert by_category["a"]["total"] == 3
        assert by_category["a"]["failed"] == 2
        assert by_category["a"]["error_rate"] == pytest.approx(2 / 3)

    def test_analyze_errors_by_difficulty(
        self, collector: MetricsCollector, mixed_error_results: list[dict]
    ):
        """测试按难度统计错误"""
        errors = collector.analyze_errors(mixed_error_results)

        by_difficulty = errors["error_by_difficulty"]

        # 难度 1: 3 个任务，2 个失败
        assert by_difficulty[1]["total"] == 3
        assert by_difficulty[1]["failed"] == 2

    def test_analyze_errors_empty(self, collector: MetricsCollector):
        """测试空结果的错误分析"""
        errors = collector.analyze_errors([])

        assert errors["error_rate"] == 0.0
        assert errors["error_types"] == {}

    def test_analyze_errors_all_success(self, collector: MetricsCollector):
        """测试全部成功的错误分析"""
        results = [{"success": True, "errors": [], "category": "a", "difficulty": 1}] * 5
        errors = collector.analyze_errors(results)

        assert errors["error_rate"] == 0.0
        assert errors["error_types"] == {}


# ============================================================================
# Anthropic Style Report Tests
# ============================================================================


class TestAnthropicStyleReport:
    """Anthropic 风格报告测试"""

    def test_generate_report_structure(
        self, collector: MetricsCollector, sample_results: list[dict]
    ):
        """测试报告结构"""
        report = collector.generate_anthropic_style_report(sample_results)

        assert "summary" in report
        assert "pass_metrics" in report
        assert "latency" in report
        assert "cost" in report
        assert "errors" in report
        assert "by_category" in report
        assert "by_difficulty" in report
        assert "recommendations" in report

    def test_generate_report_summary(self, collector: MetricsCollector, sample_results: list[dict]):
        """测试报告摘要"""
        report = collector.generate_anthropic_style_report(sample_results)

        summary = report["summary"]
        assert summary["total_tasks"] == 5
        assert summary["successful_tasks"] == 3
        assert summary["failed_tasks"] == 2
        assert summary["success_rate"] == pytest.approx(0.6)
        assert summary["model_name"] == "claude-3-opus"

    def test_generate_report_pass_metrics(
        self, collector: MetricsCollector, sample_results: list[dict]
    ):
        """测试报告中的 pass 指标"""
        report = collector.generate_anthropic_style_report(sample_results)

        pass_metrics = report["pass_metrics"]
        assert "pass_at_1" in pass_metrics
        assert "pass_at_3" in pass_metrics
        assert "pass_at_5" in pass_metrics
        assert "pass_power_3" in pass_metrics
        assert "pass_power_5" in pass_metrics

        # 验证 pass@1 等于成功率
        assert pass_metrics["pass_at_1"] == pytest.approx(0.6)

    def test_generate_report_empty(self, collector: MetricsCollector):
        """测试空结果的报告"""
        report = collector.generate_anthropic_style_report([])

        assert "error" in report

    def test_generate_report_recommendations(
        self, collector: MetricsCollector, sample_results: list[dict]
    ):
        """测试报告建议"""
        report = collector.generate_anthropic_style_report(sample_results)

        recommendations = report["recommendations"]
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0

    def test_generate_report_by_category(
        self, collector: MetricsCollector, sample_results: list[dict]
    ):
        """测试按类别统计"""
        report = collector.generate_anthropic_style_report(sample_results)

        by_category = report["by_category"]
        assert "code_generation" in by_category
        assert "debugging" in by_category
        assert "refactoring" in by_category

        # code_generation: 2 个任务，2 个成功
        assert by_category["code_generation"]["total"] == 2
        assert by_category["code_generation"]["success"] == 2

    def test_generate_report_by_difficulty(
        self, collector: MetricsCollector, sample_results: list[dict]
    ):
        """测试按难度统计"""
        report = collector.generate_anthropic_style_report(sample_results)

        by_difficulty = report["by_difficulty"]
        assert "1" in by_difficulty or 1 in by_difficulty
        assert "2" in by_difficulty or 2 in by_difficulty
        assert "3" in by_difficulty or 3 in by_difficulty


# ============================================================================
# Save and Print Tests
# ============================================================================


class TestSaveAndPrint:
    """保存和打印功能测试"""

    def test_save_anthropic_report(self, sample_results: list[dict]):
        """测试保存 Anthropic 报告"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(output_dir=tmpdir, model_name="claude-3-opus")

            output_path = collector.save_anthropic_report(sample_results, "test_report.json")

            assert output_path.exists()

            with open(output_path) as f:
                saved_report = json.load(f)

            assert "summary" in saved_report
            assert "pass_metrics" in saved_report

    def test_print_anthropic_summary(
        self, collector: MetricsCollector, sample_results: list[dict], capsys
    ):
        """测试打印 Anthropic 摘要"""
        collector.print_anthropic_summary(sample_results)

        captured = capsys.readouterr()

        assert "Anthropic Style Evaluation Report" in captured.out
        assert "Summary" in captured.out
        assert "Pass Metrics" in captured.out
        assert "Latency Distribution" in captured.out
        assert "Cost Analysis" in captured.out
        assert "Recommendations" in captured.out


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """集成测试"""

    def test_full_workflow(self, sample_results: list[dict]):
        """测试完整工作流程"""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = MetricsCollector(output_dir=tmpdir, model_name="claude-3-opus")

            # 开始收集
            collector.start_collection()

            # 记录结果
            for result in sample_results:
                collector.record_task_result(result)

            # 结束收集
            collector.end_collection()

            # 计算基本指标
            metrics = collector.calculate_metrics()
            assert metrics["core_metrics"]["total_tasks"] == 5

            # 生成 Anthropic 报告
            report = collector.generate_anthropic_style_report(sample_results)
            assert report["summary"]["total_tasks"] == 5

            # 保存报告
            output_path = collector.save_anthropic_report(sample_results)
            assert output_path.exists()

    def test_metrics_consistency(self, collector: MetricsCollector, sample_results: list[dict]):
        """测试指标一致性"""
        # 基本指标
        collector.start_collection()
        for result in sample_results:
            collector.record_task_result(result)
        collector.end_collection()

        basic_metrics = collector.calculate_metrics()

        # Anthropic 报告
        report = collector.generate_anthropic_style_report(sample_results)

        # 验证一致性
        assert basic_metrics["core_metrics"]["total_tasks"] == report["summary"]["total_tasks"]
        assert basic_metrics["core_metrics"]["success_rate"] == pytest.approx(
            report["summary"]["success_rate"]
        )


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """边界情况测试"""

    def test_single_result(self, collector: MetricsCollector):
        """测试单个结果"""
        results = [
            {"success": True, "execution_time": 1.0, "input_tokens": 100, "output_tokens": 50}
        ]

        report = collector.generate_anthropic_style_report(results)

        assert report["summary"]["total_tasks"] == 1
        assert report["pass_metrics"]["pass_at_1"] == 1.0

    def test_missing_fields(self, collector: MetricsCollector):
        """测试缺失字段"""
        results = [{"success": True}]  # 最小化结果

        report = collector.generate_anthropic_style_report(results)

        assert report["summary"]["total_tasks"] == 1
        assert report["latency"]["mean"] == 0.0
        assert report["cost"]["total_tokens"] == 0

    def test_large_dataset(self, collector: MetricsCollector):
        """测试大数据集"""
        import random

        results = [
            {
                "success": random.random() > 0.3,
                "execution_time": random.uniform(0.5, 10.0),
                "input_tokens": random.randint(100, 2000),
                "output_tokens": random.randint(50, 1000),
                "category": random.choice(["a", "b", "c"]),
                "difficulty": random.randint(1, 5),
                "errors": [] if random.random() > 0.3 else ["error"],
            }
            for _ in range(1000)
        ]

        report = collector.generate_anthropic_style_report(results)

        assert report["summary"]["total_tasks"] == 1000
        assert 0 <= report["summary"]["success_rate"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
