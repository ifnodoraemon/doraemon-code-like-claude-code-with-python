"""
完整评估运行器

整合所有评估组件，提供统一的评估入口
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ComprehensiveEvaluator:
    """完整评估系统"""

    def __init__(
        self,
        output_dir: str = "eval_results/comprehensive",
        parallel: bool = True,
        max_workers: int = 4,
        n_trials: int = 1,
        use_llm_judge: bool = True,
    ):
        """
        初始化完整评估器

        Args:
            output_dir: 输出目录
            parallel: 是否使用并行评估
            max_workers: 并行工作数
            n_trials: 每个任务运行次数
            use_llm_judge: 是否使用 LLM 评判
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.parallel = parallel
        self.max_workers = max_workers
        self.n_trials = n_trials
        self.use_llm_judge = use_llm_judge

        # 初始化组件
        from tests.evals.metrics_collector import MetricsCollector
        from tests.evals.parallel_evaluator import ParallelEvaluator
        from tests.evals.baseline_runner import BaselineRunner

        self.metrics_collector = MetricsCollector(
            output_dir=str(self.output_dir / "metrics")
        )
        self.parallel_evaluator = ParallelEvaluator(
            max_workers=max_workers, output_dir=str(self.output_dir / "parallel")
        )
        self.baseline_runner = BaselineRunner(
            output_dir=str(self.output_dir / "baseline")
        )

        if use_llm_judge:
            from tests.evals.llm_judge_evaluator import LLMJudgeEvaluator

            self.llm_judge = LLMJudgeEvaluator()

    def run_full_evaluation(
        self, agent_factory, task_files: Optional[List[str]] = None
    ) -> Dict:
        """
        运行完整评估

        Args:
            agent_factory: Agent 工厂函数
            task_files: 任务文件列表（如果为 None，使用默认任务集）

        Returns:
            完整评估报告
        """
        print("=" * 80)
        print("Doraemon Code 完整评估系统")
        print("=" * 80)

        # 使用默认任务集
        if task_files is None:
            task_files = self._get_default_task_files()

        print(f"\n评估配置:")
        print(f"  任务文件: {len(task_files)} 个")
        print(f"  并行模式: {'是' if self.parallel else '否'}")
        print(f"  并行度: {self.max_workers}")
        print(f"  试验次数: {self.n_trials}")
        print(f"  LLM 评判: {'是' if self.use_llm_judge else '否'}")

        # 开始收集指标
        self.metrics_collector.start_collection()

        # 运行评估
        if self.parallel:
            print("\n使用并行评估...")
            summary = self.parallel_evaluator.run_parallel_evaluation(
                agent_factory, task_files, self.n_trials
            )
        else:
            print("\n使用串行评估...")
            summary = self._run_serial_evaluation(agent_factory, task_files)

        # 结束收集指标
        self.metrics_collector.end_collection()

        # 生成完整报告
        report = self._generate_comprehensive_report(summary)

        # 保存报告
        self._save_report(report)

        # 打印摘要
        self._print_report_summary(report)

        return report

    def run_baseline_evaluation(self, agent_factory, task_files: Optional[List[str]] = None) -> Dict:
        """
        运行基线评估并保存

        Args:
            agent_factory: Agent 工厂函数
            task_files: 任务文件列表

        Returns:
            基线评估结果
        """
        if task_files is None:
            task_files = self._get_default_task_files()

        agent = agent_factory()
        summary = self.baseline_runner.run_baseline_evaluation(agent, task_files)
        self.baseline_runner.print_baseline_report(summary)

        return summary

    def compare_with_baseline(self, current_results: Dict) -> Dict:
        """
        与基线对比

        Args:
            current_results: 当前评估结果

        Returns:
            对比报告
        """
        return self.baseline_runner.compare_with_baseline(current_results)

    def run_model_comparison(
        self, agent_factories: Dict[str, callable], task_files: Optional[List[str]] = None
    ) -> Dict:
        """
        运行多模型对比评估

        Args:
            agent_factories: 模型名称到 Agent 工厂函数的映射
            task_files: 任务文件列表

        Returns:
            模型对比报告
        """
        from tests.evals.multi_model_comparator import MultiModelComparator

        if task_files is None:
            task_files = self._get_default_task_files()

        comparator = MultiModelComparator(
            output_dir=str(self.output_dir / "model_comparison")
        )

        print("=" * 80)
        print("多模型对比评估")
        print("=" * 80)

        comparison_results = {
            "timestamp": datetime.now().isoformat(),
            "models": {},
        }

        for model_name, agent_factory in agent_factories.items():
            print(f"\n评估模型: {model_name}")
            print("-" * 80)

            # 运行评估
            summary = self.run_full_evaluation(agent_factory, task_files)
            comparison_results["models"][model_name] = summary

        # 保存对比结果
        comparator.save_comparison(comparison_results)
        comparator.print_comparison(comparison_results)

        return comparison_results

    def _run_serial_evaluation(self, agent_factory, task_files: List[str]) -> Dict:
        """串行评估"""
        from tests.evals.agent_evaluator import AgentEvaluator

        all_results = []

        for task_file in task_files:
            print(f"\n评估任务集: {task_file}")
            evaluator = AgentEvaluator(task_file)

            agent = agent_factory()
            report = evaluator.run_evaluation(agent)

            all_results.extend(evaluator.results)

            # 记录指标
            for result in evaluator.results:
                self.metrics_collector.record_task_result(result)

        # 生成汇总
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_tasks": len(all_results),
            "successful_tasks": sum(1 for r in all_results if r["success"]),
            "results": all_results,
        }

        return summary

    def _generate_comprehensive_report(self, summary: Dict) -> Dict:
        """生成完整报告"""
        # 计算指标
        metrics = self.metrics_collector.calculate_metrics()

        # LLM 评估（如果启用）
        llm_evaluation = None
        if self.use_llm_judge and "results" in summary:
            print("\n运行 LLM 评判...")
            evaluated_results = self.llm_judge.batch_evaluate(summary["results"])
            llm_evaluation = self.llm_judge.generate_summary_report(evaluated_results)

        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "evaluation_type": "comprehensive",
                "parallel": self.parallel,
                "max_workers": self.max_workers,
                "n_trials": self.n_trials,
                "use_llm_judge": self.use_llm_judge,
            },
            "summary": summary,
            "metrics": metrics,
            "llm_evaluation": llm_evaluation,
        }

        return report

    def _save_report(self, report: Dict):
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 保存 JSON 报告
        json_file = self.output_dir / f"report_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(report, f, indent=2)

        # 保存 Markdown 报告
        md_file = self.output_dir / f"report_{timestamp}.md"
        with open(md_file, "w") as f:
            f.write(self._generate_markdown_report(report))

        print(f"\n报告已保存:")
        print(f"  JSON: {json_file}")
        print(f"  Markdown: {md_file}")

    def _generate_markdown_report(self, report: Dict) -> str:
        """生成 Markdown 格式报告"""
        lines = []

        lines.append("# Doraemon Code 评估报告\n")
        lines.append(f"**评估时间**: {report['metadata']['timestamp']}\n")
        lines.append(f"**评估类型**: {report['metadata']['evaluation_type']}\n")
        lines.append(f"**并行模式**: {'是' if report['metadata']['parallel'] else '否'}\n")
        lines.append(f"**试验次数**: {report['metadata']['n_trials']}\n")

        # 核心指标
        if "metrics" in report and "core_metrics" in report["metrics"]:
            core = report["metrics"]["core_metrics"]
            lines.append("\n## 核心指标\n")
            lines.append(f"- **总任务数**: {core['total_tasks']}\n")
            lines.append(f"- **成功任务**: {core['successful_tasks']}\n")
            lines.append(f"- **失败任务**: {core['failed_tasks']}\n")
            lines.append(f"- **成功率**: {core['success_rate']*100:.1f}%\n")
            lines.append(f"- **平均耗时**: {core['avg_execution_time']:.2f}s\n")
            lines.append(f"- **总耗时**: {core['total_execution_time']:.2f}s\n")

        # 按类别统计
        if "metrics" in report and "category_metrics" in report["metrics"]:
            lines.append("\n## 按类别统计\n")
            lines.append("| 类别 | 成功率 | 平均耗时 |\n")
            lines.append("|------|--------|----------|\n")

            for category, stats in report["metrics"]["category_metrics"][
                "by_category"
            ].items():
                lines.append(
                    f"| {category} | {stats['success_rate']*100:.1f}% | {stats['avg_execution_time']:.2f}s |\n"
                )

        # 按难度统计
        if "metrics" in report and "difficulty_metrics" in report["metrics"]:
            lines.append("\n## 按难度统计\n")
            lines.append("| 难度 | 成功率 | 平均耗时 |\n")
            lines.append("|------|--------|----------|\n")

            for difficulty, stats in sorted(
                report["metrics"]["difficulty_metrics"]["by_difficulty"].items()
            ):
                lines.append(
                    f"| {difficulty} | {stats['success_rate']*100:.1f}% | {stats['avg_execution_time']:.2f}s |\n"
                )

        # LLM 评估
        if report.get("llm_evaluation"):
            llm = report["llm_evaluation"]
            lines.append("\n## LLM 评估分数\n")
            if "average_scores" in llm:
                scores = llm["average_scores"]
                lines.append(f"- **总体评分**: {scores['overall']:.2f}/10\n")
                lines.append(f"- **任务完成**: {scores['task_completion']:.2f}/10\n")
                lines.append(f"- **工具使用**: {scores['tool_usage']:.2f}/10\n")
                lines.append(f"- **代码质量**: {scores['code_quality']:.2f}/10\n")

        # 工具使用
        if "metrics" in report and "tool_metrics" in report["metrics"]:
            tool = report["metrics"]["tool_metrics"]
            lines.append("\n## 工具使用统计\n")
            lines.append(f"- **总调用次数**: {tool['total_tool_calls']}\n")
            lines.append(f"- **平均调用/任务**: {tool['avg_tool_calls_per_task']:.1f}\n")
            lines.append("\n### 最常用工具\n")
            for tool_name, count in tool["most_used_tools"]:
                lines.append(f"- {tool_name}: {count}\n")

        return "".join(lines)

    def _print_report_summary(self, report: Dict):
        """打印报告摘要"""
        print("\n" + "=" * 80)
        print("评估完成")
        print("=" * 80)

        # 打印指标摘要
        self.metrics_collector.print_summary()

        # 打印 LLM 评估摘要
        if report.get("llm_evaluation"):
            llm = report["llm_evaluation"]
            print(f"\nLLM 评估:")
            print(f"  评估任务数: {llm['evaluated_tasks']}/{llm['total_tasks']}")
            if "average_scores" in llm:
                scores = llm["average_scores"]
                print(f"  总体评分: {scores['overall']:.2f}/10")

    def _get_default_task_files(self) -> List[str]:
        """获取默认任务文件列表"""
        base_dir = Path(__file__).parent / "tasks"

        task_files = []

        # 基础任务
        basic_file = base_dir / "basic" / "file_and_code_tasks.json"
        if basic_file.exists():
            task_files.append(str(basic_file))

        # 中级任务
        inter_file = base_dir / "intermediate" / "tasks.json"
        if inter_file.exists():
            task_files.append(str(inter_file))

        # 高级任务
        adv_file = base_dir / "advanced" / "tasks.json"
        if adv_file.exists():
            task_files.append(str(adv_file))

        # 专家任务
        exp_file = base_dir / "expert" / "tasks.json"
        if exp_file.exists():
            task_files.append(str(exp_file))

        # Agent 评估任务
        agent_file = base_dir / "agent_eval_tasks.json"
        if agent_file.exists():
            task_files.append(str(agent_file))

        return task_files


def main():
    """主函数"""
    print("完整评估系统已准备就绪")
    print("\n使用方法:")
    print("\n1. 运行完整评估:")
    print("  evaluator = ComprehensiveEvaluator()")
    print("  report = evaluator.run_full_evaluation(agent_factory)")
    print("\n2. 运行基线评估:")
    print("  evaluator = ComprehensiveEvaluator()")
    print("  baseline = evaluator.run_baseline_evaluation(agent_factory)")
    print("\n3. 多模型对比:")
    print("  evaluator = ComprehensiveEvaluator()")
    print("  comparison = evaluator.run_model_comparison(agent_factories)")
    print("\n4. 与基线对比:")
    print("  comparison = evaluator.compare_with_baseline(current_results)")


if __name__ == "__main__":
    main()
