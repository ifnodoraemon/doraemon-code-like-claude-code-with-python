"""
多层评分器系统

整合三种评分方式：代码评分、模型评分、人工评分
基于 Anthropic 评估最佳实践
"""

import json
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class GradeResult:
    """评分结果"""

    score: float  # 0-1
    grader_type: str  # 'code', 'model', 'human'
    breakdown: dict[str, float] = field(default_factory=dict)  # 各维度得分
    feedback: str = ""  # 反馈信息
    confidence: float = 1.0  # 置信度
    metadata: dict[str, Any] = field(default_factory=dict)  # 额外信息


class CodeGrader:
    """
    代码评分器 - 确定性检查

    用于验证可以通过代码确定的结果：
    - 文件存在性
    - 测试通过
    - 覆盖率达标
    - 语法正确
    - Lint 通过
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()

    def grade(self, result: dict, task: dict) -> GradeResult:
        """
        执行代码评分

        Args:
            result: 任务执行结果
            task: 任务定义

        Returns:
            GradeResult
        """
        from tests.evals.state_verifier import StateVerifier

        verifier = StateVerifier(str(self.base_dir))
        assertions = task.get("assertions", [])

        if not assertions:
            return GradeResult(
                score=1.0 if result.get("success") else 0.0,
                grader_type="code",
                feedback="无断言可验证，使用默认结果",
            )

        # 验证所有断言
        verification = verifier.verify_assertions(assertions)

        breakdown = {}
        for i, res in enumerate(verification["results"]):
            key = f"{res['type']}_{i}"
            breakdown[key] = 1.0 if res["result"].success else 0.0

        return GradeResult(
            score=verification["pass_rate"],
            grader_type="code",
            breakdown=breakdown,
            feedback=f"通过 {verification['passed']}/{verification['total']} 个断言",
            confidence=1.0,  # 代码评分是确定性的
            metadata={
                "passed": verification["passed"],
                "total": verification["total"],
                "failed_assertions": [
                    r["assertion"] for r in verification["results"] if not r["result"].success
                ],
            },
        )


class ModelGrader:
    """
    模型评分器 - LLM 质量评估

    用于评估主观质量维度：
    - 代码质量（可读性、最佳实践）
    - 解决方案完整性
    - 文档质量
    - 错误处理
    - 可维护性
    """

    def __init__(self, model_client=None):
        """
        初始化模型评分器

        Args:
            model_client: 模型客户端，如果为 None 则使用默认客户端
        """
        self.model_client = model_client
        self.calibration_data: list[dict] = []

    def grade(self, result: dict, task: dict) -> GradeResult:
        """
        执行模型评分

        Args:
            result: 任务执行结果
            task: 任务定义

        Returns:
            GradeResult
        """
        rubric = task.get("rubric", "")
        result.get("output", "")
        result.get("tool_calls", [])

        # 构建评分提示
        prompt = self._build_grading_prompt(result, task, rubric)

        # 如果没有模型客户端，使用启发式评分
        if self.model_client is None:
            return self._heuristic_grade(result, task)

        try:
            # 调用模型进行评分
            response = self.model_client.create(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
            )

            # 解析评分结果
            return self._parse_model_response(response, result, task)

        except Exception as e:
            # 回退到启发式评分
            return self._heuristic_grade(result, task, error=str(e))

    def _build_grading_prompt(self, result: dict, task: dict, rubric: str) -> str:
        """构建评分提示"""
        return f"""你是一个代码质量评估专家。请评估以下任务执行结果。

## 任务描述
{task.get("prompt", "")}

## 评分标准
{rubric if rubric else "代码质量、完整性、可维护性"}

## 执行结果
成功: {result.get("success", False)}
输出: {result.get("output", "")[:2000]}
工具调用: {len(result.get("tool_calls", []))} 次

## 评分要求
请从以下维度评分（每个维度 0-1 分）：
1. code_quality: 代码质量（可读性、最佳实践）
2. completeness: 解决方案完整性
3. documentation: 文档质量
4. error_handling: 错误处理
5. maintainability: 可维护性

请以 JSON 格式返回评分结果：
{{
    "overall_score": 0.85,
    "breakdown": {{
        "code_quality": 0.9,
        "completeness": 0.8,
        "documentation": 0.7,
        "error_handling": 0.9,
        "maintainability": 0.85
    }},
    "feedback": "简短的评价反馈",
    "confidence": 0.9
}}
"""

    def _parse_model_response(self, response: Any, result: dict, task: dict) -> GradeResult:
        """解析模型响应"""
        try:
            # 提取 JSON
            content = response.content if hasattr(response, "content") else str(response)

            # 尝试找到 JSON 块
            import re

            json_match = re.search(r"\{[\s\S]*\}", content)
            if json_match:
                data = json.loads(json_match.group())
                return GradeResult(
                    score=data.get("overall_score", 0.5),
                    grader_type="model",
                    breakdown=data.get("breakdown", {}),
                    feedback=data.get("feedback", ""),
                    confidence=data.get("confidence", 0.8),
                    metadata={"raw_response": content[:500]},
                )
        except Exception:
            pass

        # 解析失败，使用启发式
        return self._heuristic_grade(result, task, error="解析模型响应失败")

    def _heuristic_grade(self, result: dict, task: dict, error: str | None = None) -> GradeResult:
        """启发式评分（当模型不可用时）"""
        score = 0.0
        breakdown = {}

        # 基础分：任务是否成功
        if result.get("success"):
            score += 0.4
            breakdown["task_success"] = 1.0
        else:
            breakdown["task_success"] = 0.0

        # 工具使用效率
        tool_calls = result.get("tool_calls", [])
        expected_steps = task.get("expected_steps", 5)
        if tool_calls:
            efficiency = min(1.0, expected_steps / max(len(tool_calls), 1))
            score += 0.2 * efficiency
            breakdown["tool_efficiency"] = efficiency
        else:
            breakdown["tool_efficiency"] = 0.5

        # 执行时间
        execution_time = result.get("execution_time", 0)
        timeout = task.get("timeout", 60)
        if execution_time > 0 and execution_time < timeout:
            time_score = 1.0 - (execution_time / timeout)
            score += 0.2 * time_score
            breakdown["time_efficiency"] = time_score
        else:
            breakdown["time_efficiency"] = 0.5

        # 错误数量
        errors = result.get("errors", [])
        if not errors:
            score += 0.2
            breakdown["error_free"] = 1.0
        else:
            breakdown["error_free"] = max(0, 1.0 - len(errors) * 0.2)
            score += 0.2 * breakdown["error_free"]

        return GradeResult(
            score=min(1.0, score),
            grader_type="model",
            breakdown=breakdown,
            feedback=f"启发式评分{f' (错误: {error})' if error else ''}",
            confidence=0.6,  # 启发式评分置信度较低
            metadata={"heuristic": True, "error": error},
        )

    def calibrate(self, human_grades: list[dict]):
        """
        使用人工评分数据校准模型

        Args:
            human_grades: 人工评分数据列表
        """
        self.calibration_data.extend(human_grades)
        # TODO: 实现校准逻辑（调整提示、权重等）


class HumanGrader:
    """
    人工评分器 - 抽样评估

    用于：
    - 验证模型评分的准确性
    - 评估难以自动化的维度
    - 校准模型评分器
    """

    def __init__(self, sample_rate: float = 0.1, callback: Callable | None = None):
        """
        初始化人工评分器

        Args:
            sample_rate: 抽样率 (0-1)
            callback: 人工评分回调函数
        """
        self.sample_rate = sample_rate
        self.callback = callback
        self.grades: list[dict] = []

    def should_sample(self) -> bool:
        """决定是否需要人工评估"""
        return random.random() < self.sample_rate

    def grade(self, result: dict, task: dict) -> GradeResult | None:
        """
        执行人工评分

        Args:
            result: 任务执行结果
            task: 任务定义

        Returns:
            GradeResult 或 None（如果不需要人工评估）
        """
        if not self.should_sample():
            return None

        if self.callback:
            # 使用回调函数获取人工评分
            human_input = self.callback(result, task)
            if human_input:
                grade = GradeResult(
                    score=human_input.get("score", 0.5),
                    grader_type="human",
                    breakdown=human_input.get("breakdown", {}),
                    feedback=human_input.get("feedback", ""),
                    confidence=1.0,  # 人工评分置信度最高
                    metadata={"reviewer": human_input.get("reviewer", "unknown")},
                )
                self.grades.append(
                    {
                        "task_id": task.get("id"),
                        "result": result,
                        "grade": grade,
                    }
                )
                return grade

        # 如果没有回调，记录待评估项
        self.grades.append(
            {
                "task_id": task.get("id"),
                "result": result,
                "grade": None,
                "pending": True,
            }
        )
        return None

    def get_pending_reviews(self) -> list[dict]:
        """获取待人工评估的项目"""
        return [g for g in self.grades if g.get("pending")]

    def submit_review(self, task_id: str, score: float, feedback: str = ""):
        """提交人工评审结果"""
        for grade in self.grades:
            if grade.get("task_id") == task_id and grade.get("pending"):
                grade["grade"] = GradeResult(
                    score=score,
                    grader_type="human",
                    feedback=feedback,
                    confidence=1.0,
                )
                grade["pending"] = False
                break

    def calibrate_model_grader(self, model_grader: ModelGrader):
        """使用人工评估结果校准模型评分器"""
        completed_grades = [g for g in self.grades if g.get("grade") and not g.get("pending")]
        if completed_grades:
            model_grader.calibrate(completed_grades)


class MultiLayerGrader:
    """
    多层评分器 - 整合三种评分方式

    基于 Anthropic 最佳实践：
    - 代码评分：确定性检查（50%权重）
    - 模型评分：质量评估（30%权重）
    - 人工评分：抽样验证（20%权重）
    """

    def __init__(
        self,
        code_weight: float = 0.5,
        model_weight: float = 0.3,
        human_weight: float = 0.2,
        base_dir: str | None = None,
        model_client=None,
        human_sample_rate: float = 0.1,
    ):
        """
        初始化多层评分器

        Args:
            code_weight: 代码评分权重
            model_weight: 模型评分权重
            human_weight: 人工评分权重
            base_dir: 基础目录
            model_client: 模型客户端
            human_sample_rate: 人工抽样率
        """
        self.code_weight = code_weight
        self.model_weight = model_weight
        self.human_weight = human_weight

        self.code_grader = CodeGrader(base_dir)
        self.model_grader = ModelGrader(model_client)
        self.human_grader = HumanGrader(human_sample_rate)

    def grade(self, result: dict, task: dict) -> GradeResult:
        """
        执行多层评分

        Args:
            result: 任务执行结果
            task: 任务定义

        Returns:
            聚合的 GradeResult
        """
        grades = {}
        weights = {}

        # 1. 代码评分（总是执行）
        code_grade = self.code_grader.grade(result, task)
        grades["code"] = code_grade
        weights["code"] = self.code_weight

        # 2. 模型评分（总是执行）
        model_grade = self.model_grader.grade(result, task)
        grades["model"] = model_grade
        weights["model"] = self.model_weight

        # 3. 人工评分（抽样执行）
        human_grade = self.human_grader.grade(result, task)
        if human_grade:
            grades["human"] = human_grade
            weights["human"] = self.human_weight
        else:
            # 如果没有人工评分，重新分配权重
            total_weight = self.code_weight + self.model_weight
            weights["code"] = self.code_weight / total_weight
            weights["model"] = self.model_weight / total_weight

        # 聚合评分
        return self._aggregate_grades(grades, weights)

    def _aggregate_grades(
        self, grades: dict[str, GradeResult], weights: dict[str, float]
    ) -> GradeResult:
        """聚合多个评分结果"""
        total_score = 0.0
        total_confidence = 0.0
        breakdown = {}
        feedbacks = []

        for grader_type, grade in grades.items():
            weight = weights.get(grader_type, 0)
            total_score += grade.score * weight
            total_confidence += grade.confidence * weight

            # 合并 breakdown
            for key, value in grade.breakdown.items():
                breakdown[f"{grader_type}_{key}"] = value

            if grade.feedback:
                feedbacks.append(f"[{grader_type}] {grade.feedback}")

        return GradeResult(
            score=total_score,
            grader_type="multi_layer",
            breakdown=breakdown,
            feedback=" | ".join(feedbacks),
            confidence=total_confidence,
            metadata={
                "grades": {k: v.score for k, v in grades.items()},
                "weights": weights,
            },
        )

    def generate_grade_report(self, result: dict, task: dict) -> dict:
        """生成详细的评分报告"""
        grade = self.grade(result, task)

        return {
            "task_id": task.get("id"),
            "task_name": task.get("name", task.get("id")),
            "overall_score": grade.score,
            "confidence": grade.confidence,
            "grader_type": grade.grader_type,
            "breakdown": grade.breakdown,
            "feedback": grade.feedback,
            "metadata": grade.metadata,
            "weights": {
                "code": self.code_weight,
                "model": self.model_weight,
                "human": self.human_weight,
            },
        }

    def batch_grade(self, results: list[dict], tasks: list[dict]) -> list[dict]:
        """批量评分"""
        reports = []
        for result, task in zip(results, tasks, strict=False):
            report = self.generate_grade_report(result, task)
            reports.append(report)
        return reports

    def get_pending_human_reviews(self) -> list[dict]:
        """获取待人工评审的项目"""
        return self.human_grader.get_pending_reviews()

    def calibrate(self):
        """使用人工评分校准模型评分器"""
        self.human_grader.calibrate_model_grader(self.model_grader)


# 便捷函数
def grade_task_result(
    result: dict,
    task: dict,
    base_dir: str | None = None,
    model_client=None,
) -> dict:
    """
    评分任务结果

    Args:
        result: 任务执行结果
        task: 任务定义
        base_dir: 基础目录
        model_client: 模型客户端

    Returns:
        评分报告
    """
    grader = MultiLayerGrader(base_dir=base_dir, model_client=model_client)
    return grader.generate_grade_report(result, task)


if __name__ == "__main__":
    # 示例用法
    print("=== 多层评分器示例 ===")

    # 创建评分器
    grader = MultiLayerGrader()

    # 模拟任务和结果
    task = {
        "id": "test-001",
        "name": "测试任务",
        "prompt": "创建一个 hello.py 文件",
        "expected_steps": 2,
        "timeout": 30,
        "assertions": [
            {"type": "file_exists", "path": "README.md"},
        ],
        "rubric": "代码质量和完整性",
    }

    result = {
        "success": True,
        "output": "文件创建成功",
        "tool_calls": ["write_file"],
        "execution_time": 5.0,
        "errors": [],
    }

    # 执行评分
    report = grader.generate_grade_report(result, task)

    print(f"任务: {report['task_name']}")
    print(f"总分: {report['overall_score']:.2f}")
    print(f"置信度: {report['confidence']:.2f}")
    print(f"反馈: {report['feedback']}")
    print(f"详细分数: {report['breakdown']}")
