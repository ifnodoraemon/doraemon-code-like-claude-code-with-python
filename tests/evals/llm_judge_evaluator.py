"""
LLM-as-Judge 评估器

使用 LLM 作为评判者来评估 Agent 的响应质量
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.model_client import ModelClient
from src.core.model_utils import Message


class LLMJudgeEvaluator:
    """LLM 评判者评估器"""

    def __init__(self, model: str = "gemini-2.0-flash-exp"):
        self.model = model
        self.client = ModelClient.create()

    def evaluate_response(self, task: dict, agent_response: str, agent_actions: list[str]) -> dict:
        """评估 Agent 的响应"""

        judge_prompt = self._create_judge_prompt(task, agent_response, agent_actions)

        # 调用 LLM 进行评估
        messages = [Message(role="user", content=judge_prompt)]
        response = self.client.chat(messages)

        # 解析评分
        scores = self._parse_scores(response.content)

        return scores

    def _create_judge_prompt(
        self, task: dict, agent_response: str, agent_actions: list[str]
    ) -> str:
        """创建评判提示"""

        prompt = f"""
你是一个专业的 AI Agent 评估专家。请评估以下 Agent 对任务的完成情况。

## 任务信息
**任务**: {task["prompt"]}
**难度**: {task["difficulty"]}/10
**类别**: {task["category"]}
**期望工具**: {", ".join(task.get("expected_tools", []))}

## Agent 响应
{agent_response}

## Agent 执行的操作
{chr(10).join(f"- {action}" for action in agent_actions)}

## 评分标准 (每项 1-10 分)

### 1. 任务完成度 (Task Completion)
- 是否完成了任务的所有要求？
- 是否达到了预期目标？

### 2. 工具使用 (Tool Usage)
- 是否选择了正确的工具？
- 工具使用是否高效？
- 是否有不必要的工具调用？

### 3. 代码质量 (Code Quality) - 如果涉及代码
- 代码语法是否正确？
- 代码逻辑是否清晰？
- 是否遵循最佳实践？
- 是否有适当的错误处理？

### 4. 问题解决 (Problem Solving)
- 解决方案是否合理？
- 是否考虑了边界情况？
- 是否有创新性？

### 5. 用户体验 (User Experience)
- 响应是否清晰易懂？
- 是否提供了有用的信息？
- 交互是否友好？

### 6. 完整性 (Completeness)
- 是否遗漏了重要内容？
- 是否需要后续操作？

## 输出格式

请以 JSON 格式输出评分：

```json
{{
  "task_completion": {{
    "score": 8,
    "reasoning": "完成了主要任务，但缺少..."
  }},
  "tool_usage": {{
    "score": 9,
    "reasoning": "工具选择正确，使用高效"
  }},
  "code_quality": {{
    "score": 7,
    "reasoning": "代码正确但可以改进..."
  }},
  "problem_solving": {{
    "score": 8,
    "reasoning": "解决方案合理，考虑了..."
  }},
  "user_experience": {{
    "score": 9,
    "reasoning": "响应清晰，交互友好"
  }},
  "completeness": {{
    "score": 8,
    "reasoning": "基本完整，但..."
  }},
  "overall_score": 8.2,
  "overall_assessment": "总体表现良好，Agent 成功完成了任务...",
  "strengths": [
    "工具使用准确",
    "代码质量高"
  ],
  "weaknesses": [
    "缺少错误处理",
    "文档不够详细"
  ],
  "suggestions": [
    "添加更多的错误处理",
    "改进文档说明"
  ]
}}
```

请严格按照上述 JSON 格式输出评分结果。
"""
        return prompt

    def _parse_scores(self, response_content: str) -> dict:
        """解析 LLM 返回的评分"""
        try:
            # 提取 JSON 部分
            start = response_content.find("{")
            end = response_content.rfind("}") + 1

            if start >= 0 and end > start:
                json_str = response_content[start:end]
                scores = json.loads(json_str)
                return scores
            else:
                # 如果无法解析，返回默认评分
                return self._default_scores()

        except Exception as e:
            print(f"解析评分失败: {e}")
            return self._default_scores()

    def _default_scores(self) -> dict:
        """默认评分"""
        return {
            "task_completion": {"score": 5, "reasoning": "无法评估"},
            "tool_usage": {"score": 5, "reasoning": "无法评估"},
            "code_quality": {"score": 5, "reasoning": "无法评估"},
            "problem_solving": {"score": 5, "reasoning": "无法评估"},
            "user_experience": {"score": 5, "reasoning": "无法评估"},
            "completeness": {"score": 5, "reasoning": "无法评估"},
            "overall_score": 5.0,
            "overall_assessment": "评估失败",
            "strengths": [],
            "weaknesses": [],
            "suggestions": [],
        }

    def batch_evaluate(self, results: list[dict]) -> list[dict]:
        """批量评估多个任务结果"""
        evaluated_results = []

        for result in results:
            if result.get("success"):
                # 只评估成功的任务
                llm_scores = self.evaluate_response(
                    result["task"],
                    result.get("response", ""),
                    result.get("actions", []),
                )
                result["llm_evaluation"] = llm_scores

            evaluated_results.append(result)

        return evaluated_results

    def generate_summary_report(self, evaluated_results: list[dict]) -> dict:
        """生成汇总报告"""
        total_tasks = len(evaluated_results)
        evaluated_tasks = [r for r in evaluated_results if "llm_evaluation" in r]

        if not evaluated_tasks:
            return {"error": "没有可评估的任务"}

        # 计算平均分
        avg_scores = {
            "task_completion": 0,
            "tool_usage": 0,
            "code_quality": 0,
            "problem_solving": 0,
            "user_experience": 0,
            "completeness": 0,
            "overall": 0,
        }

        for result in evaluated_tasks:
            eval_data = result["llm_evaluation"]
            avg_scores["task_completion"] += eval_data["task_completion"]["score"]
            avg_scores["tool_usage"] += eval_data["tool_usage"]["score"]
            avg_scores["code_quality"] += eval_data["code_quality"]["score"]
            avg_scores["problem_solving"] += eval_data["problem_solving"]["score"]
            avg_scores["user_experience"] += eval_data["user_experience"]["score"]
            avg_scores["completeness"] += eval_data["completeness"]["score"]
            avg_scores["overall"] += eval_data["overall_score"]

        # 计算平均值
        n = len(evaluated_tasks)
        for key in avg_scores:
            avg_scores[key] = round(avg_scores[key] / n, 2)

        # 收集所有优点和缺点
        all_strengths = []
        all_weaknesses = []
        all_suggestions = []

        for result in evaluated_tasks:
            eval_data = result["llm_evaluation"]
            all_strengths.extend(eval_data.get("strengths", []))
            all_weaknesses.extend(eval_data.get("weaknesses", []))
            all_suggestions.extend(eval_data.get("suggestions", []))

        # 统计频率
        from collections import Counter

        strength_counts = Counter(all_strengths)
        weakness_counts = Counter(all_weaknesses)
        suggestion_counts = Counter(all_suggestions)

        return {
            "total_tasks": total_tasks,
            "evaluated_tasks": len(evaluated_tasks),
            "average_scores": avg_scores,
            "top_strengths": strength_counts.most_common(5),
            "top_weaknesses": weakness_counts.most_common(5),
            "top_suggestions": suggestion_counts.most_common(5),
        }


def main():
    """主函数"""
    print("LLM-as-Judge 评估器已准备就绪")
    print("\n使用方法:")
    print("  judge = LLMJudgeEvaluator()")
    print("  scores = judge.evaluate_response(task, response, actions)")
    print("  report = judge.generate_summary_report(results)")


if __name__ == "__main__":
    main()
