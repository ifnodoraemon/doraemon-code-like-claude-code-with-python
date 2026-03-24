"""
Model-based Grader for Evaluation

Uses LLM as a judge to evaluate agent outputs against rubrics.
Implements structured grading with JSON responses.
"""

import json
import os
from typing import Any

# Use new Google GenAI SDK (consistent with main CLI)
from google import genai
from src.core.config import get_required_config_value


class ModelGrader:
    """LLM-based grader for evaluating agent outputs."""

    def __init__(self, model_name: str | None = None):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set for ModelGrader")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name or get_required_config_value("model")

    def grade(self, task: str, agent_output: str, rubric: str) -> dict[str, Any]:
        """
        Use LLM as a judge to grade the Agent's output.

        Args:
            task: The original task description
            agent_output: The agent's response to evaluate
            rubric: Grading criteria

        Returns:
            Dictionary with score, reasoning, and pass status
        """
        prompt = f"""
        # Role
        You are an expert evaluator for AI agents. Your job is to grade the performance of an AI assistant based on its output.

        # Task
        {task}

        # Agent Output
        {agent_output}

        # Grading Rubric
        {rubric}

        # Instruction
        Evaluate the Agent Output against the Rubric.
        Return your response in STRICT JSON format with the following keys:
        - "score": (int) 1-5
        - "reasoning": (str) Explanation of the score
        - "pass": (bool) true if score >= 4
        """

        try:
            # Use new SDK API - contents must be a list
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(parts=[types.Part(text=prompt)])],
            )
            # Extract JSON from response - safely handle empty response
            if not response or not hasattr(response, "text") or not response.text:
                return {"score": 0, "pass": False, "reasoning": "Empty response from model"}
            text = response.text.strip()
            # More robust JSON extraction
            import re

            # Try to find JSON object in the response
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            # If text doesn't start with {, try to find a JSON object
            if not text.startswith("{"):
                json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
                if json_match:
                    text = json_match.group(0)

            return json.loads(text)
        except json.JSONDecodeError as e:
            return {"score": 0, "pass": False, "reasoning": f"JSON parsing failed: {str(e)}"}
        except Exception as e:
            return {"score": 0, "pass": False, "reasoning": f"Grading failed: {str(e)}"}


if __name__ == "__main__":
    # Test
    grader = ModelGrader()
    res = grader.grade(
        task="Write a haiku about code.",
        agent_output="Code flows like a stream\nBugs vanish in the clear water\nSoftware is alive",
        rubric="Must be a valid haiku (5-7-5 syllables). Must be about programming.",
    )
    print(res)
