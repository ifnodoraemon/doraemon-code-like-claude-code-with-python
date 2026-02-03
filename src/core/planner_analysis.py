"""
Task Analysis and Risk Assessment

Provides analysis capabilities for task planning:
- Complexity estimation from text
- Time estimation based on complexity
- Risk assessment with factors and mitigations
- Dependency analysis
- Checkpoint recommendations
"""

import logging
from typing import Any

from .planner_output import (
    Task, TaskPriority, RiskLevel, RiskAssessment
)

logger = logging.getLogger(__name__)


class TaskAnalyzer:
    """Analyzes tasks for complexity, risk, and dependencies."""

    # Complexity keywords for estimation
    COMPLEXITY_KEYWORDS = {
        5: ["refactor", "redesign", "migration", "architecture", "security"],
        4: ["integrate", "optimize", "test suite", "api", "database"],
        3: ["implement", "feature", "endpoint", "component"],
        2: ["update", "modify", "fix", "bug"],
        1: ["typo", "comment", "rename", "config", "doc"],
    }

    # Risk keywords
    RISK_KEYWORDS = {
        "high": ["delete", "drop", "remove", "migration", "production", "credentials", "security"],
        "medium": ["refactor", "update", "database", "api", "config"],
        "low": ["add", "create", "doc", "test", "comment"],
    }

    def estimate_complexity(self, text: str) -> int:
        """
        Estimate complexity from text.

        Args:
            text: Task description or title

        Returns:
            Complexity level 1-5
        """
        text_lower = text.lower()
        for complexity, keywords in self.COMPLEXITY_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                return complexity
        return 2  # Default medium

    def estimate_time(self, complexity: int) -> int:
        """
        Estimate time in minutes based on complexity.

        Args:
            complexity: Complexity level 1-5

        Returns:
            Estimated time in minutes
        """
        time_map = {1: 5, 2: 15, 3: 30, 4: 60, 5: 120}
        return time_map.get(complexity, 30)

    def assess_risk(self, task: Task) -> RiskAssessment:
        """
        Assess risk for a task.

        Args:
            task: Task to assess

        Returns:
            RiskAssessment with level, factors, and mitigations
        """
        text = f"{task.title} {task.description}".lower()

        # Check for high risk keywords
        for level, keywords in self.RISK_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                factors = []
                mitigations = []

                if "delete" in text or "remove" in text:
                    factors.append("Destructive operation")
                    mitigations.append("Create backup before proceeding")

                if "production" in text:
                    factors.append("Affects production")
                    mitigations.append("Test in staging first")

                if "database" in text or "migration" in text:
                    factors.append("Database changes")
                    mitigations.append("Create database backup")

                if "security" in text or "credentials" in text:
                    factors.append("Security-sensitive")
                    mitigations.append("Review security implications")

                return RiskAssessment(
                    level=RiskLevel(level),
                    factors=factors,
                    mitigations=mitigations,
                )

        return RiskAssessment(level=RiskLevel.LOW)

    def analyze_dependencies(self, tasks: list[Task]):
        """
        Analyze and enrich task dependencies.

        Args:
            tasks: List of tasks to analyze

        Note:
            Dependencies are set during task creation.
            This method can add cross-task dependencies if needed.
        """
        # Dependencies are set during task creation
        # This method can add cross-task dependencies if needed
        pass

    def recommend_checkpoints(self, tasks: list[Task]):
        """
        Recommend checkpoints for tasks.

        Args:
            tasks: List of tasks to analyze

        Updates:
            Sets checkpoint_recommended flag on tasks that need checkpoints
        """
        for task in tasks:
            # Recommend checkpoint for high complexity or high risk
            if task.complexity >= 4:
                task.checkpoint_recommended = True
            if task.risk and task.risk.level in [RiskLevel.MEDIUM, RiskLevel.HIGH]:
                task.checkpoint_recommended = True
            # Files affected
            if len(task.files_affected) >= 3:
                task.checkpoint_recommended = True
