"""
Intelligent Task Planner

Provides AI-powered task planning and decomposition:
- Automatic task breakdown
- Dependency analysis
- Complexity estimation
- Risk assessment
- Checkpoint suggestions

This enables smarter planning mode in Doraemon.

This module serves as the public API, re-exporting from specialized submodules:
- planner_output: Data models and serialization
- planner_analysis: Analysis and risk assessment
- planner_core: Core planning logic
"""

# Re-export all public APIs
from .planner_output import (
    TaskStatus,
    TaskPriority,
    RiskLevel,
    TaskDependency,
    RiskAssessment,
    Task,
    ExecutionPlan,
)
from .planner_core import TaskPlanner
from .planner_analysis import TaskAnalyzer

__all__ = [
    # Enums
    "TaskStatus",
    "TaskPriority",
    "RiskLevel",
    # Data classes
    "TaskDependency",
    "RiskAssessment",
    "Task",
    "ExecutionPlan",
    # Main classes
    "TaskPlanner",
    "TaskAnalyzer",
]
