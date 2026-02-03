"""Advanced comprehensive tests for planner.py - 25+ tests for improved coverage"""
import pytest
from datetime import datetime
from src.core.planner import (
    TaskStatus, TaskPriority, RiskLevel,
    TaskDependency, RiskAssessment, Task, ExecutionPlan,
    TaskPlanner
)


class TestTaskAdvanced:
    """Advanced tests for Task class."""

    def test_task_default_values(self):
        """Test task has correct default values."""
        task = Task(id="t1", title="Test", description="Desc")
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.MEDIUM
        assert task.complexity == 1
        assert task.estimated_minutes == 0
        assert task.dependencies == []
        assert task.subtasks == []
        assert task.risk is None
        assert task.files_affected == []
        assert task.checkpoint_recommended is False
        assert task.completed_at is None

    def test_task_to_dict_without_risk(self):
        """Test to_dict when risk is None."""
        task = Task(id="t1", title="Test", description="Desc")
        data = task.to_dict()
        assert data["risk"] is None

    def test_task_to_dict_with_subtasks(self):
        """Test to_dict includes subtasks."""
        subtask = Task(id="st1", title="Subtask", description="Sub")
        task = Task(
            id="t1",
            title="Parent",
            description="Parent task",
            subtasks=[subtask]
        )
        data = task.to_dict()
        assert len(data["subtasks"]) == 1
        assert data["subtasks"][0]["title"] == "Subtask"

    def test_task_from_dict_with_risk(self):
        """Test from_dict reconstructs risk."""
        data = {
            "id": "t1",
            "title": "Test",
            "description": "Desc",
            "risk": {
                "level": "high",
                "factors": ["Factor 1", "Factor 2"],
                "mitigations": ["Mit 1"]
            }
        }
        task = Task.from_dict(data)
        assert task.risk is not None
        assert task.risk.level == RiskLevel.HIGH
        assert len(task.risk.factors) == 2

    def test_task_from_dict_with_nested_subtasks(self):
        """Test from_dict with nested subtasks."""
        data = {
            "id": "t1",
            "title": "Parent",
            "description": "Parent",
            "subtasks": [
                {
                    "id": "st1",
                    "title": "Subtask",
                    "description": "Sub",
                    "subtasks": []
                }
            ]
        }
        task = Task.from_dict(data)
        assert len(task.subtasks) == 1
        assert task.subtasks[0].title == "Subtask"

    def test_task_from_dict_status_conversion(self):
        """Test from_dict converts status strings to enum."""
        data = {
            "id": "t1",
            "title": "Test",
            "description": "Desc",
            "status": "in_progress",
            "priority": "high"
        }
        task = Task.from_dict(data)
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.priority == TaskPriority.HIGH

    def test_task_from_dict_with_all_fields(self):
        """Test from_dict with all possible fields."""
        data = {
            "id": "t1",
            "title": "Complete Task",
            "description": "Full description",
            "status": "completed",
            "priority": "critical",
            "complexity": 5,
            "estimated_minutes": 120,
            "dependencies": [
                {"task_id": "t0", "type": "requires", "reason": "Must complete first"}
            ],
            "subtasks": [],
            "risk": {
                "level": "medium",
                "factors": ["Factor"],
                "mitigations": ["Mitigation"]
            },
            "files_affected": ["file1.py", "file2.py"],
            "checkpoint_recommended": True
        }
        task = Task.from_dict(data)
        assert task.complexity == 5
        assert task.estimated_minutes == 120
        assert len(task.dependencies) == 1
        assert len(task.files_affected) == 2
        assert task.checkpoint_recommended is True


class TestExecutionPlanAdvanced:
    """Advanced tests for ExecutionPlan."""

    def test_execution_plan_to_dict(self):
        """Test ExecutionPlan to_dict."""
        task = Task(id="t1", title="Task", description="Desc")
        plan = ExecutionPlan(
            id="plan1",
            goal="Complete project",
            tasks=[task],
            total_estimated_minutes=60,
            total_complexity=3,
            high_risk_count=1
        )
        data = plan.to_dict()
        assert data["id"] == "plan1"
        assert data["goal"] == "Complete project"
        assert len(data["tasks"]) == 1
        assert data["total_estimated_minutes"] == 60

    def test_execution_plan_to_markdown_empty(self):
        """Test markdown generation for empty plan."""
        plan = ExecutionPlan(
            id="plan1",
            goal="Test goal",
            tasks=[],
            total_estimated_minutes=0,
            total_complexity=0,
            high_risk_count=0
        )
        markdown = plan.to_markdown()
        assert "# Execution Plan" in markdown
        assert "Test goal" in markdown
        assert "0 minutes" in markdown

    def test_execution_plan_to_markdown_with_tasks(self):
        """Test markdown generation with tasks."""
        task = Task(
            id="t1",
            title="Implementation",
            description="Implement feature",
            complexity=3,
            estimated_minutes=30,
            priority=TaskPriority.HIGH
        )
        plan = ExecutionPlan(
            id="plan1",
            goal="Implement feature",
            tasks=[task],
            total_estimated_minutes=30,
            total_complexity=3,
            high_risk_count=0
        )
        markdown = plan.to_markdown()
        assert "Implementation" in markdown
        assert "30 min" in markdown
        assert "3/5" in markdown

    def test_execution_plan_to_markdown_with_dependencies(self):
        """Test markdown includes dependencies."""
        task1 = Task(id="t1", title="Task 1", description="First")
        task2 = Task(
            id="t2",
            title="Task 2",
            description="Second",
            dependencies=[TaskDependency("t1", "requires")]
        )
        plan = ExecutionPlan(
            id="plan1",
            goal="Multi-task",
            tasks=[task1, task2],
            total_estimated_minutes=60,
            total_complexity=3,
            high_risk_count=0
        )
        markdown = plan.to_markdown()
        assert "Depends on:" in markdown
        assert "t1" in markdown

    def test_execution_plan_to_markdown_with_risk(self):
        """Test markdown includes risk information."""
        risk = RiskAssessment(
            level=RiskLevel.HIGH,
            factors=["Destructive", "Production"],
            mitigations=["Backup", "Test"]
        )
        task = Task(
            id="t1",
            title="Risky Task",
            description="High risk",
            risk=risk
        )
        plan = ExecutionPlan(
            id="plan1",
            goal="Risky operation",
            tasks=[task],
            total_estimated_minutes=60,
            total_complexity=5,
            high_risk_count=1
        )
        markdown = plan.to_markdown()
        assert "Risk:" in markdown or "HIGH" in markdown

    def test_execution_plan_to_markdown_with_subtasks(self):
        """Test markdown includes subtasks."""
        subtask = Task(id="st1", title="Subtask", description="Sub")
        task = Task(
            id="t1",
            title="Parent",
            description="Parent task",
            subtasks=[subtask]
        )
        plan = ExecutionPlan(
            id="plan1",
            goal="With subtasks",
            tasks=[task],
            total_estimated_minutes=60,
            total_complexity=3,
            high_risk_count=0
        )
        markdown = plan.to_markdown()
        assert "Subtasks:" in markdown
        assert "Subtask" in markdown


class TestTaskPlannerComplexityEstimation:
    """Tests for complexity estimation."""

    def test_estimate_complexity_keywords(self):
        """Test complexity estimation with keywords."""
        planner = TaskPlanner()

        # Level 5
        assert planner._estimate_complexity("refactor the architecture") == 5
        assert planner._estimate_complexity("security migration") == 5

        # Level 4
        assert planner._estimate_complexity("integrate new API") == 4
        assert planner._estimate_complexity("optimize database") == 4

        # Level 3
        assert planner._estimate_complexity("implement new feature") == 3

        # Level 2
        assert planner._estimate_complexity("fix bug in code") == 2
        assert planner._estimate_complexity("fix typo") == 2  # "fix" is level 2

    def test_estimate_complexity_default(self):
        """Test complexity estimation defaults to 2."""
        planner = TaskPlanner()
        complexity = planner._estimate_complexity("random task description")
        assert complexity == 2

    def test_estimate_time_mapping(self):
        """Test time estimation for each complexity level."""
        planner = TaskPlanner()
        assert planner._estimate_time(1) == 5
        assert planner._estimate_time(2) == 15
        assert planner._estimate_time(3) == 30
        assert planner._estimate_time(4) == 60
        assert planner._estimate_time(5) == 120

    def test_estimate_time_unknown_complexity(self):
        """Test time estimation for unknown complexity."""
        planner = TaskPlanner()
        time = planner._estimate_time(10)
        assert time == 30  # Default


class TestTaskPlannerDecomposition:
    """Tests for task decomposition."""

    def test_decompose_implementation_goal(self):
        """Test decomposition of implementation goal."""
        planner = TaskPlanner()
        tasks = planner._decompose_goal("implement user authentication", {})
        assert len(tasks) == 4
        assert any("Analyze" in t.title for t in tasks)
        assert any("Design" in t.title for t in tasks)
        assert any("Implement" in t.title for t in tasks)
        assert any("Test" in t.title for t in tasks)

    def test_decompose_bugfix_goal(self):
        """Test decomposition of bug fix goal."""
        planner = TaskPlanner()
        tasks = planner._decompose_goal("fix login bug", {})
        assert len(tasks) == 4
        assert any("Reproduce" in t.title for t in tasks)
        assert any("Investigate" in t.title for t in tasks)
        assert any("Fix" in t.title for t in tasks)
        assert any("Verify" in t.title for t in tasks)

    def test_decompose_refactor_goal(self):
        """Test decomposition of refactoring goal."""
        planner = TaskPlanner()
        tasks = planner._decompose_goal("refactor database layer", {})
        assert len(tasks) == 4
        assert any("Analyze" in t.title for t in tasks)
        assert any("Plan" in t.title for t in tasks)
        assert any("Refactor" in t.title for t in tasks)
        assert any("Run Tests" in t.title for t in tasks)

    def test_decompose_testing_goal(self):
        """Test decomposition of testing goal."""
        planner = TaskPlanner()
        tasks = planner._decompose_goal("test payment module", {})
        assert len(tasks) == 3
        assert any("Identify" in t.title for t in tasks)
        assert any("Write" in t.title for t in tasks)
        assert any("Run" in t.title for t in tasks)

    def test_decompose_generic_goal(self):
        """Test decomposition of generic goal."""
        planner = TaskPlanner()
        tasks = planner._decompose_goal("do something random", {})
        assert len(tasks) == 1
        assert tasks[0].title == "do something random"


class TestTaskPlannerRiskAssessment:
    """Tests for risk assessment."""

    def test_assess_risk_low(self):
        """Test low risk assessment."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Add comment", description="Add documentation")
        risk = planner._assess_risk(task)
        assert risk.level == RiskLevel.LOW

    def test_assess_risk_delete_operation(self):
        """Test high risk for delete operations."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Delete old files", description="Remove deprecated code")
        risk = planner._assess_risk(task)
        assert risk.level == RiskLevel.HIGH
        assert any("Destructive" in f for f in risk.factors)

    def test_assess_risk_production(self):
        """Test high risk for production changes."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Deploy", description="Deploy to production")
        risk = planner._assess_risk(task)
        assert risk.level == RiskLevel.HIGH
        assert any("production" in f.lower() for f in risk.factors)

    def test_assess_risk_database(self):
        """Test medium/high risk for database changes."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Migrate", description="Database migration")
        risk = planner._assess_risk(task)
        assert risk.level in [RiskLevel.MEDIUM, RiskLevel.HIGH]
        assert any("Database" in f for f in risk.factors)

    def test_assess_risk_security(self):
        """Test high risk for security changes."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Update", description="Update security credentials")
        risk = planner._assess_risk(task)
        assert risk.level == RiskLevel.HIGH
        assert any("Security" in f for f in risk.factors)

    def test_assess_risk_has_mitigations(self):
        """Test risk assessment includes mitigations."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Delete", description="Delete database")
        risk = planner._assess_risk(task)
        assert len(risk.mitigations) > 0


class TestTaskPlannerCheckpoints:
    """Tests for checkpoint recommendations."""

    def test_recommend_checkpoints_high_complexity(self):
        """Test checkpoint recommended for high complexity."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Complex", description="Desc", complexity=4)
        tasks = [task]
        planner._recommend_checkpoints(tasks)
        assert task.checkpoint_recommended is True

    def test_recommend_checkpoints_high_risk(self):
        """Test checkpoint recommended for high risk."""
        planner = TaskPlanner()
        risk = RiskAssessment(level=RiskLevel.HIGH)
        task = Task(id="t1", title="Risky", description="Desc", risk=risk)
        tasks = [task]
        planner._recommend_checkpoints(tasks)
        assert task.checkpoint_recommended is True

    def test_recommend_checkpoints_many_files(self):
        """Test checkpoint recommended for many files."""
        planner = TaskPlanner()
        task = Task(
            id="t1",
            title="Multi-file",
            description="Desc",
            files_affected=["f1.py", "f2.py", "f3.py"]
        )
        tasks = [task]
        planner._recommend_checkpoints(tasks)
        assert task.checkpoint_recommended is True

    def test_recommend_checkpoints_low_risk(self):
        """Test no checkpoint for low risk."""
        planner = TaskPlanner()
        task = Task(
            id="t1",
            title="Simple",
            description="Desc",
            complexity=1,
            risk=RiskAssessment(level=RiskLevel.LOW)
        )
        tasks = [task]
        planner._recommend_checkpoints(tasks)
        assert task.checkpoint_recommended is False


class TestTaskPlannerTaskManagement:
    """Tests for task status management."""

    def test_update_task_status_found(self):
        """Test updating task status when found."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Test", description="Desc")
        plan = ExecutionPlan(
            id="plan1",
            goal="Test",
            tasks=[task],
            total_estimated_minutes=0,
            total_complexity=0,
            high_risk_count=0
        )
        result = planner.update_task_status(plan, "t1", TaskStatus.COMPLETED)
        assert result is True
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None

    def test_update_task_status_not_found(self):
        """Test updating task status when not found."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Test", description="Desc")
        plan = ExecutionPlan(
            id="plan1",
            goal="Test",
            tasks=[task],
            total_estimated_minutes=0,
            total_complexity=0,
            high_risk_count=0
        )
        result = planner.update_task_status(plan, "t999", TaskStatus.COMPLETED)
        assert result is False

    def test_update_subtask_status(self):
        """Test updating subtask status."""
        planner = TaskPlanner()
        subtask = Task(id="st1", title="Subtask", description="Sub")
        task = Task(
            id="t1",
            title="Parent",
            description="Desc",
            subtasks=[subtask]
        )
        plan = ExecutionPlan(
            id="plan1",
            goal="Test",
            tasks=[task],
            total_estimated_minutes=0,
            total_complexity=0,
            high_risk_count=0
        )
        result = planner.update_task_status(plan, "st1", TaskStatus.COMPLETED)
        assert result is True
        assert subtask.status == TaskStatus.COMPLETED


class TestTaskPlannerNextTasks:
    """Tests for getting next executable tasks."""

    def test_get_next_tasks_no_dependencies(self):
        """Test getting next tasks with no dependencies."""
        planner = TaskPlanner()
        t1 = Task(id="t1", title="Task 1", description="Desc")
        t2 = Task(id="t2", title="Task 2", description="Desc")
        plan = ExecutionPlan(
            id="plan1",
            goal="Test",
            tasks=[t1, t2],
            total_estimated_minutes=0,
            total_complexity=0,
            high_risk_count=0
        )
        next_tasks = planner.get_next_tasks(plan)
        assert len(next_tasks) == 2

    def test_get_next_tasks_with_dependencies(self):
        """Test getting next tasks respects dependencies."""
        planner = TaskPlanner()
        t1 = Task(id="t1", title="Task 1", description="Desc")
        t2 = Task(
            id="t2",
            title="Task 2",
            description="Desc",
            dependencies=[TaskDependency("t1", "requires")]
        )
        plan = ExecutionPlan(
            id="plan1",
            goal="Test",
            tasks=[t1, t2],
            total_estimated_minutes=0,
            total_complexity=0,
            high_risk_count=0
        )
        next_tasks = planner.get_next_tasks(plan)
        assert len(next_tasks) == 1
        assert next_tasks[0].id == "t1"

    def test_get_next_tasks_after_completion(self):
        """Test getting next tasks after completing one."""
        planner = TaskPlanner()
        t1 = Task(id="t1", title="Task 1", description="Desc", status=TaskStatus.COMPLETED)
        t2 = Task(
            id="t2",
            title="Task 2",
            description="Desc",
            dependencies=[TaskDependency("t1", "requires")]
        )
        plan = ExecutionPlan(
            id="plan1",
            goal="Test",
            tasks=[t1, t2],
            total_estimated_minutes=0,
            total_complexity=0,
            high_risk_count=0
        )
        next_tasks = planner.get_next_tasks(plan)
        assert len(next_tasks) == 1
        assert next_tasks[0].id == "t2"

    def test_get_next_tasks_priority_sorting(self):
        """Test next tasks are sorted by priority."""
        planner = TaskPlanner()
        t1 = Task(id="t1", title="Low", description="Desc", priority=TaskPriority.LOW)
        t2 = Task(id="t2", title="High", description="Desc", priority=TaskPriority.HIGH)
        t3 = Task(id="t3", title="Critical", description="Desc", priority=TaskPriority.CRITICAL)
        plan = ExecutionPlan(
            id="plan1",
            goal="Test",
            tasks=[t1, t2, t3],
            total_estimated_minutes=0,
            total_complexity=0,
            high_risk_count=0
        )
        next_tasks = planner.get_next_tasks(plan)
        assert next_tasks[0].priority == TaskPriority.CRITICAL
        assert next_tasks[1].priority == TaskPriority.HIGH
        assert next_tasks[2].priority == TaskPriority.LOW

    def test_get_next_tasks_skips_non_pending(self):
        """Test next tasks skips non-pending tasks."""
        planner = TaskPlanner()
        t1 = Task(id="t1", title="Pending", description="Desc", status=TaskStatus.PENDING)
        t2 = Task(id="t2", title="In Progress", description="Desc", status=TaskStatus.IN_PROGRESS)
        t3 = Task(id="t3", title="Completed", description="Desc", status=TaskStatus.COMPLETED)
        plan = ExecutionPlan(
            id="plan1",
            goal="Test",
            tasks=[t1, t2, t3],
            total_estimated_minutes=0,
            total_complexity=0,
            high_risk_count=0
        )
        next_tasks = planner.get_next_tasks(plan)
        assert len(next_tasks) == 1
        assert next_tasks[0].id == "t1"


class TestTaskPlannerIDGeneration:
    """Tests for ID generation."""

    def test_generate_id_uniqueness(self):
        """Test generated IDs are unique."""
        planner = TaskPlanner()
        id1 = planner._generate_id()
        id2 = planner._generate_id()
        assert id1 != id2

    def test_generate_id_format(self):
        """Test generated ID format."""
        planner = TaskPlanner()
        task_id = planner._generate_id("task")
        assert task_id.startswith("task_")
        assert len(task_id) > 5

    def test_generate_id_custom_prefix(self):
        """Test ID generation with custom prefix."""
        planner = TaskPlanner()
        plan_id = planner._generate_id("plan")
        assert plan_id.startswith("plan_")


class TestTaskPlannerFullPlanGeneration:
    """Tests for full plan generation."""

    def test_generate_plan_implementation(self):
        """Test plan generation for implementation."""
        planner = TaskPlanner()
        plan = planner.generate_plan("Implement user authentication")
        assert plan.goal == "Implement user authentication"
        assert len(plan.tasks) > 0
        assert plan.total_estimated_minutes > 0
        assert plan.total_complexity > 0

    def test_generate_plan_with_context(self):
        """Test plan generation with context."""
        planner = TaskPlanner()
        context = {"files": ["auth.py", "models.py"]}
        plan = planner.generate_plan("Implement authentication", context)
        assert plan.goal == "Implement authentication"
        assert len(plan.tasks) > 0

    def test_generate_plan_has_dependencies(self):
        """Test generated plan has task dependencies."""
        planner = TaskPlanner()
        plan = planner.generate_plan("Implement feature")
        # Most plans should have some dependencies
        has_deps = any(len(t.dependencies) > 0 for t in plan.tasks)
        assert has_deps or len(plan.tasks) == 1

    def test_generate_plan_has_risk_assessment(self):
        """Test generated plan includes risk assessment."""
        planner = TaskPlanner()
        plan = planner.generate_plan("Delete old database")
        # High-risk operations should have risk assessment
        has_risk = any(t.risk is not None for t in plan.tasks)
        assert has_risk or plan.high_risk_count == 0

    def test_generate_plan_markdown_output(self):
        """Test plan can be converted to markdown."""
        planner = TaskPlanner()
        plan = planner.generate_plan("Test something")
        markdown = plan.to_markdown()
        assert "# Execution Plan" in markdown
        assert "Test something" in markdown
