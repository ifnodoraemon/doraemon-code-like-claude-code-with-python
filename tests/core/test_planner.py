"""Comprehensive tests for planner.py"""

from src.core.planner import (
    ExecutionPlan,
    RiskAssessment,
    RiskLevel,
    Task,
    TaskDependency,
    TaskPlanner,
    TaskPriority,
    TaskStatus,
)


class TestEnums:
    """Tests for enum types."""

    def test_task_status_values(self):
        """Test TaskStatus enum values."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.BLOCKED.value == "blocked"
        assert TaskStatus.FAILED.value == "failed"

    def test_task_priority_values(self):
        """Test TaskPriority enum values."""
        assert TaskPriority.CRITICAL.value == "critical"
        assert TaskPriority.HIGH.value == "high"
        assert TaskPriority.MEDIUM.value == "medium"
        assert TaskPriority.LOW.value == "low"

    def test_risk_level_values(self):
        """Test RiskLevel enum values."""
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"


class TestTaskDependency:
    """Tests for TaskDependency."""

    def test_creation(self):
        """Test creating a task dependency."""
        dep = TaskDependency(
            task_id="task_123",
            dependency_type="requires",
            reason="Must complete first"
        )
        assert dep.task_id == "task_123"
        assert dep.dependency_type == "requires"
        assert dep.reason == "Must complete first"

    def test_creation_without_reason(self):
        """Test creating dependency without reason."""
        dep = TaskDependency(task_id="task_456", dependency_type="blocked_by")
        assert dep.task_id == "task_456"
        assert dep.reason is None


class TestRiskAssessment:
    """Tests for RiskAssessment."""

    def test_creation(self):
        """Test creating risk assessment."""
        risk = RiskAssessment(
            level=RiskLevel.HIGH,
            factors=["Destructive operation", "Production impact"],
            mitigations=["Create backup", "Test in staging"]
        )
        assert risk.level == RiskLevel.HIGH
        assert len(risk.factors) == 2
        assert len(risk.mitigations) == 2

    def test_creation_with_defaults(self):
        """Test risk assessment with default lists."""
        risk = RiskAssessment(level=RiskLevel.LOW)
        assert risk.factors == []
        assert risk.mitigations == []


class TestTask:
    """Tests for Task class."""

    def test_basic_creation(self):
        """Test creating a basic task."""
        task = Task(
            id="task_001",
            title="Test Task",
            description="A test task"
        )
        assert task.id == "task_001"
        assert task.title == "Test Task"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.MEDIUM
        assert task.complexity == 1

    def test_creation_with_all_fields(self):
        """Test creating task with all fields."""
        risk = RiskAssessment(level=RiskLevel.HIGH)
        dep = TaskDependency(task_id="task_000", dependency_type="requires")
        task = Task(
            id="task_002",
            title="Complex Task",
            description="Complex description",
            status=TaskStatus.IN_PROGRESS,
            priority=TaskPriority.HIGH,
            complexity=5,
            estimated_minutes=120,
            dependencies=[dep],
            risk=risk,
            files_affected=["file1.py", "file2.py"],
            checkpoint_recommended=True
        )
        assert task.complexity == 5
        assert task.estimated_minutes == 120
        assert len(task.dependencies) == 1
        assert task.risk.level == RiskLevel.HIGH
        assert len(task.files_affected) == 2
        assert task.checkpoint_recommended is True

    def test_to_dict(self):
        """Test converting task to dictionary."""
        task = Task(
            id="task_003",
            title="Dict Task",
            description="Test dict conversion"
        )
        data = task.to_dict()
        assert data["id"] == "task_003"
        assert data["title"] == "Dict Task"
        assert data["status"] == "pending"
        assert data["priority"] == "medium"

    def test_to_dict_with_risk(self):
        """Test dict conversion with risk."""
        risk = RiskAssessment(
            level=RiskLevel.MEDIUM,
            factors=["Factor 1"],
            mitigations=["Mitigation 1"]
        )
        task = Task(
            id="task_004",
            title="Risky Task",
            description="Has risk",
            risk=risk
        )
        data = task.to_dict()
        assert data["risk"] is not None
        assert data["risk"]["level"] == "medium"
        assert len(data["risk"]["factors"]) == 1

    def test_from_dict_basic(self):
        """Test creating task from dictionary."""
        data = {
            "id": "task_005",
            "title": "From Dict",
            "description": "Created from dict",
            "status": "completed",
            "priority": "high",
            "complexity": 3,
            "estimated_minutes": 45
        }
        task = Task.from_dict(data)
        assert task.id == "task_005"
        assert task.status == TaskStatus.COMPLETED
        assert task.priority == TaskPriority.HIGH
        assert task.complexity == 3

    def test_from_dict_with_dependencies(self):
        """Test from_dict with dependencies."""
        data = {
            "id": "task_006",
            "title": "With Deps",
            "description": "Has dependencies",
            "dependencies": [
                {"task_id": "task_001", "type": "requires", "reason": "Needs it"}
            ]
        }
        task = Task.from_dict(data)
        assert len(task.dependencies) == 1
        assert task.dependencies[0].task_id == "task_001"

    def test_from_dict_with_subtasks(self):
        """Test from_dict with subtasks."""
        data = {
            "id": "task_007",
            "title": "Parent",
            "description": "Has subtasks",
            "subtasks": [
                {"id": "sub_001", "title": "Subtask 1", "description": "First subtask"}
            ]
        }
        task = Task.from_dict(data)
        assert len(task.subtasks) == 1
        assert task.subtasks[0].id == "sub_001"


class TestExecutionPlan:
    """Tests for ExecutionPlan."""

    def test_creation(self):
        """Test creating execution plan."""
        task1 = Task(id="t1", title="Task 1", description="First")
        task2 = Task(id="t2", title="Task 2", description="Second")
        plan = ExecutionPlan(
            id="plan_001",
            goal="Complete project",
            tasks=[task1, task2],
            total_estimated_minutes=90,
            total_complexity=3,
            high_risk_count=1
        )
        assert plan.id == "plan_001"
        assert plan.goal == "Complete project"
        assert len(plan.tasks) == 2
        assert plan.total_estimated_minutes == 90

    def test_to_dict(self):
        """Test converting plan to dictionary."""
        task = Task(id="t1", title="Task", description="Desc")
        plan = ExecutionPlan(
            id="plan_002",
            goal="Test goal",
            tasks=[task],
            total_estimated_minutes=30,
            total_complexity=2,
            high_risk_count=0
        )
        data = plan.to_dict()
        assert data["id"] == "plan_002"
        assert data["goal"] == "Test goal"
        assert len(data["tasks"]) == 1

    def test_to_markdown(self):
        """Test converting plan to markdown."""
        task = Task(
            id="t1",
            title="Test Task",
            description="Test description",
            complexity=3,
            estimated_minutes=30,
            priority=TaskPriority.HIGH
        )
        plan = ExecutionPlan(
            id="plan_003",
            goal="Test markdown",
            tasks=[task],
            total_estimated_minutes=30,
            total_complexity=3,
            high_risk_count=0
        )
        markdown = plan.to_markdown()
        assert "# Execution Plan" in markdown
        assert "Test markdown" in markdown
        assert "Test Task" in markdown
        assert "30 min" in markdown


class TestTaskPlanner:
    """Tests for TaskPlanner."""

    def test_initialization(self):
        """Test TaskPlanner initialization."""
        planner = TaskPlanner()
        assert planner._id_counter == 0

    def test_generate_id(self):
        """Test ID generation."""
        planner = TaskPlanner()
        id1 = planner._generate_id()
        id2 = planner._generate_id()
        assert id1.startswith("task_")
        assert id2.startswith("task_")
        assert id1 != id2

    def test_estimate_complexity_simple(self):
        """Test complexity estimation for simple tasks."""
        planner = TaskPlanner()
        # "fix" keyword maps to complexity 2, use "typo" for complexity 1
        complexity = planner._analyzer.estimate_complexity("typo in readme")
        assert complexity == 1

    def test_estimate_complexity_medium(self):
        """Test complexity estimation for medium tasks."""
        planner = TaskPlanner()
        complexity = planner._analyzer.estimate_complexity("implement new feature")
        assert complexity == 3

    def test_estimate_complexity_high(self):
        """Test complexity estimation for high tasks."""
        planner = TaskPlanner()
        complexity = planner._analyzer.estimate_complexity("refactor entire architecture")
        assert complexity == 5

    def test_estimate_time(self):
        """Test time estimation."""
        planner = TaskPlanner()
        assert planner._analyzer.estimate_time(1) == 5
        assert planner._analyzer.estimate_time(3) == 30
        assert planner._analyzer.estimate_time(5) == 120

    def test_generate_plan_implementation(self):
        """Test generating plan for implementation task."""
        planner = TaskPlanner()
        plan = planner.generate_plan("Implement user authentication")
        assert plan.goal == "Implement user authentication"
        assert len(plan.tasks) > 0
        assert plan.total_estimated_minutes > 0

    def test_generate_plan_bugfix(self):
        """Test generating plan for bug fix."""
        planner = TaskPlanner()
        plan = planner.generate_plan("Fix login bug")
        assert len(plan.tasks) > 0
        # Bug fix should have reproduce, investigate, fix, verify steps
        task_titles = [t.title for t in plan.tasks]
        assert any("Reproduce" in title for title in task_titles)

    def test_generate_plan_refactor(self):
        """Test generating plan for refactoring."""
        planner = TaskPlanner()
        plan = planner.generate_plan("Refactor database layer")
        assert len(plan.tasks) > 0
        # Refactor should have analyze, plan, refactor, test steps
        task_titles = [t.title for t in plan.tasks]
        assert any("Analyze" in title or "Refactor" in title for title in task_titles)

    def test_generate_plan_testing(self):
        """Test generating plan for testing task."""
        planner = TaskPlanner()
        plan = planner.generate_plan("Write tests for API")
        assert len(plan.tasks) > 0

    def test_assess_risk_low(self):
        """Test risk assessment for low risk task."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Add comment", description="Add documentation comment")
        risk = planner._analyzer.assess_risk(task)
        assert risk.level == RiskLevel.LOW

    def test_assess_risk_high_delete(self):
        """Test risk assessment for destructive operation."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Delete database", description="Remove old database")
        risk = planner._analyzer.assess_risk(task)
        assert risk.level == RiskLevel.HIGH
        assert any("Destructive" in f for f in risk.factors)

    def test_assess_risk_high_production(self):
        """Test risk assessment for production changes."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Deploy", description="Deploy to production")
        risk = planner._analyzer.assess_risk(task)
        assert risk.level in [RiskLevel.MEDIUM, RiskLevel.HIGH]

    def test_recommend_checkpoints_high_complexity(self):
        """Test checkpoint recommendation for high complexity."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Complex", description="Complex task", complexity=5)
        planner._analyzer.recommend_checkpoints([task])
        assert task.checkpoint_recommended is True

    def test_recommend_checkpoints_high_risk(self):
        """Test checkpoint recommendation for high risk."""
        planner = TaskPlanner()
        risk = RiskAssessment(level=RiskLevel.HIGH)
        task = Task(id="t1", title="Risky", description="Risky task", risk=risk)
        planner._analyzer.recommend_checkpoints([task])
        assert task.checkpoint_recommended is True

    def test_recommend_checkpoints_many_files(self):
        """Test checkpoint recommendation for many affected files."""
        planner = TaskPlanner()
        task = Task(
            id="t1",
            title="Multi-file",
            description="Affects many files",
            files_affected=["f1", "f2", "f3", "f4"]
        )
        planner._analyzer.recommend_checkpoints([task])
        assert task.checkpoint_recommended is True

    def test_update_task_status(self):
        """Test updating task status."""
        planner = TaskPlanner()
        task = Task(id="t1", title="Task", description="Desc")
        plan = ExecutionPlan(
            id="p1",
            goal="Goal",
            tasks=[task],
            total_estimated_minutes=30,
            total_complexity=2,
            high_risk_count=0
        )
        result = planner.update_task_status(plan, "t1", TaskStatus.COMPLETED)
        assert result is True
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None

    def test_update_task_status_not_found(self):
        """Test updating non-existent task."""
        planner = TaskPlanner()
        plan = ExecutionPlan(
            id="p1",
            goal="Goal",
            tasks=[],
            total_estimated_minutes=0,
            total_complexity=0,
            high_risk_count=0
        )
        result = planner.update_task_status(plan, "nonexistent", TaskStatus.COMPLETED)
        assert result is False

    def test_get_next_tasks_no_dependencies(self):
        """Test getting next tasks with no dependencies."""
        planner = TaskPlanner()
        task1 = Task(id="t1", title="Task 1", description="First")
        task2 = Task(id="t2", title="Task 2", description="Second")
        plan = ExecutionPlan(
            id="p1",
            goal="Goal",
            tasks=[task1, task2],
            total_estimated_minutes=60,
            total_complexity=2,
            high_risk_count=0
        )
        next_tasks = planner.get_next_tasks(plan)
        assert len(next_tasks) == 2

    def test_get_next_tasks_with_dependencies(self):
        """Test getting next tasks with dependencies."""
        planner = TaskPlanner()
        task1 = Task(id="t1", title="Task 1", description="First")
        task2 = Task(
            id="t2",
            title="Task 2",
            description="Second",
            dependencies=[TaskDependency(task_id="t1", dependency_type="requires")]
        )
        plan = ExecutionPlan(
            id="p1",
            goal="Goal",
            tasks=[task1, task2],
            total_estimated_minutes=60,
            total_complexity=2,
            high_risk_count=0
        )
        # Initially, only task1 should be ready
        next_tasks = planner.get_next_tasks(plan)
        assert len(next_tasks) == 1
        assert next_tasks[0].id == "t1"

        # After completing task1, task2 should be ready
        task1.status = TaskStatus.COMPLETED
        next_tasks = planner.get_next_tasks(plan)
        assert len(next_tasks) == 1
        assert next_tasks[0].id == "t2"

    def test_get_next_tasks_priority_sorting(self):
        """Test that next tasks are sorted by priority."""
        planner = TaskPlanner()
        task_low = Task(id="t1", title="Low", description="Low", priority=TaskPriority.LOW)
        task_high = Task(id="t2", title="High", description="High", priority=TaskPriority.HIGH)
        task_critical = Task(id="t3", title="Critical", description="Critical", priority=TaskPriority.CRITICAL)
        plan = ExecutionPlan(
            id="p1",
            goal="Goal",
            tasks=[task_low, task_high, task_critical],
            total_estimated_minutes=90,
            total_complexity=3,
            high_risk_count=0
        )
        next_tasks = planner.get_next_tasks(plan)
        # Should be sorted: critical, high, low
        assert next_tasks[0].priority == TaskPriority.CRITICAL
        assert next_tasks[1].priority == TaskPriority.HIGH
        assert next_tasks[2].priority == TaskPriority.LOW
