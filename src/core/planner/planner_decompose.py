"""
Task Decomposition Patterns

Provides pattern-based task decomposition for different goal types:
- Implementation tasks
- Bug fix tasks
- Refactoring tasks
- Testing tasks
"""

from .planner_output import Task, TaskDependency, TaskPriority


class TaskDecomposer:
    """Decomposes goals into task sequences based on patterns."""

    def __init__(self, id_generator):
        """
        Initialize decomposer.

        Args:
            id_generator: Function to generate unique task IDs
        """
        self._generate_id = id_generator

    def create_implementation_tasks(self, goal: str, context: dict) -> list[Task]:
        """Create tasks for an implementation goal."""
        planning_task = Task(
            id=self._generate_id(),
            title="Analyze Requirements",
            description=f"Understand what needs to be built for: {goal}",
            complexity=2,
            estimated_minutes=15,
            priority=TaskPriority.HIGH,
        )

        design_task = Task(
            id=self._generate_id(),
            title="Design Solution",
            description="Plan the implementation approach and identify components",
            complexity=3,
            estimated_minutes=20,
            priority=TaskPriority.HIGH,
            dependencies=[TaskDependency(planning_task.id, "requires")],
        )

        implement_task = Task(
            id=self._generate_id(),
            title="Implement Solution",
            description=f"Write the code to: {goal}",
            complexity=4,
            estimated_minutes=60,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(design_task.id, "requires")],
            checkpoint_recommended=True,
        )

        test_task = Task(
            id=self._generate_id(),
            title="Test Implementation",
            description="Write and run tests to verify the implementation",
            complexity=2,
            estimated_minutes=20,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(implement_task.id, "requires")],
        )

        return [planning_task, design_task, implement_task, test_task]

    def create_bugfix_tasks(self, goal: str, context: dict) -> list[Task]:
        """Create tasks for a bug fix goal."""
        reproduce_task = Task(
            id=self._generate_id(),
            title="Reproduce Issue",
            description="Identify steps to reproduce the bug",
            complexity=2,
            estimated_minutes=15,
            priority=TaskPriority.HIGH,
        )

        investigate_task = Task(
            id=self._generate_id(),
            title="Investigate Root Cause",
            description="Find the source of the bug in the code",
            complexity=3,
            estimated_minutes=30,
            priority=TaskPriority.HIGH,
            dependencies=[TaskDependency(reproduce_task.id, "requires")],
        )

        fix_task = Task(
            id=self._generate_id(),
            title="Implement Fix",
            description=f"Fix: {goal}",
            complexity=3,
            estimated_minutes=30,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(investigate_task.id, "requires")],
            checkpoint_recommended=True,
        )

        verify_task = Task(
            id=self._generate_id(),
            title="Verify Fix",
            description="Confirm the fix works and doesn't break other functionality",
            complexity=2,
            estimated_minutes=15,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(fix_task.id, "requires")],
        )

        return [reproduce_task, investigate_task, fix_task, verify_task]

    def create_refactor_tasks(self, goal: str, context: dict) -> list[Task]:
        """Create tasks for a refactoring goal."""
        analyze_task = Task(
            id=self._generate_id(),
            title="Analyze Current Code",
            description="Understand the current implementation and its issues",
            complexity=2,
            estimated_minutes=20,
            priority=TaskPriority.HIGH,
        )

        plan_task = Task(
            id=self._generate_id(),
            title="Plan Refactoring",
            description="Design the target architecture and migration path",
            complexity=4,
            estimated_minutes=30,
            priority=TaskPriority.HIGH,
            dependencies=[TaskDependency(analyze_task.id, "requires")],
        )

        refactor_task = Task(
            id=self._generate_id(),
            title="Refactor Code",
            description=f"Refactor: {goal}",
            complexity=5,
            estimated_minutes=90,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(plan_task.id, "requires")],
            checkpoint_recommended=True,
        )

        test_task = Task(
            id=self._generate_id(),
            title="Run Tests",
            description="Ensure all existing tests pass after refactoring",
            complexity=2,
            estimated_minutes=20,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(refactor_task.id, "requires")],
        )

        return [analyze_task, plan_task, refactor_task, test_task]

    def create_testing_tasks(self, goal: str, context: dict) -> list[Task]:
        """Create tasks for a testing goal."""
        analyze_task = Task(
            id=self._generate_id(),
            title="Identify Test Cases",
            description="Determine what needs to be tested",
            complexity=2,
            estimated_minutes=15,
            priority=TaskPriority.HIGH,
        )

        write_task = Task(
            id=self._generate_id(),
            title="Write Tests",
            description=f"Write tests for: {goal}",
            complexity=3,
            estimated_minutes=45,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(analyze_task.id, "requires")],
        )

        run_task = Task(
            id=self._generate_id(),
            title="Run and Validate",
            description="Run tests and ensure they pass",
            complexity=1,
            estimated_minutes=10,
            priority=TaskPriority.MEDIUM,
            dependencies=[TaskDependency(write_task.id, "requires")],
        )

        return [analyze_task, write_task, run_task]

    def create_generic_task(self, goal: str, context: dict, complexity: int, time: int) -> Task:
        """Create a generic task."""
        return Task(
            id=self._generate_id(),
            title=goal[:50] + ("..." if len(goal) > 50 else ""),
            description=goal,
            complexity=complexity,
            estimated_minutes=time,
        )
