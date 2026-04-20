"""
Parallel Tool Executor

Executes independent tools in parallel for improved performance.

Features:
- Dependency analysis
- Parallel execution of independent tools
- Result aggregation
- Timeout management
- Error handling
"""

import asyncio
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Tool execution strategy."""

    SEQUENTIAL = "sequential"  # Execute one by one
    PARALLEL = "parallel"  # Execute all in parallel
    SMART = "smart"  # Analyze dependencies and parallelize


@dataclass
class ToolCall:
    """A tool call to execute."""

    id: str
    name: str
    arguments: dict[str, Any]
    depends_on: list[str] = field(default_factory=list)  # IDs of dependent calls


@dataclass
class ToolResult:
    """Result of a tool execution."""

    id: str
    name: str
    success: bool
    output: Any
    error: str | None = None
    duration: float = 0
    started_at: float = 0
    completed_at: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "duration": self.duration,
        }


@dataclass
class ExecutionPlan:
    """Plan for executing multiple tool calls."""

    stages: list[list[ToolCall]]  # Tools in each stage can run in parallel
    total_calls: int
    estimated_parallel_time: float = 0
    estimated_sequential_time: float = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [[c.name for c in stage] for stage in self.stages],
            "total_calls": self.total_calls,
            "estimated_speedup": (
                self.estimated_sequential_time / self.estimated_parallel_time
                if self.estimated_parallel_time > 0
                else 1
            ),
        }


class DependencyAnalyzer:
    """
    Analyzes tool dependencies to determine execution order.

    Rules:
    - Read operations are independent
    - Write operations depend on reads of the same file
    - Sequential operations like git commit depend on git add
    """

    # Tools that write/modify state
    WRITE_TOOLS = {
        "write",
        "run",
    }

    # Tools that read state
    READ_TOOLS = {
        "read",
        "search",
    }

    # Known dependencies (tool -> depends on)
    KNOWN_DEPENDENCIES = {
        "file_edit": [],  # Can depend on file_read of same file
    }

    def analyze(self, calls: list[ToolCall]) -> list[list[ToolCall]]:
        """
        Analyze calls and return execution stages.

        Args:
            calls: List of tool calls

        Returns:
            List of stages, where tools in each stage can run in parallel
        """
        if not calls:
            return []

        # Build dependency graph
        graph = self._build_dependency_graph(calls)

        # Topological sort into stages
        stages = self._topological_stages(calls, graph)

        return stages

    def _build_dependency_graph(self, calls: list[ToolCall]) -> dict[str, set[str]]:
        """Build dependency graph from tool calls."""
        graph: dict[str, set[str]] = {call.id: set() for call in calls}

        # Map tool names to call IDs for lookup
        name_to_ids: dict[str, list[str]] = {}
        for call in calls:
            if call.name not in name_to_ids:
                name_to_ids[call.name] = []
            name_to_ids[call.name].append(call.id)

        # Add explicit dependencies
        for call in calls:
            for dep_id in call.depends_on:
                if dep_id in graph:
                    graph[call.id].add(dep_id)

        # Add implicit dependencies based on rules
        for i, call in enumerate(calls):
            # Write depends on previous writes to same resource
            if call.name in self.WRITE_TOOLS:
                path = call.arguments.get("path", call.arguments.get("file"))
                if path:
                    for prev_call in calls[:i]:
                        prev_path = prev_call.arguments.get("path", prev_call.arguments.get("file"))
                        if prev_path == path and prev_call.name in self.WRITE_TOOLS:
                            graph[call.id].add(prev_call.id)

            # Known tool dependencies
            if call.name in self.KNOWN_DEPENDENCIES:
                for dep_name in self.KNOWN_DEPENDENCIES[call.name]:
                    if dep_name in name_to_ids:
                        # Find latest call of that tool before this one
                        for dep_id in reversed(name_to_ids[dep_name]):
                            dep_idx = next(j for j, c in enumerate(calls) if c.id == dep_id)
                            if dep_idx < i:
                                graph[call.id].add(dep_id)
                                break

        return graph

    def _topological_stages(
        self, calls: list[ToolCall], graph: dict[str, set[str]]
    ) -> list[list[ToolCall]]:
        """
        Perform topological sort and group into parallel stages.

        Returns list of stages where tools in each stage can run in parallel.
        """
        # Calculate in-degrees
        in_degree = {call.id: len(graph[call.id]) for call in calls}
        call_map = {call.id: call for call in calls}

        stages = []
        remaining = {call.id for call in calls}

        while remaining:
            # Find all nodes with no remaining dependencies
            ready = [
                cid
                for cid in remaining
                if in_degree[cid] == 0 or all(dep not in remaining for dep in graph[cid])
            ]

            if not ready:
                # Circular dependency - just take the first remaining
                ready = [next(iter(remaining))]
                logger.warning("Circular dependency detected, breaking at %s", ready)

            stage = [call_map[cid] for cid in ready]
            stages.append(stage)

            # Remove from remaining
            for cid in ready:
                remaining.remove(cid)

        return stages


class ParallelExecutor:
    """
    Executes tool calls with parallelization.

    Usage:
        executor = ParallelExecutor(tool_handler)

        # Execute with smart parallelization
        results = await executor.execute(calls, strategy=ExecutionStrategy.SMART)

        # Execute all in parallel (if you know they're independent)
        results = await executor.execute(calls, strategy=ExecutionStrategy.PARALLEL)
    """

    def __init__(
        self,
        tool_handler: Callable[[str, dict], Any],
        max_parallel: int = 5,
        timeout: float = 30.0,
    ):
        """
        Initialize executor.

        Args:
            tool_handler: Function to execute a tool (name, args) -> result
            max_parallel: Maximum parallel executions
            timeout: Default timeout per tool
        """
        self._tool_handler = tool_handler
        self._max_parallel = max_parallel
        self._timeout = timeout
        self._analyzer = DependencyAnalyzer()
        self._semaphore = asyncio.Semaphore(max_parallel)
        self._sync_executor: ThreadPoolExecutor | None = None

    async def execute(
        self,
        calls: list[ToolCall],
        strategy: ExecutionStrategy = ExecutionStrategy.SMART,
    ) -> list[ToolResult]:
        """
        Execute tool calls.

        Args:
            calls: Tool calls to execute
            strategy: Execution strategy

        Returns:
            List of results in same order as calls
        """
        if not calls:
            return []

        with self._sync_executor_scope():
            if strategy == ExecutionStrategy.SEQUENTIAL:
                return await self._execute_sequential(calls)
            elif strategy == ExecutionStrategy.PARALLEL:
                return await self._execute_parallel(calls)
            else:  # SMART
                return await self._execute_smart(calls)

    async def _execute_sequential(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute tools sequentially."""
        results = []
        for call in calls:
            result = await self._execute_single(call)
            results.append(result)
        return results

    async def _execute_parallel(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute all tools in parallel."""
        tasks = [self._execute_single(call) for call in calls]
        return await asyncio.gather(*tasks)

    async def _execute_smart(self, calls: list[ToolCall]) -> list[ToolResult]:
        """Execute with smart parallelization based on dependencies."""
        stages = self._analyzer.analyze(calls)

        # Create result map
        results_map: dict[str, ToolResult] = {}

        for stage in stages:
            if len(stage) == 1:
                # Single tool, no parallelization needed
                result = await self._execute_single(stage[0])
                results_map[stage[0].id] = result
            else:
                # Multiple tools, execute in parallel
                tasks = [self._execute_single(call) for call in stage]
                stage_results = await asyncio.gather(*tasks)
                for call, result in zip(stage, stage_results, strict=False):
                    results_map[call.id] = result

        # Return in original order
        return [results_map[call.id] for call in calls]

    async def _execute_single(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        async with self._semaphore:
            started_at = time.time()

            try:
                # Execute with timeout
                if asyncio.iscoroutinefunction(self._tool_handler):
                    output = await asyncio.wait_for(
                        self._tool_handler(call.name, call.arguments),
                        timeout=self._timeout,
                    )
                else:
                    # Sync handler
                    output = await asyncio.get_running_loop().run_in_executor(
                        self._sync_executor, self._tool_handler, call.name, call.arguments
                    )

                completed_at = time.time()

                return ToolResult(
                    id=call.id,
                    name=call.name,
                    success=True,
                    output=output,
                    duration=completed_at - started_at,
                    started_at=started_at,
                    completed_at=completed_at,
                )

            except asyncio.TimeoutError:
                return ToolResult(
                    id=call.id,
                    name=call.name,
                    success=False,
                    output=None,
                    error=f"Timeout after {self._timeout}s",
                    duration=time.time() - started_at,
                    started_at=started_at,
                    completed_at=time.time(),
                )

            except Exception as e:
                return ToolResult(
                    id=call.id,
                    name=call.name,
                    success=False,
                    output=None,
                    error=str(e),
                    duration=time.time() - started_at,
                    started_at=started_at,
                    completed_at=time.time(),
                )

    def create_plan(self, calls: list[ToolCall]) -> ExecutionPlan:
        """
        Create an execution plan without executing.

        Args:
            calls: Tool calls to plan

        Returns:
            ExecutionPlan showing stages and estimated times
        """
        stages = self._analyzer.analyze(calls)

        # Estimate times (rough estimate based on tool type)
        avg_time = 1.0  # Average time per tool
        sequential_time = len(calls) * avg_time
        parallel_time = len(stages) * avg_time

        return ExecutionPlan(
            stages=stages,
            total_calls=len(calls),
            estimated_sequential_time=sequential_time,
            estimated_parallel_time=parallel_time,
        )

    def get_execution_summary(self, results: list[ToolResult]) -> dict[str, Any]:
        """Get summary of execution results."""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        total_duration = sum(r.duration for r in results)

        # Calculate actual parallel time (max of overlapping executions)
        if results:
            min_start = min(r.started_at for r in results)
            max_end = max(r.completed_at for r in results)
            actual_time = max_end - min_start
        else:
            actual_time = 0

        return {
            "total_calls": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "total_duration": total_duration,
            "actual_time": actual_time,
            "speedup": total_duration / actual_time if actual_time > 0 else 1,
            "errors": [{"name": r.name, "error": r.error} for r in failed],
        }

    async def execute_streaming(
        self,
        calls: list[ToolCall],
        on_result: Callable[[ToolResult], None] | None = None,
    ) -> list[ToolResult]:
        """
        Execute tools and stream results as they complete.

        Unlike execute(), this yields results immediately as each tool finishes,
        allowing the caller to process results incrementally.

        Args:
            calls: Tool calls to execute
            on_result: Optional callback called for each result

        Returns:
            List of all results in order of completion
        """
        if not calls:
            return []

        with self._sync_executor_scope():
            stages = self._analyzer.analyze(calls)
            results = []
            call_to_result: dict[str, ToolResult] = {}

            for stage in stages:
                # Execute stage in parallel
                stage_results = await self._execute_stage_streaming(stage, on_result)
                results.extend(stage_results)

                # Store for dependency injection
                for r in stage_results:
                    call_to_result[r.id] = r

            return results

    def _sync_executor_scope(self):
        """Create a short-lived thread pool for sync handlers."""
        return _SyncExecutorScope(self)

    async def _execute_stage_streaming(
        self,
        stage: list[ToolCall],
        on_result: Callable[[ToolResult], None] | None,
    ) -> list[ToolResult]:
        """Execute a stage and stream results as they complete."""
        if not stage:
            return []

        # Create queue for streaming results
        result_queue: asyncio.Queue[ToolResult] = asyncio.Queue()

        async def run_one(call: ToolCall):
            result = await self._execute_single(call)
            await result_queue.put(result)
            if on_result:
                on_result(result)
            return result

        # Start all tasks
        tasks = [asyncio.create_task(run_one(call)) for call in stage]

        try:
            # Collect results as they complete
            results = []
            for _ in range(len(stage)):
                result = await result_queue.get()
                results.append(result)

            # Ensure all tasks are done
            await asyncio.gather(*tasks, return_exceptions=True)

            return results
        except BaseException:
            for t in tasks:
                if not t.done():
                    t.cancel()
            raise


class _SyncExecutorScope:
    """Context manager for per-execution sync thread pools."""

    def __init__(self, executor: ParallelExecutor):
        self._executor = executor
        self._created = False

    def __enter__(self):
        if (
            not asyncio.iscoroutinefunction(self._executor._tool_handler)
            and self._executor._sync_executor is None
        ):
            self._executor._sync_executor = ThreadPoolExecutor(
                max_workers=self._executor._max_parallel,
                thread_name_prefix="parallel_executor",
            )
            self._created = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._created and self._executor._sync_executor is not None:
            self._executor._sync_executor.shutdown(wait=True, cancel_futures=True)
            self._executor._sync_executor = None
        return False


async def execute_tools_streaming(
    tool_handler: Callable[[str, dict], Any],
    tool_calls: list[tuple[str, dict[str, Any]]],
    on_result: Callable[[ToolResult], None] | None = None,
) -> list[ToolResult]:
    """
    Convenience function to execute tools with streaming results.

    Args:
        tool_handler: Function to execute a tool (name, args) -> result
        tool_calls: List of (tool_name, arguments) tuples
        on_result: Optional callback for each result

    Returns:
        List of results in completion order
    """
    calls = [
        ToolCall(id=f"call_{i}", name=name, arguments=args)
        for i, (name, args) in enumerate(tool_calls)
    ]

    executor = ParallelExecutor(tool_handler)
    return await executor.execute_streaming(calls, on_result)
