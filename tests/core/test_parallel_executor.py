"""Comprehensive tests for parallel_executor.py

Tests cover:
- ParallelExecutor class and async execution
- Task scheduling and concurrency
- Error handling and timeouts
- Result aggregation
- DependencyAnalyzer
- ExecutionStrategy modes
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from src.core.parallel_executor import (
    DependencyAnalyzer,
    ExecutionPlan,
    ExecutionStrategy,
    ParallelExecutor,
    ToolCall,
    ToolResult,
    execute_tools_streaming,
)


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_tool_call_creation(self):
        """Test creating a ToolCall."""
        call = ToolCall(
            id="call_1",
            name="file_read",
            arguments={"path": "/test/file.txt"},
        )
        assert call.id == "call_1"
        assert call.name == "file_read"
        assert call.arguments == {"path": "/test/file.txt"}
        assert call.depends_on == []

    def test_tool_call_with_dependencies(self):
        """Test ToolCall with dependencies."""
        call = ToolCall(
            id="call_2",
            name="file_write",
            arguments={"path": "/test/file.txt", "content": "data"},
            depends_on=["call_1"],
        )
        assert call.depends_on == ["call_1"]

    def test_tool_call_multiple_dependencies(self):
        """Test ToolCall with multiple dependencies."""
        call = ToolCall(
            id="call_3",
            name="run",
            arguments={"command": "python build.py"},
            depends_on=["call_1", "call_2"],
        )
        assert len(call.depends_on) == 2
        assert "call_1" in call.depends_on
        assert "call_2" in call.depends_on


class TestToolResult:
    """Tests for ToolResult dataclass."""

    def test_tool_result_success(self):
        """Test creating a successful ToolResult."""
        result = ToolResult(
            id="call_1",
            name="file_read",
            success=True,
            output="file content",
            duration=0.5,
            started_at=100.0,
            completed_at=100.5,
        )
        assert result.success is True
        assert result.output == "file content"
        assert result.error is None
        assert result.duration == 0.5

    def test_tool_result_failure(self):
        """Test creating a failed ToolResult."""
        result = ToolResult(
            id="call_1",
            name="file_read",
            success=False,
            output=None,
            error="File not found",
            duration=0.1,
            started_at=100.0,
            completed_at=100.1,
        )
        assert result.success is False
        assert result.output is None
        assert result.error == "File not found"

    def test_tool_result_to_dict(self):
        """Test converting ToolResult to dictionary."""
        result = ToolResult(
            id="call_1",
            name="file_read",
            success=True,
            output="content",
            duration=0.5,
            started_at=100.0,
            completed_at=100.5,
        )
        result_dict = result.to_dict()
        assert result_dict["id"] == "call_1"
        assert result_dict["name"] == "file_read"
        assert result_dict["success"] is True
        assert result_dict["output"] == "content"
        assert result_dict["duration"] == 0.5


class TestExecutionPlan:
    """Tests for ExecutionPlan dataclass."""

    def test_execution_plan_creation(self):
        """Test creating an ExecutionPlan."""
        call1 = ToolCall(id="call_1", name="file_read", arguments={})
        call2 = ToolCall(id="call_2", name="file_write", arguments={})
        stages = [[call1], [call2]]

        plan = ExecutionPlan(
            stages=stages,
            total_calls=2,
            estimated_sequential_time=2.0,
            estimated_parallel_time=1.0,
        )
        assert len(plan.stages) == 2
        assert plan.total_calls == 2
        assert plan.estimated_sequential_time == 2.0
        assert plan.estimated_parallel_time == 1.0

    def test_execution_plan_to_dict(self):
        """Test converting ExecutionPlan to dictionary."""
        call1 = ToolCall(id="call_1", name="file_read", arguments={})
        call2 = ToolCall(id="call_2", name="file_write", arguments={})
        stages = [[call1], [call2]]

        plan = ExecutionPlan(
            stages=stages,
            total_calls=2,
            estimated_sequential_time=2.0,
            estimated_parallel_time=1.0,
        )
        plan_dict = plan.to_dict()
        assert plan_dict["total_calls"] == 2
        assert plan_dict["estimated_speedup"] == 2.0


class TestDependencyAnalyzer:
    """Tests for DependencyAnalyzer."""

    def test_analyzer_initialization(self):
        """Test DependencyAnalyzer initialization."""
        analyzer = DependencyAnalyzer()
        assert analyzer.WRITE_TOOLS
        assert analyzer.READ_TOOLS
        assert analyzer.KNOWN_DEPENDENCIES

    def test_analyze_empty_calls(self):
        """Test analyzing empty call list."""
        analyzer = DependencyAnalyzer()
        stages = analyzer.analyze([])
        assert stages == []

    def test_analyze_single_call(self):
        """Test analyzing single call."""
        analyzer = DependencyAnalyzer()
        call = ToolCall(id="call_1", name="file_read", arguments={})
        stages = analyzer.analyze([call])
        assert len(stages) == 1
        assert len(stages[0]) == 1
        assert stages[0][0].id == "call_1"

    def test_analyze_independent_reads(self):
        """Test analyzing independent read operations."""
        analyzer = DependencyAnalyzer()
        calls = [
            ToolCall(id="call_1", name="file_read", arguments={"path": "/file1.txt"}),
            ToolCall(id="call_2", name="file_read", arguments={"path": "/file2.txt"}),
            ToolCall(id="call_3", name="file_read", arguments={"path": "/file3.txt"}),
        ]
        stages = analyzer.analyze(calls)
        # All reads should be in same stage (parallel)
        assert len(stages) == 1
        assert len(stages[0]) == 3

    def test_analyze_write_depends_on_read(self):
        """Test that write depends on read of same file (explicit dependency)."""
        analyzer = DependencyAnalyzer()
        calls = [
            ToolCall(id="call_1", name="file_read", arguments={"path": "/file.txt"}),
            ToolCall(
                id="call_2",
                name="file_write",
                arguments={"path": "/file.txt", "content": "data"},
                depends_on=["call_1"],
            ),
        ]
        stages = analyzer.analyze(calls)
        # Should be in separate stages due to explicit dependency
        assert len(stages) == 2
        assert stages[0][0].id == "call_1"
        assert stages[1][0].id == "call_2"

    def test_analyze_sequential_writes_same_file(self):
        """Test sequential writes to same file."""
        analyzer = DependencyAnalyzer()
        calls = [
            ToolCall(
                id="call_1",
                name="write",
                arguments={"path": "/file.txt", "operation": "create", "content": "data1"},
            ),
            ToolCall(
                id="call_2",
                name="write",
                arguments={"path": "/file.txt", "operation": "create", "content": "data2"},
            ),
        ]
        stages = analyzer.analyze(calls)
        # Should be in separate stages
        assert len(stages) == 2

    def test_analyze_parallel_writes_different_files(self):
        """Test parallel writes to different files."""
        analyzer = DependencyAnalyzer()
        calls = [
            ToolCall(
                id="call_1",
                name="write",
                arguments={"path": "/file1.txt", "operation": "create", "content": "data1"},
            ),
            ToolCall(
                id="call_2",
                name="write",
                arguments={"path": "/file2.txt", "operation": "create", "content": "data2"},
            ),
        ]
        stages = analyzer.analyze(calls)
        # Should be in same stage (parallel)
        assert len(stages) == 1
        assert len(stages[0]) == 2

    def test_analyze_explicit_dependency_chain(self):
        """Test explicit dependencies create separate stages."""
        analyzer = DependencyAnalyzer()
        calls = [
            ToolCall(id="call_1", name="run", arguments={"command": "npm test"}),
            ToolCall(
                id="call_2",
                name="run",
                arguments={"command": "npm publish"},
                depends_on=["call_1"],
            ),
        ]
        stages = analyzer.analyze(calls)
        assert len(stages) == 2
        assert stages[0][0].id == "call_1"
        assert stages[1][0].id == "call_2"

    def test_analyze_explicit_dependencies(self):
        """Test explicit dependencies."""
        analyzer = DependencyAnalyzer()
        calls = [
            ToolCall(id="call_1", name="file_read", arguments={}),
            ToolCall(id="call_2", name="file_read", arguments={}, depends_on=["call_1"]),
        ]
        stages = analyzer.analyze(calls)
        # Should be in separate stages due to explicit dependency
        assert len(stages) == 2

    def test_build_dependency_graph(self):
        """Test building dependency graph."""
        analyzer = DependencyAnalyzer()
        calls = [
            ToolCall(
                id="call_1",
                name="write",
                arguments={"path": "/file.txt", "operation": "create", "content": "data1"},
            ),
            ToolCall(
                id="call_2",
                name="write",
                arguments={"path": "/file.txt", "operation": "create", "content": "data2"},
            ),
        ]
        graph = analyzer._build_dependency_graph(calls)
        assert "call_1" in graph
        assert "call_2" in graph
        # call_2 should depend on call_1 (both write to same file)
        assert "call_1" in graph["call_2"]


class TestParallelExecutor:
    """Tests for ParallelExecutor."""

    @pytest.mark.asyncio
    async def test_executor_initialization(self):
        """Test ParallelExecutor initialization."""
        async_handler = AsyncMock()
        executor = ParallelExecutor(async_handler, max_parallel=5, timeout=30.0)
        assert executor._max_parallel == 5
        assert executor._timeout == 30.0

    @pytest.mark.asyncio
    async def test_execute_empty_calls(self):
        """Test executing empty call list."""
        async_handler = AsyncMock()
        executor = ParallelExecutor(async_handler)
        results = await executor.execute([])
        assert results == []

    @pytest.mark.asyncio
    async def test_execute_single_call_success(self):
        """Test executing single successful call."""
        async_handler = AsyncMock(return_value="result")
        executor = ParallelExecutor(async_handler)
        call = ToolCall(id="call_1", name="file_read", arguments={"path": "/file.txt"})
        results = await executor.execute([call])

        assert len(results) == 1
        assert results[0].id == "call_1"
        assert results[0].success is True
        assert results[0].output == "result"

    @pytest.mark.asyncio
    async def test_execute_single_call_failure(self):
        """Test executing single failed call."""
        async_handler = AsyncMock(side_effect=ValueError("Test error"))
        executor = ParallelExecutor(async_handler)
        call = ToolCall(id="call_1", name="file_read", arguments={"path": "/file.txt"})
        results = await executor.execute([call])

        assert len(results) == 1
        assert results[0].success is False
        assert "Test error" in results[0].error

    @pytest.mark.asyncio
    async def test_execute_sequential_strategy(self):
        """Test sequential execution strategy."""
        call_order = []

        async def handler(name, args):
            call_order.append(name)
            await asyncio.sleep(0.01)
            return f"result_{name}"

        executor = ParallelExecutor(handler)
        calls = [
            ToolCall(id="call_1", name="tool_1", arguments={}),
            ToolCall(id="call_2", name="tool_2", arguments={}),
            ToolCall(id="call_3", name="tool_3", arguments={}),
        ]
        results = await executor.execute(calls, strategy=ExecutionStrategy.SEQUENTIAL)

        assert len(results) == 3
        assert call_order == ["tool_1", "tool_2", "tool_3"]

    @pytest.mark.asyncio
    async def test_execute_parallel_strategy(self):
        """Test parallel execution strategy."""
        start_times = {}

        async def handler(name, args):
            start_times[name] = time.time()
            await asyncio.sleep(0.05)
            return f"result_{name}"

        executor = ParallelExecutor(handler)
        calls = [
            ToolCall(id="call_1", name="tool_1", arguments={}),
            ToolCall(id="call_2", name="tool_2", arguments={}),
            ToolCall(id="call_3", name="tool_3", arguments={}),
        ]
        results = await executor.execute(calls, strategy=ExecutionStrategy.PARALLEL)

        assert len(results) == 3
        # All should start roughly at the same time
        start_times_list = list(start_times.values())
        time_diff = max(start_times_list) - min(start_times_list)
        assert time_diff < 0.02  # Should start within 20ms

    @pytest.mark.asyncio
    async def test_execute_smart_strategy(self):
        """Test smart execution strategy."""
        async_handler = AsyncMock(return_value="result")
        executor = ParallelExecutor(async_handler)
        calls = [
            ToolCall(id="call_1", name="file_read", arguments={"path": "/file.txt"}),
            ToolCall(id="call_2", name="file_read", arguments={"path": "/file2.txt"}),
        ]
        results = await executor.execute(calls, strategy=ExecutionStrategy.SMART)

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test execution timeout."""

        async def slow_handler(name, args):
            await asyncio.sleep(1.0)
            return "result"

        executor = ParallelExecutor(slow_handler, timeout=0.1)
        call = ToolCall(id="call_1", name="slow_tool", arguments={})
        results = await executor.execute([call])

        assert len(results) == 1
        assert results[0].success is False
        assert "Timeout" in results[0].error

    @pytest.mark.asyncio
    async def test_execute_with_sync_handler(self):
        """Test execution with synchronous handler."""

        def sync_handler(name, args):
            return f"result_{name}"

        executor = ParallelExecutor(sync_handler)
        call = ToolCall(id="call_1", name="tool_1", arguments={})
        results = await executor.execute([call])

        assert len(results) == 1
        assert results[0].success is True
        assert results[0].output == "result_tool_1"

    @pytest.mark.asyncio
    async def test_execute_multiple_calls_preserves_order(self):
        """Test that results are returned in original order."""
        async_handler = AsyncMock(return_value="result")
        executor = ParallelExecutor(async_handler)
        calls = [
            ToolCall(id="call_1", name="tool_1", arguments={}),
            ToolCall(id="call_2", name="tool_2", arguments={}),
            ToolCall(id="call_3", name="tool_3", arguments={}),
        ]
        results = await executor.execute(calls, strategy=ExecutionStrategy.PARALLEL)

        assert results[0].id == "call_1"
        assert results[1].id == "call_2"
        assert results[2].id == "call_3"

    @pytest.mark.asyncio
    async def test_execute_respects_max_parallel(self):
        """Test that max_parallel limit is respected."""
        concurrent_count = 0
        max_concurrent = 0

        async def handler(name, args):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return "result"

        executor = ParallelExecutor(handler, max_parallel=2)
        calls = [ToolCall(id=f"call_{i}", name=f"tool_{i}", arguments={}) for i in range(5)]
        results = await executor.execute(calls, strategy=ExecutionStrategy.PARALLEL)

        assert len(results) == 5
        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_execute_single_with_timing(self):
        """Test that timing information is captured."""

        async def handler(name, args):
            await asyncio.sleep(0.05)
            return "result"

        executor = ParallelExecutor(handler)
        call = ToolCall(id="call_1", name="tool_1", arguments={})
        results = await executor.execute([call])

        assert results[0].duration >= 0.05
        assert results[0].started_at > 0
        assert results[0].completed_at > results[0].started_at

    def test_create_plan(self):
        """Test creating execution plan."""
        executor = ParallelExecutor(AsyncMock())
        calls = [
            ToolCall(id="call_1", name="file_read", arguments={"path": "/file.txt"}),
            ToolCall(id="call_2", name="file_read", arguments={"path": "/file2.txt"}),
        ]
        plan = executor.create_plan(calls)

        assert isinstance(plan, ExecutionPlan)
        assert plan.total_calls == 2
        assert len(plan.stages) > 0

    def test_get_execution_summary_all_success(self):
        """Test execution summary with all successful results."""
        executor = ParallelExecutor(AsyncMock())
        results = [
            ToolResult(
                id="call_1",
                name="tool_1",
                success=True,
                output="result",
                duration=0.1,
                started_at=100.0,
                completed_at=100.1,
            ),
            ToolResult(
                id="call_2",
                name="tool_2",
                success=True,
                output="result",
                duration=0.1,
                started_at=100.0,
                completed_at=100.1,
            ),
        ]
        summary = executor.get_execution_summary(results)

        assert summary["total_calls"] == 2
        assert summary["successful"] == 2
        assert summary["failed"] == 0
        assert summary["total_duration"] == 0.2

    def test_get_execution_summary_with_failures(self):
        """Test execution summary with failures."""
        executor = ParallelExecutor(AsyncMock())
        results = [
            ToolResult(
                id="call_1",
                name="tool_1",
                success=True,
                output="result",
                duration=0.1,
                started_at=100.0,
                completed_at=100.1,
            ),
            ToolResult(
                id="call_2",
                name="tool_2",
                success=False,
                output=None,
                error="Error occurred",
                duration=0.05,
                started_at=100.0,
                completed_at=100.05,
            ),
        ]
        summary = executor.get_execution_summary(results)

        assert summary["total_calls"] == 2
        assert summary["successful"] == 1
        assert summary["failed"] == 1
        assert len(summary["errors"]) == 1
        assert summary["errors"][0]["name"] == "tool_2"

    def test_get_execution_summary_empty(self):
        """Test execution summary with empty results."""
        executor = ParallelExecutor(AsyncMock())
        summary = executor.get_execution_summary([])

        assert summary["total_calls"] == 0
        assert summary["successful"] == 0
        assert summary["failed"] == 0
        assert summary["actual_time"] == 0

    def test_get_execution_summary_speedup_calculation(self):
        """Test speedup calculation in summary."""
        executor = ParallelExecutor(AsyncMock())
        results = [
            ToolResult(
                id="call_1",
                name="tool_1",
                success=True,
                output="result",
                duration=0.1,
                started_at=100.0,
                completed_at=100.05,
            ),
            ToolResult(
                id="call_2",
                name="tool_2",
                success=True,
                output="result",
                duration=0.1,
                started_at=100.0,
                completed_at=100.05,
            ),
        ]
        summary = executor.get_execution_summary(results)

        # Total duration is 0.2, actual time is 0.05 (parallel)
        assert summary["speedup"] == pytest.approx(0.2 / 0.05)

    @pytest.mark.asyncio
    async def test_execute_streaming(self):
        """Test streaming execution works after bug fix."""
        async_handler = AsyncMock(return_value="result")
        executor = ParallelExecutor(async_handler)
        calls = [
            ToolCall(id="call_1", name="tool_1", arguments={}),
            ToolCall(id="call_2", name="tool_2", arguments={}),
        ]
        results = await executor.execute_streaming(calls)
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_streaming_with_callback(self):
        """Test streaming execution with callback works after bug fix."""
        async_handler = AsyncMock(return_value="result")
        executor = ParallelExecutor(async_handler)
        callback_results = []

        def on_result(result):
            callback_results.append(result)

        calls = [
            ToolCall(id="call_1", name="tool_1", arguments={}),
            ToolCall(id="call_2", name="tool_2", arguments={}),
        ]
        results = await executor.execute_streaming(calls, on_result=on_result)
        assert len(results) == 2
        assert len(callback_results) == 2

    @pytest.mark.asyncio
    async def test_execute_streaming_empty(self):
        """Test streaming execution with empty calls."""
        async_handler = AsyncMock()
        executor = ParallelExecutor(async_handler)
        results = await executor.execute_streaming([])

        assert results == []

    @pytest.mark.asyncio
    async def test_execute_tools_streaming_convenience_function(self):
        """Test execute_tools_streaming convenience function works after bug fix."""
        async_handler = AsyncMock(return_value="result")
        tool_calls = [
            ("tool_1", {"arg": "value1"}),
            ("tool_2", {"arg": "value2"}),
        ]
        results = await execute_tools_streaming(async_handler, tool_calls)
        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_execute_tools_streaming_with_callback(self):
        """Test execute_tools_streaming with callback works after bug fix."""
        async_handler = AsyncMock(return_value="result")
        callback_results = []

        def on_result(result):
            callback_results.append(result)

        tool_calls = [
            ("tool_1", {"arg": "value1"}),
            ("tool_2", {"arg": "value2"}),
        ]
        results = await execute_tools_streaming(async_handler, tool_calls, on_result=on_result)
        assert len(results) == 2
        assert len(callback_results) == 2

    @pytest.mark.asyncio
    async def test_multiple_errors_in_batch(self):
        """Test handling multiple errors in batch execution."""
        call_count = 0

        async def handler(name, args):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise ValueError(f"Error in {name}")
            return f"result_{name}"

        executor = ParallelExecutor(handler)
        calls = [
            ToolCall(id="call_1", name="tool_1", arguments={}),
            ToolCall(id="call_2", name="tool_2", arguments={}),
            ToolCall(id="call_3", name="tool_3", arguments={}),
            ToolCall(id="call_4", name="tool_4", arguments={}),
        ]
        results = await executor.execute(calls, strategy=ExecutionStrategy.PARALLEL)

        assert len(results) == 4
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        assert len(successful) == 2
        assert len(failed) == 2

    @pytest.mark.asyncio
    async def test_dependency_chain_execution(self):
        """Test execution with dependency chain."""
        async_handler = AsyncMock(return_value="result")
        executor = ParallelExecutor(async_handler)
        calls = [
            ToolCall(id="call_1", name="run", arguments={"command": "npm test"}),
            ToolCall(
                id="call_2",
                name="run",
                arguments={"command": "npm pack"},
                depends_on=["call_1"],
            ),
            ToolCall(
                id="call_3",
                name="run",
                arguments={"command": "npm publish"},
                depends_on=["call_2"],
            ),
        ]
        results = await executor.execute(calls, strategy=ExecutionStrategy.SMART)

        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_large_batch_execution(self):
        """Test executing large batch of calls."""
        async_handler = AsyncMock(return_value="result")
        executor = ParallelExecutor(async_handler, max_parallel=10)
        calls = [ToolCall(id=f"call_{i}", name=f"tool_{i}", arguments={}) for i in range(50)]
        results = await executor.execute(calls, strategy=ExecutionStrategy.PARALLEL)

        assert len(results) == 50
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_mixed_read_write_operations(self):
        """Test mixed read and write operations."""
        async_handler = AsyncMock(return_value="result")
        executor = ParallelExecutor(async_handler)
        calls = [
            ToolCall(id="call_1", name="file_read", arguments={"path": "/file1.txt"}),
            ToolCall(id="call_2", name="file_read", arguments={"path": "/file2.txt"}),
            ToolCall(
                id="call_3", name="file_write", arguments={"path": "/file3.txt", "content": "data"}
            ),
            ToolCall(
                id="call_4", name="file_write", arguments={"path": "/file4.txt", "content": "data"}
            ),
        ]
        results = await executor.execute(calls, strategy=ExecutionStrategy.SMART)

        assert len(results) == 4
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_exception_types_preserved(self):
        """Test that exception types are preserved in error messages."""

        async def handler(name, args):
            if name == "tool_1":
                raise RuntimeError("Runtime error")
            elif name == "tool_2":
                raise ValueError("Value error")
            return "result"

        executor = ParallelExecutor(handler)
        calls = [
            ToolCall(id="call_1", name="tool_1", arguments={}),
            ToolCall(id="call_2", name="tool_2", arguments={}),
        ]
        results = await executor.execute(calls, strategy=ExecutionStrategy.PARALLEL)

        assert results[0].error == "Runtime error"
        assert results[1].error == "Value error"

    @pytest.mark.asyncio
    async def test_result_ordering_with_dependencies(self):
        """Test that results maintain original order even with dependencies."""
        async_handler = AsyncMock(return_value="result")
        executor = ParallelExecutor(async_handler)
        calls = [
            ToolCall(id="call_3", name="tool_3", arguments={}),
            ToolCall(id="call_1", name="tool_1", arguments={}),
            ToolCall(id="call_2", name="tool_2", arguments={}),
        ]
        results = await executor.execute(calls, strategy=ExecutionStrategy.SMART)

        # Results should be in original order
        assert results[0].id == "call_3"
        assert results[1].id == "call_1"
        assert results[2].id == "call_2"

    def test_analyzer_circular_dependency_handling(self):
        """Test handling of circular dependencies."""
        analyzer = DependencyAnalyzer()
        calls = [
            ToolCall(id="call_1", name="tool_1", arguments={}, depends_on=["call_2"]),
            ToolCall(id="call_2", name="tool_2", arguments={}, depends_on=["call_1"]),
        ]
        # Should not raise, should handle gracefully
        stages = analyzer.analyze(calls)
        assert len(stages) > 0

    @pytest.mark.asyncio
    async def test_concurrent_execution_timing(self):
        """Test that concurrent execution is faster than sequential."""

        async def handler(name, args):
            await asyncio.sleep(0.05)
            return "result"

        executor = ParallelExecutor(handler)
        calls = [
            ToolCall(id="call_1", name="tool_1", arguments={}),
            ToolCall(id="call_2", name="tool_2", arguments={}),
            ToolCall(id="call_3", name="tool_3", arguments={}),
        ]

        # Sequential execution
        start = time.time()
        await executor.execute(calls, strategy=ExecutionStrategy.SEQUENTIAL)
        sequential_time = time.time() - start

        # Parallel execution
        start = time.time()
        await executor.execute(calls, strategy=ExecutionStrategy.PARALLEL)
        parallel_time = time.time() - start

        # Parallel should be significantly faster
        assert parallel_time < sequential_time * 0.7

    def test_execution_strategy_enum(self):
        """Test ExecutionStrategy enum values."""
        assert ExecutionStrategy.SEQUENTIAL.value == "sequential"
        assert ExecutionStrategy.PARALLEL.value == "parallel"
        assert ExecutionStrategy.SMART.value == "smart"

    def test_analyzer_write_tools_set(self):
        """Test that WRITE_TOOLS set contains expected tools."""
        analyzer = DependencyAnalyzer()
        assert "write" in analyzer.WRITE_TOOLS
        assert "run" in analyzer.WRITE_TOOLS

    def test_analyzer_read_tools_set(self):
        """Test that READ_TOOLS set contains expected tools."""
        analyzer = DependencyAnalyzer()
        assert "read" in analyzer.READ_TOOLS
        assert "search" in analyzer.READ_TOOLS
        assert "semantic_search" in analyzer.READ_TOOLS

    def test_analyzer_known_dependencies(self):
        """Test KNOWN_DEPENDENCIES mapping."""
        analyzer = DependencyAnalyzer()
        assert "file_edit" in analyzer.KNOWN_DEPENDENCIES
        assert analyzer.KNOWN_DEPENDENCIES["file_edit"] == []

    @pytest.mark.asyncio
    async def test_execute_with_different_timeout_values(self):
        """Test execution with different timeout values."""

        async def handler(name, args):
            await asyncio.sleep(0.01)
            return "result"

        # Test with short timeout
        executor_short = ParallelExecutor(handler, timeout=0.001)
        call = ToolCall(id="call_1", name="tool_1", arguments={})
        results = await executor_short.execute([call])
        assert results[0].success is False
        assert "Timeout" in results[0].error

        # Test with long timeout
        executor_long = ParallelExecutor(handler, timeout=10.0)
        results = await executor_long.execute([call])
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_execute_with_different_max_parallel_values(self):
        """Test execution with different max_parallel values."""
        async_handler = AsyncMock(return_value="result")

        # Test with max_parallel=1
        executor_serial = ParallelExecutor(async_handler, max_parallel=1)
        calls = [
            ToolCall(id="call_1", name="tool_1", arguments={}),
            ToolCall(id="call_2", name="tool_2", arguments={}),
        ]
        results = await executor_serial.execute(calls, strategy=ExecutionStrategy.PARALLEL)
        assert len(results) == 2

        # Test with max_parallel=10
        executor_parallel = ParallelExecutor(async_handler, max_parallel=10)
        results = await executor_parallel.execute(calls, strategy=ExecutionStrategy.PARALLEL)
        assert len(results) == 2

    def test_topological_stages_with_complex_dependencies(self):
        """Test topological sort with complex dependency graph."""
        analyzer = DependencyAnalyzer()
        calls = [
            ToolCall(id="call_1", name="write", arguments={"path": "/file1.txt"}),
            ToolCall(id="call_2", name="write", arguments={"path": "/file2.txt"}),
            ToolCall(
                id="call_3",
                name="run",
                arguments={"command": "npm test"},
                depends_on=["call_1", "call_2"],
            ),
            ToolCall(
                id="call_4",
                name="run",
                arguments={"command": "npm publish"},
                depends_on=["call_3"],
            ),
        ]
        stages = analyzer.analyze(calls)

        # Should have multiple stages
        assert len(stages) > 1
        assert any(call.name == "write" for call in stages[0])

    @pytest.mark.asyncio
    async def test_execute_single_with_exception_timing(self):
        """Test that timing is captured even on exception."""

        async def handler(name, args):
            await asyncio.sleep(0.02)
            raise RuntimeError("Test error")

        executor = ParallelExecutor(handler)
        call = ToolCall(id="call_1", name="tool_1", arguments={})
        results = await executor.execute([call])

        assert results[0].success is False
        assert results[0].duration >= 0.02
        assert results[0].error == "Test error"

    def test_tool_result_with_zero_duration(self):
        """Test ToolResult with zero duration."""
        result = ToolResult(
            id="call_1",
            name="tool_1",
            success=True,
            output="result",
            duration=0.0,
            started_at=100.0,
            completed_at=100.0,
        )
        assert result.duration == 0.0
        result_dict = result.to_dict()
        assert result_dict["duration"] == 0.0

    @pytest.mark.asyncio
    async def test_execute_with_none_output(self):
        """Test execution with None output."""

        async def handler(name, args):
            return None

        executor = ParallelExecutor(handler)
        call = ToolCall(id="call_1", name="tool_1", arguments={})
        results = await executor.execute([call])

        assert results[0].success is True
        assert results[0].output is None

    @pytest.mark.asyncio
    async def test_execute_with_complex_output_types(self):
        """Test execution with various output types."""

        async def handler(name, args):
            if name == "tool_1":
                return {"key": "value", "nested": {"data": [1, 2, 3]}}
            elif name == "tool_2":
                return [1, 2, 3, 4, 5]
            elif name == "tool_3":
                return "string result"
            return 42

        executor = ParallelExecutor(handler)
        calls = [
            ToolCall(id="call_1", name="tool_1", arguments={}),
            ToolCall(id="call_2", name="tool_2", arguments={}),
            ToolCall(id="call_3", name="tool_3", arguments={}),
            ToolCall(id="call_4", name="tool_4", arguments={}),
        ]
        results = await executor.execute(calls)

        assert isinstance(results[0].output, dict)
        assert isinstance(results[1].output, list)
        assert isinstance(results[2].output, str)
        assert isinstance(results[3].output, int)

    def test_execution_plan_with_zero_parallel_time(self):
        """Test ExecutionPlan speedup calculation with zero parallel time."""
        call = ToolCall(id="call_1", name="tool_1", arguments={})
        plan = ExecutionPlan(
            stages=[[call]],
            total_calls=1,
            estimated_sequential_time=1.0,
            estimated_parallel_time=0.0,
        )
        plan_dict = plan.to_dict()
        # Should handle division by zero gracefully
        assert plan_dict["estimated_speedup"] == 1

    @pytest.mark.asyncio
    async def test_analyze_with_multiple_explicit_dependencies(self):
        """Test analyzing multiple explicit dependencies."""
        analyzer = DependencyAnalyzer()
        calls = [
            ToolCall(id="call_1", name="write", arguments={"path": "/file1.txt"}),
            ToolCall(id="call_2", name="write", arguments={"path": "/file2.txt"}),
            ToolCall(id="call_3", name="write", arguments={"path": "/file3.txt"}),
            ToolCall(
                id="call_4",
                name="run",
                arguments={"command": "npm test"},
                depends_on=["call_1", "call_2", "call_3"],
            ),
        ]
        stages = analyzer.analyze(calls)

        assert len(stages[0]) == 3
        assert stages[1][0].name == "run"

    @pytest.mark.asyncio
    async def test_get_execution_summary_with_single_result(self):
        """Test execution summary with single result."""
        executor = ParallelExecutor(AsyncMock())
        results = [
            ToolResult(
                id="call_1",
                name="tool_1",
                success=True,
                output="result",
                duration=0.5,
                started_at=100.0,
                completed_at=100.5,
            ),
        ]
        summary = executor.get_execution_summary(results)

        assert summary["total_calls"] == 1
        assert summary["successful"] == 1
        assert summary["failed"] == 0
        assert summary["speedup"] == 1.0  # Single result, no parallelization benefit
