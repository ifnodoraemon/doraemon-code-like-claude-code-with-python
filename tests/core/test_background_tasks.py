"""Tests for src/core/background_tasks.py"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from src.core.background_tasks import (
    BackgroundTask,
    BackgroundTaskManager,
    TaskStatus,
    get_task_manager,
)


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestBackgroundTask:
    def test_to_dict(self):
        task = BackgroundTask(
            id="abc",
            name="test",
            description="desc",
            status=TaskStatus.COMPLETED,
            started_at=1000.0,
            completed_at=1010.0,
            progress=100,
            token_usage=50,
        )
        d = task.to_dict()
        assert d["id"] == "abc"
        assert d["name"] == "test"
        assert d["status"] == "completed"
        assert d["progress"] == 100
        assert d["token_usage"] == 50
        assert d["duration"] == 10.0
        assert d["output_preview"] is None
        assert d["error"] is None

    def test_to_dict_with_output(self):
        task = BackgroundTask(id="x", name="n", description="d", output="hello world")
        d = task.to_dict()
        assert d["output_preview"] == "hello world"

    def test_to_dict_long_output(self):
        task = BackgroundTask(id="x", name="n", description="d", output="x" * 600)
        d = task.to_dict()
        assert len(d["output_preview"]) <= 500

    def test_to_dict_with_error(self):
        task = BackgroundTask(id="x", name="n", description="d", error="failed")
        d = task.to_dict()
        assert d["error"] == "failed"

    def test_get_duration_not_started(self):
        task = BackgroundTask(id="x", name="n", description="d")
        assert task._get_duration() is None

    def test_get_duration_running(self):
        task = BackgroundTask(id="x", name="n", description="d", started_at=time.time() - 5)
        dur = task._get_duration()
        assert dur is not None
        assert dur >= 4.9

    def test_get_duration_completed(self):
        task = BackgroundTask(
            id="x", name="n", description="d", started_at=1000.0, completed_at=1015.0
        )
        assert task._get_duration() == 15.0

    def test_append_output(self):
        task = BackgroundTask(id="x", name="n", description="d")
        task.append_output("hello ")
        task.append_output("world")
        assert task.output == "hello world"

    def test_tool_names_default(self):
        task = BackgroundTask(id="x", name="n", description="d")
        assert task.tool_names == []


class TestBackgroundTaskManagerStartTask:
    @pytest.mark.asyncio
    async def test_start_and_complete(self):
        mgr = BackgroundTaskManager()
        task_id = await mgr.start_task(
            name="test",
            description="desc",
            coroutine=asyncio.sleep(0, result="done"),
        )
        assert task_id is not None
        await asyncio.sleep(0.05)
        task = mgr.get_task(task_id)
        assert task is not None
        assert task.status == TaskStatus.COMPLETED
        assert task.progress == 100

    @pytest.mark.asyncio
    async def test_start_task_max_concurrent(self):
        mgr = BackgroundTaskManager(max_concurrent=1)
        mgr._running_count = 1
        coro = asyncio.sleep(0)
        with pytest.raises(RuntimeError, match="Maximum concurrent"):
            await mgr.start_task(name="blocked", description="blocked", coroutine=coro)
        coro.close()
        mgr._running_count = 0

    @pytest.mark.asyncio
    async def test_start_task_failure(self):
        mgr = BackgroundTaskManager()

        async def fail_task():
            raise ValueError("task failed")

        task_id = await mgr.start_task(
            name="failing",
            description="will fail",
            coroutine=fail_task(),
        )
        await asyncio.sleep(0.05)
        task = mgr.get_task(task_id)
        assert task.status == TaskStatus.FAILED
        assert "task failed" in task.error

    @pytest.mark.asyncio
    async def test_start_task_cancel(self):
        mgr = BackgroundTaskManager()

        async def long_task():
            await asyncio.sleep(100)

        task_id = await mgr.start_task(
            name="long",
            description="long running",
            coroutine=long_task(),
        )
        await asyncio.sleep(0.01)
        mgr.cancel_task(task_id)
        await asyncio.sleep(0.05)
        task = mgr.get_task(task_id)
        assert task.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_on_complete_callback(self):
        mgr = BackgroundTaskManager()
        callback_result = []

        def on_done(task):
            callback_result.append(task.id)

        task_id = await mgr.start_task(
            name="cb",
            description="with callback",
            coroutine=asyncio.sleep(0, result="ok"),
            on_complete=on_done,
        )
        await asyncio.sleep(0.05)
        assert callback_result == [task_id]

    @pytest.mark.asyncio
    async def test_on_complete_callback_error(self):
        mgr = BackgroundTaskManager()

        def bad_callback(task):
            raise RuntimeError("callback error")

        task_id = await mgr.start_task(
            name="cberr",
            description="bad callback",
            coroutine=asyncio.sleep(0, result="ok"),
            on_complete=bad_callback,
        )
        await asyncio.sleep(0.05)
        task = mgr.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_tool_names(self):
        mgr = BackgroundTaskManager()
        task_id = await mgr.start_task(
            name="tools",
            description="with tools",
            coroutine=asyncio.sleep(0, result="ok"),
            tool_names=["read", "write"],
        )
        task = mgr.get_task(task_id)
        assert task.tool_names == ["read", "write"]


class TestBackgroundTaskManagerStartToolTask:
    @pytest.mark.asyncio
    async def test_start_tool_task_blocks_background_unsafe_tools(self):
        from src.host.tools import ToolRegistry

        manager = BackgroundTaskManager()
        registry = ToolRegistry()

        def ask_user(question: str) -> str:
            return question

        registry.register(ask_user, name="ask_user", metadata={"capability_group": "read"})

        with pytest.raises(RuntimeError, match="background task"):
            await manager.start_tool_task(
                name="Interactive prompt",
                description="Should not be allowed in background",
                coroutine=asyncio.sleep(0),
                tool_names=["ask_user"],
                tool_registry=registry,
            )

    @pytest.mark.asyncio
    async def test_start_tool_task_allows_background_safe_tools(self):
        from src.host.tools import ToolRegistry

        manager = BackgroundTaskManager()
        registry = ToolRegistry()

        def read(path: str) -> str:
            return path

        registry.register(read, name="read", metadata={"capability_group": "read"})

        task_id = await manager.start_tool_task(
            name="Read files",
            description="Background-safe read task",
            coroutine=asyncio.sleep(0, result="done"),
            tool_names=["read"],
            tool_registry=registry,
        )

        await asyncio.sleep(0.05)
        task = manager.get_task(task_id)
        assert task is not None
        assert task.tool_names == ["read"]
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_start_tool_task_no_check_method(self):
        manager = BackgroundTaskManager()
        registry = MagicMock()
        delattr(registry, "check_tool_execution")

        task_id = await manager.start_tool_task(
            name="safe",
            description="no check",
            coroutine=asyncio.sleep(0, result="ok"),
            tool_names=["read"],
            tool_registry=registry,
        )
        await asyncio.sleep(0.05)
        task = manager.get_task(task_id)
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_start_tool_task_closes_coro_on_reject(self):
        manager = BackgroundTaskManager()
        registry = MagicMock()
        registry.check_tool_execution.return_value = (False, "not allowed", None)

        coro = asyncio.sleep(0)

        with pytest.raises(RuntimeError, match="not allowed"):
            await manager.start_tool_task(
                name="blocked",
                description="blocked",
                coroutine=coro,
                tool_names=["dangerous"],
                tool_registry=registry,
            )
        coro.close()


class TestBackgroundCurrent:
    def test_background_current(self):
        mgr = BackgroundTaskManager()

        async def dummy():
            await asyncio.sleep(100)

        loop = asyncio.new_event_loop()
        try:
            task = loop.create_task(dummy())
            task_id = mgr.background_current("bg", "desc", task)
            assert task_id is not None
            assert mgr.get_task(task_id) is not None
            assert mgr.get_task(task_id).status == TaskStatus.RUNNING
            assert mgr.get_running_count() == 1
            task.cancel()
            try:
                loop.run_until_complete(task)
            except asyncio.CancelledError:
                pass
        finally:
            loop.close()

    def test_background_current_on_done_completed(self):
        mgr = BackgroundTaskManager()

        async def quick():
            return "result"

        loop = asyncio.new_event_loop()
        try:
            task = loop.create_task(quick())
            task_id = mgr.background_current("bg", "desc", task)
            loop.run_until_complete(task)
            bg_task = mgr.get_task(task_id)
            assert bg_task.status == TaskStatus.COMPLETED
            assert bg_task.output == "result"
        finally:
            loop.close()

    def test_background_current_on_done_failed(self):
        mgr = BackgroundTaskManager()

        async def fail():
            raise ValueError("oops")

        loop = asyncio.new_event_loop()
        try:
            task = loop.create_task(fail())
            task_id = mgr.background_current("bg", "desc", task)
            try:
                loop.run_until_complete(task)
            except ValueError:
                pass
            bg_task = mgr.get_task(task_id)
            assert bg_task.status == TaskStatus.FAILED
            assert "oops" in bg_task.error
        finally:
            loop.close()


class TestGetTask:
    def test_existing(self):
        mgr = BackgroundTaskManager()
        task = BackgroundTask(id="t1", name="n", description="d")
        mgr._tasks["t1"] = task
        assert mgr.get_task("t1") is task

    def test_missing(self):
        mgr = BackgroundTaskManager()
        assert mgr.get_task("nonexistent") is None


class TestGetTaskStatus:
    def test_existing(self):
        mgr = BackgroundTaskManager()
        task = BackgroundTask(id="t1", name="n", description="d", status=TaskStatus.RUNNING)
        mgr._tasks["t1"] = task
        result = mgr.get_task_status("t1")
        assert result["status"] == "running"

    def test_missing(self):
        mgr = BackgroundTaskManager()
        assert mgr.get_task_status("nonexistent") is None


class TestGetTaskOutput:
    def test_existing(self):
        mgr = BackgroundTaskManager()
        task = BackgroundTask(id="t1", name="n", description="d", output="hello")
        mgr._tasks["t1"] = task
        assert mgr.get_task_output("t1") == "hello"

    def test_missing(self):
        mgr = BackgroundTaskManager()
        assert mgr.get_task_output("nonexistent") is None

    def test_tail(self):
        mgr = BackgroundTaskManager()
        task = BackgroundTask(id="t1", name="n", description="d", output="x" * 100)
        mgr._tasks["t1"] = task
        result = mgr.get_task_output("t1", tail=10)
        assert "truncated" in result
        assert result.endswith("x" * 10)

    def test_tail_shorter_than_output(self):
        mgr = BackgroundTaskManager()
        task = BackgroundTask(id="t1", name="n", description="d", output="hi")
        mgr._tasks["t1"] = task
        assert mgr.get_task_output("t1", tail=100) == "hi"


class TestUpdateTaskProgress:
    def test_existing(self):
        mgr = BackgroundTaskManager()
        task = BackgroundTask(id="t1", name="n", description="d")
        mgr._tasks["t1"] = task
        mgr.update_task_progress("t1", 50, "partial output")
        assert task.progress == 50
        assert task.output == "partial output"

    def test_progress_clamped(self):
        mgr = BackgroundTaskManager()
        task = BackgroundTask(id="t1", name="n", description="d")
        mgr._tasks["t1"] = task
        mgr.update_task_progress("t1", 150)
        assert task.progress == 100
        mgr.update_task_progress("t1", -10)
        assert task.progress == 0

    def test_missing(self):
        mgr = BackgroundTaskManager()
        mgr.update_task_progress("nonexistent", 50)

    def test_without_output(self):
        mgr = BackgroundTaskManager()
        task = BackgroundTask(id="t1", name="n", description="d")
        mgr._tasks["t1"] = task
        mgr.update_task_progress("t1", 30)
        assert task.progress == 30
        assert task.output == ""


class TestUpdateTaskTokens:
    def test_existing(self):
        mgr = BackgroundTaskManager()
        task = BackgroundTask(id="t1", name="n", description="d", token_usage=10)
        mgr._tasks["t1"] = task
        mgr.update_task_tokens("t1", 20)
        assert task.token_usage == 30

    def test_missing(self):
        mgr = BackgroundTaskManager()
        mgr.update_task_tokens("nonexistent", 20)


class TestListTasks:
    def test_list_all(self):
        mgr = BackgroundTaskManager()
        mgr._tasks["t1"] = BackgroundTask(id="t1", name="a", description="d", created_at=1.0)
        mgr._tasks["t2"] = BackgroundTask(id="t2", name="b", description="d", created_at=2.0)
        result = mgr.list_tasks()
        assert len(result) == 2
        assert result[0]["id"] == "t2"

    def test_filter_by_status(self):
        mgr = BackgroundTaskManager()
        mgr._tasks["t1"] = BackgroundTask(id="t1", name="a", description="d", created_at=1.0, status=TaskStatus.COMPLETED)
        mgr._tasks["t2"] = BackgroundTask(id="t2", name="b", description="d", created_at=2.0, status=TaskStatus.RUNNING)
        result = mgr.list_tasks(status=TaskStatus.COMPLETED)
        assert len(result) == 1
        assert result[0]["id"] == "t1"

    def test_limit(self):
        mgr = BackgroundTaskManager()
        for i in range(5):
            mgr._tasks[f"t{i}"] = BackgroundTask(id=f"t{i}", name=f"n{i}", description="d", created_at=float(i))
        result = mgr.list_tasks(limit=2)
        assert len(result) == 2


class TestCancelTask:
    @pytest.mark.asyncio
    async def test_cancel_running(self):
        mgr = BackgroundTaskManager()
        task_id = await mgr.start_task(
            name="long",
            description="long",
            coroutine=asyncio.sleep(100),
        )
        await asyncio.sleep(0.01)
        result = mgr.cancel_task(task_id)
        assert result is True

    def test_cancel_missing(self):
        mgr = BackgroundTaskManager()
        assert mgr.cancel_task("nonexistent") is False

    def test_cancel_not_running(self):
        mgr = BackgroundTaskManager()
        task = BackgroundTask(id="t1", name="n", description="d", status=TaskStatus.COMPLETED)
        mgr._tasks["t1"] = task
        assert mgr.cancel_task("t1") is False

    def test_cancel_no_task_handle(self):
        mgr = BackgroundTaskManager()
        task = BackgroundTask(id="t1", name="n", description="d", status=TaskStatus.RUNNING, _task=None)
        mgr._tasks["t1"] = task
        assert mgr.cancel_task("t1") is False


class TestGetRunningTasks:
    def test_running(self):
        mgr = BackgroundTaskManager()
        mgr._tasks["t1"] = BackgroundTask(id="t1", name="a", description="d", status=TaskStatus.RUNNING)
        mgr._tasks["t2"] = BackgroundTask(id="t2", name="b", description="d", status=TaskStatus.COMPLETED)
        result = mgr.get_running_tasks()
        assert len(result) == 1
        assert result[0].id == "t1"

    def test_none(self):
        mgr = BackgroundTaskManager()
        assert mgr.get_running_tasks() == []


class TestGetRunningCount:
    def test_count(self):
        mgr = BackgroundTaskManager()
        mgr._running_count = 3
        assert mgr.get_running_count() == 3


class TestCleanupCompleted:
    def test_removes_old(self):
        mgr = BackgroundTaskManager()
        old = BackgroundTask(
            id="old", name="n", description="d",
            status=TaskStatus.COMPLETED, completed_at=time.time() - 7200,
        )
        recent = BackgroundTask(
            id="recent", name="n", description="d",
            status=TaskStatus.COMPLETED, completed_at=time.time(),
        )
        mgr._tasks["old"] = old
        mgr._tasks["recent"] = recent
        mgr.cleanup_completed(max_age=3600)
        assert "old" not in mgr._tasks
        assert "recent" in mgr._tasks

    def test_keeps_running(self):
        mgr = BackgroundTaskManager()
        running = BackgroundTask(
            id="running", name="n", description="d",
            status=TaskStatus.RUNNING,
        )
        mgr._tasks["running"] = running
        mgr.cleanup_completed()
        assert "running" in mgr._tasks

    def test_no_completed_at(self):
        mgr = BackgroundTaskManager()
        task = BackgroundTask(
            id="t1", name="n", description="d",
            status=TaskStatus.COMPLETED, completed_at=None,
        )
        mgr._tasks["t1"] = task
        mgr.cleanup_completed()
        assert "t1" in mgr._tasks

    @pytest.mark.asyncio
    async def test_auto_cleanup_on_start(self):
        mgr = BackgroundTaskManager()
        mgr._max_completed = 1
        for i in range(3):
            mgr._tasks[f"c{i}"] = BackgroundTask(
                id=f"c{i}", name=f"n{i}", description="d",
                status=TaskStatus.COMPLETED, completed_at=time.time() - 7200,
            )
        await mgr.start_task(name="t", description="d", coroutine=asyncio.sleep(0, result="ok"))
        remaining = sum(1 for t in mgr._tasks.values() if t.status == TaskStatus.COMPLETED)
        assert remaining < 3


class TestGetTaskManager:
    def test_returns_instance(self):
        import src.core.background_tasks as mod
        mod._task_manager = None
        mgr = get_task_manager()
        assert isinstance(mgr, BackgroundTaskManager)
        mod._task_manager = None

    def test_singleton(self):
        import src.core.background_tasks as mod
        mod._task_manager = None
        mgr1 = get_task_manager()
        mgr2 = get_task_manager()
        assert mgr1 is mgr2
        mod._task_manager = None
