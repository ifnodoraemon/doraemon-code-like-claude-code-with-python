"""Targeted coverage for webui/dashboard/api.py uncovered lines: 116,164,173,179-180,240-241,314-316,358,378,415-419,482-483,494,505,568,642-643."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from src.webui.dashboard.api import (
    EvaluationProgress,
    EvaluationRequest,
    _dashboard_page_requires_client_auth,
    _empty_task_statistics,
    _empty_trends,
    calculate_task_statistics,
    calculate_trends,
    compare_models,
    dashboard_index,
    get_categories,
    get_evaluation_details,
    get_evaluation_progress,
    get_evaluations,
    list_evaluation_files,
    load_evaluation_details,
    run_evaluation_async,
    trigger_evaluation,
)


class TestListEvaluationFilesNonDir:
    def test_skips_files_in_results_dir(self, tmp_path, monkeypatch):
        (tmp_path / "not_a_dir.txt").write_text("x")
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        result = list_evaluation_files()
        assert result == []


class TestDashboardIndexAuthShell:
    def test_requires_client_auth_when_key_is_set_without_header(self, monkeypatch):
        monkeypatch.setenv("AGENT_WEBUI_API_KEY", "secret")
        request = SimpleNamespace(headers={})

        assert _dashboard_page_requires_client_auth(request) is True

    def test_accepts_bearer_header_for_server_render(self, monkeypatch):
        monkeypatch.setenv("AGENT_WEBUI_API_KEY", "secret")
        request = SimpleNamespace(headers={"Authorization": "Bearer secret"})

        assert _dashboard_page_requires_client_auth(request) is False

    @pytest.mark.asyncio
    async def test_dashboard_index_does_not_inject_data_without_auth(self, monkeypatch):
        monkeypatch.setenv("AGENT_WEBUI_API_KEY", "secret")
        monkeypatch.setattr(
            "src.webui.dashboard.api.list_evaluation_files",
            lambda: pytest.fail("dashboard index should not load evaluations without auth"),
        )

        response = await dashboard_index(SimpleNamespace(headers={}))

        assert response.context["requires_auth"] is True
        assert response.context["evaluations"] == []
        assert response.context["trends"] == _empty_trends()
        assert response.context["task_stats"] == _empty_task_statistics()


class TestLoadEvaluationDetailsPathTraversalResolve:
    def test_path_traversal_via_resolve(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        with pytest.raises(ValueError):
            load_evaluation_details(".._.._etc")


class TestCalculateTrendsEdge:
    def test_zero_total_tasks_skips_latency(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {"timestamp": "2025-01-01", "total_tasks": 0, "success_rate": 0.0, "total_time": 0}
        (cat_dir / "summary_20250101_000000.json").write_text(json.dumps(summary))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        trends = calculate_trends()
        assert len(trends["latency_trend"]) == 0
        assert len(trends["task_count_trend"]) == 1


class TestCalculateTaskStatisticsWithLoadError:
    def test_skips_evaluation_with_value_error(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {"timestamp": "2025-01-01T00:00:00", "total_tasks": 1, "success_rate": 0.5, "total_time": 1}
        (cat_dir / "summary_20250101_000000.json").write_text(json.dumps(summary))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        with patch("src.webui.dashboard.api.load_evaluation_details", side_effect=ValueError("bad id")):
            stats = calculate_task_statistics()
            assert stats["total_evaluations"] == 1
            assert stats["total_tasks"] == 0


class TestGetEvaluationsOffsetBeyond:
    @pytest.mark.asyncio
    async def test_offset_beyond_results(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {"timestamp": "t", "total_tasks": 1, "success_rate": 0.5, "total_time": 1}
        (cat_dir / "summary_20250101_000000.json").write_text(json.dumps(summary))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        result = await get_evaluations(offset=100)
        assert result["evaluations"] == []


class TestGetEvaluationProgressMissing:
    @pytest.mark.asyncio
    async def test_missing_progress_raises_404(self):
        with pytest.raises(HTTPException):
            await get_evaluation_progress(eval_id="nonexistent_id")


class TestCompareModelsWithLoadError:
    @pytest.mark.asyncio
    async def test_skips_on_file_not_found(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {"timestamp": "t", "total_tasks": 1, "success_rate": 0.5, "total_time": 1}
        (cat_dir / "summary_20250101_000000.json").write_text(json.dumps(summary))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        result = await compare_models()
        assert "models" in result


class TestGetCategoriesNoDir:
    @pytest.mark.asyncio
    async def test_no_results_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path / "missing")
        result = await get_categories()
        assert result["categories"] == []


class TestRunEvaluationAsyncNoProgress:
    @pytest.mark.asyncio
    async def test_no_progress_entry(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        await run_evaluation_async("nonexistent_id", "default", 1, 2, None)


class TestRunEvaluationAsyncWithModel:
    @pytest.mark.asyncio
    async def test_run_with_model_arg(self, tmp_path, monkeypatch):
        progress = EvaluationProgress(id="model_test", status="pending", progress=0.0)
        from src.webui.dashboard import api as api_mod
        api_mod._running_evaluations["model_test"] = progress
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process
            await run_evaluation_async("model_test", "default", 1, 2, "gpt-4o")
            assert progress.status == "completed"
        api_mod._running_evaluations.pop("model_test", None)


class TestTriggerEvaluationBackground:
    @pytest.mark.asyncio
    async def test_trigger_adds_background_task(self, tmp_path, monkeypatch):
        from fastapi import BackgroundTasks
        req = EvaluationRequest(task_set="quick", n_trials=2, max_workers=1)
        bg = BackgroundTasks()
        result = await trigger_evaluation(req, bg)
        assert "id" in result
        assert result["status"] == "pending"
        from src.webui.dashboard import api as api_mod
        api_mod._running_evaluations.pop(result["id"], None)


class TestGetEvaluationDetailsPathTraversalCategory:
    @pytest.mark.asyncio
    async def test_slash_in_category_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        with pytest.raises(HTTPException):
            await get_evaluation_details(eval_id=".._.._etc")


class TestListEvaluationFilesNonDirSkipped:
    @pytest.mark.asyncio
    async def test_skips_subdirs_without_summaries(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "empty_cat"
        cat_dir.mkdir()
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        result = await get_evaluations()
        assert result["evaluations"] == []
