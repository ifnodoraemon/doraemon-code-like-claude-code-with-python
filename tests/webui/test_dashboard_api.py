import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.webui.dashboard.api import (
    EvaluationProgress,
    EvaluationRequest,
    calculate_trends,
    get_eval_results_dir,
    list_evaluation_files,
    load_evaluation_details,
    router,
)


class TestEvaluationRequest:
    def test_valid_defaults(self):
        req = EvaluationRequest()
        assert req.task_set == "default"
        assert req.n_trials == 1

    def test_invalid_task_set(self):
        with pytest.raises(ValueError):
            EvaluationRequest(task_set="invalid")

    def test_n_trials_out_of_range(self):
        with pytest.raises(ValueError):
            EvaluationRequest(n_trials=0)
        with pytest.raises(ValueError):
            EvaluationRequest(n_trials=11)

    def test_max_workers_out_of_range(self):
        with pytest.raises(ValueError):
            EvaluationRequest(max_workers=0)
        with pytest.raises(ValueError):
            EvaluationRequest(max_workers=9)

    def test_invalid_model_name(self):
        with pytest.raises(ValueError):
            EvaluationRequest(model="../../etc/passwd")

    def test_model_too_long(self):
        with pytest.raises(ValueError):
            EvaluationRequest(model="a" * 200)

    def test_valid_model(self):
        req = EvaluationRequest(model="gemini-3-pro")
        assert req.model == "gemini-3-pro"

    def test_model_none_ok(self):
        req = EvaluationRequest(model=None)
        assert req.model is None


class TestListEvaluationFiles:
    def test_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        result = list_evaluation_files()
        assert result == []

    def test_nonexistent_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path / "missing")
        result = list_evaluation_files()
        assert result == []

    def test_with_summary_files(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {
            "timestamp": "2025-01-01T00:00:00",
            "total_tasks": 5,
            "success_rate": 0.8,
            "total_time": 10,
        }
        (cat_dir / "summary_20250101_000000.json").write_text(json.dumps(summary))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        result = list_evaluation_files()
        assert len(result) == 1
        assert result[0]["name"] == "cat1"

    def test_invalid_summary_skipped(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "bad"
        cat_dir.mkdir()
        (cat_dir / "summary_bad.json").write_text("not json")
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        result = list_evaluation_files()
        assert result == []


class TestLoadEvaluationDetails:
    def test_invalid_id_format(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        with pytest.raises(ValueError, match="Invalid evaluation ID"):
            load_evaluation_details("short")

    def test_path_traversal_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        with pytest.raises(ValueError):
            load_evaluation_details(".._.._etc")

    def test_missing_summary_raises(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError):
            load_evaluation_details("cat1_20250101_000000")


class TestCalculateTrends:
    def test_empty_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        trends = calculate_trends()
        assert trends["success_rate_trend"] == []


class TestGetEvalResultsDir:
    def test_returns_path(self):
        result = get_eval_results_dir()
        assert isinstance(result, Path)


class TestEvaluationProgress:
    def test_creation(self):
        progress = EvaluationProgress(id="test1", status="pending", progress=0.0)
        assert progress.id == "test1"
        assert progress.status == "pending"
        assert progress.progress == 0.0


class TestLoadEvaluationDetails:
    def test_valid_load(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {"timestamp": "2025-01-01T00:00:00", "total_tasks": 5, "success_rate": 0.8}
        (cat_dir / "summary_20250101_000000.json").write_text(json.dumps(summary))
        (cat_dir / "results_20250101_000000.json").write_text(json.dumps([{"task": "t1"}]))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        result = load_evaluation_details("cat1_20250101_000000")
        assert result["summary"]["total_tasks"] == 5
        assert len(result["results"]) == 1

    def test_path_traversal_category(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        with pytest.raises(ValueError, match="Invalid"):
            load_evaluation_details(".._.._etc")


class TestCalculateTrends:
    def test_empty_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        trends = calculate_trends()
        assert trends["success_rate_trend"] == []

    def test_with_data(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {
            "timestamp": "2025-01-01T10:00:00",
            "total_tasks": 3,
            "success_rate": 0.66,
            "total_time": 9,
        }
        (cat_dir / "summary_20250101_100000.json").write_text(json.dumps(summary))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        trends = calculate_trends()
        assert len(trends["success_rate_trend"]) == 1
        assert trends["success_rate_trend"][0]["value"] == 66.0
        assert len(trends["latency_trend"]) == 1
        assert trends["latency_trend"][0]["value"] == 3.0


class TestCalculateTaskStatistics:
    def test_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)
        from src.webui.dashboard.api import calculate_task_statistics

        stats = calculate_task_statistics()
        assert stats["total_evaluations"] == 0
        assert stats["overall_success_rate"] == 0

    def test_with_data(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {
            "timestamp": "2025-01-01T00:00:00",
            "total_tasks": 5,
            "successful_tasks": 4,
            "by_category": {"core": {"total": 5, "success": 4}},
            "by_difficulty": {"easy": {"total": 3, "success": 3}},
        }
        (cat_dir / "summary_20250101_000000.json").write_text(json.dumps(summary))
        results = [{"task": "t1", "success": True}]
        (cat_dir / "results_20250101_000000.json").write_text(json.dumps(results))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import calculate_task_statistics

        stats = calculate_task_statistics()
        assert stats["total_evaluations"] == 1
        assert stats["total_tasks"] == 5
        assert stats["overall_success_rate"] == 0.8


class TestEvaluationRequestModel:
    def test_valid_model_with_special_chars(self):
        req = EvaluationRequest(model="gpt-4o:latest")
        assert req.model == "gpt-4o:latest"

    def test_model_with_special_chars_rejected(self):
        with pytest.raises(ValueError):
            EvaluationRequest(model="../../etc/passwd")


class TestTriggerEvaluation:
    @pytest.mark.asyncio
    async def test_trigger_returns_id(self, tmp_path, monkeypatch):
        from fastapi import BackgroundTasks

        from src.webui.dashboard.api import trigger_evaluation

        req = EvaluationRequest(task_set="default", n_trials=1, max_workers=2)
        bg = BackgroundTasks()
        result = await trigger_evaluation(req, bg)
        assert "id" in result


class TestEvaluationProgressModel:
    def test_all_fields(self):
        progress = EvaluationProgress(
            id="test2",
            status="running",
            progress=0.5,
            current_task="task1",
            total_tasks=10,
            completed_tasks=5,
            start_time="2025-01-01T00:00:00",
            end_time=None,
            error=None,
        )
        assert progress.progress == 0.5
        assert progress.completed_tasks == 5


class TestGetEvaluationsEndpoint:
    @pytest.mark.asyncio
    async def test_get_evaluations_with_category_filter(self, tmp_path, monkeypatch):
        cat1 = tmp_path / "cat1"
        cat1.mkdir()
        summary1 = {"timestamp": "2025-01-01T00:00:00", "total_tasks": 3, "success_rate": 0.5, "total_time": 5}
        (cat1 / "summary_20250101_000000.json").write_text(json.dumps(summary1))

        cat2 = tmp_path / "cat2"
        cat2.mkdir()
        summary2 = {"timestamp": "2025-01-02T00:00:00", "total_tasks": 1, "success_rate": 1.0, "total_time": 2}
        (cat2 / "summary_20250102_000000.json").write_text(json.dumps(summary2))

        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import get_evaluations

        result = await get_evaluations(category="cat1")
        assert result["total"] == 1

    @pytest.mark.asyncio
    async def test_get_evaluations_pagination(self, tmp_path, monkeypatch):
        cat1 = tmp_path / "cat1"
        cat1.mkdir()
        for i in range(5):
            summary = {"timestamp": f"2025-01-0{i+1}T00:00:00", "total_tasks": 1, "success_rate": 0.5, "total_time": 1}
            (cat1 / f"summary_2025010{i+1}_000000.json").write_text(json.dumps(summary))

        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import get_evaluations

        result = await get_evaluations(limit=2, offset=0)
        assert result["total"] == 5
        assert len(result["evaluations"]) == 2

        result2 = await get_evaluations(limit=2, offset=2)
        assert len(result2["evaluations"]) == 2


class TestGetEvaluationDetailsEndpoint:
    @pytest.mark.asyncio
    async def test_valid_details(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {"timestamp": "2025-01-01T00:00:00", "total_tasks": 5}
        (cat_dir / "summary_20250101_000000.json").write_text(json.dumps(summary))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import get_evaluation_details

        result = await get_evaluation_details(eval_id="cat1_20250101_000000")
        assert result["id"] == "cat1_20250101_000000"
        assert result["category"] == "cat1"

    @pytest.mark.asyncio
    async def test_missing_eval_details(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import get_evaluation_details

        with pytest.raises(Exception):
            await get_evaluation_details(eval_id="missing_20250101_000000")


class TestGetTrendsEndpoint:
    @pytest.mark.asyncio
    async def test_with_date_timestamp(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {
            "timestamp": "2025-01-15T10:30:00",
            "total_tasks": 2,
            "success_rate": 0.5,
            "total_time": 4,
        }
        (cat_dir / "summary_20250115_103000.json").write_text(json.dumps(summary))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import calculate_trends

        trends = calculate_trends()
        assert len(trends["success_rate_trend"]) == 1
        assert trends["success_rate_trend"][0]["label"] == "01/15 10:30"

    @pytest.mark.asyncio
    async def test_with_plain_timestamp(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {
            "timestamp": "2025-01-15",
            "total_tasks": 2,
            "success_rate": 0.5,
            "total_time": 4,
        }
        (cat_dir / "summary_20250115_000000.json").write_text(json.dumps(summary))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import calculate_trends

        trends = calculate_trends()
        assert trends["success_rate_trend"][0]["label"] == "2025-01-15"

    @pytest.mark.asyncio
    async def test_invalid_timestamp(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {
            "timestamp": "invalid",
            "total_tasks": 2,
            "success_rate": 0.5,
            "total_time": 4,
        }
        (cat_dir / "summary_invalid_000000.json").write_text(json.dumps(summary))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import calculate_trends

        trends = calculate_trends()
        assert len(trends["success_rate_trend"]) == 1


class TestCalculateTaskStatisticsDetailed:
    def test_with_by_difficulty(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {
            "timestamp": "2025-01-01T00:00:00",
            "total_tasks": 10,
            "successful_tasks": 8,
            "by_category": {"core": {"total": 10, "success": 8}},
            "by_difficulty": {
                "easy": {"total": 5, "success": 5},
                "hard": {"total": 5, "success": 3},
            },
        }
        (cat_dir / "summary_20250101_000000.json").write_text(json.dumps(summary))
        (cat_dir / "results_20250101_000000.json").write_text(json.dumps([]))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import calculate_task_statistics

        stats = calculate_task_statistics()
        assert stats["by_difficulty"]["easy"]["success_rate"] == 1.0
        assert stats["by_difficulty"]["hard"]["success_rate"] == 0.6

    def test_with_file_not_found(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {
            "timestamp": "2025-01-01T00:00:00",
            "total_tasks": 3,
            "success_rate": 0.5,
            "total_time": 5,
        }
        (cat_dir / "summary_20250101_000000.json").write_text(json.dumps(summary))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import calculate_task_statistics

        stats = calculate_task_statistics()
        assert stats["total_evaluations"] == 1


class TestRunEvaluationAsync:
    @pytest.mark.asyncio
    async def test_successful_run(self, tmp_path, monkeypatch):
        from src.webui.dashboard.api import EvaluationProgress, run_evaluation_async

        progress = EvaluationProgress(id="test_run", status="pending", progress=0.0)
        from src.webui.dashboard import api as api_mod

        api_mod._running_evaluations["test_run"] = progress
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(return_value=(b"", b""))
            mock_process.returncode = 0
            mock_exec.return_value = mock_process

            await run_evaluation_async("test_run", "default", 1, 2, None)
            assert progress.status == "completed"
            assert progress.progress == 1.0

    @pytest.mark.asyncio
    async def test_failed_run(self, tmp_path, monkeypatch):
        from src.webui.dashboard.api import EvaluationProgress, run_evaluation_async

        progress = EvaluationProgress(id="test_fail", status="pending", progress=0.0)
        from src.webui.dashboard import api as api_mod

        api_mod._running_evaluations["test_fail"] = progress

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate = AsyncMock(return_value=(b"", b"error output"))
            mock_process.returncode = 1
            mock_exec.return_value = mock_process

            await run_evaluation_async("test_fail", "default", 1, 2, None)
            assert progress.status == "failed"

    @pytest.mark.asyncio
    async def test_exception_during_run(self, monkeypatch):
        from src.webui.dashboard.api import EvaluationProgress, run_evaluation_async

        progress = EvaluationProgress(id="test_exc", status="pending", progress=0.0)
        from src.webui.dashboard import api as api_mod

        api_mod._running_evaluations["test_exc"] = progress

        with patch("asyncio.create_subprocess_exec", side_effect=OSError("spawn failed")):
            await run_evaluation_async("test_exc", "default", 1, 2, None)
            assert progress.status == "failed"
            assert "spawn failed" in progress.error


class TestGetEvaluationProgressEndpoint:
    @pytest.mark.asyncio
    async def test_unknown_eval_id(self):
        from src.webui.dashboard.api import get_evaluation_progress

        with pytest.raises(Exception):
            await get_evaluation_progress(eval_id="nonexistent")


class TestGetCategoriesEndpoint:
    @pytest.mark.asyncio
    async def test_with_categories(self, tmp_path, monkeypatch):
        cat1 = tmp_path / "cat1"
        cat1.mkdir()
        summary = {"timestamp": "t", "total_tasks": 1}
        (cat1 / "summary_20250101_000000.json").write_text(json.dumps(summary))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import get_categories

        result = await get_categories()
        assert len(result["categories"]) == 1
        assert result["categories"][0]["name"] == "cat1"

    @pytest.mark.asyncio
    async def test_empty_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import get_categories

        result = await get_categories()
        assert result["categories"] == []


class TestCompareModelsEndpoint:
    @pytest.mark.asyncio
    async def test_compare_models(self, tmp_path, monkeypatch):
        cat_dir = tmp_path / "cat1"
        cat_dir.mkdir()
        summary = {
            "timestamp": "2025-01-01T00:00:00",
            "model_name": "gpt-4o",
            "total_tasks": 5,
            "successful_tasks": 4,
            "total_time": 10,
        }
        (cat_dir / "summary_20250101_000000.json").write_text(json.dumps(summary))
        (cat_dir / "results_20250101_000000.json").write_text(json.dumps([]))
        monkeypatch.setattr("src.webui.dashboard.api.EVAL_RESULTS_DIR", tmp_path)

        from src.webui.dashboard.api import compare_models

        result = await compare_models()
        assert "models" in result
        assert "gpt-4o" in result["models"]
        assert result["models"]["gpt-4o"]["success_rate"] == 0.8
