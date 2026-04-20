"""Targeted coverage tests for evals.harness - TaskRunner.execute, check_assertions."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestEvaluationHarnessExecute:
    @pytest.mark.asyncio
    async def test_run_with_mocked_trial(self, tmp_path):
        from src.evals.harness import EvaluationHarness

        dataset = tmp_path / "tasks.json"
        dataset.write_text(json.dumps([
            {"id": "t1", "prompt": "p1", "assertions": []},
            {"id": "t2", "prompt": "p2", "assertions": []},
        ]))
        with patch("src.evals.harness.ModelGrader"):
            h = EvaluationHarness(str(dataset), str(tmp_path / "out"), n_trials=1)
        with patch.object(h, "run_single_trial", new_callable=AsyncMock) as mock_trial:
            mock_trial.return_value = {
                "trial_id": 0, "trace": [], "output": "ok",
                "code_grade": {"pass": True, "reasons": []},
                "rubric_grade": {"score": 0, "pass": True},
                "passed": True, "error": None,
                "metrics": {"duration_s": 1.0, "steps": 0},
                "sandbox": "/tmp/sandbox",
            }
            await h.run()
            assert mock_trial.call_count == 2


class TestCheckAssertionsEdgeCases:
    def setup_method(self):
        with patch("src.evals.harness.ModelGrader"):
            from src.evals.harness import EvaluationHarness
            h = EvaluationHarness.__new__(EvaluationHarness)
        h.grader = None
        h.results = []
        h.output_dir = "/tmp"
        h.dataset_path = ""
        h.n_trials = 1
        self.h = h

    def test_file_exists_in_subdirectory(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.py").write_text("x = 1")
        assertions = [{"type": "file_exists", "path": "sub/nested.py"}]
        result = self.h.check_assertions(assertions, [], "", str(tmp_path))
        assert result["pass"] is True

    def test_file_exists_empty_path(self, tmp_path):
        assertions = [{"type": "file_exists", "path": ""}]
        result = self.h.check_assertions(assertions, [], "", str(tmp_path))
        assert result["pass"] is True  # empty path resolves to sandbox dir which exists


class TestLoadTasksJsonl:
    def test_single_line_jsonl(self, tmp_path):
        with patch("src.evals.harness.ModelGrader"):
            from src.evals.harness import EvaluationHarness
            f = tmp_path / "tasks.jsonl"
            f.write_text(json.dumps({"id": "t1", "prompt": "hello"}))
            h = EvaluationHarness(str(f), str(tmp_path))
            result = h.load_tasks()
            assert len(result) == 1
            assert result[0]["id"] == "t1"

    def test_empty_jsonl(self, tmp_path):
        with patch("src.evals.harness.ModelGrader"):
            from src.evals.harness import EvaluationHarness
            f = tmp_path / "empty.jsonl"
            f.write_text("")
            h = EvaluationHarness(str(f), str(tmp_path))
            result = h.load_tasks()
            assert result == []
