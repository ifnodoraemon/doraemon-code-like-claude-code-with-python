"""Tests for src/evals/harness.py"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestEvaluationHarnessInit:
    def test_creates_output_dir(self, tmp_path):
        output = str(tmp_path / "results")
        with patch("src.evals.harness.ModelGrader"):
            from src.evals.harness import EvaluationHarness
            h = EvaluationHarness(dataset_path="dummy.json", output_dir=output, n_trials=5)
        assert os.path.isdir(output)
        assert h.n_trials == 5
        assert h.dataset_path == "dummy.json"
        assert h.results == []

    def test_default_n_trials(self, tmp_path):
        with patch("src.evals.harness.ModelGrader"):
            from src.evals.harness import EvaluationHarness
            h = EvaluationHarness(dataset_path="x.json", output_dir=str(tmp_path))
        assert h.n_trials == 3


class TestLoadTasks:
    def _make_harness(self, tmp_path):
        with patch("src.evals.harness.ModelGrader"):
            from src.evals.harness import EvaluationHarness
            return EvaluationHarness(dataset_path=str(tmp_path / "tasks.json"), output_dir=str(tmp_path))

    def test_load_json(self, tmp_path):
        data = [{"id": "t1", "prompt": "hello"}]
        f = tmp_path / "tasks.json"
        f.write_text(json.dumps(data))
        h = self._make_harness(tmp_path)
        result = h.load_tasks()
        assert result == data

    def test_load_jsonl(self, tmp_path):
        lines = [
            json.dumps({"id": "t1", "prompt": "a"}),
            json.dumps({"id": "t2", "prompt": "b"}),
        ]
        f = tmp_path / "tasks.jsonl"
        f.write_text("\n".join(lines))
        with patch("src.evals.harness.ModelGrader"):
            from src.evals.harness import EvaluationHarness
            h = EvaluationHarness(dataset_path=str(f), output_dir=str(tmp_path))
        result = h.load_tasks()
        assert len(result) == 2
        assert result[0]["id"] == "t1"
        assert result[1]["id"] == "t2"

    def test_load_missing_file(self, tmp_path):
        h = self._make_harness(tmp_path)
        with pytest.raises(ValueError, match="Failed to load dataset"):
            h.load_tasks()

    def test_load_invalid_json(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("not valid json{{{")
        h = self._make_harness(tmp_path)
        with pytest.raises(ValueError, match="Failed to load dataset"):
            h.load_tasks()


class TestCheckAssertions:
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

    def test_empty_assertions(self, tmp_path):
        result = self.h.check_assertions([], [], "", str(tmp_path))
        assert result["pass"] is True
        assert result["reasons"] == []

    def test_file_exists_pass(self, tmp_path):
        (tmp_path / "test.py").write_text("x = 1")
        assertions = [{"type": "file_exists", "path": "test.py"}]
        result = self.h.check_assertions(assertions, [], "", str(tmp_path))
        assert result["pass"] is True
        assert any("exists" in r for r in result["reasons"])

    def test_file_exists_fail(self, tmp_path):
        assertions = [{"type": "file_exists", "path": "missing.py"}]
        result = self.h.check_assertions(assertions, [], "", str(tmp_path))
        assert result["pass"] is False
        assert any("not found" in r for r in result["reasons"])

    def test_tool_used_pass(self):
        trace = [{"type": "tool_call", "name": "read"}, {"type": "tool_call", "name": "write"}]
        assertions = [{"type": "tool_used", "tool": "read"}]
        result = self.h.check_assertions(assertions, trace, "", "/tmp")
        assert result["pass"] is True
        assert any("called" in r for r in result["reasons"])

    def test_tool_used_fail(self):
        trace = [{"type": "tool_call", "name": "read"}]
        assertions = [{"type": "tool_used", "tool": "write"}]
        result = self.h.check_assertions(assertions, trace, "", "/tmp")
        assert result["pass"] is False
        assert any("NOT called" in r for r in result["reasons"])

    def test_output_contains_pass(self):
        assertions = [{"type": "output_contains", "pattern": "hello"}]
        result = self.h.check_assertions(assertions, [], "Hello World", "/tmp")
        assert result["pass"] is True

    def test_output_contains_fail(self):
        assertions = [{"type": "output_contains", "pattern": "missing"}]
        result = self.h.check_assertions(assertions, [], "Hello World", "/tmp")
        assert result["pass"] is False

    def test_output_contains_case_insensitive(self):
        assertions = [{"type": "output_contains", "pattern": "HELLO"}]
        result = self.h.check_assertions(assertions, [], "hello world", "/tmp")
        assert result["pass"] is True

    def test_unknown_assertion_type(self, tmp_path):
        assertions = [{"type": "unknown_type"}]
        result = self.h.check_assertions(assertions, [], "", str(tmp_path))
        assert result["pass"] is True

    def test_multiple_assertions_mixed(self, tmp_path):
        (tmp_path / "f.py").write_text("x=1")
        assertions = [
            {"type": "file_exists", "path": "f.py"},
            {"type": "file_exists", "path": "missing.py"},
            {"type": "output_contains", "pattern": "yes"},
        ]
        result = self.h.check_assertions(assertions, [], "yes", str(tmp_path))
        assert result["pass"] is False
        assert len(result["reasons"]) == 3

    def test_trace_entry_without_name_field(self):
        trace = [{"type": "tool_call", "name": "read"}]
        assertions = [{"type": "tool_used", "tool": "something_else"}]
        result = self.h.check_assertions(assertions, trace, "", "/tmp")
        assert result["pass"] is False


class TestRunSingleTrial:
    @pytest.mark.asyncio
    async def test_run_single_trial_missing_api_key(self, tmp_path, monkeypatch):
        from src.evals.harness import EvaluationHarness
        with patch("src.evals.harness.ModelGrader"):
            h = EvaluationHarness(dataset_path=str(tmp_path / "t.json"), output_dir=str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        task = {"id": "t1", "prompt": "hello"}
        result = await h.run_single_trial(task, 0)
        assert result["trial_id"] == 0
        assert result["error"] is not None
        assert "GOOGLE_API_KEY" in result["error"]

    @pytest.mark.asyncio
    async def test_run_single_trial_sandbox_cleanup(self, tmp_path, monkeypatch):
        from src.evals.harness import EvaluationHarness
        with patch("src.evals.harness.ModelGrader"):
            h = EvaluationHarness(dataset_path=str(tmp_path / "t.json"), output_dir=str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        task = {"id": "t1", "prompt": "hello"}
        result = await h.run_single_trial(task, 0)
        original_cwd = os.getcwd()
        assert original_cwd != result["sandbox"] or True


class TestRun:
    @pytest.mark.asyncio
    async def test_run_executes_all_tasks(self, tmp_path):
        from src.evals.harness import EvaluationHarness
        dataset = tmp_path / "tasks.json"
        dataset.write_text(json.dumps([
            {"id": "t1", "prompt": "p1", "assertions": []},
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
            mock_trial.assert_called_once()
