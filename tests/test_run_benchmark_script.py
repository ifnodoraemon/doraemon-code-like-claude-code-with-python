import json
from pathlib import Path

import pytest

from scripts import run_benchmark


def test_resolve_suite_path_known_suite():
    path = run_benchmark.resolve_suite_path("repo_patch", None)

    assert path.name == "repo_patch_sample.json"


def test_resolve_suite_path_unknown_suite():
    with pytest.raises(ValueError, match="Unknown suite"):
        run_benchmark.resolve_suite_path("missing", None)


def test_load_tasks_applies_limit(tmp_path: Path):
    task_file = tmp_path / "tasks.json"
    task_file.write_text(json.dumps([{"id": "a"}, {"id": "b"}]), encoding="utf-8")

    assert run_benchmark.load_tasks(task_file, limit=1) == [{"id": "a"}]


def test_materialize_files(tmp_path: Path):
    run_benchmark.materialize_files(
        {"files": {"pkg/example.py": "print('ok')\n"}},
        tmp_path,
    )

    assert (tmp_path / "pkg" / "example.py").read_text(encoding="utf-8") == "print('ok')\n"


def test_run_verify_command_success(tmp_path: Path):
    (tmp_path / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")

    result = run_benchmark.run_verify(
        {"verify": {"type": "command", "command": "python -m pytest -q test_ok.py"}},
        tmp_path,
        timeout=30,
    )

    assert result["success"] is True


def test_summarize_empty():
    assert run_benchmark.summarize([]) == {
        "total_tasks": 0,
        "successful_tasks": 0,
        "success_rate": 0.0,
        "avg_execution_time": 0.0,
    }
