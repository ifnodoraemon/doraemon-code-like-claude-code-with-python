from src.host.cli.main import _format_task_tree, _parse_orchestrate_args


def test_parse_orchestrate_args_supports_worker_flag():
    parsed = _parse_orchestrate_args(["--workers", "3", "implement", "auth", "flow"])

    assert parsed == (3, "implement auth flow")


def test_parse_orchestrate_args_rejects_missing_goal():
    assert _parse_orchestrate_args(["--workers", "2"]) is None
    assert _parse_orchestrate_args([]) is None


def test_format_task_tree_renders_nested_assignments():
    lines = _format_task_tree(
        [
            {
                "id": "root",
                "title": "Root",
                "status": "in_progress",
                "ready": True,
                "children": [
                    {
                        "id": "child",
                        "title": "Child",
                        "status": "pending",
                        "ready": False,
                        "assigned_agent": "worker-1",
                    }
                ],
            }
        ]
    )

    assert lines[0] == "- [in_progress] Root (root) ready"
    assert lines[1] == "  - [pending] Child (child) @worker-1"
