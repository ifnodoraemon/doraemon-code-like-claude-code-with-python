from src.core.rules import load_project_rules


def test_load_project_rules_supports_agents_hierarchy(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    nested = repo / "src" / "feature"
    nested.mkdir(parents=True)
    (repo / ".git").mkdir()
    (repo / "AGENTS.md").write_text("root agent instructions", encoding="utf-8")
    (repo / "src" / "AGENTS.md").write_text("nested agent instructions", encoding="utf-8")

    monkeypatch.chdir(nested)

    content = load_project_rules(nested)

    assert content is not None
    assert "root agent instructions" in content
    assert "nested agent instructions" in content
    assert content.index("root agent instructions") < content.index("nested agent instructions")
