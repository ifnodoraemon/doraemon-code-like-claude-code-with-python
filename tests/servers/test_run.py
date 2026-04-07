from src.servers.run import _run_shell


class _FakeStdout:
    def __init__(self, process, lines):
        self._process = process
        self._lines = list(lines)

    def readline(self):
        if self._lines:
            return self._lines.pop(0)
        return ""

    def close(self):
        self._process.done = True


class _FakeProcess:
    def __init__(self, lines):
        self.done = False
        self.returncode = 0
        self.stdout = _FakeStdout(self, lines)

    def poll(self):
        return self.returncode if self.done else None

    def kill(self):
        self.done = True


def test_run_shell_does_not_truncate_output(monkeypatch, tmp_path):
    large_output = "x" * 40000

    monkeypatch.setattr("src.servers.run._is_command_blocked", lambda command: False)
    monkeypatch.setattr("src.servers.run._check_git_safety", lambda command: None)
    monkeypatch.setattr("src.servers.run.validate_path", lambda path: path)
    monkeypatch.setattr(
        "src.servers.run.subprocess.Popen",
        lambda *args, **kwargs: _FakeProcess([large_output]),
    )

    result = _run_shell("echo large", timeout=5, working_dir=str(tmp_path))

    assert result == large_output
    assert "truncated" not in result.lower()
