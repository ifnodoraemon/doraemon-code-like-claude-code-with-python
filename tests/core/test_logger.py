"""Tests for src.core.logger."""

import json
import logging
from pathlib import Path
from unittest.mock import patch

from src.core.logger import (
    TraceEvent,
    TraceLogger,
    _resolve_log_level,
    configure_root_logger,
    get_logger,
    setup_logger,
)


class TestResolveLogLevel:
    def test_default(self):
        with patch.dict("os.environ", {}, clear=True):
            assert _resolve_log_level(None) == "INFO"

    def test_from_arg(self):
        assert _resolve_log_level("DEBUG") == "DEBUG"

    def test_from_env(self):
        with patch.dict("os.environ", {"AGENT_LOG_LEVEL": "WARNING"}):
            assert _resolve_log_level(None) == "WARNING"

    def test_arg_overrides_env(self):
        with patch.dict("os.environ", {"AGENT_LOG_LEVEL": "WARNING"}):
            assert _resolve_log_level("DEBUG") == "DEBUG"


class TestConfigureRootLogger:
    def test_configures_root(self, tmp_path):
        log_file = tmp_path / "test.log"
        root = configure_root_logger(level="DEBUG", log_file=str(log_file))
        assert root.level == logging.DEBUG
        assert len(root.handlers) >= 1

    def test_clears_existing_handlers(self, tmp_path):
        log_file = tmp_path / "test.log"
        configure_root_logger(log_file=str(log_file))
        root = logging.getLogger()
        before = len(root.handlers)
        configure_root_logger(log_file=str(log_file))
        assert len(root.handlers) == before


class TestSetupLogger:
    def test_creates_logger(self):
        logger = setup_logger("test.module")
        assert logger.name == "test.module"
        assert len(logger.handlers) >= 1

    def test_with_file(self, tmp_path):
        log_file = tmp_path / "mod.log"
        logger = setup_logger("test.file", log_file=str(log_file))
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)

    def test_no_duplicate_handlers(self):
        setup_logger("test.dup")
        setup_logger("test.dup")
        logger = logging.getLogger("test.dup")
        assert len(logger.handlers) <= 2


class TestGetLogger:
    def test_auto_configures(self, tmp_path):
        with patch("src.core.logger.logs_dir", return_value=tmp_path):
            logger = get_logger("test.auto")
            assert isinstance(logger, logging.Logger)

    def test_returns_existing(self):
        with patch("src.core.logger.logs_dir", return_value=Path("/tmp")):
            l1 = get_logger("test.existing")
            l2 = get_logger("test.existing")
            assert l1 is l2


class TestTraceEvent:
    def test_fields(self):
        e = TraceEvent(
            type="tool_call", name="read", data={"k": "v"}, timestamp=1.0, duration_ms=100
        )
        assert e.type == "tool_call"
        assert e.name == "read"
        assert e.duration_ms == 100


class TestTraceLogger:
    def test_log_and_export(self):
        tl = TraceLogger()
        tl.log("tool_call", "read", {"path": "f.py"}, duration_ms=50)
        exported = tl.export()
        assert len(exported) == 1
        assert exported[0]["type"] == "tool_call"
        assert exported[0]["name"] == "read"

    def test_log_multiple(self):
        tl = TraceLogger()
        tl.log("tool_call", "read", {})
        tl.log("tool_call", "write", {})
        tl.log("llm_call", "gemini", {})
        assert len(tl.export()) == 3
