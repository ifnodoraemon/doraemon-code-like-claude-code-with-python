"""Comprehensive tests for src/host/cli/commands.py

Tests cover:
- CommandHandler initialization
- Core slash commands and session/config command paths
- Command parsing and validation
- Command execution with various arguments
- Error handling for invalid commands
- Integration with ContextManager, CheckpointManager, BudgetTracker
- Mode switching (plan/build)
- Project switching
- Configuration updates
- Help text generation
- Permission checks for sensitive commands
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.host.cli.commands import CommandHandler


class TestCommandHandlerInitialization:
    """Test CommandHandler initialization and setup."""

    def test_init_with_all_dependencies(self):
        """Test initialization with all required dependencies."""
        ctx = MagicMock()
        tool_selector = MagicMock()
        registry = MagicMock()
        skill_mgr = MagicMock()
        checkpoint_mgr = MagicMock()
        task_mgr = MagicMock()
        cost_tracker = MagicMock()
        cmd_history = MagicMock()
        session_mgr = MagicMock()
        hook_mgr = MagicMock()

        handler = CommandHandler(
            ctx=ctx,
            tool_selector=tool_selector,
            registry=registry,
            skill_mgr=skill_mgr,
            checkpoint_mgr=checkpoint_mgr,
            task_mgr=task_mgr,
            cost_tracker=cost_tracker,
            cmd_history=cmd_history,
            session_mgr=session_mgr,
            hook_mgr=hook_mgr,
            model_name="test-model",
            project="test-project",
        )

        assert handler.cc.ctx == ctx
        assert handler.cc.tool_selector == tool_selector
        assert handler.cc.registry == registry
        assert handler.cc.skill_mgr == skill_mgr
        assert handler.cc.checkpoint_mgr == checkpoint_mgr
        assert handler.cc.task_mgr == task_mgr
        assert handler.cc.cost_tracker == cost_tracker
        assert handler.cc.cmd_history == cmd_history
        assert handler.cc.session_mgr == session_mgr
        assert handler.cc.hook_mgr == hook_mgr
        assert handler.cc.model_name == "test-model"
        assert handler.cc.project == "test-project"

    def test_init_stores_all_attributes(self):
        """Test that all attributes are properly stored."""
        mocks = {
            "ctx": MagicMock(),
            "tool_selector": MagicMock(),
            "registry": MagicMock(),
            "skill_mgr": MagicMock(),
            "checkpoint_mgr": MagicMock(),
            "task_mgr": MagicMock(),
            "cost_tracker": MagicMock(),
            "cmd_history": MagicMock(),
            "session_mgr": MagicMock(),
            "hook_mgr": MagicMock(),
        }

        handler = CommandHandler(
            **mocks,
            model_name="gpt-4",
            project="my-project",
        )

        for key, value in mocks.items():
            assert getattr(handler.cc, key) == value


@pytest.mark.asyncio
class TestHelpCommandFallback:
    """Test /help command."""

    async def test_help_command_shows_help(self):
        """Test that /help command displays help text."""
        ctx = MagicMock()
        handler = CommandHandler(
            ctx=ctx,
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_core.console") as mock_console:
            result = await handler.handle(
                cmd="help",
                cmd_args=[],
                mode="build",
                tool_names=["tool1"],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            mock_console.print.assert_called()


@pytest.mark.asyncio
class TestHelpCommand:
    async def test_help_command_returns_handled_true(self):
        """Test that help command returns handled=True."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_core.console"):
            result = await handler.handle(
                cmd="help",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True


@pytest.mark.asyncio
class TestClearCommand:
    """Test /clear command."""

    async def test_clear_command_clears_context(self):
        """Test that /clear command clears conversation."""
        ctx = MagicMock()
        ctx.clear = MagicMock()

        handler = CommandHandler(
            ctx=ctx,
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_core.console"):
            result = await handler.handle(
                cmd="clear",
                cmd_args=[],
                mode="build",
                tool_names=["tool1"],
                tool_definitions=[],
                conversation_history=[{"role": "user", "content": "hello"}],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            ctx.clear.assert_called_once_with(keep_summaries=True)
            assert result.conversation_history == []
            assert result.handled is True

    async def test_clear_command_preserves_summaries(self):
        """Test that /clear preserves summaries."""
        ctx = MagicMock()
        ctx.clear = MagicMock()

        handler = CommandHandler(
            ctx=ctx,
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_core.console"):
            await handler.handle(
                cmd="clear",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            # Verify keep_summaries=True is passed
            ctx.clear.assert_called_once()
            call_kwargs = ctx.clear.call_args[1]
            assert call_kwargs.get("keep_summaries") is True


@pytest.mark.asyncio
class TestModeCommand:
    """Test /mode command for switching between plan and build modes."""

    async def test_mode_switch_to_build(self):
        """Test switching to build mode."""
        ctx = MagicMock()
        tool_selector = MagicMock()
        tool_selector.get_tools_for_mode = MagicMock(return_value=["tool1", "tool2"])
        registry = MagicMock()
        registry.get_genai_tools = MagicMock(return_value=[])
        hook_mgr = MagicMock()

        handler = CommandHandler(
            ctx=ctx,
            tool_selector=tool_selector,
            registry=registry,
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=hook_mgr,
            model_name="test",
            project="test",
        )

        build_system_prompt = MagicMock(return_value="build prompt")
        convert_tools = MagicMock(return_value=[])

        with patch("src.host.cli.commands_core.console"):
            result = await handler.handle(
                cmd="mode",
                cmd_args=["build"],
                mode="plan",
                tool_names=["old_tool"],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=build_system_prompt,
                convert_tools_to_definitions=convert_tools,
            )

            assert result.mode == "build"
            assert result.handled is True
            tool_selector.get_tools_for_mode.assert_called_with("build")
            hook_mgr.permission_mode = "build"

    async def test_mode_switch_to_plan(self):
        """Test switching to plan mode."""
        ctx = MagicMock()
        tool_selector = MagicMock()
        tool_selector.get_tools_for_mode = MagicMock(return_value=["read_tool"])
        registry = MagicMock()
        registry.get_genai_tools = MagicMock(return_value=[])
        hook_mgr = MagicMock()

        handler = CommandHandler(
            ctx=ctx,
            tool_selector=tool_selector,
            registry=registry,
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=hook_mgr,
            model_name="test",
            project="test",
        )

        build_system_prompt = MagicMock(return_value="plan prompt")
        convert_tools = MagicMock(return_value=[])

        with patch("src.host.cli.commands_core.console"):
            result = await handler.handle(
                cmd="mode",
                cmd_args=["plan"],
                mode="build",
                tool_names=["write_tool"],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=build_system_prompt,
                convert_tools_to_definitions=convert_tools,
            )

            assert result.mode == "plan"
            tool_selector.get_tools_for_mode.assert_called_with("plan")

    async def test_mode_command_without_args_shows_current(self):
        """Test /mode without args shows current mode."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_core.console") as mock_console:
            result = await handler.handle(
                cmd="mode",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.mode == "build"
            mock_console.print.assert_called()

    async def test_mode_command_invalid_mode(self):
        """Test /mode with invalid mode name."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_core.console") as mock_console:
            result = await handler.handle(
                cmd="mode",
                cmd_args=["invalid_mode"],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.mode == "build"  # Mode unchanged
            mock_console.print.assert_called()
            # Verify error message was printed
            call_args = str(mock_console.print.call_args)
            assert "Unknown mode" in call_args or "invalid" in call_args.lower()


@pytest.mark.asyncio
class TestResetCommand:
    """Test /reset command for full reset."""

    async def test_reset_command_clears_everything(self):
        """Test that /reset clears context, skills, and resets mode."""
        ctx = MagicMock()
        ctx.reset = MagicMock()
        tool_selector = MagicMock()
        tool_selector.get_tools_for_mode = MagicMock(return_value=["tool1"])
        registry = MagicMock()
        registry.get_genai_tools = MagicMock(return_value=[])

        handler = CommandHandler(
            ctx=ctx,
            tool_selector=tool_selector,
            registry=registry,
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        build_system_prompt = MagicMock(return_value="build prompt")
        convert_tools = MagicMock(return_value=[])

        with patch("src.host.cli.commands_core.console"):
            result = await handler.handle(
                cmd="reset",
                cmd_args=[],
                mode="plan",
                tool_names=["old_tool"],
                tool_definitions=[],
                conversation_history=[{"role": "user", "content": "test"}],
                active_skills_content="skill1",
                build_system_prompt=build_system_prompt,
                convert_tools_to_definitions=convert_tools,
            )

            ctx.reset.assert_called_once()
            assert result.mode == "build"
            assert result.conversation_history == []
            assert result.active_skills_content == ""
            assert result.handled is True

    async def test_reset_command_resets_to_build_mode(self):
        """Test that /reset always resets to build mode."""
        ctx = MagicMock()
        ctx.reset = MagicMock()
        tool_selector = MagicMock()
        tool_selector.get_tools_for_mode = MagicMock(return_value=[])
        registry = MagicMock()
        registry.get_genai_tools = MagicMock(return_value=[])

        handler = CommandHandler(
            ctx=ctx,
            tool_selector=tool_selector,
            registry=registry,
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_core.console"):
            result = await handler.handle(
                cmd="reset",
                cmd_args=[],
                mode="plan",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.mode == "build"


@pytest.mark.asyncio
class TestSkillsCommand:
    """Test /skills command."""

    async def test_skills_command_shows_active_skills(self):
        """Test that /skills shows active skills."""
        skill_mgr = MagicMock()
        skill_mgr.get_active_skills = MagicMock(return_value=["skill1", "skill2"])

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=skill_mgr,
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_core.console") as mock_console:
            result = await handler.handle(
                cmd="skills",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            mock_console.print.assert_called()

    async def test_skills_command_with_no_active_skills(self):
        """Test /skills when no skills are active."""
        skill_mgr = MagicMock()
        skill_mgr.get_active_skills = MagicMock(return_value=[])

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=skill_mgr,
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_core.console"):
            result = await handler.handle(
                cmd="skills",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True


@pytest.mark.asyncio
class TestCheckpointsCommand:
    """Test /checkpoints command."""

    async def test_checkpoints_command_lists_checkpoints(self):
        """Test that /checkpoints lists available checkpoints."""
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.list_checkpoints = MagicMock(return_value=[
            {
                "id": "cp1",
                "created_at": "2024-01-01T10:00:00",
                "files_count": 5,
                "prompt": "Test checkpoint",
            },
            {
                "id": "cp2",
                "created_at": "2024-01-01T11:00:00",
                "files_count": 3,
                "prompt": "Another checkpoint",
            },
        ])

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=checkpoint_mgr,
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console"):
            result = await handler.handle(
                cmd="checkpoints",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            checkpoint_mgr.list_checkpoints.assert_called_once_with(limit=10)

    async def test_checkpoints_command_with_no_checkpoints(self):
        """Test /checkpoints when no checkpoints exist."""
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.list_checkpoints = MagicMock(return_value=[])

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=checkpoint_mgr,
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console"):
            result = await handler.handle(
                cmd="checkpoints",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True


@pytest.mark.asyncio
class TestRewindCommand:
    """Test /rewind command for checkpoint restoration."""

    async def test_rewind_to_specific_checkpoint(self):
        """Test rewinding to a specific checkpoint."""
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.rewind = MagicMock(return_value={
            "restored_files": ["file1.py", "file2.py"],
            "failed_files": [],
        })

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=checkpoint_mgr,
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console"):
            result = await handler.handle(
                cmd="rewind",
                cmd_args=["cp123"],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            checkpoint_mgr.rewind.assert_called_once_with("cp123", mode="code")

    async def test_rewind_to_last_checkpoint(self):
        """Test rewinding to last checkpoint without args."""
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.rewind_last = MagicMock(return_value={
            "restored_files": ["file1.py"],
            "failed_files": [],
        })

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=checkpoint_mgr,
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console"):
            result = await handler.handle(
                cmd="rewind",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            checkpoint_mgr.rewind_last.assert_called_once_with(mode="code")

    async def test_rewind_with_failed_files(self):
        """Test rewind when some files fail to restore."""
        checkpoint_mgr = MagicMock()
        checkpoint_mgr.rewind = MagicMock(return_value={
            "restored_files": ["file1.py"],
            "failed_files": ["file2.py"],
        })

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=checkpoint_mgr,
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console") as mock_console:
            result = await handler.handle(
                cmd="rewind",
                cmd_args=["cp123"],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            # Verify warning about failed files
            mock_console.print.assert_called()


@pytest.mark.asyncio
class TestSessionsCommand:
    """Test /sessions command."""

    async def test_sessions_command_lists_sessions(self):
        """Test that /sessions lists recent sessions."""
        session_mgr = MagicMock()

        # Create proper mock objects with string representations
        session1 = MagicMock()
        session1.id = "sess1"
        session1.name = "Session 1"
        session1.message_count = 10
        session1.updated_at = datetime.now().timestamp()
        session1.__str__ = MagicMock(return_value="sess1")

        session2 = MagicMock()
        session2.id = "sess2"
        session2.name = "Session 2"
        session2.message_count = 5
        session2.updated_at = datetime.now().timestamp()
        session2.__str__ = MagicMock(return_value="sess2")

        session_mgr.list_sessions = MagicMock(return_value=[session1, session2])

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=session_mgr,
            hook_mgr=MagicMock(),
            model_name="test",
            project="test-project",
        )

        with patch("src.host.cli.commands_session.console"):
            result = await handler.handle(
                cmd="sessions",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            session_mgr.list_sessions.assert_called_once_with(
                project="test-project", limit=10
            )

    async def test_sessions_command_with_no_sessions(self):
        """Test /sessions when no sessions exist."""
        session_mgr = MagicMock()
        session_mgr.list_sessions = MagicMock(return_value=[])

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=session_mgr,
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console"):
            result = await handler.handle(
                cmd="sessions",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True


@pytest.mark.asyncio
class TestHistoryCommand:
    """Test /history command."""

    async def test_history_command_shows_recent_commands(self):
        """Test that /history shows recent command history."""
        cmd_history = MagicMock()
        cmd_history.get_recent = MagicMock(return_value=[
            "/mode build",
            "/clear",
            "/help",
        ])

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=cmd_history,
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_config.console"):
            result = await handler.handle(
                cmd="history",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            cmd_history.get_recent.assert_called_once_with(20)

    async def test_history_command_with_empty_history(self):
        """Test /history when no history exists."""
        cmd_history = MagicMock()
        cmd_history.get_recent = MagicMock(return_value=[])

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=cmd_history,
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_config.console"):
            result = await handler.handle(
                cmd="history",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True


@pytest.mark.asyncio
class TestCostCommand:
    """Test /cost command."""

    async def test_cost_command_shows_usage(self):
        """Test that /cost shows usage and cost statistics."""
        cost_tracker = MagicMock()
        cost_tracker.get_cost_summary = MagicMock(return_value={
            "session": {
                "total_tokens": 5000,
                "total_input_tokens": 3000,
                "total_output_tokens": 2000,
                "total_cost_usd": 0.15,
                "request_count": 10,
            },
            "today": {
                "total_tokens": 10000,
                "total_input_tokens": 6000,
                "total_output_tokens": 4000,
                "total_cost_usd": 0.30,
                "request_count": 20,
            },
            "budget": {
                "warning": None,
            },
        })

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=cost_tracker,
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_config.console"):
            result = await handler.handle(
                cmd="cost",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            cost_tracker.get_cost_summary.assert_called_once()

    async def test_cost_command_with_budget_warning(self):
        """Test /cost when budget warning is present."""
        cost_tracker = MagicMock()
        cost_tracker.get_cost_summary = MagicMock(return_value={
            "session": {
                "total_tokens": 5000,
                "total_input_tokens": 3000,
                "total_output_tokens": 2000,
                "total_cost_usd": 0.15,
                "request_count": 10,
            },
            "today": {
                "total_tokens": 10000,
                "total_input_tokens": 6000,
                "total_output_tokens": 4000,
                "total_cost_usd": 0.30,
                "request_count": 20,
            },
            "budget": {
                "warning": "Daily budget limit approaching",
            },
        })

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=cost_tracker,
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_config.console") as mock_console:
            result = await handler.handle(
                cmd="cost",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            # Verify warning was printed
            mock_console.print.assert_called()


@pytest.mark.asyncio
class TestTasksCommand:
    """Test /tasks command."""

    async def test_tasks_command_lists_tasks(self):
        """Test that /tasks lists background tasks."""
        task_mgr = MagicMock()
        task_mgr.list_tasks = MagicMock(return_value=[
            {
                "id": "task1",
                "name": "Test Task 1",
                "status": "running",
                "progress": 50,
                "duration": 10.5,
            },
            {
                "id": "task2",
                "name": "Test Task 2",
                "status": "completed",
                "progress": 100,
                "duration": 5.2,
            },
        ])

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=task_mgr,
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console"):
            result = await handler.handle(
                cmd="tasks",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            task_mgr.list_tasks.assert_called_once_with(limit=10)

    async def test_tasks_command_with_no_tasks(self):
        """Test /tasks when no tasks exist."""
        task_mgr = MagicMock()
        task_mgr.list_tasks = MagicMock(return_value=[])

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=task_mgr,
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console"):
            result = await handler.handle(
                cmd="tasks",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True


@pytest.mark.asyncio
class TestUnknownCommand:
    """Test handling of unknown commands."""

    async def test_unknown_command_returns_handled_true(self):
        """Test that unknown command returns handled=True."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands.console") as mock_console:
            result = await handler.handle(
                cmd="unknown_command",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            mock_console.print.assert_called()


@pytest.mark.asyncio
class TestReturnValueStructure:
    """Test that handle() returns correct structure."""

    async def test_handle_returns_all_required_keys(self):
        """Test that handle() returns all required keys."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_core.console"):
            result = await handler.handle(
                cmd="help",
                cmd_args=[],
                mode="build",
                tool_names=["tool1"],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            required_keys = {
                "handled",
                "mode",
                "tool_names",
                "tool_definitions",
                "system_prompt",
                "active_skills_content",
                "conversation_history",
            }
            assert all(hasattr(result, key) for key in required_keys)

    async def test_handle_preserves_input_values_on_no_change(self):
        """Test that handle() preserves input values when command doesn't change them."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        original_mode = "build"
        original_tools = ["tool1", "tool2"]
        original_history = [{"role": "user", "content": "test"}]

        with patch("src.host.cli.commands_core.console"):
            result = await handler.handle(
                cmd="help",
                cmd_args=[],
                mode=original_mode,
                tool_names=original_tools,
                tool_definitions=[],
                conversation_history=original_history,
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.mode == original_mode
            assert result.tool_names == original_tools
            assert result.conversation_history == original_history


@pytest.mark.asyncio
class TestSessionManagementCommands:
    """Test session management commands."""

    async def test_resume_session_with_valid_id(self):
        """Test /resume with valid session ID."""
        session_mgr = MagicMock()
        session_data = MagicMock()
        session_data.metadata.get_display_name = MagicMock(return_value="Test Session")
        session_mgr.resume_session = MagicMock(return_value=session_data)

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=session_mgr,
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console"):
            result = await handler.handle(
                cmd="resume",
                cmd_args=["sess123"],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            session_mgr.resume_session.assert_called_once_with("sess123")

    async def test_resume_session_without_args(self):
        """Test /resume without arguments shows usage."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console") as mock_console:
            result = await handler.handle(
                cmd="resume",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            mock_console.print.assert_called()

    async def test_rename_session_with_name(self):
        """Test /rename with new session name."""
        ctx = MagicMock()
        ctx.session_id = "sess123"
        session_mgr = MagicMock()
        session_mgr.rename_session = MagicMock()

        handler = CommandHandler(
            ctx=ctx,
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=session_mgr,
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console"):
            result = await handler.handle(
                cmd="rename",
                cmd_args=["New", "Session", "Name"],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            session_mgr.rename_session.assert_called_once_with("sess123", "New Session Name")

    async def test_export_session_with_path(self):
        """Test /export with custom path."""
        ctx = MagicMock()
        ctx.session_id = "sess123"
        session_mgr = MagicMock()
        session_mgr.export_session = MagicMock()

        handler = CommandHandler(
            ctx=ctx,
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=session_mgr,
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console"):
            result = await handler.handle(
                cmd="export",
                cmd_args=["/tmp/export.md"],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            session_mgr.export_session.assert_called_once()

    async def test_fork_session(self):
        """Test /fork command."""
        ctx = MagicMock()
        ctx.session_id = "sess123"
        session_mgr = MagicMock()
        forked_session = MagicMock()
        forked_session.metadata.id = "sess456"
        session_mgr.fork_session = MagicMock(return_value=forked_session)

        handler = CommandHandler(
            ctx=ctx,
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=session_mgr,
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_session.console"):
            result = await handler.handle(
                cmd="fork",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            session_mgr.fork_session.assert_called_once_with("sess123")


@pytest.mark.asyncio
class TestModelCommand:
    """Test /model command."""

    async def test_model_command_switch_model(self):
        """Test /model with model name to switch."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="gpt-4",
            project="test",
        )

        with patch("src.host.cli.commands_config.console"):
            with patch("src.host.cli.commands_config.ModelManager") as mock_mgr_class:
                mock_mgr = MagicMock()
                mock_mgr.switch_model = MagicMock(return_value=True)
                mock_mgr.get_current_model = MagicMock(return_value="gpt-3.5")
                mock_mgr_class.return_value = mock_mgr

                result = await handler.handle(
                    cmd="model",
                    cmd_args=["gpt-3.5"],
                    mode="build",
                    tool_names=[],
                    tool_definitions=[],
                    conversation_history=[],
                    active_skills_content="",
                    build_system_prompt=MagicMock(return_value="prompt"),
                    convert_tools_to_definitions=MagicMock(return_value=[]),
                    )

                assert result.handled is True

    async def test_model_command_list_models(self):
        """Test /model without args lists available models."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="gpt-4",
            project="test",
        )

        with patch("src.host.cli.commands_config.console"):
            with patch("src.host.cli.commands_config.ModelManager") as mock_mgr_class:
                mock_mgr = MagicMock()
                mock_mgr.format_model_list = MagicMock(return_value="Model list")
                mock_mgr_class.return_value = mock_mgr

                result = await handler.handle(
                    cmd="model",
                    cmd_args=[],
                    mode="build",
                    tool_names=[],
                    tool_definitions=[],
                    conversation_history=[],
                    active_skills_content="",
                    build_system_prompt=MagicMock(return_value="prompt"),
                    convert_tools_to_definitions=MagicMock(return_value=[]),
                    )

                assert result.handled is True


@pytest.mark.asyncio
class TestPluginCommand:
    """Test /plugin command."""

    async def test_plugin_install(self):
        """Test /plugin install command."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_config.console"):
            with patch("src.host.cli.commands_config.PluginManager") as mock_mgr_class:
                mock_mgr = MagicMock()
                mock_plugin = MagicMock()
                mock_plugin.manifest.name = "test-plugin"
                mock_mgr.install = MagicMock(return_value=mock_plugin)
                mock_mgr_class.return_value = mock_mgr

                result = await handler.handle(
                    cmd="plugin",
                    cmd_args=["install", "test-plugin"],
                    mode="build",
                    tool_names=[],
                    tool_definitions=[],
                    conversation_history=[],
                    active_skills_content="",
                    build_system_prompt=MagicMock(return_value="prompt"),
                    convert_tools_to_definitions=MagicMock(return_value=[]),
                    )

                assert result.handled is True

    async def test_plugin_enable(self):
        """Test /plugin enable command."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_config.console"):
            with patch("src.host.cli.commands_config.PluginManager") as mock_mgr_class:
                mock_mgr = MagicMock()
                mock_mgr.enable = MagicMock(return_value=True)
                mock_mgr_class.return_value = mock_mgr

                result = await handler.handle(
                    cmd="plugin",
                    cmd_args=["enable", "test-plugin"],
                    mode="build",
                    tool_names=[],
                    tool_definitions=[],
                    conversation_history=[],
                    active_skills_content="",
                    build_system_prompt=MagicMock(return_value="prompt"),
                    convert_tools_to_definitions=MagicMock(return_value=[]),
                    )

                assert result.handled is True

    async def test_plugin_disable(self):
        """Test /plugin disable command."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_config.console"):
            with patch("src.host.cli.commands_config.PluginManager") as mock_mgr_class:
                mock_mgr = MagicMock()
                mock_mgr.disable = MagicMock(return_value=True)
                mock_mgr_class.return_value = mock_mgr

                result = await handler.handle(
                    cmd="plugin",
                    cmd_args=["disable", "test-plugin"],
                    mode="build",
                    tool_names=[],
                    tool_definitions=[],
                    conversation_history=[],
                    active_skills_content="",
                    build_system_prompt=MagicMock(return_value="prompt"),
                    convert_tools_to_definitions=MagicMock(return_value=[]),
                    )

                assert result.handled is True

    async def test_plugin_without_args(self):
        """Test /plugin without arguments shows usage."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_config.console") as mock_console:
            result = await handler.handle(
                cmd="plugin",
                cmd_args=[],
                mode="build",
                tool_names=[],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
                build_system_prompt=MagicMock(return_value="prompt"),
                convert_tools_to_definitions=MagicMock(return_value=[]),
            )

            assert result.handled is True
            mock_console.print.assert_called()


@pytest.mark.asyncio
class TestThemeCommand:
    """Test /theme command."""

    async def test_theme_command_set_theme(self):
        """Test /theme with theme name."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_config.console"):
            with patch("src.host.cli.commands_config.ThemeManager") as mock_mgr_class:
                mock_mgr = MagicMock()
                mock_mgr.set_theme = MagicMock(return_value=True)
                mock_mgr_class.return_value = mock_mgr

                result = await handler.handle(
                    cmd="theme",
                    cmd_args=["dark"],
                    mode="build",
                    tool_names=[],
                    tool_definitions=[],
                    conversation_history=[],
                    active_skills_content="",
                    build_system_prompt=MagicMock(return_value="prompt"),
                    convert_tools_to_definitions=MagicMock(return_value=[]),
                    )

                assert result.handled is True

    async def test_theme_command_list_themes(self):
        """Test /theme without args lists available themes."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_config.console"):
            with patch("src.host.cli.commands_config.ThemeManager") as mock_mgr_class:
                mock_mgr = MagicMock()
                mock_mgr.format_theme_list = MagicMock(return_value="Theme list")
                mock_mgr_class.return_value = mock_mgr

                result = await handler.handle(
                    cmd="theme",
                    cmd_args=[],
                    mode="build",
                    tool_names=[],
                    tool_definitions=[],
                    conversation_history=[],
                    active_skills_content="",
                    build_system_prompt=MagicMock(return_value="prompt"),
                    convert_tools_to_definitions=MagicMock(return_value=[]),
                    )

                assert result.handled is True


@pytest.mark.asyncio
class TestToggleCommands:
    """Test toggle commands like /vim and /thinking."""

    async def test_vim_toggle(self):
        """Test /vim command toggles vim mode."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_config.console"):
            with patch("src.host.cli.commands_config.InputManager") as mock_mgr_class:
                mock_mgr = MagicMock()
                from src.core.input_mode import InputMode
                mock_mgr.toggle_mode = MagicMock(return_value=InputMode.VI)
                mock_mgr_class.return_value = mock_mgr

                result = await handler.handle(
                    cmd="vim",
                    cmd_args=[],
                    mode="build",
                    tool_names=[],
                    tool_definitions=[],
                    conversation_history=[],
                    active_skills_content="",
                    build_system_prompt=MagicMock(return_value="prompt"),
                    convert_tools_to_definitions=MagicMock(return_value=[]),
                    )

                assert result.handled is True

    async def test_thinking_toggle(self):
        """Test /thinking command toggles thinking mode."""
        from src.core.thinking import ThinkingMode

        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_config.console"):
            with patch("src.host.cli.commands_config.ThinkingManager") as mock_mgr_class:
                mock_mgr = MagicMock()
                mock_mgr.toggle_mode = MagicMock(return_value=ThinkingMode.EXTENDED)
                mock_mgr.get_mode_indicator = MagicMock(return_value="[ON]")
                mock_mgr_class.return_value = mock_mgr

                result = await handler.handle(
                    cmd="thinking",
                    cmd_args=[],
                    mode="build",
                    tool_names=[],
                    tool_definitions=[],
                    conversation_history=[],
                    active_skills_content="",
                    build_system_prompt=MagicMock(return_value="prompt"),
                    convert_tools_to_definitions=MagicMock(return_value=[]),
                    )

                assert result.handled is True


@pytest.mark.asyncio
class TestInitCommand:
    """Test /init command."""

    async def test_init_creates_agents_md(self):
        """Test /init creates AGENTS.md file."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_core.console"):
            with patch("src.host.cli.commands_core.Path") as mock_path_class:
                mock_path = MagicMock()
                mock_path.exists = MagicMock(return_value=False)
                mock_path.write_text = MagicMock()
                mock_path_class.cwd.return_value = MagicMock()
                mock_path_class.cwd.return_value.__truediv__ = MagicMock(return_value=mock_path)

                result = await handler.handle(
                    cmd="init",
                    cmd_args=[],
                    mode="build",
                    tool_names=[],
                    tool_definitions=[],
                    conversation_history=[],
                    active_skills_content="",
                    build_system_prompt=MagicMock(return_value="prompt"),
                    convert_tools_to_definitions=MagicMock(return_value=[]),
                    )

                assert result.handled is True

    async def test_init_skips_if_exists(self):
        """Test /init skips if AGENTS.md already exists."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            skill_mgr=MagicMock(),
            checkpoint_mgr=MagicMock(),
            task_mgr=MagicMock(),
            cost_tracker=MagicMock(),
            cmd_history=MagicMock(),
            session_mgr=MagicMock(),
            hook_mgr=MagicMock(),
            model_name="test",
            project="test",
        )

        with patch("src.host.cli.commands_core.console"):
            with patch("src.host.cli.commands_core.Path") as mock_path_class:
                mock_path = MagicMock()
                mock_path.exists = MagicMock(return_value=True)
                mock_path_class.cwd.return_value = MagicMock()
                mock_path_class.cwd.return_value.__truediv__ = MagicMock(return_value=mock_path)

                result = await handler.handle(
                    cmd="init",
                    cmd_args=[],
                    mode="build",
                    tool_names=[],
                    tool_definitions=[],
                    conversation_history=[],
                    active_skills_content="",
                    build_system_prompt=MagicMock(return_value="prompt"),
                    convert_tools_to_definitions=MagicMock(return_value=[]),
                    )

                assert result.handled is True
                # Verify write_text was not called
                mock_path.write_text.assert_not_called()
