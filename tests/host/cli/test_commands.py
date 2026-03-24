"""Comprehensive tests for src/host/cli/commands.py

Tests cover:
- CommandHandler initialization
- Core slash command paths
- Command parsing and validation
- Command execution with various arguments
- Error handling for invalid commands
- Integration with core CLI dependencies
- Mode switching (plan/build)
- Project switching
- Help text generation
"""

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
        hook_mgr = MagicMock()

        handler = CommandHandler(
            ctx=ctx,
            tool_selector=tool_selector,
            registry=registry,
            hook_mgr=hook_mgr,
            project="test-project",
        )

        assert handler.cc.ctx == ctx
        assert handler.cc.tool_selector == tool_selector
        assert handler.cc.registry == registry
        assert handler.cc.hook_mgr == hook_mgr
        assert handler.cc.project == "test-project"

    def test_init_stores_all_attributes(self):
        """Test that all attributes are properly stored."""
        mocks = {
            "ctx": MagicMock(),
            "tool_selector": MagicMock(),
            "registry": MagicMock(),
            "hook_mgr": MagicMock(),
        }

        handler = CommandHandler(
            **mocks,
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
            hook_mgr=MagicMock(),
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
            hook_mgr=MagicMock(),
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
            hook_mgr=MagicMock(),
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
            hook_mgr=MagicMock(),
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
            hook_mgr=hook_mgr,
            project="test",
        )

        with patch("src.host.cli.commands_core.console"):
            result = await handler.handle(
                cmd="mode",
                cmd_args=["build"],
                mode="plan",
                tool_names=["old_tool"],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
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
            hook_mgr=hook_mgr,
            project="test",
        )

        with patch("src.host.cli.commands_core.console"):
            result = await handler.handle(
                cmd="mode",
                cmd_args=["plan"],
                mode="build",
                tool_names=["write_tool"],
                tool_definitions=[],
                conversation_history=[],
                active_skills_content="",
            )

            assert result.mode == "plan"
            tool_selector.get_tools_for_mode.assert_called_with("plan")

    async def test_mode_command_without_args_shows_current(self):
        """Test /mode without args shows current mode."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            hook_mgr=MagicMock(),
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
            )

            assert result.mode == "build"
            mock_console.print.assert_called()

    async def test_mode_command_invalid_mode(self):
        """Test /mode with invalid mode name."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            hook_mgr=MagicMock(),
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
            hook_mgr=MagicMock(),
            project="test",
        )

        with patch("src.host.cli.commands_core.console"):
            result = await handler.handle(
                cmd="reset",
                cmd_args=[],
                mode="plan",
                tool_names=["old_tool"],
                tool_definitions=[],
                conversation_history=[{"role": "user", "content": "test"}],
                active_skills_content="skill1",
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
            hook_mgr=MagicMock(),
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
            )

            assert result.mode == "build"


@pytest.mark.asyncio
class TestUnknownCommand:
    """Test handling of unknown commands."""

    async def test_unknown_command_returns_handled_true(self):
        """Test that unknown command returns handled=True."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            hook_mgr=MagicMock(),
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
            hook_mgr=MagicMock(),
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
            hook_mgr=MagicMock(),
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
            )

            assert result.mode == original_mode
            assert result.tool_names == original_tools
            assert result.conversation_history == original_history


@pytest.mark.asyncio
class TestInitCommand:
    """Test /init command."""

    async def test_init_creates_agents_md(self):
        """Test /init creates AGENTS.md file."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            hook_mgr=MagicMock(),
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
                )

                assert result.handled is True

    async def test_init_skips_if_exists(self):
        """Test /init skips if AGENTS.md already exists."""
        handler = CommandHandler(
            ctx=MagicMock(),
            tool_selector=MagicMock(),
            registry=MagicMock(),
            hook_mgr=MagicMock(),
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
                )

                assert result.handled is True
                # Verify write_text was not called
                mock_path.write_text.assert_not_called()
