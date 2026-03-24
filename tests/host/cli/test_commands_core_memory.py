"""Tests for /memory command handling in CoreCommandHandler."""

from unittest.mock import MagicMock, patch

import pytest

from src.host.cli.command_context import CommandContext
from src.host.cli.commands_core import CoreCommandHandler


def _make_handler(project: str = "demo-project") -> CoreCommandHandler:
    cc = CommandContext(
        ctx=MagicMock(),
        tool_selector=MagicMock(),
        registry=MagicMock(),
        hook_mgr=MagicMock(),
        project=project,
        permission_mgr=MagicMock(),
    )
    return CoreCommandHandler(cc)


@pytest.mark.asyncio
class TestMemoryCommand:
    async def test_memory_notes_uses_current_project(self):
        handler = _make_handler("alpha")

        with (
            patch("src.host.cli.commands_core.list_notes", return_value="notes") as mock_list,
            patch("src.host.cli.commands_core.console") as mock_console,
        ):
            await handler._handle_memory(["notes"])

        mock_list.assert_called_once_with(collection_name="alpha")
        mock_console.print.assert_called_with("notes")

    async def test_memory_search_uses_current_project(self):
        handler = _make_handler("beta")

        with (
            patch("src.host.cli.commands_core.search_notes", return_value="result") as mock_search,
            patch("src.host.cli.commands_core.console") as mock_console,
        ):
            await handler._handle_memory(["search", "hello", "world"])

        mock_search.assert_called_once_with(query="hello world", collection_name="beta")
        mock_console.print.assert_called_with("result")

    async def test_memory_persona_set_updates_persona(self):
        handler = _make_handler()

        with (
            patch(
                "src.host.cli.commands_core.update_user_persona", return_value="ok"
            ) as mock_update,
            patch("src.host.cli.commands_core.console") as mock_console,
        ):
            await handler._handle_memory(["persona", "set", "role", "staff", "engineer"])

        mock_update.assert_called_once_with(key="role", value="staff engineer")
        mock_console.print.assert_called_with("ok")

    async def test_memory_delete_uses_current_project(self):
        handler = _make_handler("gamma")

        with (
            patch("src.host.cli.commands_core.delete_note", return_value="deleted") as mock_delete,
            patch("src.host.cli.commands_core.console") as mock_console,
        ):
            await handler._handle_memory(["delete", "old", "note"])

        mock_delete.assert_called_once_with(title="old note", collection_name="gamma")
        mock_console.print.assert_called_with("deleted")

    async def test_memory_export_writes_to_path(self, tmp_path):
        handler = _make_handler("delta")
        export_path = tmp_path / "notes.json"

        with (
            patch(
                "src.host.cli.commands_core.export_notes", return_value='[{"title":"x"}]'
            ) as mock_export,
            patch("src.host.cli.commands_core.console") as mock_console,
        ):
            await handler._handle_memory(["export", str(export_path)])

        mock_export.assert_called_once_with(collection_name="delta", export_format="json")
        assert export_path.read_text(encoding="utf-8") == '[{"title":"x"}]'
        mock_console.print.assert_called_with(f"[green]Exported notes to {export_path}[/green]")
