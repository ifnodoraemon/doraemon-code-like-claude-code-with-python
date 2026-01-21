"""
Textual-based TUI for Polymath.

This provides a modern terminal user interface with:
- Split-screen layout (chat + sidebar)
- Real-time Markdown rendering
- Keyboard shortcuts
- Mode switching (Tab key)
"""

from rich.markdown import Markdown
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static


class ChatArea(ScrollableContainer):
    """Main chat area showing conversation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log = RichLog(highlight=True, markup=True)

    def compose(self) -> ComposeResult:
        yield self.log

    def add_message(self, role: str, content: str):
        """Add a message to the chat."""
        if role == "user":
            self.log.write(f"[bold cyan]You:[/bold cyan] {content}")
        elif role == "assistant":
            # Render as Markdown
            md = Markdown(content)
            self.log.write(md)
        elif role == "system":
            self.log.write(f"[dim]{content}[/dim]")


class Sidebar(Vertical):
    """Sidebar showing status and info."""

    mode = reactive("default")
    server_count = reactive(0)

    def compose(self) -> ComposeResult:
        yield Static("[bold]Polymath v0.4.0[/bold]", id="title")
        yield Static("", id="mode-display")
        yield Static("", id="server-display")
        yield Static("", id="status")

    def watch_mode(self, new_mode: str):
        """Update mode display when mode changes."""
        self.query_one("#mode-display", Static).update(f"Mode: [green]{new_mode}[/green]")

    def watch_server_count(self, count: int):
        """Update server count."""
        self.query_one("#server-display", Static).update(f"Servers: {count}")


class PolymathTUI(App):
    """Polymath Textual User Interface."""

    CSS = """
    Screen {
        background: $background;
    }

    #main-container {
        layout: horizontal;
        height: 100%;
    }

    #chat-container {
        width: 3fr;
        border: solid $primary;
    }

    #sidebar {
        width: 1fr;
        background: $surface;
        border: solid $primary;
        padding: 1;
    }

    #input-area {
        dock: bottom;
        height: 3;
        border: solid $accent;
    }

    Input {
        width: 100%;
    }

    #title {
        text-align: center;
        margin-bottom: 1;
    }

    #mode-display, #server-display, #status {
        margin-top: 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=True),
        Binding("tab", "switch_mode", "Switch Mode", show=True),
        Binding("ctrl+l", "clear", "Clear Chat", show=False),
    ]

    TITLE = "Polymath AI Assistant"
    SUB_TITLE = "Powered by MCP"

    current_mode = reactive("default")
    modes = ["default", "coder", "architect"]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()

        with Container(id="main-container"):
            with Vertical(id="chat-container"):
                yield ChatArea(id="chat-area")
                yield Input(placeholder="Type your message... (Tab to switch mode)", id="input")

            yield Sidebar(id="sidebar")

        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app."""
        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.mode = self.current_mode
        sidebar.server_count = 8  # TODO: Get actual count

        # Welcome message
        chat = self.query_one("#chat-area", ChatArea)
        chat.add_message("system", "Polymath TUI v0.4.0 - Type /help for commands")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input."""
        message = event.value.strip()
        if not message:
            return

        # Clear input
        event.input.value = ""

        # Add user message
        chat = self.query_one("#chat-area", ChatArea)
        chat.add_message("user", message)

        # Handle slash commands
        if message.startswith("/"):
            self.handle_command(message)
        else:
            # TODO: Send to AI
            chat.add_message("assistant", f"*Processing: {message}*\n\n(AI integration pending)")

    def handle_command(self, command: str):
        """Handle slash commands."""
        chat = self.query_one("#chat-area", ChatArea)

        if command == "/help":
            help_text = """# Available Commands

- `/help` - Show this help message
- `/clear` - Clear chat history
- `/mode <name>` - Switch mode (default, coder, architect)
- `/init` - Initialize project (create AGENTS.md)
- `/quit` - Exit application

## Keyboard Shortcuts
- **Tab** - Switch between modes
- **Ctrl+L** - Clear chat
- **Ctrl+C** - Quit
"""
            chat.add_message("assistant", help_text)

        elif command == "/clear":
            chat.log.clear()
            chat.add_message("system", "Chat cleared")

        elif command.startswith("/mode"):
            parts = command.split()
            if len(parts) > 1:
                new_mode = parts[1]
                if new_mode in self.modes:
                    self.current_mode = new_mode
                    sidebar = self.query_one("#sidebar", Sidebar)
                    sidebar.mode = new_mode
                    chat.add_message("system", f"Switched to {new_mode} mode")
                else:
                    chat.add_message("system", f"Unknown mode: {new_mode}")

        elif command == "/quit":
            self.exit()

        else:
            chat.add_message("system", f"Unknown command: {command}")

    def action_switch_mode(self):
        """Switch to next mode (Tab key)."""
        current_idx = self.modes.index(self.current_mode)
        next_idx = (current_idx + 1) % len(self.modes)
        self.current_mode = self.modes[next_idx]

        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.mode = self.current_mode

        chat = self.query_one("#chat-area", ChatArea)
        chat.add_message("system", f"Switched to {self.current_mode} mode")

    def action_clear(self):
        """Clear chat area."""
        chat = self.query_one("#chat-area", ChatArea)
        chat.log.clear()
        chat.add_message("system", "Chat cleared")


def run_tui():
    """Run the Textual UI."""
    app = PolymathTUI()
    app.run()


if __name__ == "__main__":
    run_tui()
