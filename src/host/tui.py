"""
Textual-based TUI for Polymath.

This provides a modern terminal user interface with:
- Split-screen layout (chat + sidebar)
- Real-time Markdown rendering
- Keyboard shortcuts
- Mode switching (Tab key)
- AI chat integration via ModelClient
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.markdown import Markdown
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, ScrollableContainer, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static

# Load environment
load_dotenv()

# Import ModelClient for AI integration
from src.core.model_client import (
    ModelClient,
    ClientConfig,
    ClientMode,
    Message,
    ToolDefinition,
)
from src.core.prompts import get_system_prompt
from src.host.tools import get_default_registry


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

    mode = reactive("build")
    server_count = reactive(0)

    def compose(self) -> ComposeResult:
        yield Static("[bold]Polymath v0.7.0[/bold]", id="title")
        yield Static("", id="mode-display")
        yield Static("", id="server-display")
        yield Static("[green]Ready[/green]", id="status")

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
    SUB_TITLE = "Powered by ModelClient"

    current_mode = reactive("build")
    modes = ["build", "plan"]

    # AI client and conversation state
    model_client = None
    conversation_history: list[Message] = []
    system_prompt: str = ""
    model_name: str = ""

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Header()

        with Container(id="main-container"):
            with Vertical(id="chat-container"):
                yield ChatArea(id="chat-area")
                yield Input(placeholder="Type your message... (Tab to switch mode)", id="input")

            yield Sidebar(id="sidebar")

        yield Footer()

    async def on_mount(self) -> None:
        """Initialize the app."""
        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.mode = self.current_mode

        # Get tool count from registry
        try:
            registry = get_default_registry()
            sidebar.server_count = len(registry._tools)
        except Exception:
            sidebar.server_count = 0

        # Initialize AI client
        chat = self.query_one("#chat-area", ChatArea)

        try:
            self.model_client = await ModelClient.create()
            self.model_name = os.getenv("POLYMATH_MODEL", "gemini-2.5-flash-preview")
            self.system_prompt = get_system_prompt(self.current_mode, {})

            # Show mode info
            mode_info = ModelClient.get_mode_info()
            if mode_info.get("mode") == "gateway":
                chat.add_message("system", f"Polymath TUI v0.7.0 - Gateway Mode ({mode_info.get('gateway_url')})")
            else:
                providers = [p for p, v in mode_info.get("providers", {}).items() if v]
                chat.add_message("system", f"Polymath TUI v0.7.0 - Direct Mode ({', '.join(providers)})")

            chat.add_message("system", "Type /help for commands")
        except Exception as e:
            chat.add_message("system", f"[red]Failed to initialize AI client: {e}[/red]")
            chat.add_message("system", "Check your API configuration in .env file")

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
            # Send to AI
            if self.model_client:
                self.run_worker(self.send_to_ai(message))
            else:
                chat.add_message("assistant", "[red]AI client not initialized[/red]")

    async def send_to_ai(self, message: str) -> None:
        """Send message to AI and display response."""
        chat = self.query_one("#chat-area", ChatArea)
        sidebar = self.query_one("#sidebar", Sidebar)

        # Update status
        sidebar.query_one("#status", Static).update("[cyan]Thinking...[/cyan]")

        try:
            # Add user message to history
            self.conversation_history.append(Message(role="user", content=message))

            # Build messages with system prompt
            messages = [Message(role="system", content=self.system_prompt)]
            messages.extend(self.conversation_history)

            # Send to model
            response = await self.model_client.chat(
                messages,
                model=self.model_name,
            )

            # Add assistant response to history
            self.conversation_history.append(Message(
                role="assistant",
                content=response.content,
            ))

            # Display response
            if response.content:
                chat.add_message("assistant", response.content)
            else:
                chat.add_message("assistant", "[dim](Empty response)[/dim]")

            # Show usage if available
            if response.usage:
                usage_text = (
                    f"[dim]Tokens: {response.usage.get('prompt_tokens', 0)} in / "
                    f"{response.usage.get('completion_tokens', 0)} out[/dim]"
                )
                sidebar.query_one("#status", Static).update(usage_text)
            else:
                sidebar.query_one("#status", Static).update("[green]Ready[/green]")

        except Exception as e:
            chat.add_message("assistant", f"[red]Error: {e}[/red]")
            sidebar.query_one("#status", Static).update("[red]Error[/red]")

    def handle_command(self, command: str):
        """Handle slash commands."""
        chat = self.query_one("#chat-area", ChatArea)

        if command == "/help":
            help_text = """# Available Commands

- `/help` - Show this help message
- `/clear` - Clear chat history
- `/mode <name>` - Switch mode (build, plan)
- `/model` - Show current model
- `/quit` - Exit application

## Keyboard Shortcuts
- **Tab** - Switch between modes
- **Ctrl+L** - Clear chat
- **Ctrl+C** - Quit
"""
            chat.add_message("assistant", help_text)

        elif command == "/clear":
            chat.log.clear()
            self.conversation_history.clear()
            chat.add_message("system", "Chat and history cleared")

        elif command.startswith("/mode"):
            parts = command.split()
            if len(parts) > 1:
                new_mode = parts[1]
                if new_mode in self.modes:
                    self.current_mode = new_mode
                    self.system_prompt = get_system_prompt(new_mode, {})
                    sidebar = self.query_one("#sidebar", Sidebar)
                    sidebar.mode = new_mode
                    chat.add_message("system", f"Switched to {new_mode} mode")
                else:
                    chat.add_message("system", f"Unknown mode: {new_mode}. Available: {', '.join(self.modes)}")
            else:
                chat.add_message("system", f"Current mode: {self.current_mode}")

        elif command == "/model":
            chat.add_message("system", f"Current model: {self.model_name}")

        elif command == "/quit":
            self.exit()

        else:
            chat.add_message("system", f"Unknown command: {command}")

    def action_switch_mode(self):
        """Switch to next mode (Tab key)."""
        current_idx = self.modes.index(self.current_mode)
        next_idx = (current_idx + 1) % len(self.modes)
        self.current_mode = self.modes[next_idx]
        self.system_prompt = get_system_prompt(self.current_mode, {})

        sidebar = self.query_one("#sidebar", Sidebar)
        sidebar.mode = self.current_mode

        chat = self.query_one("#chat-area", ChatArea)
        chat.add_message("system", f"Switched to {self.current_mode} mode")

    def action_clear(self):
        """Clear chat area."""
        chat = self.query_one("#chat-area", ChatArea)
        chat.log.clear()
        self.conversation_history.clear()
        chat.add_message("system", "Chat and history cleared")


def run_tui():
    """Run the Textual UI."""
    app = PolymathTUI()
    app.run()


if __name__ == "__main__":
    run_tui()
