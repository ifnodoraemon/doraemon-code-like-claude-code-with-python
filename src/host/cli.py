"""
Polymath CLI - Main Command Line Interface

Provides the primary interactive interface for Polymath AI agent.
Features:
- Multi-mode support (plan/build/coder/architect)
- HITL (Human-in-the-loop) approval for sensitive operations
- Rich terminal UI with markdown rendering
- Vector memory (ChromaDB) for long-term recall
- Direct tool calls (no subprocess overhead)
- Automatic context summarization for unlimited conversation length
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from google import genai
from google.genai import types
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

# Direct imports (no DI container needed)
from src.core.config import load_config
from src.core.context_manager import ContextManager, ContextConfig
from src.core.diff import print_diff
from src.core.prompts import get_system_prompt
from src.core.rules import format_instructions_for_prompt, load_all_instructions
from src.core.skills import SkillManager
from src.host.tools import get_default_registry

# Fix encoding
for stream in [sys.stdin, sys.stdout, sys.stderr]:
    if stream and hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

app = typer.Typer()
console = Console()

# Mode colors (only two modes: plan and build)
MODE_COLORS = {
    "plan": "blue",
    "build": "green",
}


def create_chat(
    client: genai.Client,
    mode: str,
    tools: list[types.FunctionDeclaration],
    history: list | None = None,
    skills_content: str = "",
):
    """Create a chat session with the given mode, tools, and skills."""
    config = load_config()
    persona = config.get("persona", {})

    # Build system prompt
    system_prompt = get_system_prompt(mode, persona)

    # Add project rules (AGENTS.md)
    instructions = load_all_instructions(config)
    if instructions:
        system_prompt += format_instructions_for_prompt(instructions)

    # Add active skills (loaded on-demand based on context)
    if skills_content:
        system_prompt += f"\n\n{skills_content}"

    # Create chat
    gen_config = types.GenerateContentConfig(
        tools=[types.Tool(function_declarations=tools)],
        system_instruction=system_prompt,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    model_name = os.getenv("POLYMATH_MODEL", "gemini-2.0-flash")
    return client.chats.create(model=model_name, config=gen_config, history=history or [])


async def chat_loop(project: str = "default"):
    """Main chat loop with automatic context management."""

    # 1. Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        console.print("[red]Error: GOOGLE_API_KEY not set[/red]")
        return

    # 2. Initialize client
    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        console.print(f"[red]Failed to initialize client: {e}[/red]")
        return

    # 3. Initialize tools (direct function calls, no MCP)
    registry = get_default_registry()
    tools = registry.get_genai_tools()
    sensitive_tools = registry.get_sensitive_tools()

    # 4. Initialize context manager
    ctx_config = ContextConfig(
        max_context_tokens=100_000,
        summarize_threshold=0.7,
        keep_recent_messages=6,
        auto_save=True,
    )
    ctx = ContextManager(project=project, config=ctx_config)

    # 5. Initialize skill manager
    skill_mgr = SkillManager(project_dir=Path.cwd(), max_skill_tokens=5000)
    active_skills_content = ""  # Will be populated based on context

    # Show startup info
    console.print(f"[bold yellow]Project: {project}[/bold yellow]")
    console.print(f"[dim]Tools loaded: {len(tools)}[/dim]")

    stats = ctx.get_context_stats()
    if stats["messages"] > 0 or stats["summaries"] > 0:
        console.print(
            f"[dim]Restored context: {stats['messages']} messages, "
            f"{stats['summaries']} summaries[/dim]"
        )

    # 6. State
    mode = "build"  # Default to build mode
    turn_count = 0

    # Create initial chat with restored history
    history = ctx.get_history_for_api()
    chat = create_chat(client, mode, tools, history, active_skills_content)

    console.print(
        Panel.fit(
            f"[bold blue]Polymath[/bold blue]\n"
            f"[dim]Type /help for commands. Mode: {mode}[/dim]",
            border_style="blue",
        )
    )

    # 5. Main loop
    while True:
        mode_color = MODE_COLORS.get(mode, "yellow")
        user_input = Prompt.ask(f"\n[bold {mode_color}]You ({mode})[/bold {mode_color}]")

        # Exit
        if user_input.lower() in ["exit", "quit", "/exit"]:
            break

        # Slash commands
        if user_input.startswith("/"):
            cmd = user_input[1:].split()[0].lower()

            if cmd == "help":
                console.print("""
[bold]Commands:[/bold]
  /mode <name>  - Switch mode (plan/build)
  /context      - Show context/memory statistics
  /skills       - Show loaded skills
  /clear        - Clear conversation (keeps summaries)
  /reset        - Full reset (clears everything)
  /tools        - List available tools
  /debug        - Show debug info
  /exit         - Exit

[bold]Modes:[/bold]
  plan   - Analyze requirements, investigate code, create plans (read-only)
  build  - Implement solutions, write code, execute tasks
""")
                continue

            elif cmd == "mode":
                parts = user_input.split()
                if len(parts) > 1:
                    new_mode = parts[1].lower()
                    if new_mode in MODE_COLORS:
                        mode = new_mode
                        # Rebuild chat with new mode but KEEP context history
                        history = ctx.get_history_for_api()
                        chat = create_chat(client, mode, tools, history)
                        console.print(f"[green]Switched to {mode} mode (context preserved)[/green]")
                    else:
                        console.print(f"[red]Unknown mode: {new_mode}[/red]")
                else:
                    console.print(f"Current mode: {mode}")
                continue

            elif cmd == "context":
                stats = ctx.get_context_stats()
                table = Table(title="Context Statistics", show_header=False)
                table.add_column("Key", style="cyan")
                table.add_column("Value", style="green")
                table.add_row("Session ID", stats["session_id"])
                table.add_row("Messages", str(stats["messages"]))
                table.add_row("Summaries", str(stats["summaries"]))
                table.add_row("Total Ever", str(stats["total_messages_ever"]))
                table.add_row("Est. Tokens", f"{stats['estimated_tokens']:,}")
                table.add_row("Last Prompt", f"{stats['last_prompt_tokens']:,}")
                table.add_row("Threshold", f"{stats['threshold_tokens']:,}")
                table.add_row("Usage", f"{stats['usage_percent']}%")
                active = skill_mgr.get_active_skills()
                table.add_row("Active Skills", ", ".join(active) if active else "(none)")
                if stats["needs_summary"]:
                    table.add_row("Status", "[yellow]Summary needed[/yellow]")
                console.print(table)
                continue

            elif cmd == "skills":
                # Show available and active skills
                console.print("[bold]Skills System[/bold]")
                active = skill_mgr.get_active_skills()
                if active:
                    console.print(f"  [green]Active:[/green] {', '.join(active)}")
                else:
                    console.print("  [dim]No skills currently active[/dim]")
                console.print("\n[dim]Skills are loaded automatically based on conversation context.[/dim]")
                console.print("[dim]Put SKILL.md files in .polymath/skills/<name>/ to add custom skills.[/dim]")
                continue

            elif cmd == "clear":
                ctx.clear(keep_summaries=True)
                history = ctx.get_history_for_api()
                chat = create_chat(client, mode, tools, history, active_skills_content)
                console.print("[green]Conversation cleared (summaries preserved)[/green]")
                continue

            elif cmd == "reset":
                ctx.reset()
                active_skills_content = ""  # Reset skills too
                chat = create_chat(client, mode, tools, [], "")
                turn_count = 0
                console.print("[green]Full reset complete[/green]")
                continue

            elif cmd == "tools":
                tool_names = registry.get_tool_names()
                console.print(f"[bold]Available tools ({len(tool_names)}):[/bold]")
                for name in sorted(tool_names):
                    marker = "🔒" if name in sensitive_tools else "  "
                    console.print(f"  {marker} {name}")
                continue

            elif cmd == "debug":
                console.print(f"Mode: {mode}")
                console.print(f"Turn: {turn_count}")
                console.print(f"Tools: {len(tools)}")
                console.print(f"Project: {project}")
                stats = ctx.get_context_stats()
                console.print(f"Context: {stats['messages']} msgs, {stats['summaries']} summaries")
                continue

            else:
                console.print(f"[yellow]Unknown command: {cmd}[/yellow]")
                continue

        # Track user message in context
        ctx.add_user_message(user_input)

        # Check if we need to load/update skills based on user input
        new_skills_content = skill_mgr.get_skills_for_context(user_input)
        if new_skills_content != active_skills_content:
            active_skills_content = new_skills_content
            new_active = skill_mgr.get_active_skills()
            if new_active:
                console.print(f"[dim cyan]Skills loaded: {', '.join(new_active)}[/dim cyan]")
            # Rebuild chat with new skills
            history = ctx.get_history_for_api()
            chat = create_chat(client, mode, tools, history, active_skills_content)

        # Send message
        try:
            with console.status(f"[bold {mode_color}]Thinking...[/bold {mode_color}]"):
                response = chat.send_message(user_input)
        except Exception as e:
            console.print(f"[red]API Error: {e}[/red]")
            continue

        # Process response (tool loop)
        accumulated_text = ""  # Accumulate all text parts for context tracking

        while True:
            if not response.candidates:
                console.print("[red]Empty response[/red]")
                break

            parts = response.candidates[0].content.parts
            has_tool_call = False
            tool_results = []

            for part in parts:
                # Text response
                if part.text:
                    accumulated_text += part.text
                    console.print(
                        Panel(
                            Markdown(part.text),
                            title="[bold purple]Response[/bold purple]",
                            border_style="purple",
                            expand=False,
                        )
                    )

                # Tool call
                if part.function_call:
                    has_tool_call = True
                    fc = part.function_call
                    tool_name = fc.name
                    args = dict(fc.args.items()) if hasattr(fc.args, "items") else {}

                    # Inject project context for memory tools
                    if tool_name in ["save_note", "search_notes"]:
                        args["collection_name"] = project

                    # Show diff for write operations
                    if tool_name == "write_file" and "content" in args and "path" in args:
                        console.print(
                            f"\n[bold yellow]📝 Proposing changes:[/bold yellow] {args['path']}"
                        )
                        print_diff(args["path"], args["content"])

                    # HITL approval for sensitive tools
                    tool_result = None
                    if tool_name in sensitive_tools:
                        console.print(f"\n[bold red]⚠️ Sensitive:[/bold red] {tool_name}")
                        if tool_name != "write_file":
                            console.print(
                                f"[dim]{json.dumps(args, indent=2, ensure_ascii=False)}[/dim]"
                            )

                        if Prompt.ask("Execute?", choices=["y", "n"], default="n") != "y":
                            tool_result = "User denied the operation."
                            console.print("[red]Cancelled[/red]")
                        else:
                            console.print(f"[cyan]Running {tool_name}...[/cyan]")
                            tool_result = await registry.call_tool(tool_name, args)
                    else:
                        console.print(f"[cyan]Running {tool_name}...[/cyan]")
                        tool_result = await registry.call_tool(tool_name, args)

                    tool_results.append({"name": tool_name, "result": {"result": tool_result}})

            # No more tool calls - done with this turn
            if not has_tool_call:
                turn_count += 1
                usage = response.usage_metadata

                # Track assistant response in context with actual token counts
                prompt_tokens = usage.prompt_token_count if usage else None
                completion_tokens = usage.candidates_token_count if usage else None

                # Record messages before summary check
                prev_summary_count = len(ctx.summaries)
                ctx.add_assistant_message(
                    accumulated_text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )

                # Check if summarization occurred - need to rebuild chat
                if len(ctx.summaries) > prev_summary_count:
                    console.print(
                        "[dim yellow]Context summarized to save memory. "
                        "Rebuilding conversation...[/dim yellow]"
                    )
                    history = ctx.get_history_for_api()
                    chat = create_chat(client, mode, tools, history, active_skills_content)

                # Show usage stats
                if usage:
                    stats = ctx.get_context_stats()
                    console.print(
                        f"\n[dim]Turn {turn_count} | "
                        f"In: {usage.prompt_token_count:,} | "
                        f"Out: {usage.candidates_token_count:,} | "
                        f"Ctx: {stats['usage_percent']}%[/dim]"
                    )
                break

            # Send tool results back
            response_parts = [
                types.Part.from_function_response(name=tr["name"], response=tr["result"])
                for tr in tool_results
            ]

            with console.status("[cyan]Processing...[/cyan]"):
                response = chat.send_message(response_parts)


@app.command()
def start(project: str = "default"):
    """Start Polymath CLI."""
    asyncio.run(chat_loop(project=project))


@app.command()
def version():
    """Show version information."""
    console.print("[bold]Polymath v0.5.0[/bold]")
    console.print("[dim]Simplified architecture with direct tool calls[/dim]")


def entry_point():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()
