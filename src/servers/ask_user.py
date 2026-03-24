"""
AskUser Tool — LLM 通过工具调用向用户提问

让 AI 在需要用户输入时主动提问, 而不是猜测或假设。
支持自由文本输入和选项列表。
"""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

console = Console()


def ask_user(
    question: str,
    options: str = "",
    multi_select: bool = False,
) -> str:
    """Ask the user a question and wait for their response.

    Use this tool when you need user input to proceed. Supports free-text input
    or a list of predefined options.

    Args:
        question: The question to ask the user.
        options: Comma-separated list of options (e.g. "Yes,No,Maybe").
                 Leave empty for free-text input.
        multi_select: When True with options, allow selecting multiple choices.

    Returns:
        The user's answer text.
    """
    # Display the question prominently
    console.print()
    console.print(
        Panel(
            f"[bold]{question}[/bold]",
            title="[bold cyan]AI Question[/bold cyan]",
            border_style="cyan",
            expand=False,
        )
    )

    # No options — free text input
    if not options.strip():
        answer = Prompt.ask("[cyan]Your answer[/cyan]")
        return answer.strip()

    # Parse options
    option_list = [o.strip() for o in options.split(",") if o.strip()]
    if not option_list:
        answer = Prompt.ask("[cyan]Your answer[/cyan]")
        return answer.strip()

    # Display numbered options
    for i, opt in enumerate(option_list, 1):
        console.print(f"  [bold]{i}.[/bold] {opt}")
    console.print(f"  [bold]{len(option_list) + 1}.[/bold] [dim]Other (custom input)[/dim]")

    if multi_select:
        console.print("[dim]Enter numbers separated by commas (e.g. 1,3)[/dim]")
        raw = Prompt.ask("[cyan]Your choice(s)[/cyan]")
        selected = []
        for part in raw.split(","):
            part = part.strip()
            if part.isdigit():
                idx = int(part)
                if 1 <= idx <= len(option_list):
                    selected.append(option_list[idx - 1])
                elif idx == len(option_list) + 1:
                    custom = Prompt.ask("[cyan]Custom input[/cyan]")
                    selected.append(custom.strip())
            else:
                # Treat as literal text
                selected.append(part)
        return ", ".join(selected) if selected else raw.strip()
    else:
        raw = Prompt.ask("[cyan]Your choice[/cyan]")
        raw = raw.strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(option_list):
                return option_list[idx - 1]
            elif idx == len(option_list) + 1:
                custom = Prompt.ask("[cyan]Custom input[/cyan]")
                return custom.strip()
        # Treat as literal text (user typed the option name directly)
        return raw
