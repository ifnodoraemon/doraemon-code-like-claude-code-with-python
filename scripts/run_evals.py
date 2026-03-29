#!/usr/bin/env python3
"""
Evaluation Runner Script

Usage:
    python scripts/run_evals.py --category basic
    python scripts/run_evals.py --category advanced
    python scripts/run_evals.py --all
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

console = Console()


def load_tasks(category: str = None) -> list[dict]:
    """Load evaluation tasks from JSON files."""
    tasks = []
    tasks_dir = Path("tasks")

    if category:
        pattern = f"{category}/*.json"
    else:
        pattern = "**/*.json"

    for task_file in tasks_dir.glob(pattern):
        with open(task_file) as f:
            file_tasks = json.load(f)
            if isinstance(file_tasks, list):
                tasks.extend(file_tasks)
            else:
                tasks.append(file_tasks)

    return tasks


def display_task_summary(tasks: list[dict]):
    """Display summary of loaded tasks."""
    table = Table(title="Evaluation Tasks Summary")
    table.add_column("Category", style="cyan")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Count", style="green")

    # Group by category and difficulty
    summary = {}
    for task in tasks:
        cat = task.get("category", "unknown")
        diff = task.get("difficulty", "unknown")
        key = (cat, diff)
        summary[key] = summary.get(key, 0) + 1

    for (cat, diff), count in sorted(summary.items()):
        table.add_row(cat, diff, str(count))

    console.print(table)
    console.print(f"\n[bold]Total Tasks:[/bold] {len(tasks)}")


def main():
    parser = argparse.ArgumentParser(description="Run Doraemon Code evaluations")
    parser.add_argument("--category", choices=["basic", "advanced", "adversarial"],
                       help="Task category to run")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--list", action="store_true", help="List tasks without running")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per task")

    args = parser.parse_args()

    if args.all:
        tasks = load_tasks()
    elif args.category:
        tasks = load_tasks(args.category)
    else:
        console.print("[red]Error: Specify --category or --all[/red]")
        return 1

    if args.list:
        display_task_summary(tasks)
        return 0

    console.print(f"[bold]Loaded {len(tasks)} evaluation tasks.[/bold]\n")
    console.print("[red]scripts/run_evals.py is not wired to the active evaluation harness.[/red]")
    console.print("Use one of these instead:")
    console.print("  `python3 scripts/ci_eval_runner.py --scope quick`")
    console.print(
        "  `REAL_API_BASE=... REAL_API_KEY=... REAL_MODEL=... python3 -m pytest -q tests/integration/test_real_protocols.py`"
    )
    console.print(
        "[yellow]Reason:[/yellow] this script currently only loads task files and would otherwise report a false success."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
