"""Lightweight agent loop policy and state tracking.

This module keeps the runtime "more agentic" without forcing a rigid workflow.
The model remains free to choose the next action, while the host tracks the
current situation and injects compact guidance into the prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

from src.core.model_utils import ChatResponse


class AgentPhase(str, Enum):
    """High-level intent, not a hard workflow state."""

    GATHERING = "gathering"
    EDITING = "editing"
    VERIFYING = "verifying"
    FINISHING = "finishing"


class RecommendedShift(str, Enum):
    """Preferred next shift when the loop needs stronger guidance."""

    CONTINUE = "continue"
    READ_NEW_SURFACE = "read_new_surface"
    VERIFY_NOW = "verify_now"
    SUMMARIZE_BLOCKER = "summarize_blocker"


VALIDATION_MARKERS = (
    "pytest",
    "ruff",
    "mypy",
    "eslint",
    "vitest",
    "jest",
    "npm test",
    "npm run test",
    "npm run build",
    "pnpm test",
    "pnpm lint",
    "pnpm build",
    "yarn test",
    "yarn lint",
    "cargo test",
    "go test",
    "tox",
)

READ_LIKE_TOOLS = {
    "read",
    "search",
    "semantic_search",
    "web_search",
    "browse_page",
    "take_screenshot",
    "fetch_url",
    "db_read_query",
    "db_list_tables",
    "db_describe_table",
    "search_notes",
    "ask_user",
}

WRITE_LIKE_TOOLS = {
    "write",
    "run",
    "db_write_query",
    "github_create_issue",
    "save_note",
}


def is_validation_action(tool_name: str, args: dict) -> bool:
    """Return whether a tool call is performing verification work."""
    if tool_name in {
        "lint",
        "lint_python_ruff",
        "typecheck_python_mypy",
        "lint_javascript_eslint",
        "lint_all",
        "get_lint_summary",
        "lsp_diagnostics",
        "check_security",
        "code_complexity",
    }:
        return True

    if tool_name != "run":
        return False

    command = str(args.get("command") or "").lower()
    return any(marker in command for marker in VALIDATION_MARKERS)


@dataclass
class AgentState:
    """Runtime state for a single interactive session."""

    project: str
    mode: str
    current_goal: str = ""
    phase: AgentPhase = AgentPhase.GATHERING
    turn_count: int = 0
    tool_iterations: int = 0
    files_modified: set[str] = field(default_factory=set)
    verification_performed: bool = False
    verification_nudged: bool = False
    last_tool_names: list[str] = field(default_factory=list)
    recent_tool_signatures: list[str] = field(default_factory=list)
    recent_failures: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    recommended_shift: RecommendedShift = RecommendedShift.CONTINUE
    strategy_hint: str = ""
    parallel_opportunities: list[str] = field(default_factory=list)
    speculative_batches: list[str] = field(default_factory=list)
    is_stuck: bool = False

    def start_turn(self, goal: str) -> None:
        """Reset per-turn state while keeping session-level counters."""
        self.turn_count += 1
        self.current_goal = goal.strip()
        self.phase = AgentPhase.GATHERING
        self.tool_iterations = 0
        self.files_modified.clear()
        self.verification_performed = False
        self.verification_nudged = False
        self.last_tool_names = []
        self.recent_tool_signatures = []
        self.recent_failures = []
        self.evidence = []
        self.recommended_shift = RecommendedShift.CONTINUE
        self.strategy_hint = ""
        self.parallel_opportunities = []
        self.speculative_batches = []
        self.is_stuck = False


class AgentPolicyEngine:
    """Compact policy layer to bias the loop without hard-coding a workflow."""

    MAX_RECENT_SIGNATURES = 6

    def _build_tool_signature(self, pending_calls: list[tuple[str, dict, str]]) -> str:
        normalized = [
            {"name": tool_name, "args": args}
            for tool_name, args, _ in pending_calls
        ]
        return json.dumps(normalized, sort_keys=True, ensure_ascii=True)

    def determine_phase(
        self,
        pending_tool_names: list[str],
        files_modified: set[str],
        verification_performed: bool,
        response_has_tool_calls: bool,
    ) -> AgentPhase:
        """Infer the current intent from pending work and recent outcomes."""
        if any(name in WRITE_LIKE_TOOLS for name in pending_tool_names):
            if all(name == "run" for name in pending_tool_names):
                return AgentPhase.VERIFYING
            return AgentPhase.EDITING

        if pending_tool_names:
            if all(name in READ_LIKE_TOOLS for name in pending_tool_names):
                return AgentPhase.GATHERING
            if any(name == "run" for name in pending_tool_names):
                return AgentPhase.VERIFYING

        if files_modified and not verification_performed:
            return AgentPhase.VERIFYING

        if not response_has_tool_calls:
            return AgentPhase.FINISHING

        return AgentPhase.GATHERING

    def should_nudge_verification(self, state: AgentState) -> bool:
        """Ask the model to verify only when edits happened and verification did not."""
        return (
            state.recommended_shift == RecommendedShift.VERIFY_NOW
            and not state.verification_nudged
        )

    def recommend_shift(self, state: AgentState) -> RecommendedShift:
        """Choose the strongest useful strategy shift for the current state."""
        if state.is_stuck:
            if state.recent_failures:
                return RecommendedShift.SUMMARIZE_BLOCKER
            return RecommendedShift.READ_NEW_SURFACE

        if state.files_modified and not state.verification_performed:
            return RecommendedShift.VERIFY_NOW

        return RecommendedShift.CONTINUE

    def evaluate_strategy_hint(self, state: AgentState) -> str:
        """Suggest a strategy shift when the loop is stuck or can batch work."""
        if state.recommended_shift == RecommendedShift.SUMMARIZE_BLOCKER:
            return (
                "You appear to be blocked. Stop repeating the same reads or failed actions. "
                "Summarize the blocker, identify the missing evidence, then change strategy."
            )

        if state.recommended_shift == RecommendedShift.READ_NEW_SURFACE:
            return (
                "You are not making progress with the current surface. Read one different file, "
                "run one targeted search, or perform one validating check before deciding the next move."
            )

        if state.recommended_shift == RecommendedShift.VERIFY_NOW:
            return (
                "You have already changed files. Verify now with the highest-signal checks before doing more work."
            )

        if state.phase == AgentPhase.GATHERING:
            return (
                "If multiple reads or searches are independent, batch them into parallel tool calls "
                "instead of issuing them one by one."
            )

        if state.phase == AgentPhase.VERIFYING:
            return (
                "Prefer the highest-signal verification first. If several checks are independent and cheap, "
                "batch them in parallel."
            )

        return ""

    def suggest_parallel_opportunities(self, state: AgentState) -> list[str]:
        """Suggest high-value batching opportunities without hard-coding a workflow."""
        suggestions: list[str] = []

        if state.phase == AgentPhase.GATHERING:
            suggestions.append(
                "Batch independent file reads and searches in one step when you need multiple facts before acting."
            )
            if state.current_goal:
                suggestions.append(
                    "If the task spans multiple modules, read the entrypoint, the likely implementation file, "
                    "and a targeted search result in parallel."
                )

        if state.phase == AgentPhase.VERIFYING:
            suggestions.append(
                "Batch independent cheap checks first, such as lint plus targeted tests, instead of running them serially."
            )
            if state.files_modified:
                suggestions.append(
                    "Prioritize verification that covers modified files, and parallelize checks that do not depend on each other."
                )

        if state.is_stuck:
            suggestions.append(
                "Change the shape of the batch: switch from repeated reads to one read plus one search plus one verification command."
            )

        return suggestions[:2]

    def suggest_speculative_batches(self, state: AgentState) -> list[str]:
        """Suggest structured batch shapes the model can adopt directly."""
        batches: list[str] = []

        if state.phase == AgentPhase.GATHERING:
            batches.append(
                "batch_1 = [read(target file), search(symbol or error), read(adjacent implementation)]"
            )
            if state.current_goal:
                batches.append(
                    "batch_2 = [read(entrypoint), read(likely hot file), search(goal keyword)]"
                )

        if state.phase == AgentPhase.VERIFYING:
            batches.append(
                "batch_1 = [run(lint or typecheck), run(targeted test or build)]"
            )
            if state.files_modified:
                batches.append(
                    "batch_2 = [run(check covering modified files), run(second independent lightweight check)]"
                )

        if state.is_stuck:
            batches.insert(
                0,
                "recovery_batch = [read(one new file), search(one targeted pattern), run(one validating command)]"
            )

        return batches[:2]

    def update_stuck_state(self, state: AgentState, pending_calls: list[tuple[str, dict, str]]) -> None:
        """Detect repeated, low-progress behavior and mark the loop as stuck."""
        if not pending_calls:
            state.is_stuck = False
            self.refresh_guidance(state)
            return

        signature = self._build_tool_signature(pending_calls)
        state.recent_tool_signatures.append(signature)
        if len(state.recent_tool_signatures) > self.MAX_RECENT_SIGNATURES:
            state.recent_tool_signatures = state.recent_tool_signatures[-self.MAX_RECENT_SIGNATURES:]

        recent = state.recent_tool_signatures[-3:]
        repeated_plan = len(recent) == 3 and len(set(recent)) == 1
        too_many_tool_iterations = state.tool_iterations >= 4 and not state.files_modified
        repeated_failures = len(state.recent_failures) >= 2

        state.is_stuck = repeated_plan or too_many_tool_iterations or repeated_failures
        self.refresh_guidance(state)

    def refresh_guidance(self, state: AgentState) -> None:
        """Recompute guidance derived from the current state."""
        state.recommended_shift = self.recommend_shift(state)
        state.strategy_hint = self.evaluate_strategy_hint(state)
        state.parallel_opportunities = self.suggest_parallel_opportunities(state)
        state.speculative_batches = self.suggest_speculative_batches(state)

    def build_prompt_suffix(self, state: AgentState) -> str:
        """Build a small execution-context block for the current turn."""
        modified = ", ".join(sorted(state.files_modified)) if state.files_modified else "none"
        recent_tools = ", ".join(state.last_tool_names[-5:]) if state.last_tool_names else "none"
        failures = " | ".join(state.recent_failures[-2:]) if state.recent_failures else "none"
        strategy_hint = state.strategy_hint or "none"
        parallel_suggestions = (
            "\n".join(f"- {item}" for item in state.parallel_opportunities)
            if state.parallel_opportunities
            else "- none"
        )
        speculative_batches = (
            "\n".join(f"- {item}" for item in state.speculative_batches)
            if state.speculative_batches
            else "- none"
        )

        return (
            "\n\n=== EXECUTION CONTEXT ===\n\n"
            f"Project: {state.project}\n"
            f"Mode: {state.mode}\n"
            f"Current goal: {state.current_goal or 'unspecified'}\n"
            f"Current phase: {state.phase.value}\n"
            f"Modified files this turn: {modified}\n"
            f"Verification: {'done' if state.verification_performed else 'pending'}\n"
            f"Stuck detector: {'triggered' if state.is_stuck else 'clear'}\n"
            f"Recommended shift: {state.recommended_shift.value}\n"
            f"Recent tools: {recent_tools}\n"
            f"Recent failures: {failures}\n\n"
            f"Strategy hint: {strategy_hint}\n\n"
            f"Parallel opportunities:\n{parallel_suggestions}\n\n"
            f"Speculative batches:\n{speculative_batches}\n\n"
            "Guidance:\n"
            "- Choose the next best action; do not follow a rigid workflow.\n"
            "- Gather only enough context to act safely.\n"
            "- After edits, verify before finalizing when feasible.\n"
            "- Batch independent reads, searches, and checks into parallel tool calls when useful.\n"
            "- Reuse a speculative batch shape when it matches the task; adjust arguments to the real files or commands.\n"
            "- If you are repeating yourself or blocked, summarize the blocker and change strategy.\n"
        )


@dataclass
class AgentLoopController:
    """Session-level controller combining state tracking and policy decisions."""

    state: AgentState
    policy: AgentPolicyEngine = field(default_factory=AgentPolicyEngine)

    @classmethod
    def create(cls, project: str, mode: str) -> AgentLoopController:
        return cls(state=AgentState(project=project, mode=mode))

    def update_mode(self, mode: str) -> None:
        self.state.mode = mode

    def begin_turn(self, goal: str) -> None:
        self.state.start_turn(goal)

    def record_model_response(self, response: ChatResponse) -> None:
        if response.content:
            preview = response.content.strip().replace("\n", " ")[:160]
            if preview:
                self.state.evidence.append(preview)

    def record_tool_plan(self, pending_calls: list[tuple[str, dict, str]], response: ChatResponse) -> None:
        tool_names = [tool_name for tool_name, _, _ in pending_calls]
        self.state.last_tool_names.extend(tool_names)
        self.state.tool_iterations += 1
        self.state.phase = self.policy.determine_phase(
            pending_tool_names=tool_names,
            files_modified=self.state.files_modified,
            verification_performed=self.state.verification_performed,
            response_has_tool_calls=bool(response.tool_calls),
        )
        self.policy.update_stuck_state(self.state, pending_calls)

    def record_tool_outcome(
        self,
        tool_name: str,
        args: dict,
        result_text: str,
        modified_paths: list[str] | None = None,
    ) -> None:
        if modified_paths:
            self.state.files_modified.update(modified_paths)
        if is_validation_action(tool_name, args):
            self.state.verification_performed = True
            self.state.phase = AgentPhase.FINISHING
        elif self.state.files_modified:
            self.state.phase = AgentPhase.EDITING

        lowered = result_text.lower()
        if "error" in lowered or "failed" in lowered or "denied" in lowered:
            self.state.recent_failures.append(result_text[:200])
        self.policy.refresh_guidance(self.state)

    def finalize_response(self, response: ChatResponse) -> None:
        self.state.phase = self.policy.determine_phase(
            pending_tool_names=[],
            files_modified=self.state.files_modified,
            verification_performed=self.state.verification_performed,
            response_has_tool_calls=bool(response.tool_calls),
        )
        self.policy.refresh_guidance(self.state)

    def should_nudge_verification(self) -> bool:
        return self.policy.should_nudge_verification(self.state)

    def mark_verification_nudged(self) -> None:
        self.state.verification_nudged = True
        self.state.phase = AgentPhase.VERIFYING
        self.policy.refresh_guidance(self.state)

    def compose_system_prompt(self, base_prompt: str) -> str:
        return f"{base_prompt}{self.policy.build_prompt_suffix(self.state)}"
