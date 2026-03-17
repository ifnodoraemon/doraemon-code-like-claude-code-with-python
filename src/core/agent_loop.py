"""Lightweight agent loop state tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.core.model_utils import ChatResponse

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
    turn_count: int = 0
    tool_iterations: int = 0
    files_modified: set[str] = field(default_factory=set)
    verification_performed: bool = False
    last_tool_names: list[str] = field(default_factory=list)
    recent_tool_signatures: list[str] = field(default_factory=list)
    recent_failures: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    is_stuck: bool = False

    def start_turn(self, goal: str) -> None:
        """Reset per-turn state while keeping session-level counters."""
        self.turn_count += 1
        self.current_goal = goal.strip()
        self.tool_iterations = 0
        self.files_modified.clear()
        self.verification_performed = False
        self.last_tool_names = []
        self.recent_tool_signatures = []
        self.recent_failures = []
        self.evidence = []
        self.is_stuck = False


class AgentPolicyEngine:
    """State inference without workflow-level nudging."""

    MAX_RECENT_SIGNATURES = 6

    def _build_tool_signature(self, pending_calls: list[tuple[str, dict, str]]) -> str:
        normalized = [{"name": tool_name, "args": args} for tool_name, args, _ in pending_calls]
        return json.dumps(normalized, sort_keys=True, ensure_ascii=True)

    def update_stuck_state(self, state: AgentState, pending_calls: list[tuple[str, dict, str]]) -> None:
        """Detect repeated, low-progress behavior and mark the loop as stuck."""
        if not pending_calls:
            state.is_stuck = False
            return

        signature = self._build_tool_signature(pending_calls)
        state.recent_tool_signatures.append(signature)
        if len(state.recent_tool_signatures) > self.MAX_RECENT_SIGNATURES:
            state.recent_tool_signatures = state.recent_tool_signatures[-self.MAX_RECENT_SIGNATURES :]

        recent = state.recent_tool_signatures[-3:]
        repeated_plan = len(recent) == 3 and len(set(recent)) == 1
        too_many_tool_iterations = state.tool_iterations >= 4 and not state.files_modified
        repeated_failures = len(state.recent_failures) >= 2
        state.is_stuck = repeated_plan or too_many_tool_iterations or repeated_failures

    def build_prompt_suffix(self, state: AgentState) -> str:
        """Build a compact execution-state block for the current turn."""
        modified = ", ".join(sorted(state.files_modified)) if state.files_modified else "none"
        recent_tools = ", ".join(state.last_tool_names[-5:]) if state.last_tool_names else "none"
        failures = " | ".join(state.recent_failures[-2:]) if state.recent_failures else "none"

        return (
            "\n\n=== EXECUTION CONTEXT ===\n\n"
            f"Project: {state.project}\n"
            f"Mode: {state.mode}\n"
            f"Current goal: {state.current_goal or 'unspecified'}\n"
            f"Modified files this turn: {modified}\n"
            f"Verification: {'done' if state.verification_performed else 'pending'}\n"
            f"Stuck detector: {'triggered' if state.is_stuck else 'clear'}\n"
            f"Tool iterations: {state.tool_iterations}\n"
            f"Recent tools: {recent_tools}\n"
            f"Recent failures: {failures}\n"
        )


@dataclass
class AgentLoopController:
    """Session-level controller combining state tracking and stuck detection."""

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
        del response
        tool_names = [tool_name for tool_name, _, _ in pending_calls]
        self.state.last_tool_names.extend(tool_names)
        self.state.tool_iterations += 1
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

        lowered = result_text.lower()
        if "error" in lowered or "failed" in lowered or "denied" in lowered:
            self.state.recent_failures.append(result_text[:200])

    def finalize_response(self, response: ChatResponse) -> None:
        del response

    def compose_system_prompt(self, base_prompt: str) -> str:
        return f"{base_prompt}{self.policy.build_prompt_suffix(self.state)}"
