from src.core.agent_loop import (
    AgentLoopController,
    AgentPhase,
    AgentPolicyEngine,
    RecommendedShift,
)
from src.core.model_utils import ChatResponse


def test_policy_moves_to_verifying_after_edits_without_checks():
    policy = AgentPolicyEngine()

    phase = policy.determine_phase(
        pending_tool_names=[],
        files_modified={"src/app.py"},
        verification_performed=False,
        response_has_tool_calls=False,
    )

    assert phase == AgentPhase.VERIFYING


def test_controller_tracks_verification_and_prompt_context():
    controller = AgentLoopController.create(project="demo", mode="build")
    controller.begin_turn("optimize performance")
    controller.record_model_response(ChatResponse(content="I will inspect the hot path."))
    controller.record_tool_plan(
        [("read", {"path": "src/app.py"}, "call_1"), ("search", {"pattern": "slow"}, "call_2")],
        ChatResponse(tool_calls=[{"id": "call_1"}, {"id": "call_2"}]),
    )

    assert controller.state.phase == AgentPhase.GATHERING

    controller.record_tool_outcome(
        tool_name="write",
        args={"path": "src/app.py"},
        result_text="updated file",
        modified_paths=["src/app.py"],
    )

    assert "src/app.py" in controller.state.files_modified
    assert controller.should_nudge_verification() is True
    assert controller.state.recommended_shift == RecommendedShift.VERIFY_NOW

    controller.record_tool_outcome(
        tool_name="run",
        args={"command": "pytest -q"},
        result_text="tests passed",
        modified_paths=[],
    )

    assert controller.state.verification_performed is True
    prompt = controller.compose_system_prompt("Base prompt")
    assert "Current phase:" in prompt
    assert "Verification: done" in prompt
    assert "Recommended shift: continue" in prompt
    assert "parallel tool calls" in prompt
    assert "Parallel opportunities:" in prompt
    assert "Speculative batches:" in prompt


def test_controller_marks_repeated_tool_plan_as_stuck():
    controller = AgentLoopController.create(project="demo", mode="build")
    controller.begin_turn("debug flaky test")
    repeated_calls = [("read", {"path": "src/app.py"}, "call_1")]

    for _ in range(3):
        controller.record_tool_plan(repeated_calls, ChatResponse(tool_calls=[{"id": "call_1"}]))

    assert controller.state.is_stuck is True
    assert controller.state.recommended_shift == RecommendedShift.READ_NEW_SURFACE
    prompt = controller.compose_system_prompt("Base prompt")
    assert "Stuck detector: triggered" in prompt
    assert "Recommended shift: read_new_surface" in prompt
    assert "not making progress with the current surface" in prompt
    assert "recovery_batch =" in prompt


def test_controller_suggests_parallel_gathering_and_verification_batches():
    controller = AgentLoopController.create(project="demo", mode="build")
    controller.begin_turn("trace a latency regression")
    controller.record_tool_plan(
        [("read", {"path": "src/app.py"}, "call_1"), ("search", {"pattern": "latency"}, "call_2")],
        ChatResponse(tool_calls=[{"id": "call_1"}, {"id": "call_2"}]),
    )

    gather_prompt = controller.compose_system_prompt("Base prompt")
    assert "Batch independent file reads and searches" in gather_prompt
    assert "batch_1 = [read(target file), search(symbol or error), read(adjacent implementation)]" in gather_prompt

    controller.record_tool_outcome(
        tool_name="write",
        args={"path": "src/app.py"},
        result_text="updated file",
        modified_paths=["src/app.py"],
    )
    controller.mark_verification_nudged()

    verify_prompt = controller.compose_system_prompt("Base prompt")
    assert "Recommended shift: verify_now" in verify_prompt
    assert "Batch independent cheap checks first" in verify_prompt
    assert "batch_1 = [run(lint or typecheck), run(targeted test or build)]" in verify_prompt


def test_recommended_shift_moves_to_summarize_blocker_on_failures():
    controller = AgentLoopController.create(project="demo", mode="build")
    controller.begin_turn("unblock flaky CI")

    repeated_calls = [("run", {"command": "pytest tests/test_ci.py -q"}, "call_1")]
    for _ in range(3):
        controller.record_tool_plan(repeated_calls, ChatResponse(tool_calls=[{"id": "call_1"}]))
        controller.record_tool_outcome(
            tool_name="run",
            args={"command": "pytest tests/test_ci.py -q"},
            result_text="failed: timeout in CI",
            modified_paths=[],
        )

    assert controller.state.is_stuck is True
    assert controller.state.recommended_shift == RecommendedShift.SUMMARIZE_BLOCKER
    prompt = controller.compose_system_prompt("Base prompt")
    assert "Recommended shift: summarize_blocker" in prompt
    assert "Summarize the blocker" in prompt
