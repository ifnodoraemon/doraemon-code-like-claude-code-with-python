from src.core.agent_loop import AgentLoopController, AgentPolicyEngine
from src.core.model_utils import ChatResponse


def test_controller_tracks_verification_and_prompt_context():
    controller = AgentLoopController.create(project="demo", mode="build")
    controller.begin_turn("optimize performance")
    controller.record_model_response(ChatResponse(content="I will inspect the hot path."))
    controller.record_tool_plan(
        [("read", {"path": "src/app.py"}, "call_1"), ("search", {"pattern": "slow"}, "call_2")],
        ChatResponse(tool_calls=[{"id": "call_1"}, {"id": "call_2"}]),
    )

    controller.record_tool_outcome(
        tool_name="write",
        args={"path": "src/app.py"},
        result_text="updated file",
        modified_paths=["src/app.py"],
    )

    assert "src/app.py" in controller.state.files_modified

    controller.record_tool_outcome(
        tool_name="run",
        args={"command": "pytest -q"},
        result_text="tests passed",
        modified_paths=[],
    )

    assert controller.state.verification_performed is True
    prompt = controller.compose_system_prompt("Base prompt")
    assert "Verification: done" in prompt
    assert "Tool iterations: 1" in prompt
    assert "Recent tools:" in prompt
    assert "Recent failures:" in prompt


def test_controller_marks_repeated_tool_plan_as_stuck():
    controller = AgentLoopController.create(project="demo", mode="build")
    controller.begin_turn("debug flaky test")
    repeated_calls = [("read", {"path": "src/app.py"}, "call_1")]

    for _ in range(3):
        controller.record_tool_plan(repeated_calls, ChatResponse(tool_calls=[{"id": "call_1"}]))

    assert controller.state.is_stuck is True
    prompt = controller.compose_system_prompt("Base prompt")
    assert "Stuck detector: triggered" in prompt
    assert "Tool iterations: 3" in prompt
    assert "Recent tools: read, read, read" in prompt


def test_policy_marks_loop_stuck_after_repeated_failures():
    policy = AgentPolicyEngine()
    controller = AgentLoopController.create(project="demo", mode="build")
    controller.begin_turn("fix flaky command")
    controller.state.tool_iterations = 4
    controller.state.recent_failures = ["failed once", "failed twice"]

    policy.update_stuck_state(controller.state, [("run", {"command": "pytest -q"}, "call_1")])

    assert controller.state.is_stuck is True
