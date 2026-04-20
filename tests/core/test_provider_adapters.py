import base64

from src.core.llm.provider_adapters import (
    GoogleAdapter,
    _deserialize_gemini_thought_signature,
    _serialize_gemini_thought_signature,
)


def test_gemini_thought_signature_round_trip_bytes():
    raw = b"\x00\x01signature"

    encoded = _serialize_gemini_thought_signature(raw)

    assert encoded == f"base64:{base64.b64encode(raw).decode('ascii')}"
    assert _deserialize_gemini_thought_signature(encoded) == raw


def test_google_parse_candidate_serializes_thought_signature():
    part = type(
        "Part",
        (),
        {
            "text": None,
            "thought": None,
            "function_call": type(
                "FunctionCall",
                (),
                {"name": "write", "args": {"path": "add.py"}},
            )(),
            "thought_signature": b"\x01sig",
        },
    )()
    candidate = type(
        "Candidate",
        (),
        {"content": type("Content", (), {"parts": [part]})()},
    )()

    _content, _thought, tool_calls = GoogleAdapter.parse_candidate(candidate)

    assert tool_calls is not None
    expected_sig = "base64:" + base64.b64encode(b"\x01sig").decode("ascii")
    assert tool_calls[0]["thought_signature"] == expected_sig


def test_google_convert_messages_maps_tool_result_to_user_function_response():
    class FunctionResponse:
        def __init__(self, name=None, response=None):
            self.name = name
            self.response = response

    class Part:
        def __init__(
            self, text=None, function_call=None, thought_signature=None, function_response=None
        ):
            self.text = text
            self.function_call = function_call
            self.thought_signature = thought_signature
            self.function_response = function_response
            self.thought = None

    class Content:
        def __init__(self, role=None, parts=None):
            self.role = role
            self.parts = parts or []

    types_module = type(
        "TypesModule",
        (),
        {
            "Part": Part,
            "Content": Content,
            "FunctionCall": lambda name=None, args=None: type(
                "FunctionCall", (), {"name": name, "args": args}
            )(),
            "FunctionResponse": FunctionResponse,
        },
    )

    _system, contents = GoogleAdapter.convert_messages(
        [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": "read", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "1", "name": "read", "content": "file body"},
        ],
        None,
        types_module=types_module,
    )

    assert contents[0].role == "model"
    assert contents[1].role == "user"
    assert [part.text for part in contents[1].parts if part.text] == []
    function_responses = [
        part.function_response for part in contents[1].parts if part.function_response
    ]
    assert len(function_responses) == 1
    assert function_responses[0].name == "read"
