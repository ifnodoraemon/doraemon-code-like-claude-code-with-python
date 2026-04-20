from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.webui.routes.chat import ChatRequest, chat_endpoint


class TestChatRequest:
    def test_defaults(self):
        req = ChatRequest(message="hello")
        assert req.project == "default"
        assert req.execution_mode == "turn"
        assert req.model is None

    def test_validated_message(self):
        req = ChatRequest(message="test msg")
        assert req.validated_message == "test msg"

    def test_custom_values(self):
        req = ChatRequest(
            message="hi",
            session_id="sess-1",
            project="myproj",
            model="gemini-3-pro",
        )
        assert req.session_id == "sess-1"
        assert req.model == "gemini-3-pro"


class TestChatEndpoint:
    @pytest.mark.asyncio
    async def test_invalid_session_id(self):
        req = ChatRequest(message="hi", session_id="../../etc/passwd")
        with pytest.raises(HTTPException) as exc_info:
            await chat_endpoint(req)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_invalid_resume_run_id(self):
        req = ChatRequest(message="hi", resume_run_id="../../bad")
        with pytest.raises(HTTPException) as exc_info:
            await chat_endpoint(req)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_valid_session_id_format(self, monkeypatch):
        class StubSession:
            session_id = "valid-id"

            async def turn_stream(self, msg):
                yield {"type": "response", "content": "ok"}
                yield {"type": "done"}

            async def aclose(self):
                pass

            def get_orchestration_state(self):
                return None

        monkeypatch.setattr("src.webui.routes.chat.AgentSession", lambda **kw: StubSession())
        monkeypatch.setattr("src.webui.routes.sessions._ACTIVE_STREAMS", {})

        req = ChatRequest(message="hello", session_id="valid-id")
        result = await chat_endpoint(req)
        assert result.media_type == "text/event-stream"
