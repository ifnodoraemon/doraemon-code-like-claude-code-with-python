"""Tests for src.core.streaming."""

import asyncio
import time

import pytest

from src.core.streaming import (
    StreamBuffer,
    StreamChunk,
    StreamManager,
    StreamState,
    StreamStats,
    StreamingChat,
    create_streaming_printer,
    stream_to_string,
)


class TestStreamState:
    def test_values(self):
        assert StreamState.IDLE.value == "idle"
        assert StreamState.STREAMING.value == "streaming"
        assert StreamState.PAUSED.value == "paused"
        assert StreamState.COMPLETED.value == "completed"
        assert StreamState.ERROR.value == "error"
        assert StreamState.CANCELLED.value == "cancelled"


class TestStreamChunk:
    def test_defaults(self):
        chunk = StreamChunk(text="hello", index=0)
        assert chunk.text == "hello"
        assert chunk.index == 0
        assert chunk.is_final is False
        assert chunk.metadata == {}

    def test_with_metadata(self):
        chunk = StreamChunk(text="x", index=5, is_final=True, metadata={"k": "v"})
        assert chunk.index == 5
        assert chunk.is_final is True
        assert chunk.metadata == {"k": "v"}


class TestStreamStats:
    def test_duration_no_times(self):
        stats = StreamStats()
        assert stats.duration == 0

    def test_duration_with_times(self):
        stats = StreamStats(start_time=100.0, end_time=105.0)
        assert stats.duration == 5.0

    def test_time_to_first_chunk_no_times(self):
        stats = StreamStats()
        assert stats.time_to_first_chunk == 0

    def test_time_to_first_chunk_with_times(self):
        stats = StreamStats(start_time=100.0, first_chunk_time=102.5)
        assert stats.time_to_first_chunk == 2.5

    def test_chars_per_second(self):
        stats = StreamStats(total_chars=100, start_time=1.0, end_time=11.0)
        assert stats.chars_per_second == 10.0

    def test_chars_per_second_zero_duration(self):
        stats = StreamStats(total_chars=100)
        assert stats.chars_per_second == 0.0

    def test_to_dict(self):
        stats = StreamStats(
            total_chunks=5,
            total_chars=50,
            total_tokens=12,
            start_time=1.0,
            end_time=11.0,
            first_chunk_time=2.0,
        )
        d = stats.to_dict()
        assert d["total_chunks"] == 5
        assert d["total_chars"] == 50
        assert d["total_tokens"] == 12
        assert d["duration"] == 10.0
        assert d["time_to_first_chunk"] == 1.0
        assert d["chars_per_second"] == 5.0


class TestStreamBuffer:
    def test_add_and_flush(self):
        buf = StreamBuffer(flush_threshold=3)
        buf.add(StreamChunk(text="a", index=0))
        buf.add(StreamChunk(text="b", index=1))
        text = buf.flush()
        assert text == "ab"

    def test_should_flush(self):
        buf = StreamBuffer(flush_threshold=2)
        buf.add(StreamChunk(text="a", index=0))
        assert not buf.should_flush()
        buf.add(StreamChunk(text="b", index=1))
        assert buf.should_flush()

    def test_get_total(self):
        buf = StreamBuffer()
        buf.add(StreamChunk(text="hello", index=0))
        buf.add(StreamChunk(text=" world", index=1))
        assert buf.get_total() == "hello world"

    def test_get_total_after_flush(self):
        buf = StreamBuffer()
        buf.add(StreamChunk(text="hi", index=0))
        buf.flush()
        assert buf.get_total() == "hi"

    def test_clear(self):
        buf = StreamBuffer()
        buf.add(StreamChunk(text="x", index=0))
        buf.clear()
        assert buf.get_total() == ""


class TestStreamManager:
    def test_initial_state(self):
        mgr = StreamManager()
        assert mgr.state == StreamState.IDLE

    def test_cancel(self):
        mgr = StreamManager()
        mgr.cancel()
        assert mgr.state == StreamState.CANCELLED

    def test_reset(self):
        mgr = StreamManager()
        mgr.cancel()
        mgr.reset()
        assert mgr.state == StreamState.IDLE
        assert mgr.stats.total_chunks == 0

    @pytest.mark.asyncio
    async def test_stream_with_strings(self):
        async def gen():
            for s in ["hello", " ", "world"]:
                yield s

        mgr = StreamManager()
        chunks = []
        async for chunk in mgr.stream(gen()):
            chunks.append(chunk)

        assert mgr.state == StreamState.COMPLETED
        assert len(chunks) == 3
        assert chunks[0].text == "hello"
        assert chunks[1].text == " "
        assert chunks[2].text == "world"

    @pytest.mark.asyncio
    async def test_stream_with_dicts(self):
        async def gen():
            yield {"text": "a"}
            yield {"content": "b"}

        mgr = StreamManager()
        chunks = []
        async for chunk in mgr.stream(gen()):
            chunks.append(chunk)

        assert len(chunks) == 2
        assert chunks[0].text == "a"
        assert chunks[1].text == "b"

    @pytest.mark.asyncio
    async def test_stream_with_objects(self):
        class Resp:
            text = "obj"

        async def gen():
            yield Resp()

        mgr = StreamManager()
        chunks = []
        async for chunk in mgr.stream(gen()):
            chunks.append(chunk)

        assert chunks[0].text == "obj"

    @pytest.mark.asyncio
    async def test_stream_skips_empty(self):
        async def gen():
            yield "hello"
            yield ""
            yield "world"

        mgr = StreamManager()
        chunks = []
        async for chunk in mgr.stream(gen()):
            chunks.append(chunk)

        assert len(chunks) == 2

    @pytest.mark.asyncio
    async def test_stream_cancel(self):
        async def gen():
            for i in range(100):
                yield f"chunk{i}"
                await asyncio.sleep(0.01)

        mgr = StreamManager()
        chunks = []
        async for chunk in mgr.stream(gen()):
            chunks.append(chunk)
            if len(chunks) == 2:
                mgr.cancel()

        assert mgr.state in (StreamState.CANCELLED, StreamState.COMPLETED)
        assert len(chunks) <= 4

    @pytest.mark.asyncio
    async def test_stream_error(self):
        async def gen():
            yield "ok"
            raise ValueError("boom")

        mgr = StreamManager()
        with pytest.raises(ValueError, match="boom"):
            async for _ in mgr.stream(gen()):
                pass
        assert mgr.state == StreamState.ERROR

    @pytest.mark.asyncio
    async def test_stream_callback(self):
        async def gen():
            yield "a"
            yield "b"

        received = []
        mgr = StreamManager(on_chunk=lambda c: received.append(c.text))
        async for _ in mgr.stream(gen()):
            pass
        assert received == ["a", "b"]

    @pytest.mark.asyncio
    async def test_get_stats(self):
        async def gen():
            yield "hello"

        mgr = StreamManager()
        async for _ in mgr.stream(gen()):
            pass
        stats = mgr.get_stats()
        assert stats["total_chunks"] == 1
        assert stats["total_chars"] == 5

    @pytest.mark.asyncio
    async def test_get_accumulated_text(self):
        async def gen():
            yield "abc"
            yield "def"

        mgr = StreamManager()
        async for _ in mgr.stream(gen()):
            pass
        assert mgr.get_accumulated_text() == "abcdef"


class TestStreamingChat:
    @pytest.mark.asyncio
    async def test_send_message_streaming(self):
        class MockResponse:
            text = "reply"

        class MockClient:
            class aio:
                @staticmethod
                class models:
                    @staticmethod
                    async def generate_content_stream(**kwargs):
                        async def _gen():
                            yield MockResponse()

                        return _gen()

        chat = StreamingChat(MockClient(), "model-1")
        chunks = []
        async for text in chat.send_message_streaming("hello"):
            chunks.append(text)
        assert chunks == ["reply"]
        assert len(chat.get_history()) == 2

    def test_clear_history(self):
        chat = StreamingChat(type("C", (), {})(), "m")
        chat._history = [{"role": "user", "parts": []}]
        chat.clear_history()
        assert chat.get_history() == []


def test_create_streaming_printer():
    class FakeConsole:
        printed = []

        def print(self, text, **kwargs):
            self.printed.append(text)

    console = FakeConsole()
    printer = create_streaming_printer(console)
    printer("hello")
    assert console.printed == ["hello"]


@pytest.mark.asyncio
async def test_stream_to_string():
    async def gen():
        for s in ["a", "b", "c"]:
            yield s

    result = await stream_to_string(gen())
    assert result == "abc"
