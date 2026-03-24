"""
Streaming Response System

Real-time streaming of model responses.

Features:
- Token-by-token streaming
- Progress indicators
- Chunk buffering
- Stream interruption
"""

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StreamState(Enum):
    """Stream state."""

    IDLE = "idle"
    STREAMING = "streaming"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class StreamChunk:
    """A chunk from the stream."""

    text: str
    index: int
    timestamp: float = field(default_factory=time.time)
    is_final: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamStats:
    """Statistics about the stream."""

    total_chunks: int = 0
    total_chars: int = 0
    total_tokens: int = 0  # Estimated
    start_time: float = 0
    end_time: float = 0
    first_chunk_time: float = 0

    @property
    def duration(self) -> float:
        """Total duration in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0

    @property
    def time_to_first_chunk(self) -> float:
        """Time to first chunk in seconds."""
        if self.first_chunk_time and self.start_time:
            return self.first_chunk_time - self.start_time
        return 0

    @property
    def chars_per_second(self) -> float:
        """Characters per second."""
        duration = self.duration
        if duration > 0:
            return self.total_chars / duration
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_chunks": self.total_chunks,
            "total_chars": self.total_chars,
            "total_tokens": self.total_tokens,
            "duration": self.duration,
            "time_to_first_chunk": self.time_to_first_chunk,
            "chars_per_second": self.chars_per_second,
        }


class StreamBuffer:
    """Buffer for accumulating stream chunks."""

    def __init__(self, flush_threshold: int = 10):
        """
        Initialize buffer.

        Args:
            flush_threshold: Number of chunks before auto-flush
        """
        self._chunks: list[StreamChunk] = []
        self._flush_threshold = flush_threshold
        self._total_text = ""

    def add(self, chunk: StreamChunk):
        """Add a chunk to the buffer."""
        self._chunks.append(chunk)
        self._total_text += chunk.text

    def should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        return len(self._chunks) >= self._flush_threshold

    def flush(self) -> str:
        """Flush buffer and return accumulated text."""
        text = "".join(c.text for c in self._chunks)
        self._chunks.clear()
        return text

    def get_total(self) -> str:
        """Get total accumulated text."""
        return self._total_text

    def clear(self):
        """Clear the buffer."""
        self._chunks.clear()
        self._total_text = ""


class StreamManager:
    """
    Manages streaming responses from the model.

    Usage:
        stream_mgr = StreamManager()

        # Stream with callback
        async for chunk in stream_mgr.stream(response_iterator):
            print(chunk.text, end="", flush=True)

        # Get stats
        stats = stream_mgr.get_stats()
    """

    def __init__(
        self,
        on_chunk: Callable[[StreamChunk], None] | None = None,
        buffer_size: int = 10,
    ):
        """
        Initialize stream manager.

        Args:
            on_chunk: Callback for each chunk
            buffer_size: Buffer size for batching
        """
        self._on_chunk = on_chunk
        self._buffer = StreamBuffer(buffer_size)
        self._state = StreamState.IDLE
        self._stats = StreamStats()
        self._cancel_event = asyncio.Event()

    @property
    def state(self) -> StreamState:
        """Get current stream state."""
        return self._state

    @property
    def stats(self) -> StreamStats:
        """Get stream statistics."""
        return self._stats

    def cancel(self):
        """Cancel the current stream."""
        self._cancel_event.set()
        self._state = StreamState.CANCELLED

    def reset(self):
        """Reset for a new stream."""
        self._buffer.clear()
        self._state = StreamState.IDLE
        self._stats = StreamStats()
        self._cancel_event.clear()

    async def stream(self, response_iterator: AsyncIterator[Any]) -> AsyncIterator[StreamChunk]:
        """
        Stream responses from an async iterator.

        Args:
            response_iterator: Async iterator yielding response chunks

        Yields:
            StreamChunk objects
        """
        self.reset()
        self._state = StreamState.STREAMING
        self._stats.start_time = time.time()
        chunk_index = 0

        try:
            async for response in response_iterator:
                # Check for cancellation
                if self._cancel_event.is_set():
                    self._state = StreamState.CANCELLED
                    break

                # Extract text from response
                text = self._extract_text(response)
                if not text:
                    continue

                # Record first chunk time
                if chunk_index == 0:
                    self._stats.first_chunk_time = time.time()

                # Create chunk
                chunk = StreamChunk(
                    text=text,
                    index=chunk_index,
                    is_final=False,
                )

                # Update stats
                self._stats.total_chunks += 1
                self._stats.total_chars += len(text)
                self._stats.total_tokens += len(text) // 4  # Rough estimate

                # Buffer and callback
                self._buffer.add(chunk)
                if self._on_chunk:
                    self._on_chunk(chunk)

                yield chunk
                chunk_index += 1

            # Final chunk
            self._state = StreamState.COMPLETED
            self._stats.end_time = time.time()

        except Exception as e:
            self._state = StreamState.ERROR
            self._stats.end_time = time.time()
            logger.error(f"Stream error: {e}")
            raise

    def _extract_text(self, response: Any) -> str:
        """Extract text from various response formats."""
        # Google GenAI format
        if hasattr(response, "text"):
            return response.text

        # Dict format
        if isinstance(response, dict):
            return response.get("text", response.get("content", ""))

        # String format
        if isinstance(response, str):
            return response

        return ""

    def get_accumulated_text(self) -> str:
        """Get all accumulated text."""
        return self._buffer.get_total()

    def get_stats(self) -> dict[str, Any]:
        """Get streaming statistics."""
        return self._stats.to_dict()


class StreamingChat:
    """
    Wrapper for streaming chat with Gemini.

    Usage:
        chat = StreamingChat(client, model)

        async for chunk in chat.send_message_streaming("Hello"):
            print(chunk, end="", flush=True)
    """

    def __init__(self, client, model: str, config: dict | None = None):
        """
        Initialize streaming chat.

        Args:
            client: Google GenAI client
            model: Model name
            config: Generation config
        """
        self._client = client
        self._model = model
        self._config = config or {}
        self._history: list[dict] = []
        self._stream_manager = StreamManager()

    async def send_message_streaming(
        self,
        message: str,
        on_chunk: Callable[[str], None] | None = None,
    ) -> AsyncIterator[str]:
        """
        Send a message and stream the response.

        Args:
            message: User message
            on_chunk: Optional callback for each text chunk

        Yields:
            Text chunks
        """
        # Add user message to history
        self._history.append({"role": "user", "parts": [{"text": message}]})

        try:
            # Create streaming response
            response = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=self._history,
                config=self._config,
            )

            full_response = ""

            async for chunk in response:
                if hasattr(chunk, "text") and chunk.text:
                    text = chunk.text
                    full_response += text

                    if on_chunk:
                        on_chunk(text)

                    yield text

            # Add assistant response to history
            self._history.append({"role": "model", "parts": [{"text": full_response}]})

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise

    def get_history(self) -> list[dict]:
        """Get conversation history."""
        return self._history.copy()

    def clear_history(self):
        """Clear conversation history."""
        self._history.clear()


def create_streaming_printer(console) -> Callable[[str], None]:
    """
    Create a streaming printer for Rich console.

    Args:
        console: Rich console instance

    Returns:
        Callback function for printing chunks
    """

    def printer(text: str):
        console.print(text, end="", highlight=False)

    return printer


async def stream_to_string(iterator: AsyncIterator[str]) -> str:
    """
    Collect all chunks from a stream into a string.

    Args:
        iterator: Async iterator of text chunks

    Returns:
        Complete text
    """
    chunks = []
    async for chunk in iterator:
        chunks.append(chunk)
    return "".join(chunks)
