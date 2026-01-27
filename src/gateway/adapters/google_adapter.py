"""
Google Gemini Adapter

Translates unified API to Google GenAI format.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ..schema import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    FinishReason,
    ModelInfo,
    Role,
    StreamChunk,
    ToolCall,
    Usage,
)
from .base import BaseAdapter

logger = logging.getLogger(__name__)


# Google Gemini models
GOOGLE_MODELS = [
    ModelInfo(
        id="gemini-2.5-pro-preview",
        name="Gemini 2.5 Pro",
        provider="google",
        description="Google's most capable model",
        context_window=1_000_000,
        max_output=64_000,
        input_price=2.50,
        output_price=15.00,
        capabilities=["text", "vision", "code", "reasoning", "function_calling"],
        aliases=["gemini-2.5-pro", "g25p"],
    ),
    ModelInfo(
        id="gemini-2.5-flash-preview",
        name="Gemini 2.5 Flash",
        provider="google",
        description="Fast and efficient",
        context_window=1_000_000,
        max_output=64_000,
        input_price=0.15,
        output_price=0.60,
        capabilities=["text", "vision", "code", "reasoning", "function_calling"],
        aliases=["gemini-2.5-flash", "g25f", "flash"],
    ),
    ModelInfo(
        id="gemini-2.0-flash",
        name="Gemini 2.0 Flash",
        provider="google",
        description="Previous generation flash model",
        context_window=1_000_000,
        max_output=8_192,
        input_price=0.10,
        output_price=0.40,
        capabilities=["text", "vision", "code", "function_calling"],
        aliases=["gemini-2-flash", "g2f"],
    ),
]


class GoogleAdapter(BaseAdapter):
    """Adapter for Google Gemini models."""

    provider_name = "google"

    async def initialize(self) -> None:
        """Initialize Google GenAI client."""
        try:
            from google import genai

            self._client = genai.Client(api_key=self.config.api_key)
            logger.info("Google adapter initialized")
        except ImportError as e:
            raise ImportError("google-genai package required: pip install google-genai") from e

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Send chat request to Gemini."""

        # Convert messages to Gemini format
        contents, system_instruction = self._convert_messages(request.messages)

        # Build config
        gen_config = self._build_config(request, system_instruction)

        # Make request with error handling
        try:
            response = await self._client.aio.models.generate_content(
                model=request.model,
                contents=contents,
                config=gen_config,
            )
        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise RuntimeError(f"Google API request failed: {e}") from e

        # Convert response
        return self._convert_response(response, request.model)

    async def chat_stream(
        self, request: ChatRequest
    ) -> AsyncIterator[StreamChunk]:
        """Stream chat response from Gemini."""

        contents, system_instruction = self._convert_messages(request.messages)
        gen_config = self._build_config(request, system_instruction)

        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        try:
            response = await self._client.aio.models.generate_content_stream(
                model=request.model,
                contents=contents,
                config=gen_config,
            )
        except Exception as e:
            logger.error(f"Google stream API error: {e}")
            raise RuntimeError(f"Google stream API request failed: {e}") from e

        async for chunk in response:
            text = None
            tool_calls = None

            if hasattr(chunk, "text") and chunk.text:
                text = chunk.text

            if hasattr(chunk, "candidates") and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, "content") and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, "function_call") and part.function_call:
                            fc = part.function_call
                            # Safely convert args to dict
                            try:
                                args = dict(fc.args) if fc.args else {}
                            except (TypeError, ValueError):
                                logger.warning(f"Failed to convert function call args: {fc.args}")
                                args = {}
                            tool_calls = [
                                ToolCall(
                                    id=f"call_{uuid.uuid4().hex[:8]}",
                                    name=fc.name,
                                    arguments=args,
                                )
                            ]

            # Check for finish reason
            finish_reason = None
            if hasattr(chunk, "candidates") and chunk.candidates:
                candidate = chunk.candidates[0]
                if hasattr(candidate, "finish_reason") and candidate.finish_reason:
                    finish_reason = self._convert_finish_reason(candidate.finish_reason)

            yield StreamChunk(
                id=response_id,
                model=request.model,
                delta_content=text,
                delta_tool_calls=tool_calls,
                finish_reason=finish_reason,
            )

    def get_models(self) -> list[ModelInfo]:
        """Get available Google models."""
        return GOOGLE_MODELS

    def _convert_messages(self, messages: list[ChatMessage]) -> tuple[list[dict], str | None]:
        """Convert unified messages to Gemini format.
        
        Returns:
            Tuple of (contents, system_instruction)
        """
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.role if isinstance(msg.role, str) else msg.role.value

            # System message becomes system_instruction
            if role == "system":
                system_instruction = msg.content
                continue

            # Map roles
            gemini_role = "user" if role == "user" else "model"

            parts = []

            # Text content
            if msg.content:
                parts.append({"text": msg.content})

            # Tool calls (function responses)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append({
                        "function_call": {
                            "name": tc.name,
                            "args": tc.arguments,
                        }
                    })

            # Tool results
            if role == "tool" and msg.tool_call_id and msg.content:
                parts.append({
                    "function_response": {
                        "name": msg.name or "function",
                        "response": {"result": msg.content},
                    }
                })

            if parts:
                contents.append({"role": gemini_role, "parts": parts})

        return contents, system_instruction

    def _build_config(self, request: ChatRequest, system_instruction: str | None = None) -> Any:
        """Build Gemini generation config."""
        from google.genai import types

        config_dict = {
            "temperature": request.temperature,
        }

        if request.max_tokens:
            config_dict["max_output_tokens"] = request.max_tokens

        if request.stop:
            config_dict["stop_sequences"] = request.stop

        if request.top_p is not None:
            config_dict["top_p"] = request.top_p

        # Add system instruction if present
        if system_instruction:
            config_dict["system_instruction"] = system_instruction

        # Add tools if present
        if request.tools:
            function_declarations = []
            for tool in request.tools:
                function_declarations.append(
                    types.FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters=tool.parameters,
                    )
                )
            config_dict["tools"] = [types.Tool(function_declarations=function_declarations)]
            config_dict["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(
                disable=True
            )

        return types.GenerateContentConfig(**config_dict)

    def _convert_response(self, response: Any, model: str) -> ChatResponse:
        """Convert Gemini response to unified format."""
        response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

        # Extract content and tool calls
        content = None
        tool_calls = None
        finish_reason = FinishReason.STOP

        if response.candidates:
            candidate = response.candidates[0]

            # Get text
            if hasattr(candidate, "text"):
                content = candidate.text
            elif hasattr(candidate, "content") and candidate.content:
                texts = []
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        texts.append(part.text)
                    elif hasattr(part, "function_call") and part.function_call:
                        fc = part.function_call
                        if tool_calls is None:
                            tool_calls = []
                        tool_calls.append(
                            ToolCall(
                                id=f"call_{uuid.uuid4().hex[:8]}",
                                name=fc.name,
                                arguments=dict(fc.args) if fc.args else {},
                            )
                        )
                if texts:
                    content = "".join(texts)

            # Finish reason
            if tool_calls:
                finish_reason = FinishReason.TOOL_CALLS
            elif hasattr(candidate, "finish_reason"):
                finish_reason = self._convert_finish_reason(candidate.finish_reason)

        # Usage
        usage = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = Usage(
                prompt_tokens=response.usage_metadata.prompt_token_count or 0,
                completion_tokens=response.usage_metadata.candidates_token_count or 0,
                total_tokens=response.usage_metadata.total_token_count or 0,
            )

        message = ChatMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=tool_calls,
        )

        return ChatResponse(
            id=response_id,
            model=model,
            choices=[Choice(index=0, message=message, finish_reason=finish_reason)],
            usage=usage,
        )

    def _convert_finish_reason(self, reason: Any) -> str:
        """Convert Gemini finish reason to unified format."""
        reason_str = str(reason).lower()
        if "stop" in reason_str:
            return FinishReason.STOP.value
        elif "length" in reason_str or "max" in reason_str:
            return FinishReason.LENGTH.value
        elif "tool" in reason_str or "function" in reason_str:
            return FinishReason.TOOL_CALLS.value
        return FinishReason.STOP.value
