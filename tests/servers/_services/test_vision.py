"""Tests for src/servers/_services/vision.py"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.servers._services.vision import (
    GoogleAdapter,
    OpenAIAdapter,
    VisionAdapter,
    get_vision_adapter,
    process_image,
    process_image_async,
)


class TestGoogleAdapterInit:
    def test_init_with_api_key(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "google_api_key": "test-key",
            "model": "gemini-pro-vision",
        }):
            with patch("src.servers._services.vision.genai.Client"):
                adapter = GoogleAdapter()
                assert adapter.client is not None
                assert adapter.model_name == "gemini-pro-vision"

    def test_init_without_api_key(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "model": "gemini-pro-vision",
        }):
            adapter = GoogleAdapter()
            assert adapter.client is None

    def test_init_without_model(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "google_api_key": "test-key",
        }):
            with patch("src.servers._services.vision.genai.Client"):
                adapter = GoogleAdapter()
                assert adapter.model_name is None


class TestGoogleAdapterProcess:
    def test_no_client(self):
        with patch("src.servers._services.vision.load_config", return_value={}):
            adapter = GoogleAdapter()
            result = adapter.process("img.png", "Describe")
            assert "GOOGLE_API_KEY not set" in result

    def test_no_model(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "google_api_key": "test-key",
        }):
            with patch("src.servers._services.vision.genai.Client"):
                adapter = GoogleAdapter()
                adapter.client = MagicMock()
                result = adapter.process("img.png", "Describe")
                assert "No Google vision model" in result

    def test_file_not_found(self, tmp_path):
        with patch("src.servers._services.vision.load_config", return_value={
            "google_api_key": "test-key",
            "model": "gemini-pro-vision",
        }):
            with patch("src.servers._services.vision.genai.Client"):
                adapter = GoogleAdapter()
                adapter.client = MagicMock()
                adapter.model_name = "gemini-pro-vision"
                result = adapter.process(str(tmp_path / "missing.png"), "Describe")
                assert "not found" in result

    def test_permission_error(self, tmp_path):
        with patch("src.servers._services.vision.load_config", return_value={
            "google_api_key": "test-key",
            "model": "gemini-pro-vision",
        }):
            with patch("src.servers._services.vision.genai.Client"):
                adapter = GoogleAdapter()
                adapter.client = MagicMock()
                adapter.model_name = "gemini-pro-vision"
                with patch("builtins.open", side_effect=PermissionError("denied")):
                    result = adapter.process("img.png", "Describe")
                    assert "Permission denied" in result

    def test_successful_process(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "A test image"
        mock_client.models.generate_content.return_value = mock_response

        with patch("src.servers._services.vision.load_config", return_value={
            "google_api_key": "test-key",
            "model": "gemini-pro-vision",
        }):
            with patch("src.servers._services.vision.genai.Client", return_value=mock_client):
                adapter = GoogleAdapter()
                adapter.client = mock_client
                adapter.model_name = "gemini-pro-vision"
                result = adapter.process(str(img), "What is this?")
                assert result == "A test image"

    def test_mime_type_jpg(self, tmp_path):
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "JPEG image"
        mock_client.models.generate_content.return_value = mock_response

        with patch("src.servers._services.vision.load_config", return_value={
            "google_api_key": "test-key",
            "model": "gemini-pro-vision",
        }):
            with patch("src.servers._services.vision.genai.Client", return_value=mock_client):
                adapter = GoogleAdapter()
                adapter.client = mock_client
                adapter.model_name = "gemini-pro-vision"
                result = adapter.process(str(img), "Describe")
                assert result == "JPEG image"

    def test_generic_exception(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n")

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("API error")

        with patch("src.servers._services.vision.load_config", return_value={
            "google_api_key": "test-key",
            "model": "gemini-pro-vision",
        }):
            with patch("src.servers._services.vision.genai.Client", return_value=mock_client):
                adapter = GoogleAdapter()
                adapter.client = mock_client
                adapter.model_name = "gemini-pro-vision"
                result = adapter.process(str(img), "Describe")
                assert "Gemini Error" in result


class TestOpenAIAdapterInit:
    def test_init_with_api_key(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "openai_api_key": "test-key",
            "model": "gpt-4-vision-preview",
        }):
            with patch("src.servers._services.vision.OpenAI"):
                adapter = OpenAIAdapter()
                assert adapter.client is not None
                assert adapter.model_name == "gpt-4-vision-preview"

    def test_init_without_api_key(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "model": "gpt-4-vision-preview",
        }):
            adapter = OpenAIAdapter()
            assert adapter.client is None


class TestOpenAIAdapterHelpers:
    def test_encode_image(self, tmp_path):
        import base64

        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG test data")
        with patch("src.servers._services.vision.load_config", return_value={}):
            adapter = OpenAIAdapter()
            result = adapter._encode_image(str(img))
            decoded = base64.b64decode(result)
            assert decoded == b"\x89PNG test data"

    def test_get_mime_type_png(self):
        with patch("src.servers._services.vision.load_config", return_value={}):
            adapter = OpenAIAdapter()
            assert adapter._get_mime_type("image.png") == "image/png"

    def test_get_mime_type_jpg(self):
        with patch("src.servers._services.vision.load_config", return_value={}):
            adapter = OpenAIAdapter()
            assert adapter._get_mime_type("photo.jpg") == "image/jpeg"

    def test_get_mime_type_jpeg(self):
        with patch("src.servers._services.vision.load_config", return_value={}):
            adapter = OpenAIAdapter()
            assert adapter._get_mime_type("photo.jpeg") == "image/jpeg"

    def test_get_mime_type_gif(self):
        with patch("src.servers._services.vision.load_config", return_value={}):
            adapter = OpenAIAdapter()
            assert adapter._get_mime_type("anim.gif") == "image/gif"

    def test_get_mime_type_webp(self):
        with patch("src.servers._services.vision.load_config", return_value={}):
            adapter = OpenAIAdapter()
            assert adapter._get_mime_type("img.webp") == "image/webp"

    def test_get_mime_type_unknown(self):
        with patch("src.servers._services.vision.load_config", return_value={}):
            adapter = OpenAIAdapter()
            assert adapter._get_mime_type("file.xyz") == "image/jpeg"

    def test_get_mime_type_no_extension(self):
        with patch("src.servers._services.vision.load_config", return_value={}):
            adapter = OpenAIAdapter()
            assert adapter._get_mime_type("noext") == "image/jpeg"


class TestOpenAIAdapterProcess:
    def test_no_client(self):
        with patch("src.servers._services.vision.load_config", return_value={}):
            adapter = OpenAIAdapter()
            result = adapter.process("img.png", "Describe")
            assert "OPENAI_API_KEY not set" in result

    def test_no_model(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "openai_api_key": "test-key",
        }):
            with patch("src.servers._services.vision.OpenAI"):
                adapter = OpenAIAdapter()
                adapter.client = MagicMock()
                result = adapter.process("img.png", "Describe")
                assert "No OpenAI vision model" in result

    def test_file_not_found(self, tmp_path):
        with patch("src.servers._services.vision.load_config", return_value={
            "openai_api_key": "test-key",
            "model": "gpt-4-vision-preview",
        }):
            with patch("src.servers._services.vision.OpenAI"):
                adapter = OpenAIAdapter()
                adapter.client = MagicMock()
                adapter.model_name = "gpt-4-vision-preview"
                result = adapter.process(str(tmp_path / "missing.png"), "Describe")
                assert "not found" in result

    def test_permission_error(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "openai_api_key": "test-key",
            "model": "gpt-4-vision-preview",
        }):
            with patch("src.servers._services.vision.OpenAI"):
                adapter = OpenAIAdapter()
                adapter.client = MagicMock()
                adapter.model_name = "gpt-4-vision-preview"
                with patch("builtins.open", side_effect=PermissionError("denied")):
                    result = adapter.process("img.png", "Describe")
                    assert "Permission denied" in result

    def test_successful_process(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG test data")

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A test image via OpenAI"
        mock_client.chat.completions.create.return_value = mock_response

        with patch("src.servers._services.vision.load_config", return_value={
            "openai_api_key": "test-key",
            "model": "gpt-4-vision-preview",
        }):
            with patch("src.servers._services.vision.OpenAI", return_value=mock_client):
                adapter = OpenAIAdapter()
                adapter.client = mock_client
                adapter.model_name = "gpt-4-vision-preview"
                result = adapter.process(str(img), "Describe")
                assert result == "A test image via OpenAI"

    def test_generic_exception(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG test data")

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API error")

        with patch("src.servers._services.vision.load_config", return_value={
            "openai_api_key": "test-key",
            "model": "gpt-4-vision-preview",
        }):
            with patch("src.servers._services.vision.OpenAI", return_value=mock_client):
                adapter = OpenAIAdapter()
                adapter.client = mock_client
                adapter.model_name = "gpt-4-vision-preview"
                result = adapter.process(str(img), "Describe")
                assert "OpenAI Error" in result


class TestGetVisionAdapter:
    def test_returns_google_by_default(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "google_api_key": "key",
            "model": "gemini-pro-vision",
        }):
            with patch("src.servers._services.vision.genai.Client"):
                adapter = get_vision_adapter()
                assert isinstance(adapter, GoogleAdapter)

    def test_returns_openai_for_gpt_model(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "openai_api_key": "key",
            "model": "gpt-4-vision",
        }):
            with patch("src.servers._services.vision.OpenAI"):
                adapter = get_vision_adapter()
                assert isinstance(adapter, OpenAIAdapter)

    def test_returns_openai_for_o1_model(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "openai_api_key": "key",
            "model": "o1-preview",
        }):
            with patch("src.servers._services.vision.OpenAI"):
                adapter = get_vision_adapter()
                assert isinstance(adapter, OpenAIAdapter)

    def test_returns_openai_for_o3_model(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "openai_api_key": "key",
            "model": "o3-mini",
        }):
            with patch("src.servers._services.vision.OpenAI"):
                adapter = get_vision_adapter()
                assert isinstance(adapter, OpenAIAdapter)

    def test_returns_openai_for_o4_model(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "openai_api_key": "key",
            "model": "o4-mini",
        }):
            with patch("src.servers._services.vision.OpenAI"):
                adapter = get_vision_adapter()
                assert isinstance(adapter, OpenAIAdapter)

    def test_returns_openai_when_only_openai_key(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "openai_api_key": "key",
            "model": "some-model",
        }):
            with patch("src.servers._services.vision.OpenAI"):
                adapter = get_vision_adapter()
                assert isinstance(adapter, OpenAIAdapter)

    def test_returns_google_when_both_keys(self):
        with patch("src.servers._services.vision.load_config", return_value={
            "google_api_key": "gkey",
            "openai_api_key": "okey",
            "model": "some-model",
        }):
            with patch("src.servers._services.vision.genai.Client"):
                adapter = get_vision_adapter()
                assert isinstance(adapter, GoogleAdapter)


class TestProcessImage:
    def test_delegates_to_adapter(self):
        with patch("src.servers._services.vision.get_vision_adapter") as mock_get:
            mock_adapter = MagicMock()
            mock_adapter.process.return_value = "result text"
            mock_get.return_value = mock_adapter
            result = process_image("img.png", "prompt")
            mock_adapter.process.assert_called_once_with("img.png", "prompt")
            assert result == "result text"


class TestProcessImageAsync:
    @pytest.mark.asyncio
    async def test_async_wrapper(self):
        with patch("src.servers._services.vision.process_image", return_value="async result") as mock_proc:
            result = await process_image_async("img.png", "prompt")
            mock_proc.assert_called_once_with("img.png", "prompt")
            assert result == "async result"

    @pytest.mark.asyncio
    async def test_async_with_custom_prompt(self):
        with patch("src.servers._services.vision.process_image", return_value="ok") as mock_proc:
            result = await process_image_async("img.png", "custom prompt")
            mock_proc.assert_called_once_with("img.png", "custom prompt")
