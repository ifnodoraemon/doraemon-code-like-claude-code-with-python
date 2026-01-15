import os
import base64
from abc import ABC, abstractmethod
from typing import Optional

from google import genai
from google.genai import types
from PIL import Image
from openai import OpenAI


class VisionAdapter(ABC):
    @abstractmethod
    def process(self, image_path: str, prompt: str) -> str:
        pass


class GoogleAdapter(VisionAdapter):
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = genai.Client(api_key=api_key) if api_key else None
        self.model_name = os.getenv("VISION_MODEL", "gemini-2.0-flash")
        
    def process(self, image_path: str, prompt: str) -> str:
        if not self.client:
            return "[Error: GOOGLE_API_KEY not set]"
        try:
            # Read image as bytes
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            
            # Determine mime type
            ext = image_path.lower().split('.')[-1]
            mime_types = {
                'jpg': 'image/jpeg',
                'jpeg': 'image/jpeg',
                'png': 'image/png',
                'gif': 'image/gif',
                'webp': 'image/webp'
            }
            mime_type = mime_types.get(ext, 'image/jpeg')
            
            # Create content with image
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[
                    types.Content(
                        parts=[
                            types.Part.from_text(prompt),
                            types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
                        ]
                    )
                ]
            )
            return response.text
        except Exception as e:
            return f"[Gemini Error: {e}]"


class OpenAIAdapter(VisionAdapter):
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None
        self.model_name = os.getenv("VISION_MODEL", "gpt-4o")

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process(self, image_path: str, prompt: str) -> str:
        if not self.client:
            return "[Error: OPENAI_API_KEY not set]"
        try:
            base64_image = self._encode_image(image_path)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            },
                        ],
                    }
                ],
                max_tokens=1000,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"[OpenAI Error: {e}]"


def get_vision_adapter() -> VisionAdapter:
    provider = os.getenv("VISION_PROVIDER", "google").lower()
    if provider == "openai":
        return OpenAIAdapter()
    return GoogleAdapter()


def process_image(path: str, prompt: str = "Extract all text and describe any diagrams in this image.") -> str:
    adapter = get_vision_adapter()
    return adapter.process(path, prompt)
