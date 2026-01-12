import os
import base64
from abc import ABC, abstractmethod
from mcp.server.fastmcp import FastMCP
from PIL import Image
import io

# ---------------------------------------------------------
# Adapter Architecture
# ---------------------------------------------------------
class VisionAdapter(ABC):
    @abstractmethod
    def process(self, image_path: str, prompt: str) -> str:
        pass

class GoogleAdapter(VisionAdapter):
    def __init__(self):
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        genai.configure(api_key=api_key)
        self.model_name = os.getenv("VISION_MODEL", "gemini-1.5-flash")
        
    def process(self, image_path: str, prompt: str) -> str:
        model = genai.GenerativeModel(self.model_name)
        image = Image.open(image_path)
        response = model.generate_content([prompt, image])
        return response.text

class OpenAIAdapter(VisionAdapter):
    def __init__(self):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.client = OpenAI(api_key=api_key)
        self.model_name = os.getenv("VISION_MODEL", "gpt-4o")

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def process(self, image_path: str, prompt: str) -> str:
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
            max_tokens=300,
        )
        return response.choices[0].message.content

# ---------------------------------------------------------
# MCP Server Definition
# ---------------------------------------------------------
mcp = FastMCP("PolymathVision")

def get_adapter() -> VisionAdapter:
    provider = os.getenv("VISION_PROVIDER", "google").lower()
    if provider == "openai":
        return OpenAIAdapter()
    return GoogleAdapter()

@mcp.tool()
def ocr_image(image_path: str) -> str:
    """Extract all text from an image using the configured AI provider."""
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"
    
    try:
        adapter = get_adapter()
        return adapter.process(image_path, "Please extract all text from this image exactly as it appears. Output only the text.")
    except Exception as e:
        return f"OCR Error: {str(e)}"

@mcp.tool()
def describe_image(image_path: str, question: str = "Describe this image in detail.") -> str:
    """Analyze an image and answer questions about it."""
    if not os.path.exists(image_path):
        return f"Error: Image not found at {image_path}"
        
    try:
        adapter = get_adapter()
        return adapter.process(image_path, question)
    except Exception as e:
        return f"Analysis Error: {str(e)}"

if __name__ == "__main__":
    mcp.run()