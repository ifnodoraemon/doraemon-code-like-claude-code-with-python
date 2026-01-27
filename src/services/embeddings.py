import logging
import os
from typing import Any, List

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google.generativeai").setLevel(logging.WARNING)

class RemoteEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function that uses remote APIs (Google GenAI or OpenAI).
    Defaults to a dummy implementation if no keys are found, but warns the user.
    """

    def __init__(self):
        self.provider = "none"
        self.api_key = None
        
        # Check for Google API Key
        if os.getenv("GOOGLE_API_KEY"):
            self.provider = "google"
            self.api_key = os.getenv("GOOGLE_API_KEY")
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.client = genai
            except ImportError as e:
                logger.warning(f"google-genai package not found or failed to load: {e}. Install it to use Google embeddings.")
                self.provider = "none"
                
        # Check for OpenAI API Key (fallback or preference)
        elif os.getenv("OPENAI_API_KEY"):
            self.provider = "openai"
            self.api_key = os.getenv("OPENAI_API_KEY")
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError as e:
                logger.warning(f"openai package not found or failed to load: {e}. Install it to use OpenAI embeddings.")
                self.provider = "none"

    def __call__(self, input: Documents) -> Embeddings:
        if self.provider == "google":
            try:
                # Batch embedding support for Gemini
                # Note: text-embedding-004 is deprecated as of Jan 2026. Using text-embedding-005 by default.
                model = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/text-embedding-005")
                embeddings = []
                for text in input:
                    result = self.client.embed_content(
                        model=model,
                        content=text,
                    )
                    embeddings.append(result['embedding'])
                return embeddings
            except Exception as e:
                logger.error(f"Google embedding error: {e}")
                return [[] for _ in input]  # Return empty on failure
                
        elif self.provider == "openai":
            try:
                response = self.client.embeddings.create(
                    input=input,
                    model="text-embedding-3-small"
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                logger.error(f"OpenAI embedding error: {e}")
                return [[] for _ in input]

        else:
            logger.warning("No remote embedding provider configured. Memory search will not work correctly.")
            # Return dummy embeddings for functionality without crashing
            # Dimension must match what Chroma expects if it was already created.
            # 768 is common for Google, 1536 for OpenAI.
            # We'll use 768 as a safe default for now, but really this shouldn't happen if configured.
            return [[0.0] * 768 for _ in input]
