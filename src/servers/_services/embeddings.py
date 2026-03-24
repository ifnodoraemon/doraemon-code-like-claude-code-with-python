"""
Embedding functions for semantic search.

This module provides embedding functions that work with ChromaDB.
chromadb is an optional dependency - if not installed, returns dummy embeddings.
"""

import logging
from typing import Any

from src.core.config.config import load_config

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)

HAS_CHROMA = False
try:
    from chromadb.api.types import Documents, Embeddings

    HAS_CHROMA = True
except ImportError:
    Documents = list[str]
    Embeddings = list[list[float]]

EmbeddingFunction = Any


class RemoteEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function that uses remote APIs (Google GenAI or OpenAI).

    Gracefully degrades if:
    - chromadb not installed
    - No API keys configured
    - embedding_model not configured

    Returns dummy embeddings in all error cases.
    """

    def __init__(self):
        config = load_config(validate=False)
        self.provider = "none"
        self.api_key = None
        self._embedding_model = config.get("embedding_model")

        if not HAS_CHROMA:
            logger.warning("chromadb not installed. Semantic search will not work.")
            return

        model_name = (self._embedding_model or "").lower()

        if model_name.startswith("text-embedding-3") and config.get("openai_api_key"):
            self.provider = "openai"
            self.api_key = config.get("openai_api_key")
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
            except ImportError as e:
                logger.warning(f"openai package not found: {e}")
                self.provider = "none"
        elif config.get("google_api_key"):
            self.provider = "google"
            self.api_key = config.get("google_api_key")
            try:
                from google import genai

                self.client = genai.Client(api_key=self.api_key)
            except ImportError as e:
                logger.warning(f"google-genai package not found: {e}")
                self.provider = "none"
        elif config.get("openai_api_key"):
            self.provider = "openai"
            self.api_key = config.get("openai_api_key")
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
            except ImportError as e:
                logger.warning(f"openai package not found: {e}")
                self.provider = "none"

    def __call__(self, input: Documents) -> Embeddings:
        if not HAS_CHROMA:
            return [[0.0] * 768 for _ in input]

        if self.provider == "google":
            try:
                embed_model = self._embedding_model
                if not embed_model:
                    logger.warning("embedding_model not configured")
                    return [[0.0] * 768 for _ in input]
                embeddings = []
                for text in input:
                    result = self.client.models.embed_content(
                        model=embed_model,
                        contents=text,
                    )
                    embeddings.append(result.embeddings[0].values)
                return embeddings
            except Exception as e:
                logger.error(f"Google embedding error: {e}")
                return [[0.0] * 768 for _ in input]

        elif self.provider == "openai":
            try:
                embed_model = self._embedding_model
                if not embed_model:
                    logger.warning("embedding_model not configured")
                    return [[0.0] * 1536 for _ in input]
                response = self.client.embeddings.create(input=input, model=embed_model)
                return [data.embedding for data in response.data]
            except Exception as e:
                logger.error(f"OpenAI embedding error: {e}")
                return [[0.0] * 1536 for _ in input]

        else:
            logger.warning("No embedding provider configured. Semantic search will not work.")
            return [[0.0] * 768 for _ in input]
