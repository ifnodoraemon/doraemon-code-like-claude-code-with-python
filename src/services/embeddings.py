import logging

from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from src.core.config import load_config

logger = logging.getLogger(__name__)

# Suppress noisy HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google.genai").setLevel(logging.WARNING)


class RemoteEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function that uses remote APIs (Google GenAI or OpenAI).
    Defaults to a dummy implementation if no keys are found, but warns the user.
    """

    def __init__(self):
        config = load_config(validate=False)
        self.provider = "none"
        self.api_key = None
        self._embedding_model = config.get("embedding_model")

        model_name = (self._embedding_model or "").lower()

        if model_name.startswith("text-embedding-3") and config.get("openai_api_key"):
            self.provider = "openai"
            self.api_key = config.get("openai_api_key")
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
            except ImportError as e:
                logger.warning(
                    f"openai package not found or failed to load: {e}. Install it to use OpenAI embeddings."
                )
                self.provider = "none"
        elif config.get("google_api_key"):
            self.provider = "google"
            self.api_key = config.get("google_api_key")
            try:
                from google import genai

                self.client = genai.Client(api_key=self.api_key)
            except ImportError as e:
                logger.warning(
                    f"google-genai package not found or failed to load: {e}. Install it to use Google embeddings."
                )
                self.provider = "none"
        elif config.get("openai_api_key"):
            self.provider = "openai"
            self.api_key = config.get("openai_api_key")
            try:
                from openai import OpenAI

                self.client = OpenAI(api_key=self.api_key)
            except ImportError as e:
                logger.warning(
                    f"openai package not found or failed to load: {e}. Install it to use OpenAI embeddings."
                )
                self.provider = "none"

    def __call__(self, input: Documents) -> Embeddings:
        from typing import cast

        if self.provider == "google":
            try:
                embed_model = self._embedding_model
                if not embed_model:
                    logger.warning("embedding_model is not configured")
                    return cast(Embeddings, [[0.0] * 768 for _ in input])
                embeddings = []
                for text in input:
                    result = self.client.models.embed_content(
                        model=embed_model,
                        contents=text,
                    )
                    # New SDK returns embeddings[0].values
                    embeddings.append(result.embeddings[0].values)
                return cast(Embeddings, embeddings)
            except Exception as e:
                logger.error(f"Google embedding error: {e}")
                return cast(Embeddings, [[0.0] * 768 for _ in input])

        elif self.provider == "openai":
            try:
                embed_model = self._embedding_model
                if not embed_model:
                    logger.warning("embedding_model is not configured")
                    return cast(Embeddings, [[0.0] * 1536 for _ in input])
                response = self.client.embeddings.create(input=input, model=embed_model)
                return cast(Embeddings, [data.embedding for data in response.data])
            except Exception as e:
                logger.error(f"OpenAI embedding error: {e}")
                return cast(Embeddings, [[0.0] * 1536 for _ in input])

        else:
            logger.warning(
                "No remote embedding provider configured. Memory search will not work correctly."
            )
            return cast(Embeddings, [[0.0] * 768 for _ in input])
