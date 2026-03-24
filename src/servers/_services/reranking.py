"""Optional note reranking with safe local-only behavior."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_terms(text: str) -> list[str]:
    return [term for term in text.lower().split() if term]


def _fallback_score(query: str, candidate: dict[str, Any]) -> float:
    """Cheap lexical rerank that does not require a model."""
    query_terms = _normalize_terms(query)
    title = str(candidate.get("title", ""))
    content = str(candidate.get("content", ""))
    tags = candidate.get("tags", [])
    tags_text = " ".join(tags if isinstance(tags, list) else [str(tags)])
    haystack = " ".join([title, content, tags_text]).lower()

    overlap = sum(haystack.count(term) for term in query_terms)
    title_bonus = sum(title.lower().count(term) * 2 for term in query_terms)
    phrase_bonus = 3 if query.lower() in haystack else 0
    source_bonus = float(candidate.get("base_score", 0.0))
    return float(overlap + title_bonus + phrase_bonus) + source_bonus


class NoteReranker:
    """Rerank note candidates using a local model when available."""

    def __init__(self, model_name: str | None):
        self.model_name = model_name
        self._cross_encoder = None

        if not model_name or model_name == "simple":
            return

        model_path = Path(model_name)
        if not model_path.exists():
            logger.warning(
                "rerank_model '%s' is not a local path; falling back to lexical reranking.",
                model_name,
            )
            return

        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(str(model_path), local_files_only=True)
        except Exception as e:
            logger.warning(
                "Failed to initialize local reranker '%s': %s. Falling back to lexical reranking.",
                model_name,
                e,
            )

    def rerank(
        self, query: str, candidates: list[dict[str, Any]], limit: int
    ) -> list[dict[str, Any]]:
        """Return candidates in reranked order."""
        if len(candidates) <= 1:
            return candidates[:limit]

        if self._cross_encoder is not None:
            pairs = [(query, str(candidate.get("content", ""))) for candidate in candidates]
            try:
                scores = self._cross_encoder.predict(pairs)
                ranked = sorted(
                    zip(scores, candidates, strict=True),
                    key=lambda item: float(item[0]),
                    reverse=True,
                )
                return [candidate for _score, candidate in ranked[:limit]]
            except Exception as e:
                logger.warning("Local reranker inference failed: %s", e)

        ranked = sorted(
            candidates,
            key=lambda candidate: _fallback_score(query, candidate),
            reverse=True,
        )
        return ranked[:limit]
