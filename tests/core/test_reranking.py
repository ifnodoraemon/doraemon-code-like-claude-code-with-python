"""Tests for optional note reranking."""

from src.servers._services.reranking import NoteReranker


class TestNoteReranker:
    """Tests for reranking behavior and fallbacks."""

    def test_simple_reranker_prefers_stronger_lexical_match(self):
        reranker = NoteReranker("simple")
        candidates = [
            {"title": "Alpha", "content": "misc content", "tags": [], "base_score": 3.0},
            {"title": "Python Testing", "content": "python testing guide", "tags": ["pytest"]},
        ]

        ranked = reranker.rerank("python testing", candidates, limit=2)

        assert ranked[0]["title"] == "Python Testing"

    def test_non_local_model_falls_back_without_crashing(self):
        reranker = NoteReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        candidates = [
            {"title": "First", "content": "alpha beta", "tags": []},
            {"title": "Second", "content": "python testing", "tags": []},
        ]

        ranked = reranker.rerank("python", candidates, limit=1)

        assert len(ranked) == 1
        assert ranked[0]["title"] == "Second"
