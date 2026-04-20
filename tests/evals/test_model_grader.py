import json

import pytest

from src.evals.model_grader import ModelGrader


class TestModelGrader:
    @pytest.fixture(autouse=True)
    def _mock_genai(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_API_KEY", "test-key")

        class MockClient:
            class Models:
                def generate_content(self, **kwargs):
                    return type(
                        "Resp",
                        (),
                        {"text": json.dumps({"score": 4, "reasoning": "Good", "pass": True})},
                    )()

            models = Models()

        class MockGenAIModule:
            class Client:
                def __init__(self, **kwargs):
                    self.models = MockClient.Models()

        monkeypatch.setattr("src.evals.model_grader.genai", MockGenAIModule)
        monkeypatch.setattr(
            "src.evals.model_grader.get_required_config_value",
            lambda k: "test-model",
        )

    def test_grade_returns_parsed_json(self):
        grader = ModelGrader()
        result = grader.grade("task", "output", "rubric")
        assert result["score"] == 4
        assert result["pass"] is True

    def test_grade_handles_markdown_json(self, monkeypatch):
        class MockModels:
            def generate_content(self, **kwargs):
                return type(
                    "Resp",
                    (),
                    {"text": '```json\n{"score": 5, "reasoning": "Perfect", "pass": true}\n```'},
                )()

        grader = ModelGrader()
        monkeypatch.setattr(type(grader.client), "models", MockModels(), raising=False)
        monkeypatch.setattr(grader.client, "models", MockModels())
        result = grader.grade("task", "output", "rubric")
        assert result["score"] == 5

    def test_grade_handles_invalid_json(self, monkeypatch):
        class MockModels:
            def generate_content(self, **kwargs):
                return type("Resp", (), {"text": "not json at all"})()

        grader = ModelGrader()
        monkeypatch.setattr(grader.client, "models", MockModels())
        result = grader.grade("task", "output", "rubric")
        assert result["score"] == 0
        assert result["pass"] is False

    def test_grade_handles_empty_response(self, monkeypatch):
        class MockModels:
            def generate_content(self, **kwargs):
                return type("Resp", (), {"text": ""})()

        grader = ModelGrader()
        monkeypatch.setattr(grader.client, "models", MockModels())
        result = grader.grade("task", "output", "rubric")
        assert result["score"] == 0
        assert "Empty" in result["reasoning"]

    def test_no_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            ModelGrader()

    def test_grade_handles_none_response(self, monkeypatch):
        class MockModels:
            def generate_content(self, **kwargs):
                return None

        grader = ModelGrader()
        monkeypatch.setattr(grader.client, "models", MockModels())
        result = grader.grade("task", "output", "rubric")
        assert result["score"] == 0
        assert "Empty" in result["reasoning"]

    def test_grade_handles_response_no_text_attr(self, monkeypatch):
        class MockModels:
            def generate_content(self, **kwargs):
                return type("Resp", (), {"text": None})()

        grader = ModelGrader()
        monkeypatch.setattr(grader.client, "models", MockModels())
        result = grader.grade("task", "output", "rubric")
        assert result["score"] == 0

    def test_grade_handles_json_decode_error(self, monkeypatch):
        class MockModels:
            def generate_content(self, **kwargs):
                return type("Resp", (), {"text": "{invalid"})()

        grader = ModelGrader()
        monkeypatch.setattr(grader.client, "models", MockModels())
        result = grader.grade("task", "output", "rubric")
        assert result["score"] == 0
        assert "JSON" in result["reasoning"]

    def test_grade_handles_generic_exception(self, monkeypatch):
        class MockModels:
            def generate_content(self, **kwargs):
                raise RuntimeError("API failure")

        grader = ModelGrader()
        monkeypatch.setattr(grader.client, "models", MockModels())
        result = grader.grade("task", "output", "rubric")
        assert result["score"] == 0
        assert "Grading failed" in result["reasoning"]

    def test_grade_main_block(self, monkeypatch):
        import src.evals.model_grader as mod
        assert hasattr(mod, "ModelGrader")

    def test_grade_extracts_json_from_text(self, monkeypatch):
        class MockModels:
            def generate_content(self, **kwargs):
                return type(
                    "Resp",
                    (),
                    {"text": 'Some text before {"score": 3, "reasoning": "ok", "pass": false} and after'},
                )()

        grader = ModelGrader()
        monkeypatch.setattr(grader.client, "models", MockModels())
        result = grader.grade("task", "output", "rubric")
        assert result["score"] == 3
        assert result["pass"] is False
