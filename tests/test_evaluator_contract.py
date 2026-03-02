from types import SimpleNamespace

from src.core.headline_quality_evaluator_prompt import (
    HEADLINE_EVALUATOR_USER_PROMPT_TEMPLATE,
    build_evaluator_prompt,
)
from src.core.schemas import LLMEvalItem, LLMEvalResponse
from src.services.headline_quality_evaluator import evaluate_headlines_llm


def _mk_item(idx: int) -> LLMEvalItem:
    return LLMEvalItem(
        headline_index=idx,
        ctr_potential=70,
        clarity=75,
        seo_fit=80,
        risk_flags=["none"],
        rationale="OK",
    )


class TestEvaluatorPromptContract:
    def test_user_prompt_requires_items_wrapper(self):
        assert '"items": [' in HEADLINE_EVALUATOR_USER_PROMPT_TEMPLATE
        assert "dokladnie 5" in HEADLINE_EVALUATOR_USER_PROMPT_TEMPLATE.lower()

    def test_built_prompt_contains_items_schema(self):
        _, user_prompt = build_evaluator_prompt(
            article_text="abc",
            headlines=["h1", "h2", "h3", "h4", "h5"],
            styles=["s1", "s2", "s3", "s4", "s5"],
        )
        assert '"items": [' in user_prompt


class TestEvaluatorParsingContract:
    def test_valid_items_wrapper_parses(self):
        payload = {
            "items": [_mk_item(i).model_dump() for i in range(5)],
        }
        parsed = LLMEvalResponse.model_validate(payload)
        assert len(parsed.items) == 5

    def test_array_root_rejected(self):
        payload = [_mk_item(i).model_dump() for i in range(5)]
        try:
            LLMEvalResponse.model_validate(payload)
            assert False, "Expected validation failure for array root"
        except Exception:
            assert True

    def test_wrong_item_count_rejected_in_evaluator_flow(self, monkeypatch):
        monkeypatch.setattr(
            "src.services.headline_quality_evaluator._evaluate_openai",
            lambda *args, **kwargs: LLMEvalResponse(items=[_mk_item(i) for i in range(4)]),
        )

        captured = {}

        def _capture_log(**kwargs):
            captured.update(kwargs)

        monkeypatch.setattr("src.utils.telemetry.log_headline_evaluation", _capture_log)

        fake_client = SimpleNamespace()
        result = evaluate_headlines_llm(
            headlines=["h1", "h2", "h3", "h4", "h5"],
            article_text="tekst",
            client=fake_client,
            model="dummy",
            source_mode="test",
        )

        assert result is None
        assert captured.get("success") is False
        assert captured.get("error_type") == "wrong_item_count"
