import os
import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

try:
    import google.genai as genai
    from google.genai import types as genai_types

    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False

from src.core.schemas import (
    PackagingOutput,
    HeadlineAssessmentItem,
    HeadlineScores,
    UrlAnalysisResponse,
)
from src.core.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    URL_ARTICLE_SYSTEM_PROMPT,
    URL_ARTICLE_USER_PROMPT_TEMPLATE,
)
from src.core.config import (
    ENABLE_HEADLINE_EVALUATOR,
    ENABLE_HEURISTIC_ONLY_FALLBACK,
    EVALUATOR_MODEL_NAME,
)
# from src.utils.parser import extract_json (removed: migrated to Structured Outputs)

from src.services.headline_quality_heuristics import assess_headline_heuristics
from src.services.headline_quality_evaluator import evaluate_headlines_llm
from src.services.headline_quality_merge import merge_assessments

load_dotenv()

logger = logging.getLogger(__name__)

_HEADLINE_STYLES = ["Pilny", "Pytanie", "Liczbowy", "Luka ciekawości", "Bezpośredni"]


class LLMProvider(ABC):
    @abstractmethod
    def generate_packaging(self, article_text: str) -> Optional[PackagingOutput]:
        pass

    @abstractmethod
    def analyze_url_content(
        self,
        page_text: str,
        url: str = "",
        skip_assessment: bool = False,
        is_article_confident: bool = False,
    ) -> dict:
        """Classify whether page_text is an article; if yes, return packaging.

        Args:
            page_text: extracted text
            url: source URL
            skip_assessment: skip Phase 2 assessment
            is_article_confident: if True, we believe it's an article deterministically.
        """
        pass


_URL_CHAR_LIMIT = 8000  # characters sent to LLM to stay within token windows


# _parse_url_response removed: replaced by native model_dump() from Structured Outputs


def run_headline_assessment(
    result: PackagingOutput,
    article_text: str,
    seo_tags: Optional[List[str]] = None,
    banned_phrases: Optional[List[str]] = None,
    client=None,
    model: str = EVALUATOR_MODEL_NAME,
    source_mode: str = "unknown",
) -> PackagingOutput:
    """
    Run heuristics + (optionally) LLM evaluator + merge.
    Attaches results to result.headline_assessment.
    Never raises — failures degrade gracefully to heuristic-only.
    """
    headlines = result.headlines
    effective_seo = seo_tags or result.seo_tags

    try:
        heuristic_results = assess_headline_heuristics(
            headlines=headlines,
            article_text=article_text,
            seo_tags=effective_seo,
            banned_phrases=banned_phrases,
        )
    except Exception as exc:
        logger.exception(
            "Heuristic assessment failed (%s); skipping assessment entirely.", exc
        )
        return result

    llm_results = None
    if ENABLE_HEADLINE_EVALUATOR and client is not None:
        try:
            llm_results = evaluate_headlines_llm(
                headlines=headlines,
                article_text=article_text,
                seo_tags=effective_seo,
                client=client,
                model=model,
                source_mode=source_mode,
            )
        except Exception:
            logger.exception(
                "LLM evaluator raised unexpectedly; falling back to heuristics."
            )
            llm_results = None

    if llm_results is None and not ENABLE_HEURISTIC_ONLY_FALLBACK:
        logger.info("Heuristic-only fallback disabled; omitting headline_assessment.")
        return result

    try:
        assessment = merge_assessments(
            heuristic_results=heuristic_results,
            llm_results=llm_results,
            headlines=headlines,
            headline_styles=_HEADLINE_STYLES,
        )
        result.headline_assessment = assessment
    except Exception as exc:
        logger.exception(
            "Merge of headline assessments failed (%s); assessment omitted.", exc
        )

    return result


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        generation_model: str = "gpt-5-nano-2025-08-07",
        extraction_model: str = "gpt-5-nano-2025-08-07",
        evaluator_model: str = EVALUATOR_MODEL_NAME,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        self.generation_model = generation_model
        self.extraction_model = extraction_model
        self.evaluator_model = evaluator_model

    def generate_packaging(
        self, article_text: str, skip_assessment: bool = False
    ) -> Optional[PackagingOutput]:
        if not self.client:
            logger.error("OpenAI client not initialized (missing API key).")
            return None

        prompt = USER_PROMPT_TEMPLATE.format(article_text=article_text)
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.generation_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=PackagingOutput,
            )
            result = response.choices[0].message.parsed
            if not result:
                logger.error("Empty or invalid structured response from OpenAI")
                return None
        except Exception as e:
            logger.exception(f"Error generating packaging with OpenAI: {e}")
            return None

        if skip_assessment:
            return result

        # ── Headline quality assessment ──────────────────────────────────────
        result = run_headline_assessment(
            result=result,
            article_text=article_text,
            client=self.client,
            model=self.evaluator_model,
            source_mode="paste",
        )
        return result

    def assess_packaging(
        self, result: PackagingOutput, article_text: str, source_mode: str = "paste"
    ) -> PackagingOutput:
        """Run assessment on an already-generated PackagingOutput (two-phase rendering)."""
        return run_headline_assessment(
            result=result,
            article_text=article_text,
            client=self.client,
            model=self.evaluator_model,
            source_mode=source_mode,
        )

    def analyze_url_content(
        self,
        page_text: str,
        url: str = "",
        skip_assessment: bool = False,
        is_article_confident: bool = False,
    ) -> dict:
        if not self.client:
            return {
                "is_article": is_article_confident,
                "error": "OpenAI client not initialized (missing API key).",
            }

        # If we are confident it's an article, we can use a slightly more direct prompt
        # but for simplicity we keep the same system prompt and just rely on the model.
        # However, if we HAD separate prompts, we'd switch here.
        # For now, we always use the classifier schema but we can pass the hint in user prompt.
        hint = " [CONFIRMED ARTICLE]" if is_article_confident else ""
        prompt = URL_ARTICLE_USER_PROMPT_TEMPLATE.format(
            url=url,
            page_text=page_text[:_URL_CHAR_LIMIT] + hint,
            char_limit=_URL_CHAR_LIMIT,
        )
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.extraction_model,
                messages=[
                    {"role": "system", "content": URL_ARTICLE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format=UrlAnalysisResponse,
            )
            result = response.choices[0].message.parsed
            if not result:
                return {
                    "is_article": False,
                    "error": "Empty or invalid structured response from OpenAI.",
                }
            result_dict = result.model_dump()
        except Exception as e:
            logger.exception("OpenAI URL analysis error: %s", e)
            return {"is_article": False, "error": str(e)}

        if skip_assessment:
            return result_dict

        return self.assess_url_result(result_dict, page_text, url)

    def assess_url_result(
        self, result_dict: dict, page_text: str, url: str = ""
    ) -> dict:
        """Run assessment on a result_dict obtained with skip_assessment=True."""
        source_mode = "portal" if url else "url"
        if result_dict.get("is_article") and result_dict.get("headlines"):
            try:
                # Defensively clamp seo_tags to schema maximum before construction
                if isinstance(result_dict.get("seo_tags"), list):
                    result_dict["seo_tags"] = result_dict["seo_tags"][:7]
                packaging = PackagingOutput(**result_dict)
                packaging = run_headline_assessment(
                    result=packaging,
                    article_text=page_text,
                    client=self.client,
                    model=self.evaluator_model,
                    source_mode=source_mode,
                )
                result_dict["headline_assessment"] = (
                    [a.model_dump() for a in packaging.headline_assessment]
                    if packaging.headline_assessment
                    else None
                )
            except Exception:
                logger.exception(
                    "assess_url_result: assessment failed; continuing without assessment."
                )
        return result_dict


# ─────────────────────────────────────────────────────────────────────────────
# GeminiProvider
# ─────────────────────────────────────────────────────────────────────────────


# LEGACY: _GeminiChatAdapter removed because evaluate_headlines_llm now handles genai.Client directly.


class GeminiProvider(LLMProvider):
    """LLM provider backed by Google Gemini via google-genai."""

    def __init__(
        self,
        generation_model: str = "gemini-3-flash-preview",
        extraction_model: str = "gemini-3-flash-preview",
        evaluator_model: str = EVALUATOR_MODEL_NAME,
    ):
        if not _GENAI_AVAILABLE:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            )
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GEMINI_API_KEY not found in environment.")
            self._configured = False
            self._client = None
        else:
            self._client = genai.Client(api_key=api_key)
            self._configured = True

        self.generation_model = generation_model
        self.extraction_model = extraction_model
        self.evaluator_model = evaluator_model

    def _generate_structured(
        self,
        model_name: str,
        system_prompt: str,
        user_prompt: str,
        schema_cls: type,
    ) -> Optional[object]:
        """Call Gemini with a Pydantic schema for structured output."""
        if not self._configured:
            logger.error("Gemini client not configured (missing API key).")
            return None
        try:
            response = self._client.models.generate_content(
                model=model_name,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=schema_cls,
                ),
            )
            return response.parsed
        except Exception as e:
            logger.exception("Gemini Structured Output error: %s", e)
            return None

    def generate_packaging(
        self, article_text: str, skip_assessment: bool = False
    ) -> Optional["PackagingOutput"]:
        if not self._configured:
            return None

        prompt = USER_PROMPT_TEMPLATE.format(article_text=article_text)
        result = self._generate_structured(
            self.generation_model, SYSTEM_PROMPT, prompt, PackagingOutput
        )
        if not result or not isinstance(result, PackagingOutput):
            logger.error("Failed to generate structured result from Gemini")
            return None

        if skip_assessment:
            return result

        return run_headline_assessment(
            result=result,
            article_text=article_text,
            client=self._client,
            model=self.evaluator_model,
            source_mode="paste",
        )

    def assess_packaging(
        self, result: "PackagingOutput", article_text: str, source_mode: str = "paste"
    ) -> "PackagingOutput":
        return run_headline_assessment(
            result=result,
            article_text=article_text,
            client=self._client,
            model=self.evaluator_model,
            source_mode=source_mode,
        )

    def analyze_url_content(
        self,
        page_text: str,
        url: str = "",
        skip_assessment: bool = False,
        is_article_confident: bool = False,
    ) -> dict:
        if not self._configured:
            return {
                "is_article": is_article_confident,
                "error": "Gemini client not configured (missing API key).",
            }

        hint = " [CONFIRMED ARTICLE]" if is_article_confident else ""
        prompt = URL_ARTICLE_USER_PROMPT_TEMPLATE.format(
            url=url,
            page_text=page_text[:_URL_CHAR_LIMIT] + hint,
            char_limit=_URL_CHAR_LIMIT,
        )
        result = self._generate_structured(
            self.extraction_model,
            URL_ARTICLE_SYSTEM_PROMPT,
            prompt,
            UrlAnalysisResponse,
        )
        if not result or not isinstance(result, UrlAnalysisResponse):
            return {
                "is_article": False,
                "error": "Failed to get structured URL analysis.",
            }

        result_dict = result.model_dump()

        if skip_assessment:
            return result_dict

        return self.assess_url_result(result_dict, page_text, url)

    def assess_url_result(
        self, result_dict: dict, page_text: str, url: str = ""
    ) -> dict:
        source_mode = "portal" if url else "url"
        if result_dict.get("is_article") and result_dict.get("headlines"):
            try:
                # Defensively clamp seo_tags to schema maximum before construction
                if isinstance(result_dict.get("seo_tags"), list):
                    result_dict["seo_tags"] = result_dict["seo_tags"][:7]
                packaging = PackagingOutput(**result_dict)
                packaging = run_headline_assessment(
                    result=packaging,
                    article_text=page_text,
                    client=self._client,
                    model=self.evaluator_model,
                    source_mode=source_mode,
                )
                result_dict["headline_assessment"] = (
                    [a.model_dump() for a in packaging.headline_assessment]
                    if packaging.headline_assessment
                    else None
                )
            except Exception:
                logger.exception(
                    "GeminiProvider.assess_url_result: assessment failed; continuing without."
                )
        return result_dict


class MockProvider(LLMProvider):
    """Used for testing UI without API calls."""

    def generate_packaging(self, article_text: str) -> Optional[PackagingOutput]:
        time.sleep(1.5)  # Simulate network latency
        data = {
            "headlines": [
                "PILNE: Rynki globalne w szoku po niespodziewanej decyzji Fed",
                "Czy Twoje oszczędności przeżyją najnowszą podwyżkę stóp procentowych?",
                "5 kluczowych akcji, które tracą po dzisiejszej decyzji Rezerwy Federalnej",
                "Ukryty sygnał ekonomiczny za nagłym ruchem Fed",
                "Rezerwa Federalna podwyższa stopy procentowe o 0,5% w walce z inflacją",
            ],
            "lead_summary": "Globalne rynki zareagowały gwałtownymi spadkami na niespodziewaną decyzję Rezerwy Federalnej o podwyższeniu stóp procentowych o 0,5%. Analitycy wskazują, że ten agresywny ruch sygnalizuje znaczącą zmianę polityki monetarnej mającej na celu ograniczenie uporczywej inflacji.",
            "seo_tags": [
                "rezerwa federalna",
                "stopy procentowe",
                "krach rynkowy",
                "inflacja",
                "finanse",
            ],
            "social_posts": {
                "x_twitter": "🚨 PILNE: Fed podwyższa stopy o 0,5%, wysyłając rynki w spiralę. Oto co to znaczy dla Twojego portfela. #Finanse #Fed",
                "facebook": "Rezerwa Federalna zaskoczyła inwestorów dzisiaj podwyżką stóp o 0,5%. Akcje technologiczne prowadzą wyprzedaż gdy rentowności obligacji rosną. Przeczytaj naszą pełną analizę.",
            },
        }
        result = PackagingOutput(**data)

        # Attach a static mock assessment
        result.headline_assessment = _build_mock_assessment(result.headlines)
        return result

    def assess_packaging(
        self, result: PackagingOutput, article_text: str, source_mode: str = "paste"
    ) -> PackagingOutput:
        """MockProvider: assessment already attached in generate_packaging."""
        return result

    def assess_url_result(
        self, result_dict: dict, page_text: str, url: str = ""
    ) -> dict:
        """MockProvider: assessment already attached in analyze_url_content."""
        return result_dict

    def analyze_url_content(
        self,
        page_text: str,
        url: str = "",
        skip_assessment: bool = False,
        is_article_confident: bool = False,
    ) -> dict:
        """Mock: simulates article detection based on text length or confidence."""
        time.sleep(1.5)
        if not is_article_confident and len(page_text.split()) < 50:
            return {
                "is_article": False,
                "reason": "[Demo] The page appears to be a homepage or navigation-only page, not an article.",
            }
        headlines = [
            "PILNE: Rynki globalne w szoku po niespodziewanej decyzji Fed",
            "Czy Twoje oszczędności przeżyją najnowszą podwyżkę stóp procentowych?",
            "5 kluczowych akcji, które tracą po dzisiejszej decyzji Rezerwy Federalnej",
            "Ukryty sygnał ekonomiczny za nagłym ruchem Fed",
            "Rezerwa Federalna podwyższa stopy procentowe o 0,5% w walce z inflacją",
        ]
        assessment = _build_mock_assessment(headlines)
        return {
            "is_article": True,
            "headlines": headlines,
            "lead_summary": "Globalne rynki zareagowały gwałtownymi spadkami na niespodziewaną decyzję Rezerwy Federalnej o podwyższeniu stóp procentowych o 0,5%. Analitycy wskazują, że ten agresywny ruch sygnalizuje znaczącą zmianę polityki monetarnej mającej na celu ograniczenie uporczywej inflacji.",
            "seo_tags": [
                "rezerwa federalna",
                "stopy procentowe",
                "krach rynkowy",
                "inflacja",
                "finanse",
            ],
            "social_posts": {
                "x_twitter": "🚨 PILNE: Fed podwyższa stopy o 0,5%, wysyłając rynki w spiralę. Oto co to znaczy dla Twojego portfela. #Finanse #Fed",
                "facebook": "Rezerwa Federalna zaskoczyła inwestorów dzisiaj podwyżką stóp o 0,5%. Akcje technologiczne prowadzą wyprzedaż gdy rentowności obligacji rosną. Przeczytaj naszą pełną analizę.",
            },
            "headline_assessment": [a.model_dump() for a in assessment],
        }


def _build_mock_assessment(headlines: List[str]) -> List[HeadlineAssessmentItem]:
    """Return a static plausible assessment for the 5 mock headlines."""
    styles = _HEADLINE_STYLES
    mock_data = [
        {
            "ctr": 82,
            "clarity": 78,
            "seo": 85,
            "flags": ["none"],
            "rationale": "Nagłówek pilny z konkretnymi encjami — dobry potencjał klikalności i SEO.",
        },
        {
            "ctr": 75,
            "clarity": 85,
            "seo": 72,
            "flags": ["too_vague"],
            "rationale": "Pytanie angażuje czytelnika, choć brakuje konkretnych danych liczbowych.",
        },
        {
            "ctr": 88,
            "clarity": 90,
            "seo": 88,
            "flags": ["none"],
            "rationale": "Nagłówek liczbowy z konkretnymi danymi — najwyższy potencjał CTR i jasność.",
        },
        {
            "ctr": 68,
            "clarity": 60,
            "seo": 55,
            "flags": ["too_vague", "clickbait_risk"],
            "rationale": "Luka ciekawości — niskie SEO fit i ryzyko clickbait przez ukrywanie tematu.",
        },
        {
            "ctr": 70,
            "clarity": 92,
            "seo": 90,
            "flags": ["none"],
            "rationale": "Bezpośredni i faktyczny — najlepsza czytelność i SEO, niższy hook emocjonalny.",
        },
    ]
    items = []
    for i, (headline, style, md) in enumerate(zip(headlines, styles, mock_data)):
        items.append(
            HeadlineAssessmentItem(
                headline_index=i,
                headline_style=style,
                headline=headline,
                scores=HeadlineScores(
                    ctr_potential=md["ctr"],
                    clarity=md["clarity"],
                    seo_fit=md["seo"],
                    credibility=80,
                ),
                risk_flags=md["flags"],
                rationale=md["rationale"],
            )
        )
    return items
