"""
LLM-based headline quality evaluator.
Makes a second API call (separate from generation) to score each headline.
Uses native Structured Outputs — no manual JSON parsing or retry loop needed.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

from src.core.headline_quality_evaluator_prompt import (
    EVALUATOR_PROMPT_VERSION,
    build_evaluator_prompt,
)
from src.core.schemas import LLMEvalItem, LLMEvalResponse

logger = logging.getLogger(__name__)

# ── LLM result dataclass ─────────────────────────────────────────────────────


@dataclass
class LLMEvalResult:
    """Raw output from the LLM evaluator for a single headline."""

    headline_index: int
    ctr_potential: int
    clarity: int
    seo_fit: int
    risk_flags: List[str] = field(default_factory=lambda: ["none"])
    rationale: str = ""


# ── Safety helper ─────────────────────────────────────────────────────────────


def _clamp(value: Any, lo: int = 0, hi: int = 100) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return 50


def _item_to_result(item: LLMEvalItem) -> LLMEvalResult:
    """Convert a validated Pydantic LLMEvalItem to the internal LLMEvalResult."""
    flags = item.risk_flags if item.risk_flags else ["none"]
    return LLMEvalResult(
        headline_index=item.headline_index,
        ctr_potential=_clamp(item.ctr_potential),
        clarity=_clamp(item.clarity),
        seo_fit=_clamp(item.seo_fit),
        risk_flags=flags,
        rationale=str(item.rationale)[:250],
    )


# ── Provider detection ────────────────────────────────────────────────────────


def _is_gemini_client(client: Any) -> bool:
    """Return True if *client* is a google.genai.Client instance."""
    try:
        import google.genai as genai  # noqa: PLC0415

        return isinstance(client, genai.Client)
    except ImportError:
        return False


# ── Gemini path ───────────────────────────────────────────────────────────────


def _evaluate_gemini(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Optional[LLMEvalResponse]:
    """Call Gemini with response_schema=LLMEvalResponse; return parsed object or None."""
    try:
        from google.genai import types as genai_types  # noqa: PLC0415

        response = client.models.generate_content(
            model=model,
            contents=user_prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=LLMEvalResponse,
            ),
        )
        return LLMEvalResponse.model_validate_json(response.text)
    except Exception as exc:
        logger.exception("Gemini evaluator call failed: %s", exc)
        return None


# ── OpenAI path ───────────────────────────────────────────────────────────────


def _evaluate_openai(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> Optional[LLMEvalResponse]:
    """Call OpenAI with response_format=LLMEvalResponse (.parse); return parsed object or None."""
    try:
        response = client.beta.chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=LLMEvalResponse,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            logger.error("OpenAI evaluator: .parsed is None (refusal or empty).")
            return None
        return parsed
    except Exception as exc:
        logger.exception("OpenAI evaluator call failed: %s", exc)
        return None


# ── Public entrypoint ────────────────────────────────────────────────────────


def evaluate_headlines_llm(
    headlines: List[str],
    article_text: str,
    seo_tags: Optional[List[str]] = None,
    client=None,
    model: str = "gemini-2.0-flash-preview",
    source_mode: str = "unknown",
) -> Optional[List[LLMEvalResult]]:
    """
    Call the LLM evaluator and return scored results for all 5 headlines.
    Returns None on unrecoverable failure so the caller can fall back gracefully.

    Uses native Structured Outputs — schema compliance is guaranteed by the provider.
    """
    from src.utils.telemetry import log_headline_evaluation  # avoid circular import

    if client is None:
        logger.warning("LLM evaluator: no client provided, skipping.")
        return None

    styles = ["Pilny", "Pytanie", "Liczbowy", "Luka ciekawości", "Bezpośredni"]

    system_prompt, user_prompt = build_evaluator_prompt(
        article_text=article_text,
        headlines=headlines,
        styles=styles,
    )

    start = time.time()
    last_error: Optional[str] = None
    parsed_response: Optional[LLMEvalResponse] = None

    if _is_gemini_client(client):
        parsed_response = _evaluate_gemini(client, model, system_prompt, user_prompt)
        if parsed_response is None:
            last_error = "gemini_call_failed"
    else:
        parsed_response = _evaluate_openai(client, model, system_prompt, user_prompt)
        if parsed_response is None:
            last_error = "openai_call_failed"

    latency_ms = int((time.time() - start) * 1000)

    results: Optional[List[LLMEvalResult]] = None
    if parsed_response is not None:
        items = parsed_response.items
        if len(items) != 5:
            logger.error(
                "LLM evaluator: expected 5 items, got %d. Discarding.", len(items)
            )
            last_error = "wrong_item_count"
        else:
            results = [_item_to_result(item) for item in items]

    success = results is not None

    log_headline_evaluation(
        source_mode=source_mode,
        model_name=model,
        prompt_version=EVALUATOR_PROMPT_VERSION,
        article_length=len(article_text),
        headlines_count=len(headlines),
        latency_ms=latency_ms,
        success=success,
        error_type=last_error if not success else None,
        retry_count=0,  # no retries — schema compliance is provider-side
    )

    if not success:
        logger.error("LLM evaluator failed. Error: %s", last_error)

    return results
