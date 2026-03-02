"""
Merge / fusion layer: combines heuristic flags with LLM evaluation scores.
"""

from __future__ import annotations

from typing import List, Optional

from src.core.schemas import HeadlineAssessmentItem, HeadlineScores
from src.services.headline_quality_heuristics import HeuristicResult
from src.services.headline_quality_evaluator import LLMEvalResult

_HEADLINE_STYLES = ["Pilny", "Pytanie", "Liczbowy", "Luka ciekawości", "Bezpośredni"]

_FALLBACK_RATIONALE = "Ocena jakości częściowo niedostępna (LLM evaluator error)."


def _merge_flags(heuristic_flags: List[str], llm_flags: List[str]) -> List[str]:
    """
    Combine heuristic and LLM flags:
    - Union of both sets
    - Remove "none" if any real flag is present
    - Preserve consistent ordering (heuristic flags first, then LLM-only flags)
    - Deduplicate
    """
    seen: set[str] = set()
    combined: List[str] = []
    for f in heuristic_flags + llm_flags:
        if f != "none" and f not in seen:
            seen.add(f)
            combined.append(f)
    return combined if combined else ["none"]


def _avg_clamp(*values: int) -> int:
    """Return rounded integer average clamped to [0, 100]."""
    if not values:
        return 50
    return max(0, min(100, round(sum(values) / len(values))))


def merge_assessments(
    heuristic_results: List[HeuristicResult],
    llm_results: Optional[List[LLMEvalResult]],
    headlines: List[str],
    headline_styles: Optional[List[str]] = None,
) -> List[HeadlineAssessmentItem]:
    """
    Merge heuristic and LLM results into a list of HeadlineAssessmentItem.

    Merge rules (v1):
    - ctr_potential : LLM value; fallback 50
    - clarity       : LLM value; fallback 50
    - seo_fit       : avg(llm_seo, heuristic_seo) if LLM available, else heuristic_seo
    - credibility   : 100 - clickbait_score (ML News probability × 100)
    - overall       : mean(ctr, clarity, seo_fit, credibility)
    - risk_flags    : union of both, deduplicated, "none" removed when real flags exist
    - rationale     : LLM value; fallback constant message
    """
    styles = headline_styles or _HEADLINE_STYLES
    llm_map: dict[int, LLMEvalResult] = (
        {r.headline_index: r for r in llm_results} if llm_results else {}
    )

    items: List[HeadlineAssessmentItem] = []
    for hr in heuristic_results:
        i = hr.headline_index
        llm = llm_map.get(i)
        style = styles[i] if i < len(styles) else "Nieznany"
        headline_text = headlines[i] if i < len(headlines) else ""

        # ── Scores & Penalties ──────────────────────────────────────────────
        ctr = llm.ctr_potential if llm else 50
        clarity = llm.clarity if llm else 50
        seo_fit = _avg_clamp(llm.seo_fit, hr.seo_fit_score) if llm else hr.seo_fit_score

        # ── ML Credibility (news probability) ──────────────────────────────
        # clickbait_score 0-100 = P(clickbait). credibility = P(news) * 100.
        credibility = max(0, 100 - hr.clickbait_score)

        # ── Flags ───────────────────────────────────────────────────────────
        llm_flags = llm.risk_flags if llm else []
        merged_flags = _merge_flags(hr.flags, llm_flags)

        # Apply Penalties
        if "too_long" in merged_flags:
            ctr = min(ctr, 50)
            clarity = min(clarity, 50)

        if "banned_phrase_detected" in merged_flags:
            ctr = min(ctr, 20)
            clarity = min(clarity, 20)
            seo_fit = min(seo_fit, 20)

        if "clickbait_risk" in merged_flags:
            # We also apply a continuous penalty based on the severity
            # hr.clickbait_score is 0-100 indicating the probability of clickbait.
            # If it's flagged (>75), we penalize it proportionally.
            max_ctr_allowed = 80 - (hr.clickbait_score - 75) * 2
            ctr = min(ctr, int(max_ctr_allowed))
            ctr = max(0, ctr)

        if (
            "shouting_detected" in merged_flags
            or "excessive_punctuation" in merged_flags
        ):
            ctr = min(ctr, 60)

        # ── Rationale ───────────────────────────────────────────────────────
        rationale = llm.rationale if llm and llm.rationale else _FALLBACK_RATIONALE

        items.append(
            HeadlineAssessmentItem(
                headline_index=i,
                headline_style=style,
                headline=headline_text,
                scores=HeadlineScores(
                    ctr_potential=ctr,
                    clarity=clarity,
                    seo_fit=seo_fit,
                    credibility=credibility,
                ),
                risk_flags=merged_flags,
                rationale=rationale,
            )
        )

    return items
