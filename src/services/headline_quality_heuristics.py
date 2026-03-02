"""
Heuristic quality checks for generated headlines.
All checks are deterministic / local — no LLM required.

SEO fit uses Polish lemmatization (spaCy pl_core_news_sm) so correctly
inflected keywords are matched rather than penalised.  If spaCy is not
installed or the Polish model is missing the code falls back silently to
the previous exact-lowercase matching.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from src.core.config import (
    DUPLICATE_SIMILARITY_THRESHOLD,
    HEADLINE_MAX_CHARS,
    HEADLINE_MAX_WORDS,
)

logger = logging.getLogger(__name__)

# ── Polish stopwords — loaded from `stop-words` library ──────────────────────


def _load_pl_stopwords() -> set[str]:
    """Load Polish stopwords from the `stop-words` library (pip install stop-words)."""
    from stop_words import get_stop_words  # type: ignore[import-untyped]

    return set(get_stop_words("pl"))


_PL_STOPWORDS: set[str] | None = None


# ── spaCy lemmatizer — lazy-loaded, graceful fallback ────────────────────────

_SPACY_NLP = None  # module-level cache; None = not yet attempted
_SPACY_AVAILABLE: bool | None = None  # tri-state: None=unchecked, True/False


def _get_spacy_nlp():
    """
    Return a cached spaCy Language pipeline for Polish or *None* if unavailable.
    Loads at most once per interpreter lifetime.
    """
    global _SPACY_NLP, _SPACY_AVAILABLE

    if _SPACY_AVAILABLE is not None:
        return _SPACY_NLP  # already resolved (may be None)

    try:
        import spacy  # noqa: PLC0415

        _SPACY_NLP = spacy.load(
            "pl_core_news_sm",
            disable=["parser", "ner"],  # only morphology/lemmatizer needed
        )
        _SPACY_AVAILABLE = True
        logger.debug("spaCy pl_core_news_sm loaded — lemmatization enabled.")
    except OSError:
        _SPACY_AVAILABLE = False
        logger.warning(
            "spaCy model 'pl_core_news_sm' not found. "
            "Run: python -m spacy download pl_core_news_sm\n"
            "Falling back to exact-lowercase SEO matching."
        )
    except ImportError:
        _SPACY_AVAILABLE = False
        logger.warning(
            "spaCy not installed. Falling back to exact-lowercase SEO matching."
        )

    return _SPACY_NLP


def _get_pl_stopwords() -> set[str]:
    """Lazy-load Polish stopwords."""
    global _PL_STOPWORDS
    if _PL_STOPWORDS is None:
        _PL_STOPWORDS = _load_pl_stopwords()
    return _PL_STOPWORDS


def _lemmatize(text: str) -> list[str]:
    """
    Return a list of lowercase lemmas for alphabetic tokens (≥3 chars).
    Falls back to simple lowercased split if spaCy is unavailable.
    """
    nlp = _get_spacy_nlp()
    if nlp is not None:
        doc = nlp(text)
        return [
            token.lemma_.lower()
            for token in doc
            if token.is_alpha and len(token.text) >= 3
        ]
    # Fallback: naive split
    return [w.lower() for w in text.split() if len(w) >= 3]


# ── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class HeuristicResult:
    """Heuristic assessment outcome for a single headline."""

    headline_index: int
    flags: List[str] = field(default_factory=list)
    seo_fit_score: int = 50  # 0–100, heuristic only
    clickbait_score: int = 0  # 0-100, local ML model probability


# ── Individual checks ────────────────────────────────────────────────────────


def contains_banned_phrase(headline: str, banned_phrases: List[str]) -> bool:
    """Return True if headline contains any banned phrase (case-insensitive)."""
    h_lower = headline.lower()
    return any(phrase.lower() in h_lower for phrase in banned_phrases)


def is_too_long(
    headline: str,
    max_chars: int = HEADLINE_MAX_CHARS,
    max_words: int = HEADLINE_MAX_WORDS,
) -> bool:
    """Return True if headline exceeds max character or word count."""
    if len(headline) > max_chars:
        return True
    word_count = len(headline.split())
    return word_count > max_words


def check_style_violations(headline: str) -> List[str]:
    """
    Check for linguistic/style issues like shouting, excessive punctuation.
    Returns a list of flags found.
    """
    flags: List[str] = []

    # 1. Shouting (too many ALL CAPS words > 3 chars)
    words = re.findall(r"\b[A-ZĄĆĘŁŃÓŚŹŻ]{4,}\b", headline)
    if words:
        flags.append("shouting_detected")

    # 2. Excessive punctuation (!!!, ???, ...)
    if "!!!" in headline or "???" in headline:
        flags.append("excessive_punctuation")

    # 3. Excessive ellipsis
    if "...." in headline:
        flags.append("excessive_ellipsis")

    return flags


def detect_duplicate_like_headlines(
    headlines: List[str],
    threshold: float = DUPLICATE_SIMILARITY_THRESHOLD,
) -> Dict[int, bool]:
    """
    Return a mapping of {index: is_duplicate_like}.
    An index is True if it shares similarity >= threshold with any other headline.
    Uses difflib.SequenceMatcher — stdlib only.
    """
    n = len(headlines)
    flagged: Dict[int, bool] = {i: False for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            ratio = SequenceMatcher(
                None, headlines[i].lower(), headlines[j].lower()
            ).ratio()
            if ratio >= threshold:
                flagged[i] = True
                flagged[j] = True
    return flagged


def extract_keyword_candidates(
    article_text: str,
    seo_tags: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Extract likely topical keywords with weights.
    Returns {lemma: weight}.
    Weights: 2.0 for SEO tags, 1.0 for top frequency words.

    Keywords are stored as **lemmas** (base grammatical form) so that
    inflected variants match correctly at scoring time.
    """
    keywords: Dict[str, float] = {}

    # SEO tags — lemmatize each phrase/word before storing
    if seo_tags:
        stopwords = _get_pl_stopwords()
        for tag in seo_tags:
            for lemma in _lemmatize(tag):
                if lemma not in stopwords:
                    # Take highest weight in case a tag was already seen
                    keywords[lemma] = max(keywords.get(lemma, 0.0), 2.0)

    # Frequency-based extraction from article text
    article_lemmas = _lemmatize(article_text)
    stopwords = _get_pl_stopwords()
    freq = Counter(
        lem for lem in article_lemmas if len(lem) >= 4 and lem not in stopwords
    )

    for lemma, _ in freq.most_common(20):
        if lemma not in keywords:
            keywords[lemma] = 1.0

    return keywords


def compute_seo_fit_score(headline: str, keywords: Dict[str, float]) -> int:
    """
    Score 0–100 based on weighted keywords (lemmas) present in the headline.
    Returns 50 as neutral when no keywords are available.

    The headline is lemmatized before comparison so correctly inflected
    words are recognised.
    """
    if not keywords:
        return 50

    headline_lemmas = set(_lemmatize(headline))
    total_weight = sum(keywords.values())
    earned_weight = sum(
        weight for kw, weight in keywords.items() if kw in headline_lemmas
    )

    ratio = earned_weight / total_weight

    # Scale: 0 hits → 10, full coverage → 100. Capped at 100.
    score = int(10 + ratio * 90)
    return min(score, 100)


def is_too_vague(headline: str, keywords: Dict[str, float]) -> bool:
    """Return True if the headline shares zero keyword lemmas with the article."""
    if not keywords:
        return False  # Can't assess without keywords — don't flag
    headline_lemmas = set(_lemmatize(headline))
    return not any(kw in headline_lemmas for kw in keywords)


# ── Local ML Clickbait Model ─────────────────────────────────────────────────

_CLICKBAIT_MODEL = None
_CLICKBAIT_MODEL_AVAILABLE: bool | None = None


def _get_clickbait_model():
    """Lazy-load the trained Scikit-Learn clickbait model."""
    global _CLICKBAIT_MODEL, _CLICKBAIT_MODEL_AVAILABLE
    if _CLICKBAIT_MODEL_AVAILABLE is not None:
        return _CLICKBAIT_MODEL

    try:
        import joblib  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        project_root = Path(__file__).parent.parent.parent
        model_path = project_root / "data" / "clickbait_model.pkl"

        if model_path.exists():
            _CLICKBAIT_MODEL = joblib.load(model_path)
            _CLICKBAIT_MODEL_AVAILABLE = True
            logger.debug("Local ML clickbait model loaded successfully.")
        else:
            _CLICKBAIT_MODEL_AVAILABLE = False
            logger.warning(
                f"Clickbait model not found at {model_path}. ML clickbait check disabled."
            )
    except Exception as exc:
        _CLICKBAIT_MODEL_AVAILABLE = False
        logger.warning(f"Failed to load local ML clickbait model: {exc}")

    return _CLICKBAIT_MODEL


def compute_ml_clickbait_score(headline: str) -> int:
    """
    Score 0-100 representing the probability of the headline being clickbait.
    Returns 0 if the model is unavailable or fails.
    """
    model = _get_clickbait_model()
    if model is None:
        return 0

    try:
        # predict_proba returns [[prob_0, prob_1]]
        probs = model.predict_proba([headline])
        # We assume class '1' is clickbait (from our training script)
        # We try to get the probability of 'clickbait'. Our target mapping was 1=clickbait.
        # Checking classes_ attribute if possible, otherwise assuming index 1 is clickbait.
        classes = list(model.classes_)
        if 1 in classes:
            idx = classes.index(1)
        elif "clickbait" in classes:
            idx = classes.index("clickbait")
        else:
            idx = 1  # fallback

        prob_clickbait = probs[0][idx]
        return int(prob_clickbait * 100)
    except Exception as exc:
        logger.warning(f"Error predicting clickbait score: {exc}")
        return 0


# ── Orchestrator ─────────────────────────────────────────────────────────────


def assess_headline_heuristics(
    headlines: List[str],
    article_text: str,
    seo_tags: Optional[List[str]] = None,
    banned_phrases: Optional[List[str]] = None,
) -> List[HeuristicResult]:
    """
    Run all heuristic checks for each headline and return a list of
    HeuristicResult objects (one per headline, same order as input).
    """
    banned_phrases = banned_phrases or []
    keywords = extract_keyword_candidates(article_text, seo_tags)
    duplicate_map = detect_duplicate_like_headlines(headlines)

    results: List[HeuristicResult] = []
    for i, headline in enumerate(headlines):
        flags: List[str] = []

        if is_too_long(headline):
            flags.append("too_long")

        if banned_phrases and contains_banned_phrase(headline, banned_phrases):
            flags.append("banned_phrase_detected")

        if duplicate_map.get(i, False):
            flags.append("duplicate_like_other_headline")

        if is_too_vague(headline, keywords):
            flags.append("too_vague")

        # Linguistic / style checks
        style_flags = check_style_violations(headline)
        flags.extend(style_flags)

        seo_score = compute_seo_fit_score(headline, keywords)
        clickbait_score = compute_ml_clickbait_score(headline)

        if clickbait_score > 75:
            flags.append("clickbait_risk")

        results.append(
            HeuristicResult(
                headline_index=i,
                flags=flags,
                seo_fit_score=seo_score,
                clickbait_score=clickbait_score,
            )
        )

    return results
