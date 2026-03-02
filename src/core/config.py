"""
Application-wide configuration constants.
Override via environment variables where noted.
"""

import os

# ── LLM provider selection ────────────────────────────────────────────────────
# Set LLM_PROVIDER=openai in .env to switch back to OpenAI
DEFAULT_LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")

# ── Headline quality assessment ──────────────────────────────────────────────

# Maximum headline length in characters before the too_long flag fires
HEADLINE_MAX_CHARS: int = int(os.getenv("HEADLINE_MAX_CHARS", "80"))

# Maximum headline length in words before the too_long flag fires
HEADLINE_MAX_WORDS: int = int(os.getenv("HEADLINE_MAX_WORDS", "14"))

# Minimum SequenceMatcher ratio to consider two headlines near-duplicates
DUPLICATE_SIMILARITY_THRESHOLD: float = float(
    os.getenv("DUPLICATE_SIMILARITY_THRESHOLD", "0.78")
)

# Enable / disable the second LLM evaluator call
ENABLE_HEADLINE_EVALUATOR: bool = (
    os.getenv("ENABLE_HEADLINE_EVALUATOR", "true").lower() == "true"
)

# When the LLM evaluator fails, still show heuristic-only flags instead of nothing
ENABLE_HEURISTIC_ONLY_FALLBACK: bool = (
    os.getenv("ENABLE_HEURISTIC_ONLY_FALLBACK", "true").lower() == "true"
)

# Include a Polish one-sentence rationale per headline in the UI
ENABLE_RATIONALE: bool = os.getenv("ENABLE_RATIONALE", "true").lower() == "true"

# Model used by the evaluator (can differ from the generation model)
EVALUATOR_MODEL_NAME: str = os.getenv("EVALUATOR_MODEL_NAME", "gemini-3-flash-preview")
