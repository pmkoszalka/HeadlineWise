from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SocialPosts(BaseModel):
    """Container for social media post suggestions."""

    x_twitter: str = Field(..., description="Short, engaging post for X/Twitter")
    facebook: str = Field(..., description="Informative post for Facebook")


class HeadlineScores(BaseModel):
    """Heuristic + LLM-derived quality scores for a single headline (all 0-100)."""

    ctr_potential: int = Field(
        ..., ge=0, le=100, description="Heuristic proxy for click-through potential"
    )
    clarity: int = Field(
        ..., ge=0, le=100, description="How clear and unambiguous the headline is"
    )
    seo_fit: int = Field(
        ..., ge=0, le=100, description="How well the headline matches article keywords"
    )
    credibility: int = Field(
        ...,
        ge=0,
        le=100,
        description="ML-based credibility score (news probability); higher = less clickbaity",
    )


class HeadlineAssessmentItem(BaseModel):
    """Quality assessment for one of the 5 generated headlines."""

    headline_index: int = Field(..., ge=0, le=4)
    headline_style: str = Field(
        ...,
        description="Style label: Pilny | Pytanie | Liczbowy | Luka ciekawości | Bezpośredni",
    )
    headline: str
    scores: HeadlineScores
    risk_flags: List[str] = Field(
        ...,
        min_length=1,
        description='List of risk flag strings, or ["none"] when no risks detected',
    )
    rationale: str = Field(
        ...,
        description="Short Polish rationale (1 sentence); fallback message when evaluator failed",
    )

    @field_validator("risk_flags")
    @classmethod
    def validate_flags(cls, v: List[str]) -> List[str]:
        """Validate and deduplicate risk flags, defaulting to ['brak'] if empty."""
        # Remove duplicates while preserving order
        seen: set = set()
        deduped = [f for f in v if not (f in seen or seen.add(f))]  # type: ignore[func-returns-value]
        return deduped if deduped else ["brak"]


class PackagingOutput(BaseModel):
    """The complete result package containing headlines, summary, SEO, and social info."""

    headlines: List[str] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Exactly 5 headlines: Urgent, Question, Number-based, Curiosity gap, Direct",
    )
    lead_summary: str = Field(..., description="Max 3 sentences summarizing the lead")
    seo_tags: List[str] = Field(
        ...,
        min_length=3,
        max_length=7,
        description="3-7 normalized keywords/tags",
    )
    social_posts: SocialPosts
    headline_assessment: Optional[List[HeadlineAssessmentItem]] = Field(
        default=None,
        description="Quality assessment for each of the 5 headlines (populated after evaluation)",
    )


# ── URL-analysis structured response ─────────────────────────────────────────


class UrlAnalysisResponse(BaseModel):
    """Schema for the URL-classification LLM call."""

    is_article: bool
    headlines: Optional[List[str]] = None
    lead_summary: Optional[str] = None
    seo_tags: Optional[List[str]] = None
    social_posts: Optional[SocialPosts] = None
    reason: Optional[str] = None  # explanation when is_article=False


# ── Headline evaluator structured response ────────────────────────────────────


class LLMEvalItem(BaseModel):
    """Schema for one headline scored by the LLM evaluator."""

    headline_index: int
    ctr_potential: int = Field(..., ge=0, le=100)
    clarity: int = Field(..., ge=0, le=100)
    seo_fit: int = Field(..., ge=0, le=100)
    risk_flags: List[str] = Field(
        default_factory=lambda: ["none"],
        description='Risk flag strings or ["none"] when clean',
    )
    rationale: str = Field(default="", description="One Polish sentence, max 200 chars")


class LLMEvalResponse(BaseModel):
    """Root wrapper — both Gemini and OpenAI require an object (not a bare array)."""

    items: List[LLMEvalItem]
