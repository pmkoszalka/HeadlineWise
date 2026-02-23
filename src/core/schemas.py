from typing import List
from pydantic import BaseModel, Field


class SocialPosts(BaseModel):
    x_twitter: str = Field(..., description="Short, engaging post for X/Twitter")
    facebook: str = Field(..., description="Informative post for Facebook")


class PackagingOutput(BaseModel):
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
        max_length=5,
        description="3-5 normalized keywords/tags",
    )
    social_posts: SocialPosts
