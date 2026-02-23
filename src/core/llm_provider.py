import os
import logging
import time
from abc import ABC, abstractmethod
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

from src.core.schemas import PackagingOutput
from src.core.prompts import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    URL_ARTICLE_SYSTEM_PROMPT,
    URL_ARTICLE_USER_PROMPT_TEMPLATE,
)
from src.utils.parser import extract_json

load_dotenv()

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    def generate_packaging(self, article_text: str) -> Optional[PackagingOutput]:
        pass

    @abstractmethod
    def analyze_url_content(self, page_text: str, url: str = "") -> dict:
        """Classify whether page_text is an article; if yes, return packaging.

        Returns:
            dict with keys:
              - is_article (bool)
              - If True:  headlines, lead_summary, seo_tags, social_posts
              - If False: reason (str)
              - error (str) on unexpected failure
        """
        pass


_URL_CHAR_LIMIT = 8000  # characters sent to LLM to stay within token windows


def _parse_url_response(text: str) -> dict:
    """Parse the LLM's URL-analysis JSON response."""
    data = extract_json(text)
    if not data:
        return {"is_article": False, "error": "Could not parse LLM response."}
    if not isinstance(data.get("is_article"), bool):
        return {"is_article": False, "error": "LLM returned unexpected schema."}
    return data


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        generation_model: str = "gpt-5-mini-2025-08-07",
        extraction_model: str = "gpt-5-mini-2025-08-07",
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment.")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
        self.generation_model = generation_model
        self.extraction_model = extraction_model

    def generate_packaging(self, article_text: str) -> Optional[PackagingOutput]:
        if not self.client:
            logger.error("OpenAI client not initialized (missing API key).")
            return None

        prompt = USER_PROMPT_TEMPLATE.format(article_text=article_text)
        try:
            response = self.client.chat.completions.create(
                model=self.generation_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                logger.error("Empty response from OpenAI")
                return None

            data = extract_json(content)
            if data:
                return PackagingOutput(**data)

            logger.error(
                f"Failed to parse JSON from OpenAI response: {content[:200]}..."
            )
            return None
        except Exception as e:
            logger.exception(f"Error generating packaging with OpenAI: {e}")
            return None

    def analyze_url_content(self, page_text: str, url: str = "") -> dict:
        if not self.client:
            return {
                "is_article": False,
                "error": "OpenAI client not initialized (missing API key).",
            }

        prompt = URL_ARTICLE_USER_PROMPT_TEMPLATE.format(
            url=url,
            page_text=page_text[:_URL_CHAR_LIMIT],
            char_limit=_URL_CHAR_LIMIT,
        )
        try:
            response = self.client.chat.completions.create(
                model=self.extraction_model,
                messages=[
                    {"role": "system", "content": URL_ARTICLE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                return {"is_article": False, "error": "Empty response from OpenAI."}
            return _parse_url_response(content)
        except Exception as e:
            logger.exception("OpenAI URL analysis error: %s", e)
            return {"is_article": False, "error": str(e)}


class MockProvider(LLMProvider):
    """Used for testing UI without API calls."""

    def generate_packaging(self, article_text: str) -> Optional[PackagingOutput]:
        time.sleep(1.5)  # Simulate network latency
        data = {
            "headlines": [
                "URGENT: Global Markets Reel After Surprise Fed Rate Hike",
                "Will Your Savings Survive the Latest Interest Rate Jump?",
                "5 Key Stocks Crashing After Today's Federal Reserve News",
                "The Hidden Economic Signal Behind the Sudden Fed Move",
                "Federal Reserve Raises Interest Rates by 0.5% to Combat Inflation",
            ],
            "lead_summary": "Global markets have reacted with sharp declines following the Federal Reserve's unexpected decision to raise interest rates by 0.5%. Analysts suggest this aggressive move signals a significant shift in monetary policy aimed at curbing persistent inflation.",
            "seo_tags": [
                "federal reserve",
                "interest rates",
                "market crash",
                "inflation",
                "finance news",
            ],
            "social_posts": {
                "x_twitter": "🚨 BREAKING: The Fed just raised rates by 0.5%, sending markets into a tailspin. Here's what it means for your wallet. #Finance #FedUpdate",
                "facebook": "The Federal Reserve caught investors off guard today with a 0.5% rate hike. Tech stocks are leading the sell-off as yields spike. Read our full analysis of the market impact.",
            },
        }
        return PackagingOutput(**data)

    def analyze_url_content(self, page_text: str, url: str = "") -> dict:
        """Mock: simulates article detection based on text length."""
        time.sleep(1.5)
        if len(page_text.split()) < 50:
            return {
                "is_article": False,
                "reason": "[Demo] The page appears to be a homepage or navigation-only page, not an article.",
            }
        return {
            "is_article": True,
            "headlines": [
                "URGENT: Global Markets Reel After Surprise Fed Rate Hike",
                "Will Your Savings Survive the Latest Interest Rate Jump?",
                "5 Key Stocks Crashing After Today's Federal Reserve News",
                "The Hidden Economic Signal Behind the Sudden Fed Move",
                "Federal Reserve Raises Interest Rates by 0.5% to Combat Inflation",
            ],
            "lead_summary": "Global markets have reacted with sharp declines following the Federal Reserve's unexpected decision to raise interest rates by 0.5%. Analysts suggest this aggressive move signals a significant shift in monetary policy aimed at curbing persistent inflation.",
            "seo_tags": [
                "federal reserve",
                "interest rates",
                "market crash",
                "inflation",
                "finance news",
            ],
            "social_posts": {
                "x_twitter": "🚨 BREAKING: The Fed just raised rates by 0.5%, sending markets into a tailspin. Here's what it means for your wallet. #Finance #FedUpdate",
                "facebook": "The Federal Reserve caught investors off guard today with a 0.5% rate hike. Tech stocks are leading the sell-off as yields spike. Read our full analysis of the market impact.",
            },
        }
