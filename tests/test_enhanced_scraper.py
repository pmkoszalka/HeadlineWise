import logging

import pytest

from src.core.llm_provider import MockProvider
from src.utils.scraper import _extract_text_enhanced, fetch_article_text

logging.basicConfig(level=logging.INFO)


class TestEnhancedScraperOffline:
    def test_article_like_html_extracts_text_and_is_article(self):
        html = """
        <html><head><title>Test</title><link rel='canonical' href='https://example.com/news/1'/></head>
        <body>
          <article>
            <h1>Tytul</h1>
            <p>Pierwszy akapit zawiera wystarczajaco duzo slow aby przejsc prog testowy dla artykulu informacyjnego.</p>
            <p>Drugi akapit rowniez opisuje temat i rozwija kontekst wydarzenia z dodatkowymi szczegolami.</p>
            <p>Trzeci akapit zamyka historie i utrzymuje strukture typowa dla artykulu redakcyjnego.</p>
          </article>
        </body></html>
        """

        text, canonical, is_article, _ = _extract_text_enhanced(html, "https://example.com/news/1")

        assert text
        assert canonical == "https://example.com/news/1"
        assert is_article is True

    def test_non_article_html_not_detected_as_article(self):
        html = """
        <html><body>
          <nav>Home Sport Biznes</nav>
          <main>
            <a href='/x'>Link</a>
            <a href='/y'>Inny Link</a>
          </main>
        </body></html>
        """

        text, canonical, is_article, _ = _extract_text_enhanced(html, "https://example.com/")

        assert canonical == ""
        assert isinstance(text, str)
        assert is_article is False

    def test_mock_provider_pipeline_short_text_not_article(self):
        provider = MockProvider()
        result = provider.analyze_url_content(
            "Too short",
            url="https://example.com/",
            skip_assessment=True,
            is_article_confident=False,
        )
        assert result.get("is_article") is False


@pytest.mark.live
class TestEnhancedScraperLive:
    @pytest.mark.parametrize(
        "url, expected_article, force_short",
        [
            (
                "https://tvn24.pl/swiat/usa-izba-reprezentantow-przyjela-pakiet-pomocy-dla-ukrainy-izraela-i-tajwanu-st8143282",
                True,
                False,
            ),
            ("https://tvn24.pl/", False, True),
        ],
    )
    def test_article_extraction_pipeline_live(self, url, expected_article, force_short):
        provider = MockProvider()
        scrape = fetch_article_text(url)

        if not scrape.success:
            pytest.skip(f"Scrape failed (likely network/blocked): {scrape.error}")

        text_to_analyze = "Too short" if force_short else scrape.text
        result = provider.analyze_url_content(
            text_to_analyze,
            url=url,
            skip_assessment=True,
            is_article_confident=scrape.is_article_deterministic if not force_short else False,
        )

        assert result.get("is_article", False) == expected_article
        if expected_article:
            assert len(result.get("headlines", [])) == 5
            assert result.get("lead_summary")
            assert len(result.get("seo_tags", [])) > 0
