import logging

from src.core.llm_provider import MockProvider
from src.utils.scraper import _extract_text_enhanced

logging.basicConfig(level=logging.INFO)


class TestEnhancedScraperOffline:
    def test_article_like_html_extracts_text_and_is_article(self):
        html = """
        <html><head><title>Test</title><link rel='canonical' href='https://example.com/news/1'/>
        <script type="application/ld+json">{"@type": "NewsArticle"}</script>
        </head>
        <body>
          <article>
            <h1>Tytul</h1>
            <p>Pierwszy akapit zawiera wystarczajaco duzo slow aby przejsc prog testowy dla artykulu informacyjnego. Pierwszy akapit zawiera wystarczajaco duzo slow aby przejsc prog testowy dla artykulu informacyjnego. Pierwszy akapit zawiera wystarczajaco duzo slow aby przejsc prog testowy dla artykulu informacyjnego.</p>
            <p>Drugi akapit rowniez opisuje temat i rozwija kontekst wydarzenia z dodatkowymi szczegolami. Drugi akapit rowniez opisuje temat i rozwija kontekst wydarzenia z dodatkowymi szczegolami. Drugi akapit rowniez opisuje temat i rozwija kontekst wydarzenia z dodatkowymi szczegolami.</p>
            <p>Trzeci akapit zamyka historie i utrzymuje strukture typowa dla artykulu redakcyjnego. Treści jest na tyle dużo, aby pomyślnie przejść nowo zdefiniowane testy minimalnej długości, które chronią przed przetwarzaniem bardzo krótkich materiałów takich jak nawigacja i stopki. Treści jest na tyle dużo, aby pomyślnie przejść nowo zdefiniowane testy minimalnej długości.</p>
            <p>Czwarty akapit dodany dla pewnosci ze przekroczymy bariere stu slow w tekscie wyciagnietym przez trafilature, zeby zapewnic poprawne zachowanie mechanizmow zdefiniowanych w scraper.py. Akapit ten zupelnie nic nie wnosi do tresci, ale sluzy jedynie jako sztuczny zapychacz slownikowy.</p>
          </article>
        </body></html>
        """

        text, canonical, is_article, _ = _extract_text_enhanced(
            html, "https://example.com/news/1"
        )

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

        text, canonical, is_article, _ = _extract_text_enhanced(
            html, "https://example.com/"
        )

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
