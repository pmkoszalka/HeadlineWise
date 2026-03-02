"""
Scraper utility: fetches a URL and extracts readable article text,
and scrapes article headlines from supported news portals.
"""

import logging
from dataclasses import dataclass, field
from urllib.parse import urljoin

import re
import requests
from bs4 import BeautifulSoup
# import trafilatura (moved to lazy-load inside _extract_text_enhanced)

logger = logging.getLogger(__name__)

# Tags containing structural chrome rather than article content
_NOISE_TAGS = [
    "nav",
    "footer",
    "aside",
    "header",
    "script",
    "style",
    "noscript",
    "figure",
    "figcaption",
    "iframe",
    "form",
    "button",
    "input",
    "advertisement",
]

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

_REQUEST_TIMEOUT = 10
_MIN_TEXT_LENGTH = 300  # Increased for better quality threshold


@dataclass
class ScrapeResult:
    """Represents the results of scraping an article page."""

    success: bool
    text: str = ""
    error: str = ""
    url: str = ""
    canonical_url: str = ""
    is_article_deterministic: bool = False
    word_count: int = field(init=False, default=0)

    def __post_init__(self):
        self.word_count = len(self.text.split()) if self.text else 0


def _is_valid_url(url: str) -> bool:
    """Check if the URL is valid and has a scheme/netloc."""
    try:
        if not url.startswith(("http://", "https://")):
            return False
        return len(url.split("://")[1].split("/")[0]) > 0
    except Exception:
        return False


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace: remove excessive newlines and spaces."""
    if not text:
        return ""
    # Replace 3+ newlines with 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Replace multiple spaces with one
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _deduplicate_paragraphs(text: str) -> str:
    """Simple paragraph-level deduplication to remove repeated snippets."""
    if not text:
        return ""
    paragraphs = text.split("\n\n")
    seen = set()
    unique_paragraphs = []
    for p in paragraphs:
        p_clean = p.strip()
        if not p_clean:
            continue
        if p_clean not in seen:
            unique_paragraphs.append(p_clean)
            seen.add(p_clean)
    return "\n\n".join(unique_paragraphs)


def check_is_article_deterministic(text: str, soup: BeautifulSoup) -> bool:
    """
    Deterministic check if the content looks like an article.
    Heuristics: word count, paragraph count, presence of <article> or ld+json.
    """
    if not text or len(text) < _MIN_TEXT_LENGTH:
        return False

    words = text.split()
    if len(words) < 100:
        return False

    # Check for ld+json NewsArticle/Article
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        try:
            content = script.string.lower()
            if "newsarticle" in content or '"article"' in content:
                return True
        except Exception:
            continue

    # Paragraph density check
    paragraphs = [p for p in text.split("\n\n") if len(p.split()) > 10]
    if len(paragraphs) >= 3:
        return True

    return False


def _extract_text_enhanced(
    html_content: str, url: str
) -> tuple[str, str, bool, BeautifulSoup]:
    """
    Extract text using trafilatura (primary) with BS4 fallback.
    Returns (text, canonical_url, is_article_deterministic, soup).
    """
    soup = BeautifulSoup(html_content, "lxml")

    # 1. Trafilatura extraction (highly robust) — Lazy-loaded to avoid ~1s startup delay
    import trafilatura  # noqa: PLC0415

    text = trafilatura.extract(
        html_content,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_precision=True,
    )

    # 2. Manual Fallback logic if trafilatura fails or returns too little
    if not text or len(text) < _MIN_TEXT_LENGTH:
        # Re-initialize clean soup for fallback
        for tag in soup.find_all(_NOISE_TAGS):
            tag.decompose()

        # Priority containers
        for selector in (
            "article",
            "main",
            '[role="main"]',
            ".article-body",
            "#content",
        ):
            container = soup.select_one(selector)
            if container:
                text = container.get_text(separator="\n\n", strip=True)
                if len(text) >= _MIN_TEXT_LENGTH:
                    break

        if not text:
            body = soup.find("body")
            text = body.get_text(separator="\n\n", strip=True) if body else ""

    # 3. Post-processing
    text = _normalize_whitespace(text)
    text = _deduplicate_paragraphs(text)

    # 4. Canonical URL
    canonical = ""
    link_canon = soup.find("link", rel="canonical")
    if link_canon and link_canon.get("href"):
        canonical = link_canon["href"]
    else:
        # Fallback to metadata
        meta_og = soup.find("meta", property="og:url")
        if meta_og:
            canonical = meta_og.get("content", "")

    # 5. Deterministic check
    is_article = check_is_article_deterministic(text, soup)

    return text, canonical, is_article, soup


def fetch_article_text(url: str) -> ScrapeResult:
    """
    Fetch a URL and extract the readable text content.

    Returns a ScrapeResult with:
    - success=True  + text  if extraction succeeded
    - success=False + error if something went wrong
    """
    url = url.strip()

    if not _is_valid_url(url):
        return ScrapeResult(
            success=False,
            error=f"Invalid URL: '{url}'. URL must start with http:// or https://",
            url=url,
        )

    try:
        headers = {"User-Agent": _USER_AGENT}
        response = requests.get(url, headers=headers, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ScrapeResult(
                success=False,
                error=f"URL returned non-HTML content ({content_type}). Only web pages are supported.",
                url=url,
            )

        text, canonical, is_article, soup = _extract_text_enhanced(
            response.content, url
        )

        if not text or len(text) < _MIN_TEXT_LENGTH:
            return ScrapeResult(
                success=False,
                error=(
                    f"Could not extract enough text from the page (len={len(text) if text else 0}). "
                    "The site may block scraping or require JavaScript to render."
                ),
                url=url,
            )

        logger.info("Scraped %s: %d words extracted", url, len(text.split()))
        return ScrapeResult(
            success=True,
            text=text,
            url=url,
            canonical_url=canonical,
            is_article_deterministic=is_article,
        )

    except requests.exceptions.ConnectionError:
        return ScrapeResult(
            success=False,
            error=f"Could not connect to '{url}'. Please check the URL and your internet connection.",
            url=url,
        )
    except requests.exceptions.Timeout:
        return ScrapeResult(
            success=False,
            error=f"Request timed out after {_REQUEST_TIMEOUT}s. The server may be slow or unreachable.",
            url=url,
        )
    except requests.exceptions.HTTPError as e:
        return ScrapeResult(
            success=False,
            error=f"HTTP error: {e}",
            url=url,
        )
    except Exception as e:
        logger.exception("Unexpected scraper error for %s", url)
        return ScrapeResult(
            success=False,
            error=f"Unexpected error: {e}",
            url=url,
        )


# ── Portal headline scraping ─────────────────────────────────────────────────

# Config: for each portal define the homepage URL and URL path fragments
# that identify article links (vs. navigation, category pages, etc.)
PORTAL_CONFIGS: dict[str, dict] = {
    "TVN24": {
        "homepage": "https://tvn24.pl",
        "article_path_fragments": [
            "/polska/",
            "/swiat/",
            "/biznes/",
            "/sport/",
            "/tvnmeteo/",
            "/kultura/",
            "/technologie/",
        ],
        "min_title_len": 25,
        "max_headlines": 10,
    },
    "Eurosport": {
        "homepage": "https://eurosport.tvn24.pl",
        "article_path_fragments": [
            "/pilka-nozna/",
            "/tenis/",
            "/skoki-narciarskie/",
            "/formula-1/",
            "/koszykowka/",
            "/biathlon/",
            "/narciarstwo-alpejskie/",
            "/biegi-narciarskie/",
            "/lekkoatletyka/",
            "/siatkowka/",
            "/boks/",
            "/igrzyska-olimpijskie/",
            "/motorcycle-racing/",
        ],
        "min_title_len": 30,
        "max_headlines": 10,
    },
}

_MAX_HEADLINES = 25  # max articles to return per portal


def fetch_portal_headlines(portal_name: str) -> list[dict]:
    """
    Scrape article headlines from a supported portal's homepage.

    Returns a list of dicts: [{"title": str, "url": str}, ...]
    Returns [] on failure (logs the error).
    """
    config = PORTAL_CONFIGS.get(portal_name)
    if not config:
        logger.error("Unknown portal: %s", portal_name)
        return []

    homepage = config["homepage"]
    path_fragments = config["article_path_fragments"]
    min_title_len = config["min_title_len"]
    max_limit = config.get("max_headlines", _MAX_HEADLINES)

    try:
        headers = {"User-Agent": _USER_AGENT}
        response = requests.get(homepage, headers=headers, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "lxml")
    except Exception as e:
        logger.error("Failed to fetch %s homepage: %s", portal_name, e)
        return []

    seen_urls: set[str] = set()
    results: list[dict] = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"].strip()

        # Resolve relative URLs
        full_url = urljoin(homepage, href)

        # Must be on the same domain
        if not full_url.startswith(homepage):
            continue

        # Must match one of the article path fragments
        path = full_url.replace(homepage, "")
        if not any(frag in path for frag in path_fragments):
            continue

        # Deduplicate
        if full_url in seen_urls:
            continue

        # Extract title — prefer heading inside the link, else link text
        heading = a_tag.find(["h1", "h2", "h3", "h4"])
        title = (
            heading.get_text(strip=True)
            if heading
            else a_tag.get_text(separator=" ", strip=True)
        )

        # Filter out non-title text (too short, all-caps nav labels, etc.)
        if len(title) < min_title_len:
            continue

        seen_urls.add(full_url)
        results.append({"title": title, "url": full_url})

        if len(results) >= max_limit:
            break

    logger.info(
        "fetch_portal_headlines(%s): %d articles found", portal_name, len(results)
    )
    return results
