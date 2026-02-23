"""
Scraper utility: fetches a URL and extracts readable article text,
and scrapes article headlines from supported news portals.
"""

import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin

import requests
from bs4 import BeautifulSoup

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

# Realistic browser User-Agent to avoid bot-blocking
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

_REQUEST_TIMEOUT = 10  # seconds
_MIN_TEXT_LENGTH = 200  # chars – below this we consider extraction failed


@dataclass
class ScrapeResult:
    success: bool
    text: str = ""
    error: str = ""
    url: str = ""
    word_count: int = field(init=False, default=0)

    def __post_init__(self):
        self.word_count = len(self.text.split()) if self.text else 0


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _extract_text(soup: BeautifulSoup) -> str:
    """Remove noise tags, then prefer <article>/<main>, fall back to <body>."""
    for tag in soup.find_all(_NOISE_TAGS):
        tag.decompose()

    # Priority containers
    for selector in (
        "article",
        "main",
        '[role="main"]',
        ".article-body",
        ".post-content",
        ".entry-content",
        "#content",
    ):
        container = soup.select_one(selector)
        if container:
            text = container.get_text(separator="\n", strip=True)
            if len(text) >= _MIN_TEXT_LENGTH:
                return text

    # Last resort: body
    body = soup.find("body")
    if body:
        return body.get_text(separator="\n", strip=True)

    return soup.get_text(separator="\n", strip=True)


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

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            return ScrapeResult(
                success=False,
                error=f"URL returned non-HTML content ({content_type}). Only web pages are supported.",
                url=url,
            )

        soup = BeautifulSoup(response.content, "lxml")
        text = _extract_text(soup)

        if len(text) < _MIN_TEXT_LENGTH:
            return ScrapeResult(
                success=False,
                error=(
                    "Could not extract enough text from the page. "
                    "The site may block scraping or require JavaScript to render."
                ),
                url=url,
            )

        logger.info("Scraped %s: %d words extracted", url, len(text.split()))
        return ScrapeResult(success=True, text=text, url=url)

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

        if len(results) >= _MAX_HEADLINES:
            break

    logger.info(
        "fetch_portal_headlines(%s): %d articles found", portal_name, len(results)
    )
    return results
