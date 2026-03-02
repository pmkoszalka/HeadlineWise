"""
Portal connector pattern.

Add a new portal by sub-classing PortalConnector and registering
an instance in PORTAL_REGISTRY — no changes elsewhere required.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
_REQUEST_TIMEOUT = 10  # seconds
_MAX_HEADLINES = 25


class PortalConnector(ABC):
    """
    Interface every portal connector must implement.

    Sub-classes only need to set the class-level attributes;
    the shared `fetch_headlines()` method handles the actual scraping.
    Override `fetch_headlines()` for portals that need custom logic.
    """

    #: Human-readable name shown in the UI
    name: str

    #: Homepage URL to scrape
    homepage: str

    #: URL path fragments that identify article links
    article_path_fragments: list[str]

    #: Minimum title length to filter nav labels and category names
    min_title_len: int = 25

    #: Maximum number of headlines to fetch
    max_headlines: int = _MAX_HEADLINES

    def fetch_headlines(self) -> list[dict]:
        """
        Scrape the portal homepage and return article headlines.

        Returns:
            list of {"title": str, "url": str} dicts, up to _MAX_HEADLINES.
            Empty list on any failure.
        """
        try:
            response = requests.get(
                self.homepage,
                headers={"User-Agent": _USER_AGENT},
                timeout=_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "lxml")
        except Exception as exc:
            logger.error("[%s] Failed to fetch homepage: %s", self.name, exc)
            return []

        seen: set[str] = set()
        results: list[dict] = []

        for a_tag in soup.find_all("a", href=True):
            full_url = urljoin(self.homepage, a_tag["href"].strip())

            if not full_url.startswith(self.homepage):
                continue

            path = full_url.replace(self.homepage, "")
            if not any(frag in path for frag in self.article_path_fragments):
                continue

            if full_url in seen:
                continue

            heading = a_tag.find(["h1", "h2", "h3", "h4"])
            title = (
                heading.get_text(strip=True)
                if heading
                else a_tag.get_text(separator=" ", strip=True)
            )

            if len(title) < self.min_title_len:
                continue

            seen.add(full_url)
            results.append({"title": title, "url": full_url})

            if len(results) >= self.max_headlines:
                break

        logger.info("[%s] %d articles found", self.name, len(results))
        return results

    @abstractmethod
    def _marker(self) -> None:
        """Forces sub-class declaration (prevents bare instantiation)."""


# ── Concrete connectors ───────────────────────────────────────────────────────


class TVN24Connector(PortalConnector):
    """Retrieves news articles from the TVN24 portal."""

    name = "TVN24"
    homepage = "https://tvn24.pl"
    article_path_fragments = [
        "/polska/",
        "/swiat/",
        "/biznes/",
        "/sport/",
        "/tvnmeteo/",
        "/kultura/",
        "/technologie/",
    ]
    min_title_len = 25
    max_headlines = 10

    def _marker(self) -> None:
        pass


class EurosportConnector(PortalConnector):
    """Retrieves sports articles from the Eurosport TVN24 portal."""

    name = "Eurosport"
    homepage = "https://eurosport.tvn24.pl"
    article_path_fragments = [
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
    ]
    min_title_len = 30
    max_headlines = 10

    def _marker(self) -> None:
        pass


# ── Registry ──────────────────────────────────────────────────────────────────
# Maps UI display name → connector instance.
# To add a new portal: create a PortalConnector sub-class and add it here.

PORTAL_REGISTRY: dict[str, PortalConnector] = {
    TVN24Connector.name: TVN24Connector(),
    EurosportConnector.name: EurosportConnector(),
}
