import streamlit as st
import time
import logging
from src.core.llm_provider import OpenAIProvider
from src.utils.telemetry import log_generation
from src.utils.scraper import fetch_article_text, fetch_portal_headlines, PORTAL_CONFIGS
from src.data.dummy import DUMMY_ARTICLE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Projekt AntiGravity | AI Copilot Redakcyjny",
    page_icon="🗞️",
    layout="wide",
)

# Initialize Session State
if "article_input" not in st.session_state:
    st.session_state.article_input = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "url_result" not in st.session_state:
    st.session_state.url_result = None
# Portal browser tab state
if "portal_headlines" not in st.session_state:
    st.session_state.portal_headlines = []
if "portal_selected_url" not in st.session_state:
    st.session_state.portal_selected_url = None
if "portal_result" not in st.session_state:
    st.session_state.portal_result = None
if "portal_name" not in st.session_state:
    st.session_state.portal_name = list(PORTAL_CONFIGS.keys())[0]


def load_dummy():
    st.session_state.article_input = DUMMY_ARTICLE


def _get_provider():
    """Return the OpenAI provider."""
    return OpenAIProvider(), "gpt-5-mini-2025-08-07"


def render_packaging_result(res):
    """Render a PackagingOutput or a dict with packaging keys."""
    headline_types = ["Pilny", "Pytanie", "Liczbowy", "Luka ciekawości", "Bezpośredni"]

    # Support both PackagingOutput objects and raw dicts
    if hasattr(res, "headlines"):
        headlines = res.headlines
        lead_summary = res.lead_summary
        seo_tags = res.seo_tags
        social = res.social_posts
        x_post = social.x_twitter
        fb_post = social.facebook
    else:
        headlines = res.get("headlines", [])
        lead_summary = res.get("lead_summary", "")
        seo_tags = res.get("seo_tags", [])
        social = res.get("social_posts", {})
        x_post = social.get("x_twitter", "") if isinstance(social, dict) else ""
        fb_post = social.get("facebook", "") if isinstance(social, dict) else ""

    st.write("### 📢 Nagłówki")
    for i, h in enumerate(headlines):
        h_type = headline_types[i] if i < len(headline_types) else "Dodatkowy"
        st.markdown(f"**{h_type}:** {h}")

    st.divider()

    st.write("### ✍️ Lead (zajawka)")
    st.info(lead_summary)

    st.divider()

    col_seo, col_soc = st.columns([1, 1])
    with col_seo:
        st.write("### 🔍 Tagi SEO")
        st.write(", ".join(seo_tags))

    with col_soc:
        st.write("### 📱 Posty w mediach społecznościowych")
        with st.expander("Pokaż treść postów", expanded=True):
            st.markdown(f"**X (Twitter):**\n{x_post}")
            st.markdown("---")
            st.markdown(f"**Facebook:**\n{fb_post}")


# ── Header ─────────────────────────────────────────────────────────────────
st.title("🗞️ AI Copilot Redakcyjny")
st.markdown("---")

tab_paste, tab_url, tab_portal = st.tabs(
    ["✍️ Wklej artykuł", "🔗 Pobierz z URL", "📰 Przeglądaj portal"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Paste Article
# ══════════════════════════════════════════════════════════════════════════════
with tab_paste:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.subheader("Treść artykułu")
        article = st.text_area(
            "Wklej artykuł tutaj:",
            value=st.session_state.article_input,
            height=400,
            placeholder="Dawno, dawno temu w pewnej redakcji...",
        )

        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            if st.button("Wczytaj przykładowy artykuł", use_container_width=True):
                load_dummy()
                st.rerun()

        with col_btn2:
            generate_clicked = st.button(
                "🚀 Generuj sugestie", type="primary", use_container_width=True
            )

    with col_right:
        st.subheader("Sugestie redakcyjne")

        if generate_clicked:
            if not article.strip():
                st.warning("Proszę najpierw wkleić artykuł.")
            else:
                with st.spinner("Analizuję i pakuję artykuł..."):
                    start_time = time.time()
                    provider, model_display = _get_provider()
                    result = provider.generate_packaging(article)
                    latency = time.time() - start_time

                    log_generation(
                        article_length=len(article),
                        model_name=model_display,
                        success=result is not None,
                        latency=latency,
                    )

                    if result:
                        st.session_state.last_result = result
                    else:
                        st.error(
                            "Nie udało się wygenerować opakowania. "
                            "Sprawdź logi lub klucz API (włącz tryb Demo jeśli nie masz klucza)."
                        )

        if st.session_state.last_result:
            render_packaging_result(st.session_state.last_result)
        else:
            st.info("Wklej artykuł i kliknij 'Generuj sugestie', aby zobaczyć wyniki.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Scrape from URL
# ══════════════════════════════════════════════════════════════════════════════
with tab_url:
    col_left_url, col_right_url = st.columns([1, 1], gap="large")

    with col_left_url:
        st.subheader("URL artykułu")
        url_input = st.text_input(
            "Wklej adres URL artykułu:",
            placeholder="https://www.tvn24.pl/...",
        )

        fetch_clicked = st.button(
            "🔍 Pobierz i przeanalizuj", type="primary", use_container_width=True
        )

        if fetch_clicked:
            if not url_input.strip():
                st.warning("Proszę najpierw podać adres URL.")
            else:
                # Step 1 – Scrape
                with st.spinner("⏳ Pobieranie strony..."):
                    scrape = fetch_article_text(url_input.strip())

                if not scrape.success:
                    st.error(f"❌ Nie udało się pobrać strony: {scrape.error}")
                    st.session_state.url_result = None
                else:
                    st.success(
                        f"✅ Pobrano {scrape.word_count:,} słów ze strony. "
                        "Wysyłam do modelu…"
                    )

                    # Step 2 – LLM classification + packaging
                    with st.spinner("🤖 Analizuję treść za pomocą modelu..."):
                        start_time = time.time()
                        provider, model_display = _get_provider()
                        result = provider.analyze_url_content(
                            scrape.text, url=url_input.strip()
                        )
                        latency = time.time() - start_time

                        log_generation(
                            article_length=len(scrape.text),
                            model_name=model_display,
                            success=result.get("is_article", False),
                            latency=latency,
                        )

                    if "error" in result and not result.get("is_article"):
                        st.error(f"❌ Błąd modelu: {result['error']}")
                        st.session_state.url_result = None
                    else:
                        st.session_state.url_result = result

    with col_right_url:
        st.subheader("Wyniki analizy")

        if st.session_state.url_result:
            res = st.session_state.url_result

            if res.get("is_article"):
                render_packaging_result(res)
            else:
                reason = res.get("reason", "Brak podanego powodu.")
                st.warning(
                    f"⚠️ **Ta strona nie wygląda jak artykuł informacyjny.**\n\n"
                    f"**Powód:** {reason}"
                )
        else:
            st.info(
                "Podaj adres URL i kliknij 'Pobierz i przeanalizuj', aby rozpocząć."
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Portal Browser
# ══════════════════════════════════════════════════════════════════════════════
with tab_portal:
    col_left_p, col_right_p = st.columns([1, 1], gap="large")

    with col_left_p:
        st.subheader("Wybierz portal")

        portal_name = st.radio(
            "Portal",
            options=list(PORTAL_CONFIGS.keys()),
            horizontal=True,
            key="portal_radio",
        )

        if st.button("📥 Pobierz nagłówki", use_container_width=True):
            st.session_state.portal_result = None
            st.session_state.portal_selected_url = None
            with st.spinner(f"Pobieranie nagłówków z {portal_name}..."):
                headlines = fetch_portal_headlines(portal_name)
            if not headlines:
                st.error(
                    f"Nie udało się pobrać nagłówków z {portal_name}. "
                    "Portal może blokować scraping lub wymagać JavaScriptu."
                )
                st.session_state.portal_headlines = []
            else:
                st.session_state.portal_headlines = headlines
                st.session_state.portal_name = portal_name
                st.success(f"Znaleziono {len(headlines)} artykułów.")

        # Article list
        if st.session_state.portal_headlines:
            st.markdown(f"**Artykuły z {st.session_state.portal_name}:**")
            for i, article in enumerate(st.session_state.portal_headlines):
                if st.button(
                    article["title"],
                    key=f"portal_article_{i}",
                    use_container_width=True,
                ):
                    st.session_state.portal_selected_url = article["url"]
                    st.session_state.portal_result = None

                    # Fetch article text and run analysis immediately
                    with st.spinner("Pobieram artykuł…"):
                        scrape = fetch_article_text(article["url"])

                    if not scrape.success:
                        st.error(f"❌ Nie udało się pobrać artykułu: {scrape.error}")
                    else:
                        with st.spinner("🤖 Analizuję artykuł…"):
                            start = time.time()
                            provider, model_display = _get_provider()
                            result = provider.analyze_url_content(
                                scrape.text, url=article["url"]
                            )
                            latency = time.time() - start
                            log_generation(
                                article_length=len(scrape.text),
                                model_name=model_display,
                                success=result.get("is_article", False),
                                latency=latency,
                            )
                        st.session_state.portal_result = result
                        st.rerun()

    with col_right_p:
        st.subheader("Wyniki analizy")

        if st.session_state.portal_selected_url:
            st.caption(f"🔗 {st.session_state.portal_selected_url}")

        if st.session_state.portal_result:
            res = st.session_state.portal_result
            if res.get("is_article"):
                render_packaging_result(res)
            else:
                reason = res.get("reason", "Brak podanego powodu.")
                st.warning(
                    f"⚠️ **Ta strona nie wygląda jak artykuł informacyjny.**\n\n"
                    f"**Powód:** {reason}"
                )
        elif st.session_state.portal_headlines:
            st.info("Kliknij tytuł artykułu po lewej stronie, aby go przeanalizować.")
        else:
            st.info("Wybierz portal i kliknij 'Pobierz nagłówki', aby rozpocząć.")
