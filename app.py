import streamlit as st
import time
import logging
import hashlib
from src.core.llm_provider import GeminiProvider, OpenAIProvider
from src.core.config import DEFAULT_LLM_PROVIDER, ENABLE_HEADLINE_EVALUATOR
from src.utils.telemetry import (
    log_generation,
    get_generation_logs,
    get_evaluation_logs,
    load_persistent_cache,
    save_to_persistent_cache,
)
from src.utils.scraper import fetch_article_text, fetch_portal_headlines, PORTAL_CONFIGS
from src.data.dummy import DUMMY_ARTICLE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Nagłówek AI",
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
if "portal_original_headline" not in st.session_state:
    st.session_state.portal_original_headline = None
if "portal_article_text" not in st.session_state:
    st.session_state.portal_article_text = None
# Two-phase rendering & URL caching
if "article_cache" not in st.session_state:
    st.session_state.article_cache = load_persistent_cache()  # url/hash -> result dict
if "pending_paste_assessment" not in st.session_state:
    st.session_state.pending_paste_assessment = (
        None  # PackagingOutput awaiting assessment
    )
if "pending_url_assessment" not in st.session_state:
    st.session_state.pending_url_assessment = None  # (result_dict, page_text, url)
if "pending_portal_assessment" not in st.session_state:
    st.session_state.pending_portal_assessment = None  # (result_dict, page_text, url)


def load_dummy():
    st.session_state.article_input = DUMMY_ARTICLE


def _get_provider():
    """Return the configured LLM provider and a display name for telemetry."""
    if DEFAULT_LLM_PROVIDER == "openai":
        return OpenAIProvider(), "gpt-5-nano-2025-08-07"
    return GeminiProvider(), "gemini-3-flash-preview"


def _render_flag_chip(flag: str) -> str:
    """Return an emoji-prefixed flag label for display."""
    if flag == "none":
        return "🟢 brak ryzyk"
    labels = {
        "clickbait_risk": "🔴 clickbait risk",
        "possible_unsupported_claim": "🔴 niezweryfikowane twierdzenie",
        "too_vague": "🟡 zbyt ogólny",
        "too_long": "🟡 zbyt długi",
        "banned_phrase_detected": "🔴 zakazana fraza",
        "duplicate_like_other_headline": "🟠 podobny do innego",
    }
    return labels.get(flag, f"⚠️ {flag}")


def _heuristic_ctr(headline: str) -> int:
    """Simple CTR heuristic: word count optimality + numbers/questions."""
    import re as _re

    words = headline.split()
    n = len(words)
    if 6 <= n <= 10:
        length_score = 100
    elif n < 6:
        length_score = max(30, 50 + n * 8)
    else:
        length_score = max(30, 100 - (n - 10) * 5)
    bonus = (
        (15 if _re.search(r"\d", headline) else 0)
        + (10 if "?" in headline else 0)
        + (5 if "!" in headline else 0)
    )
    return min(100, int(length_score * 0.7 + bonus + 15))


def _heuristic_clarity(headline: str) -> int:
    """Simple Clarity heuristic: shorter = clearer, penalize ellipsis/many commas."""
    words = headline.split()
    n = len(words)
    base = max(40, 100 - max(0, n - 5) * 3)
    if "..." in headline or "\u2026" in headline:
        base -= 10
    if headline.count(",") > 2:
        base -= 5
    return max(0, min(100, base))


def _render_original_headline_assessment(headline: str, article_text: str) -> None:
    """Render a heuristic + ML assessment card for the original portal headline."""
    from src.services.headline_quality_heuristics import assess_headline_heuristics

    results = assess_headline_heuristics([headline], article_text or "")
    if not results:
        return
    hr = results[0]
    credibility = max(0, 100 - hr.clickbait_score)
    seo = hr.seo_fit_score
    ctr = _heuristic_ctr(headline)
    clarity = _heuristic_clarity(headline)
    overall = round((ctr + clarity + seo + credibility) / 4)
    flags = hr.flags or ["none"]

    with st.container():
        st.markdown(f"#### {chr(0x1F4F0)} Oryginalny nagłówek portalu")
        st.markdown(f"> **{headline}**")
        col_ctr, col_cla, col_seo, col_cred, col_ovr = st.columns(5)
        col_ctr.metric("CTR", f"{ctr}/100")
        col_cla.metric("Clarity", f"{clarity}/100")
        col_seo.metric("SEO fit", f"{seo}/100")
        col_cred.metric(f"{chr(0x1F6E1)} Credibility", f"{credibility}/100")
        col_ovr.metric(f"{chr(0x2B50)} Overall", f"{overall}/100")
        flag_chips = " &nbsp; ".join(_render_flag_chip(f) for f in flags)
        st.markdown(f"<small>{flag_chips}</small>", unsafe_allow_html=True)
        st.divider()


def _render_headline_assessment_item(item, is_best: bool = False) -> None:
    """Render a single headline assessment card (works for both dict and object)."""
    # Support both Pydantic objects and dicts (from URL/Portal mode raw dict)
    if isinstance(item, dict):
        h_style = item.get("headline_style", "N/A")
        headline = item.get("headline", "")
        scores = item.get("scores", {})
        ctr = (
            scores.get("ctr_potential", 50)
            if isinstance(scores, dict)
            else getattr(scores, "ctr_potential", 50)
        )
        clarity = (
            scores.get("clarity", 50)
            if isinstance(scores, dict)
            else getattr(scores, "clarity", 50)
        )
        seo = (
            scores.get("seo_fit", 50)
            if isinstance(scores, dict)
            else getattr(scores, "seo_fit", 50)
        )
        credibility = (
            scores.get("credibility", 50)
            if isinstance(scores, dict)
            else getattr(scores, "credibility", 50)
        )
        flags = item.get("risk_flags", ["none"])
        rationale = item.get("rationale", "")
    else:
        h_style = getattr(item, "headline_style", "N/A")
        headline = getattr(item, "headline", "")
        scores = getattr(item, "scores", None)
        ctr = getattr(scores, "ctr_potential", 50) if scores else 50
        clarity = getattr(scores, "clarity", 50) if scores else 50
        seo = getattr(scores, "seo_fit", 50) if scores else 50
        credibility = getattr(scores, "credibility", 50) if scores else 50
        flags = getattr(item, "risk_flags", ["none"])
        rationale = getattr(item, "rationale", "")

    style_badge = f"**`{h_style}`**"
    trophy_html = (
        "<span style='color: #FF4B4B; font-weight: bold;'> 🏆 NAJLEPSZY</span>"
        if is_best
        else ""
    )
    with st.container():
        st.markdown(
            f"{style_badge}{trophy_html} &nbsp; {headline}", unsafe_allow_html=True
        )

        # Scores row
        overall = round((ctr + clarity + seo + credibility) / 4)
        col_ctr, col_cla, col_seo, col_cred, col_ovr = st.columns(5)
        col_ctr.metric("CTR", f"{ctr}/100")
        col_cla.metric("Clarity", f"{clarity}/100")
        col_seo.metric("SEO fit", f"{seo}/100")
        col_cred.metric("🛡️ Credibility", f"{credibility}/100")
        col_ovr.metric("⭐ Overall", f"{overall}/100")

        # Risk flags
        flag_chips = " &nbsp; ".join(_render_flag_chip(f) for f in flags)
        st.markdown(f"<small>{flag_chips}</small>", unsafe_allow_html=True)

        # Rationale
        if rationale:
            st.caption(f"💡 {rationale}")

        st.divider()


def render_packaging_result(res, assessment_pending: bool = False) -> None:
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
        assessment = getattr(res, "headline_assessment", None)
    else:
        headlines = res.get("headlines", [])
        lead_summary = res.get("lead_summary", "")
        seo_tags = res.get("seo_tags", [])
        social = res.get("social_posts", {})
        x_post = social.get("x_twitter", "") if isinstance(social, dict) else ""
        fb_post = social.get("facebook", "") if isinstance(social, dict) else ""
        assessment = res.get("headline_assessment", None)

    st.write("### 📢 Nagłówki i ocena jakości")

    if assessment:
        # Find the index with the highest overall score for the trophy badge
        def _overall(item):
            if isinstance(item, dict):
                s = item.get("scores", {})
                ctr = (
                    s.get("ctr_potential", 50)
                    if isinstance(s, dict)
                    else getattr(s, "ctr_potential", 50)
                )
                cla = (
                    s.get("clarity", 50)
                    if isinstance(s, dict)
                    else getattr(s, "clarity", 50)
                )
                seo = (
                    s.get("seo_fit", 50)
                    if isinstance(s, dict)
                    else getattr(s, "seo_fit", 50)
                )
                cred = (
                    s.get("credibility", 50)
                    if isinstance(s, dict)
                    else getattr(s, "credibility", 50)
                )
            else:
                sc = getattr(item, "scores", None)
                ctr = getattr(sc, "ctr_potential", 50) if sc else 50
                cla = getattr(sc, "clarity", 50) if sc else 50
                seo = getattr(sc, "seo_fit", 50) if sc else 50
                cred = getattr(sc, "credibility", 50) if sc else 50
            return (ctr + cla + seo + cred) / 4

        best_idx = max(range(len(assessment)), key=lambda i: _overall(assessment[i]))

        # Sort toggle
        sort_by_score = st.toggle("📊 Sortuj od najwyższego Overall", value=False)
        display_order = (
            sorted(
                range(len(assessment)),
                key=lambda i: _overall(assessment[i]),
                reverse=True,
            )
            if sort_by_score
            else list(range(len(assessment)))
        )

        # Rich assessment view
        for idx in display_order:
            item = assessment[idx]
            _render_headline_assessment_item(item, is_best=(idx == best_idx))
    else:
        # Fallback: plain list (no assessment available)
        for i, h in enumerate(headlines):
            h_type = headline_types[i] if i < len(headline_types) else "Dodatkowy"
            st.markdown(f"**{h_type}:** {h}")

        if ENABLE_HEADLINE_EVALUATOR and not assessment_pending:
            st.warning(
                "⚠️ Ocena jakości nagłówków niedostępna. Sprawdź logi lub klucz API.",
                icon="⚠️",
            )

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


# ── Sidebar — Navigation ──────────────────────────────────────────────────
st.sidebar.title("Nawigacja")
view_mode = st.sidebar.radio(
    "Wybierz widok", ["🚀 Nagłówek AI", "📊 Statystyki Telemetrii"], index=0
)

if view_mode == "📊 Statystyki Telemetrii":
    st.title("📊 Statystyki i Telemetria")
    st.markdown("---")

    gen_logs = get_generation_logs()
    eval_logs = get_evaluation_logs()

    if not gen_logs and not eval_logs:
        st.info("Brak danych telemetrycznych do wyświetlenia.")
    else:
        # Generation Stats
        if gen_logs:
            st.subheader("🚀 Generowanie propozycji (Faza 1)")
            total_gen = len(gen_logs)
            success_gen = sum(1 for entry in gen_logs if entry.get("success"))
            avg_lat_gen = (
                sum(entry.get("latency_seconds", 0) for entry in gen_logs) / total_gen
                if total_gen
                else 0
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Łącznie zapytań", total_gen)
            c2.metric(
                "Skuteczność",
                f"{round(success_gen / total_gen * 100)}%" if total_gen else "0%",
            )
            c3.metric("Śr. Opóźnienie", f"{round(avg_lat_gen, 2)}s")

            # Chart-like display for latency
            st.caption("Ostatnie 20 zapytań (opóźnienie):")
            st.line_chart([entry.get("latency_seconds") for entry in gen_logs[-20:]])

        st.divider()

        # Evaluation Stats
        if eval_logs:
            st.subheader("⚖️ Ocena jakości nagłówków (Faza 2)")
            total_eval = len(eval_logs)
            success_eval = sum(1 for entry in eval_logs if entry.get("success"))
            avg_lat_eval = (
                sum(entry.get("latency_ms", 0) for entry in eval_logs) / total_eval
                if total_eval
                else 0
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Łącznie ocen", total_eval)
            c2.metric(
                "Skuteczność",
                f"{round(success_eval / total_eval * 100)}%" if total_eval else "0%",
            )
            c3.metric("Śr. Opóźnienie", f"{round(avg_lat_eval / 1000, 2)}s")

            # Chart-like display for latency
            st.caption("Ostatnie 20 ocen (opóźnienie):")
            st.line_chart(
                [entry.get("latency_ms", 0) / 1000 for entry in eval_logs[-20:]]
            )

            # Flags distribution
            all_flags = {}
            for entry in eval_logs:
                for flag, count in entry.get("flag_counts", {}).items():
                    all_flags[flag] = all_flags.get(flag, 0) + count

            if all_flags:
                st.caption("Wykryte ryzyka (nagłówki):")
                st.bar_chart(all_flags)

    st.stop()  # Stop rendering the main app if in stats mode

# ── Header ─────────────────────────────────────────────────────────────────
st.title("🗞️ Nagłówek AI")
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
                article_key = hashlib.md5(article.strip().encode("utf-8")).hexdigest()
                if article_key in st.session_state.article_cache:
                    st.session_state.last_result = st.session_state.article_cache[
                        article_key
                    ]
                    st.session_state.pending_paste_assessment = None
                    st.success("✅ Wynik z cache.")
                else:
                    # Phase 1 – generate packaging (fast, no assessment yet)
                    with st.spinner("Przygotowuję propozycje..."):
                        start_time = time.time()
                        provider, model_display = _get_provider()
                        result = provider.generate_packaging(
                            article, skip_assessment=True
                        )
                        latency = time.time() - start_time
                        log_generation(
                            article_length=len(article),
                            model_name=model_display,
                            success=result is not None,
                            latency=latency,
                        )
                    if result:
                        st.session_state.last_result = result
                        st.session_state.pending_paste_assessment = (result, article)
                        st.rerun()  # show packaging immediately before Phase 2
                    else:
                        st.session_state.pending_paste_assessment = None
                        st.error(
                            "Nie udało się wygenerować opakowania. "
                            "Sprawdź logi lub klucz API."
                        )

        # Top-of-column status banner while assessment is running
        _paste_pending = st.session_state.pending_paste_assessment is not None
        if _paste_pending:
            st.warning("⏳ Trwa ocena jakości nagłówków...")

        # Show packaging immediately (suppress 'unavailable' warning while pending)
        if st.session_state.last_result:
            render_packaging_result(
                st.session_state.last_result, assessment_pending=_paste_pending
            )
        else:
            st.info("Wklej artykuł i kliknij 'Generuj sugestie', aby zobaczyć wyniki.")

        # Phase 2 – blocking call goes BELOW the already-rendered packaging
        if _paste_pending:
            _pending_result, _pending_article = (
                st.session_state.pending_paste_assessment
            )
            provider, _ = _get_provider()
            assessed = provider.assess_packaging(
                _pending_result, _pending_article, source_mode="paste"
            )

            st.session_state.last_result = assessed
            article_key = hashlib.md5(
                _pending_article.strip().encode("utf-8")
            ).hexdigest()
            st.session_state.article_cache[article_key] = assessed
            save_to_persistent_cache(article_key, assessed)
            st.session_state.pending_paste_assessment = None
            st.rerun()  # re-render with assessment scores

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
                url_key = url_input.strip()
                if url_key in st.session_state.article_cache:
                    # Cache hit – instant result, no LLM call needed
                    st.session_state.url_result = st.session_state.article_cache[
                        url_key
                    ]
                    st.session_state.pending_url_assessment = None
                    st.success("✅ Wynik z cache — analiza natychmiastowa.")
                else:
                    # Phase 1 – Scrape
                    with st.spinner("⏳ Pobieranie strony..."):
                        scrape = fetch_article_text(url_key)

                    if not scrape.success:
                        st.error(f"❌ Nie udało się pobrać strony: {scrape.error}")
                        st.session_state.url_result = None
                        st.session_state.pending_url_assessment = None
                    else:
                        st.success(
                            f"✅ Pobrano {scrape.word_count:,} słów ze strony. "
                            "Wysyłam do modelu…"
                        )

                        # Phase 1 – LLM packaging
                        with st.spinner("🤖 Przygotowuję propozycje..."):
                            start_time = time.time()
                            provider, model_display = _get_provider()
                            result = provider.analyze_url_content(
                                scrape.text,
                                url=url_key,
                                skip_assessment=True,
                                is_article_confident=scrape.is_article_deterministic,
                            )
                            # Pass flags to result for UI
                            result["canonical_url"] = scrape.canonical_url
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
                            st.session_state.pending_url_assessment = None
                        else:
                            st.session_state.url_result = result
                            if result.get("is_article"):
                                st.session_state.pending_url_assessment = (
                                    result,
                                    scrape.text,
                                    url_key,
                                )
                            else:
                                st.session_state.pending_url_assessment = None

    with col_right_url:
        st.subheader("Wyniki analizy")

        _url_pending = st.session_state.pending_url_assessment is not None
        if _url_pending:
            st.warning("⏳ Trwa ocena jakości nagłówków...")

        if st.session_state.url_result:
            res = st.session_state.url_result
            if res.get("is_article"):
                render_packaging_result(res, assessment_pending=_url_pending)
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

        # Phase 2 – blocking call goes BELOW the already-rendered packaging
        if _url_pending:
            _r, _txt, _url = st.session_state.pending_url_assessment
            provider, _ = _get_provider()
            _r = provider.assess_url_result(_r, _txt, _url)

            st.session_state.url_result = _r
            st.session_state.article_cache[_url] = _r
            save_to_persistent_cache(_url, _r)
            st.session_state.pending_url_assessment = None
            st.rerun()  # re-render with assessment scores

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
                    st.session_state.portal_original_headline = article["title"]
                    st.session_state.portal_result = None

                    article_url = article["url"]
                    if article_url in st.session_state.article_cache:
                        # Cache hit – show instantly
                        st.session_state.portal_result = st.session_state.article_cache[
                            article_url
                        ]
                        st.session_state.pending_portal_assessment = None
                        st.rerun()
                    else:
                        # Phase 1 – scrape
                        with st.spinner("Pobieram artykuł…"):
                            scrape = fetch_article_text(article_url)

                        if not scrape.success:
                            st.error(
                                f"❌ Nie udało się pobrać artykułu: {scrape.error}"
                            )
                        else:
                            # Phase 1 – LLM packaging
                            with st.spinner("🤖 Przygotowuję propozycje…"):
                                start = time.time()
                                provider, model_display = _get_provider()
                                result = provider.analyze_url_content(
                                    scrape.text,
                                    url=article_url,
                                    skip_assessment=True,
                                    is_article_confident=scrape.is_article_deterministic,
                                )
                                # Pass flags to result for UI
                                result["canonical_url"] = scrape.canonical_url
                                latency = time.time() - start
                                log_generation(
                                    article_length=len(scrape.text),
                                    model_name=model_display,
                                    success=result.get("is_article", False),
                                    latency=latency,
                                )
                            st.session_state.portal_result = result
                            st.session_state.portal_article_text = scrape.text
                            if result.get("is_article"):
                                st.session_state.pending_portal_assessment = (
                                    result,
                                    scrape.text,
                                    article_url,
                                )
                            else:
                                st.session_state.pending_portal_assessment = None
                            st.rerun()

    with col_right_p:
        st.subheader("Wyniki analizy")

        if st.session_state.portal_selected_url:
            st.caption(f"🔗 {st.session_state.portal_selected_url}")

        _portal_pending = st.session_state.pending_portal_assessment is not None
        if _portal_pending:
            st.warning("⏳ Trwa ocena jakości nagłówków...")

        if st.session_state.portal_result:
            res = st.session_state.portal_result
            if res.get("is_article"):
                # Show original headline assessment if available
                orig_headline = st.session_state.get("portal_original_headline")
                orig_text = st.session_state.get("portal_article_text", "")
                if orig_headline:
                    _render_original_headline_assessment(orig_headline, orig_text or "")
                render_packaging_result(res, assessment_pending=_portal_pending)
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

        # Phase 2 – blocking call goes BELOW the already-rendered packaging
        if _portal_pending:
            _r, _txt, _url = st.session_state.pending_portal_assessment
            provider, _ = _get_provider()
            _r = provider.assess_url_result(_r, _txt, _url)

            st.session_state.portal_result = _r
            st.session_state.article_cache[_url] = _r
            save_to_persistent_cache(_url, _r)
            st.session_state.pending_portal_assessment = None
            st.rerun()  # re-render with assessment scores
