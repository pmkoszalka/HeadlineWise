"""
tests/test_headline_assessment.py

Unit + integration tests for the headline quality assessment feature.
Run with:
    pytest tests/test_headline_assessment.py -v
"""

from __future__ import annotations

from src.core.llm_provider import MockProvider
from src.services.headline_quality_evaluator import LLMEvalResult
from src.services.headline_quality_heuristics import (
    HeuristicResult,
    assess_headline_heuristics,
    check_style_violations,
    compute_seo_fit_score,
    contains_banned_phrase,
    detect_duplicate_like_headlines,
    extract_keyword_candidates,
    is_too_long,
    is_too_vague,
)
from src.services.headline_quality_merge import _merge_flags, merge_assessments

# ---------------------------------------------------------------------------
# Heuristics â€” unit tests
# ---------------------------------------------------------------------------


class TestContainsBannedPhrase:
    def test_exact_match(self):
        assert contains_banned_phrase("Breaking news today", ["breaking news"])

    def test_case_insensitive(self):
        assert contains_banned_phrase("BREAKING NEWS update", ["breaking news"])

    def test_no_match(self):
        assert not contains_banned_phrase("Regular story", ["breaking news"])

    def test_empty_banned_list(self):
        assert not contains_banned_phrase("Any headline", [])

    def test_substring_match(self):
        # banned phrase is a substring
        assert contains_banned_phrase("This is clickbait material", ["clickbait"])


class TestIsTooLong:
    def test_short_headline_not_flagged(self):
        assert not is_too_long("Fed raises rates", max_chars=80, max_words=14)

    def test_exactly_at_char_limit_ok(self):
        # 79 chars â€” under limit
        headline = "a" * 79
        assert not is_too_long(headline, max_chars=80, max_words=14)

    def test_exceeds_char_limit(self):
        headline = "a" * 81
        assert is_too_long(headline, max_chars=80, max_words=14)

    def test_exactly_at_word_limit_ok(self):
        headline = " ".join(["word"] * 14)
        assert not is_too_long(headline, max_chars=80, max_words=14)

    def test_exceeds_word_limit(self):
        headline = " ".join(["word"] * 15)
        assert is_too_long(headline, max_chars=80, max_words=14)

    def test_both_under_limits(self):
        assert not is_too_long("Short title", max_chars=80, max_words=14)


class TestStyleViolations:
    def test_shouting_flagged(self):
        flags = check_style_violations("PILNE: WIELKIE LITERY")
        assert "krzyk_wersalikami" in flags

    def test_excessive_punctuation_flagged(self):
        assert "nadmierna_interpunkcja" in check_style_violations("Wow!!!")
        assert "nadmierna_interpunkcja" in check_style_violations("Really???")

    def test_excessive_ellipsis_flagged(self):
        assert "nadmierne_wielokropki" in check_style_violations(
            "Something happened...."
        )

    def test_normal_headline_no_flags(self):
        assert not check_style_violations("ZwykĹ‚y nagĹ‚Ăłwek bez krzyku")


class TestDetectDuplicateLikeHeadlines:
    def test_identical_headlines_both_flagged(self):
        headlines = ["Fed podnosi stopy", "Fed podnosi stopy", "CoĹ› innego"]
        result = detect_duplicate_like_headlines(headlines, threshold=0.78)
        assert result[0] is True
        assert result[1] is True
        assert result[2] is False

    def test_very_similar_both_flagged(self):
        headlines = [
            "Fed podnosi stopy procentowe o 0,5%",
            "Fed podnosi stopy procentowe o 0,5% dziĹ›",
        ]
        result = detect_duplicate_like_headlines(headlines, threshold=0.78)
        assert result[0] is True
        assert result[1] is True

    def test_dissimilar_headlines_not_flagged(self):
        headlines = [
            "Inflacja bije rekordy w Polsce",
            "Premier ogĹ‚asza nowe inwestycje",
            "Katastrofa ekologiczna na BaĹ‚tyku",
        ]
        result = detect_duplicate_like_headlines(headlines, threshold=0.78)
        assert not any(result.values())

    def test_single_headline_never_flagged(self):
        result = detect_duplicate_like_headlines(["Only one"])
        assert result == {0: False}


class TestExtractKeywordCandidates:
    def test_seo_tags_always_included(self):
        keywords = extract_keyword_candidates(
            "some article text", seo_tags=["inflacja", "Fed"]
        )
        assert "inflacja" in keywords
        assert "fed" in keywords  # lowercased

    def test_frequency_words_included(self):
        text = (
            "inflacja inflacja inflacja gospodarka gospodarka rynki rynki rynki rynki"
        )
        keywords = extract_keyword_candidates(text, seo_tags=None)
        assert "inflacja" in keywords
        assert keywords["inflacja"] == 1.0

    def test_weighted_seo_tags(self):
        keywords = extract_keyword_candidates("text", seo_tags=["inflacja"])
        assert keywords["inflacja"] == 2.0

    def test_returns_dict(self):
        result = extract_keyword_candidates("test artykuĹ‚", seo_tags=["tag"])
        assert isinstance(result, dict)


class TestComputeSeoFitScore:
    def test_no_keywords_returns_neutral(self):
        assert compute_seo_fit_score("Any headline", {}) == 50

    def test_perfect_keyword_coverage_high_score(self):
        # Keywords must be in lemma form (as produced by extract_keyword_candidates).
        # _lemmatize() lowercases all output: 'Fed' -> 'fed', 'stopy' -> 'stopa'
        keywords = {"fed": 2.0, "stopa": 2.0}
        headline = "Fed podnosi stopy procentowe"
        score = compute_seo_fit_score(headline, keywords)
        assert score == 100

    def test_no_keyword_overlap_low_score(self):
        keywords = {"inflacja": 1.0, "gospodarka": 1.0}
        headline = "Pogoda na weekend"
        score = compute_seo_fit_score(headline, keywords)
        assert score == 10  # 10 + 0/2 * 90

    def test_weighted_impact(self):
        # SEO tag (2.0) hit vs text word (1.0) hit in different headlines.
        # _lemmatize lowercases all: 'Fed' in headline -> lemma 'fed'
        keywords = {"fed": 2.0, "rynek": 1.0}
        score_fed = compute_seo_fit_score("CzeĹ›Ä‡ Fed dzisiaj", keywords)
        score_rynki = compute_seo_fit_score("CzeĹ›Ä‡ rynkach dzisiaj", keywords)
        assert score_fed > score_rynki

    def test_partial_coverage_medium_score(self):
        keywords = {"inflacja": 1.0, "fed": 1.0, "rynki": 1.0, "stopy": 1.0}
        headline = "Fed podnosi stopy"
        score = compute_seo_fit_score(headline, keywords)
        assert 10 < score < 100

    def test_score_clamped_to_100(self):
        keywords = {"a": 1.0}
        headline = "a b c"
        score = compute_seo_fit_score(headline, keywords)
        assert 0 <= score <= 100


class TestIsTooVague:
    def test_no_keyword_match_returns_true(self):
        assert is_too_vague("CoĹ› siÄ™ wydarzyĹ‚o", {"inflacja": 1.0})

    def test_keyword_present_returns_false(self):
        assert not is_too_vague("Fed podnosi stopy", {"fed": 1.0})

    def test_empty_keywords_returns_false(self):
        assert not is_too_vague("Some", {})

    def test_case_insensitive_match(self):
        # keywords are lowercase; headline uses upper case
        assert not is_too_vague("Inflacja rośnie", {"inflacja": 1.0})


class TestAssessHeadlineHeuristics:
    """Integration over the orchestrator."""

    def _make_headlines(self):
        return [
            "Fed podnosi stopy procentowe â€” rynki reagujÄ… spadkami",
            "Inflacja bije rekordy w Polsce po raz trzeci z rzÄ™du",
            "Premier ogĹ‚asza nowy pakiet stymulacyjny dla gospodarki",
            "CoĹ› siÄ™ staĹ‚o i to waĹĽne dla wszystkich mieszkaĹ„cĂłw kraju",
            "Fed podnosi stopy procentowe â€” rynki reagujÄ… spadkami",  # duplicate of [0]
        ]

    def test_returns_one_result_per_headline(self):
        headlines = self._make_headlines()
        results = assess_headline_heuristics(
            headlines=headlines,
            article_text="inflacja gospodarka rynki",
            seo_tags=["inflacja", "fed"],
        )
        assert len(results) == len(headlines)

    def test_duplicate_flagged(self):
        headlines = self._make_headlines()
        results = assess_headline_heuristics(
            headlines=headlines,
            article_text="inflacja gospodarka rynki stopy",
            seo_tags=["fed", "stopy"],
        )
        dup_result_0 = results[0]
        dup_result_4 = results[4]
        assert "duplicate_like_other_headline" in dup_result_0.flags
        assert "duplicate_like_other_headline" in dup_result_4.flags

    def test_too_long_headline_flagged(self):
        long_headline = "a " * 50  # 100 chars
        results = assess_headline_heuristics(
            headlines=[long_headline],
            article_text="some text about something",
        )
        assert "zbyt_dlugi" in results[0].flags

    def test_banned_phrase_flagged(self):
        results = assess_headline_heuristics(
            headlines=["SzokujÄ…ce wiadomoĹ›ci z rynkĂłw"],
            article_text="rynki finanse gospodarka",
            banned_phrases=["szokujÄ…ce"],
        )
        assert "banned_phrase_detected" in results[0].flags

    def test_seo_score_in_range(self):
        headlines = ["Fed podnosi stopy"]
        results = assess_headline_heuristics(
            headlines=headlines,
            article_text="Fed inflacja stopy rynki gospodarka",
            seo_tags=["fed", "stopy"],
        )
        assert 0 <= results[0].seo_fit_score <= 100


# ---------------------------------------------------------------------------
# Merge â€” unit tests
# ---------------------------------------------------------------------------


class TestMergeFlags:
    def test_none_removed_when_real_flags_exist(self):
        result = _merge_flags(["brak"], ["zbyt_ogolny"])
        assert "brak" not in result
        assert "zbyt_ogolny" in result

    def test_both_none_returns_none(self):
        result = _merge_flags(["brak"], ["brak"])
        assert result == ["brak"]

    def test_deduplication(self):
        result = _merge_flags(["zbyt_ogolny", "zbyt_dlugi"], ["zbyt_ogolny"])
        assert result.count("zbyt_ogolny") == 1

    def test_empty_inputs_return_none(self):
        result = _merge_flags([], [])
        assert result == ["brak"]

    def test_heuristic_flags_come_first(self):
        result = _merge_flags(["zbyt_dlugi"], ["ryzyko_clickbait"])
        assert result[0] == "zbyt_dlugi"
        assert result[1] == "ryzyko_clickbait"


class TestMergeAssessments:
    def _heuristic(self, idx: int, flags=None, seo=60, clickbait_score=0):
        return HeuristicResult(
            headline_index=idx,
            flags=flags or [],
            seo_fit_score=seo,
            clickbait_score=clickbait_score,
        )

    def _llm(self, idx: int, ctr=75, clarity=80, seo=70, flags=None, rationale="OK."):
        return LLMEvalResult(
            headline_index=idx,
            ctr_potential=ctr,
            clarity=clarity,
            seo_fit=seo,
            risk_flags=flags or ["brak"],
            rationale=rationale,
        )

    def test_happy_path_produces_correct_count(self):
        h_results = [self._heuristic(i) for i in range(5)]
        l_results = [self._llm(i) for i in range(5)]
        headlines = [f"Headline {i}" for i in range(5)]
        items = merge_assessments(h_results, l_results, headlines)
        assert len(items) == 5

    def test_llm_none_fallback_scores(self):
        h_results = [self._heuristic(0, seo=60)]
        items = merge_assessments(h_results, None, ["Headline 0"])
        assert items[0].scores.ctr_potential == 50
        assert items[0].scores.clarity == 50
        assert items[0].scores.seo_fit == 60  # heuristic-only

    def test_seo_averaged_when_llm_available(self):
        h_results = [self._heuristic(0, seo=60)]
        l_results = [self._llm(0, seo=80)]
        items = merge_assessments(h_results, l_results, ["Headline 0"])
        # avg(60, 80) = 70
        assert items[0].scores.seo_fit == 70

    def test_fallback_rationale_when_llm_none(self):
        h_results = [self._heuristic(0)]
        items = merge_assessments(h_results, None, ["Headline 0"])
        assert "LLM evaluator error" in items[0].rationale

    def test_flag_union_and_dedup(self):
        h_results = [self._heuristic(0, flags=["zbyt_dlugi"])]
        l_results = [self._llm(0, flags=["zbyt_dlugi", "ryzyko_clickbait"])]
        items = merge_assessments(h_results, l_results, ["Headline 0"])
        flags = items[0].risk_flags
        assert flags.count("zbyt_dlugi") == 1
        assert "ryzyko_clickbait" in flags

    def test_headline_style_assigned(self):
        h_results = [self._heuristic(i) for i in range(5)]
        items = merge_assessments(h_results, None, [f"H{i}" for i in range(5)])
        styles = [item.headline_style for item in items]
        assert "Pilny" in styles
        assert "Bezpośredni" in styles

    def test_scores_clamped_to_100(self):
        """Merge should never exceed 0â€“100."""
        h_results = [self._heuristic(0, seo=100)]
        l_results = [self._llm(0, ctr=100, clarity=100, seo=100)]
        items = merge_assessments(h_results, l_results, ["Headline 0"])
        assert items[0].scores.ctr_potential <= 100
        assert items[0].scores.clarity <= 100
        assert items[0].scores.seo_fit <= 100

    def test_penalties_applied(self):
        h_res = [self._heuristic(0, flags=["zbyt_dlugi"])]
        llm_res = [self._llm(0, ctr=90, clarity=90)]
        it = merge_assessments(h_res, llm_res, ["Short"])
        assert it[0].scores.ctr_potential <= 50
        assert it[0].scores.clarity <= 50

        # Test banned_phrase penalty
        h_res = [self._heuristic(0, flags=["banned_phrase_detected"])]
        llm_res = [self._llm(0, ctr=90, clarity=90, seo=90)]
        it = merge_assessments(h_res, llm_res, ["Banned"])
        assert it[0].scores.ctr_potential <= 20
        assert it[0].scores.clarity <= 20
        assert it[0].scores.seo_fit <= 20

        # Test ryzyko_clickbait penalty
        # Since we changed CTR penalty to be continuous based on clickbait_score: 80 - (score - 75) * 2
        # If clickbait_score is 85, max_ctr = 80 - (10) * 2 = 60
        h_res = [self._heuristic(0, flags=["ryzyko_clickbait"], clickbait_score=85)]
        llm_res = [self._llm(0, ctr=90)]
        it = merge_assessments(h_res, llm_res, ["Clickbait"])
        assert it[0].scores.ctr_potential == 60


# ---------------------------------------------------------------------------
# Integration â€” MockProvider
# ---------------------------------------------------------------------------


class TestMockProviderIntegration:
    def test_generate_packaging_has_assessment(self):
        provider = MockProvider()
        result = provider.generate_packaging("test article text")
        assert result is not None
        assert result.headline_assessment is not None
        assert len(result.headline_assessment) == 5

    def test_assessment_scores_in_range(self):
        provider = MockProvider()
        result = provider.generate_packaging("test")
        for item in result.headline_assessment:
            assert 0 <= item.scores.ctr_potential <= 100
            assert 0 <= item.scores.clarity <= 100
            assert 0 <= item.scores.seo_fit <= 100

    def test_assessment_flags_not_empty(self):
        provider = MockProvider()
        result = provider.generate_packaging("test")
        for item in result.headline_assessment:
            assert len(item.risk_flags) >= 1

    def test_assessment_rationale_not_empty(self):
        provider = MockProvider()
        result = provider.generate_packaging("test")
        for item in result.headline_assessment:
            assert item.rationale.strip()

    def test_analyze_url_content_has_assessment(self):
        provider = MockProvider()
        long_text = "ArtykuĹ‚ o inflacji i stopach procentowych. " * 20
        result = provider.analyze_url_content(long_text, url="http://example.com")
        assert result["is_article"] is True
        assert "headline_assessment" in result
        assert result["headline_assessment"] is not None
        assert len(result["headline_assessment"]) == 5

    def test_analyze_url_content_short_text_not_article(self):
        provider = MockProvider()
        result = provider.analyze_url_content("Too short", url="")
        assert result["is_article"] is False


# ---------------------------------------------------------------------------
# Lemmatization â€” only run when spaCy model is available
# ---------------------------------------------------------------------------


class TestLemmatization:
    """
    These tests verify that inflected Polish keyword forms are correctly
    matched after lemmatization.  Each test is skipped if spaCy or the
    Polish model is not installed.
    """

    @staticmethod
    def _require_spacy():
        """Skip the test if spaCy NLP pipeline couldn't be loaded."""
        import pytest

        from src.services.headline_quality_heuristics import _get_spacy_nlp

        if _get_spacy_nlp() is None:
            pytest.skip(
                "spaCy pl_core_news_sm not available â€” skipping lemmatization tests"
            )

    def test_inflected_verb_seo_tag_matches_headline(self):
        """
        SEO tag 'biegaÄ‡' (to run, infinitive) should match 'biegaĹ‚' in headline.
        Without lemmatization the score would be 10 (no match).
        """
        self._require_spacy()
        from src.services.headline_quality_heuristics import (
            compute_seo_fit_score,
            extract_keyword_candidates,
        )

        keywords = extract_keyword_candidates("", seo_tags=["biegaÄ‡"])
        score = compute_seo_fit_score("MaratoĹ„czyk biegaĹ‚ przez caĹ‚Ä… noc", keywords)
        assert score > 10, f"Expected lemma match, got score={score}"

    def test_inflected_noun_not_too_vague(self):
        """
        SEO tag 'inflacja' should recognise 'inflacji' (genitive) in headline.
        Without lemmatization is_too_vague would return True.
        """
        self._require_spacy()
        from src.services.headline_quality_heuristics import (
            extract_keyword_candidates,
            is_too_vague,
        )

        keywords = extract_keyword_candidates("", seo_tags=["inflacja"])
        assert not is_too_vague("Wzrost inflacji niepokoi ekonomistĂłw", keywords), (
            "Inflected noun 'inflacji' should not trigger too_vague"
        )

    def test_inflected_article_word_contributes_to_score(self):
        """
        Article text containing 'inflacji' (genitive of inflacja) should
        contribute the lemma 'inflacja' to keywords, and headline
        containing 'inflacjÄ…' (instrumental) should match the same lemma.
        """
        self._require_spacy()
        from src.services.headline_quality_heuristics import (
            compute_seo_fit_score,
            extract_keyword_candidates,
        )

        # article uses genitive 'inflacji' repeated so it becomes a top keyword
        article = "inflacji inflacji inflacji gospodarka gospodarka rynki rynki rynki"
        keywords = extract_keyword_candidates(article, seo_tags=None)
        # headline uses instrumental 'inflacją'
        score = compute_seo_fit_score("Rząd walczy z inflacją w tym roku", keywords)
        # Both inflacji and inflacją should yield lemma 'inflacja' -> match
        assert score > 10, (
            f"Expected lemma match for inflacji/inflacją, got score={score}"
        )
