"""
Prompt templates for the headline quality evaluator LLM call.
"""

EVALUATOR_PROMPT_VERSION = "v1.1"

# Allowed risk flags - keep this list stable for UI/log consistency
ALLOWED_FLAGS = [
    "clickbait_risk",
    "possible_unsupported_claim",
    "too_vague",
    "too_long",
    "shouting_detected",
    "excessive_punctuation",
    "excessive_ellipsis",
    "none",
]

HEADLINE_EVALUATOR_SYSTEM_PROMPT = """\
Jestes ekspertem od jakosci tresci redakcyjnych.
Twoje zadanie: ocen 5 naglowkow wygenerowanych dla podanego artykulu.

ZASADY:
1. Baza ocen to WYLACZNIE dostarczony tekst artykulu. Nie wymyslaj faktow.
2. Dla kazdego naglowka zwroc score'y int 0-100 i flagi ryzyk.
3. ctr_potential to heurystyczny PROXY klikalnosci, NIE rzeczywisty CTR.
4. Rationale: jedno zdanie po polsku, max 200 znakow.
5. Dopuszczalne flagi ryzyk: clickbait_risk, possible_unsupported_claim, too_vague, too_long.
   Jesli brak ryzyk -> [\"none\"]. Bez duplikatow w liscie.
6. Zwroc WYLACZNIE surowy JSON. Zadnego markdown, zadnych komentarzy.

DEFINICJE SCORE'OW (skala 0-100):
- ctr_potential:
    90+: Wybitny hook, konkretny, obiecuje wysoka wartosc.
    50: Przecietny, poprawny ale malo angazujacy.
    30-: Bardzo nudny, generyczny lub odpychajacy clickbait.
- clarity:
    90+: Natychmiast zrozumialy sens bez czytania artykulu.
    50: Wymaga chwili zastanowienia, ale sens jest jasny.
    30-: Belkotliwy, zbyt zagadkowy lub mylacy.
- seo_fit:
    90+: Zawiera najwazniejsze osoby/firmy/tematy (encje) z tekstu.
    50: Zawiera poboczne tematy, ale trzyma sie kontekstu.
    30-: Brak slow kluczowych, temat kompletnie niezwiązany.
"""

HEADLINE_EVALUATOR_USER_PROMPT_TEMPLATE = """\
Artykul (fragment, max 6000 znakow):
\"\"\"
{article_text}
\"\"\"

Naglowki do oceny (indeks 0-4):
{headlines_block}

Zwroc JSON jako obiekt z polem `items` zawierajacym DOKLADNIE 5 obiektow:
{{
  "items": [
    {{
      "headline_index": 0,
      "ctr_potential": <int 0-100>,
      "clarity": <int 0-100>,
      "seo_fit": <int 0-100>,
      "risk_flags": ["<flag>" | "none"],
      "rationale": "<jedno zdanie po polsku, max 200 znakow>"
    }},
    ...
  ]
}}
"""


def build_headlines_block(headlines: list[str], styles: list[str]) -> str:
    """Format the numbered headlines list for the evaluator prompt."""
    lines = []
    for i, (h, s) in enumerate(zip(headlines, styles)):
        lines.append(f"  {i}. [{s}] {h}")
    return "\n".join(lines)


def build_evaluator_prompt(
    article_text: str,
    headlines: list[str],
    styles: list[str],
) -> tuple[str, str]:
    """
    Build (system_prompt, user_prompt) for the evaluator call.
    article_text is capped at 6000 chars to stay within token limits.
    """
    headlines_block = build_headlines_block(headlines, styles)
    user_prompt = HEADLINE_EVALUATOR_USER_PROMPT_TEMPLATE.format(
        article_text=article_text[:6000],
        headlines_block=headlines_block,
    )
    return HEADLINE_EVALUATOR_SYSTEM_PROMPT, user_prompt
