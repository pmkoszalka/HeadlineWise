# Nagłówek AI

Zaawansowane narzędzie do scrapowania, analizy i optymalizacji nagłówków artykułów. Łączy moc dużych modeli językowych (LLM - Google Gemini) z lokalnym modelem Machine Learning (Scikit-Learn) do oceny jakości i wiarygodności tekstów.


## 🚀 Główne funkcje

- **Ekstrakcja treści (Scraping):** Pobieranie treści artykułów na podstawie URL lub masowe pobieranie prosto ze strony głównej (TVN24, Eurosport). Omija reklamy, paywalle i duplikaty.
- **Ocena nagłówków w locie:** Pasek boczny do szybkiego wklejenia własnego tekstu i wygenerowania 5 propozycji nagłówków.
- **Przeglądarka portalu (Portal Browser):** Podgląd na żywo najnowszych artykułów z wybranych stron. Ocena oryginalnego nagłówka i propozycje optymalizacji.
- **Moduł ML Clickbait Detection:** Lokalny, szybki model ML analizujący prawdopodobieństwo, czy dany cel jest "clickbaitem" (wyświetlany jako **Credibility Score**).
- **Skrypty Data Science:** Zestaw skryptów do masowego tłumaczenia na język polski zbiorów danych (`googletrans`, `gemini`) oraz trenowania własnego modelu wykrywającego clickbaity.

## Wskaźniki jakości (Scorowanie)

Każdy oceniany lub generowany nagłówek otrzymuje od 0 do 100 punktów w 4 kluczowych kategoriach:

1. **CTR (Click-Through Rate):** Potencjał klikalności (długość, liczby, znaki zapytania).
2. **Clarity (Jasność):** Zrozumiałość przekazu i zwięzłość treści.
3. **SEO fit:** Dopasowanie do słów kluczowych i głównych tematów z oryginalnego artykułu.
4. **🛡️ Credibility:** Prawdopodobieństwo, że nagłówek to rzetelny "news", a nie "clickbait" (liczony przez lokalny Machine Learning).

Na koniec liczony jest **⭐ Overall Score** będący średnią z powyższych metryk.

---

## 🗂️ Struktura repozytorium

```
HeadlineWise/
├── app.py                  # Główna aplikacja Streamlit (UI)
├── conftest.py             # Konfiguracja pytest (fixtures)
├── pyproject.toml          # Metadane projektu i zależności (uv/pip)
├── requirements.txt        # Lista zależności kompatybilna z pip
├── uv.lock                 # Zablokowane wersje zależności (uv)
├── .env.example            # Przykładowy plik env z wymaganymi zmiennymi
│
├── data/                   # Dane treningowe ML i wytrenowany model
│   ├── train_pl.csv        # Główny zbiór danych (clickbait PL)
│   └── clickbait_model.pkl # Wytrenowany pipeline scikit-learn (TF-IDF + LR)
│
├── src/
│   ├── core/               # Konfiguracja, schematy danych, szablony promptów, dostawcy LLM
│   ├── services/           # Logika oceny nagłówków (LLM + heurystyki + merge)
│   ├── utils/              # Scraper, konektory portali, telemetria
│   ├── data/               # Artykuł Demo
│   └── scripts/            # Skrypty Data Science: tłumaczenie zbiorów, trening modelu
│
├── tests/                  # Testy jednostkowe (pytest)
└── telemetry/              # Cache wyników (JSON)
```

---

## 💻 Instalacja

Aplikacja działa w Pythonie 3.12+ i wymaga środowiska wspierającego scikit-learn oraz biblioteki do wizualizacji.

1. **Sklonuj repozytorium**

2. **Zainstaluj zależności:**
   Plik `requirements.txt` instaluje wszystkie pakiety potrzebne dla Streamlit, ML, LLM i scrapowania.
   ```bash
   pip install -r requirements.txt
   ```

3. **Klucze API — dostawca LLM:**

   Nagłówek AI obsługuje dwóch dostawców LLM. Wybierz jednego i ustaw odpowiednią zmienną w pliku `.env`:

   | Dostawca | Zmienna środowiskowa | Model domyślny |
   |---|---|---|
   | **Google Gemini** *(domyślny)* | `GEMINI_API_KEY` | `gemini-2.0-flash-lite` |
   | **OpenAI** | `OPENAI_API_KEY` | `gpt-4o-mini` |

   Przykładowy plik `.env` (skopiuj z `.env.example`):
   ```env
   # --- Dostawcy LLM (wymagany co najmniej jeden) ---
   GEMINI_API_KEY=twoj-klucz-google-gemini
   OPENAI_API_KEY=twoj-klucz-openai

   # --- Wybór aktywnego dostawcy ---
   DEFAULT_LLM_PROVIDER=gemini   # lub: openai
   ```

4. **Ściągnij model języka polskiego spaCy (dla dokładniejszego analizowania SEO):**
   ```bash
   python -m spacy download pl_core_news_sm
   ```

### Alternatywnie — instalacja przez `uv` (szybciej)

Jeśli masz zainstalowany [`uv`](https://github.com/astral-sh/uv):

```bash
uv pip install -r requirements.txt
# lub z pyproject.toml (polski model spaCy wbudowany):
uv sync
```

---

## 🧪 Testy i jakość kodu

### Uruchomienie testów

```bash
pytest
```

Testy jednostkowe obejmują scraper, evaluator, schemat danych i cache telemetrii.

### Linter

```bash
ruff check
```

Ruff sprawdza formatowanie i spójność kodu. Konfiguracja w sekcji `[tool.ruff.lint]` w `pyproject.toml`.

---

## 🏃 Uruchomienie Aplikacji (UI Streamlit)

Aplikacja oparta jest o framework Streamlit. Odpalisz ją z terminala z głównego folderu poleceniem:

```bash
streamlit run app.py
```
Aplikacja uruchomi się pod lokalnym adresem: `http://localhost:8501`.

### Struktura Aplikacji:
- **Zakładka "Twórz (wklej artykuł)":** Przydatne przy szybkim pisaniu tekstu. Wklejasz zawartość, system sugeruje paczkę nagłówków.
- **Zakładka "Optymalizuj z URL":** Podajesz link do opublikowanego już artykułu, system zaciąga tekst, usuwa reklamy i optymalizuje.
- **Zakładka "Przeglądaj portal":** Świeże newsy. Wczytywanie po API zawartości strony głównej portalu (TVN24, Eurosport). System pokazuje aktualny tytuł na stronie i punktuje jego jakość (Credibility, SEO).

---

## 🧠 Cykl ML & Data Pipeline

Projekt posiada narzędzia do pracy z danymi na wypadek chęci dotrenowania modeli. Wszystko znajduje się w folderze `src/scripts`.

### 1. Przygotowanie danych (Tłumaczenie)
Projekt został przygotowany z użyciem zbiorów angielskich clickbaitów przetłumaczonych na polski. Istnieją tu dwa skrypty:

- `translate_datasets.py`: Tłumaczy zbiory danych za pomocą darmowej biblioteki `deep-translator`. Działa wolno ze względu na throttling IP przez serwery Google.
- `fill_translations_gemini.py`: Alternatywny skrypt korzystający z płatnego/szybszego API Gemini (`gemini-3-flash-preview`) do błyskawicznego spolszczenia brakujących próbek.

### 2. Trenowanie modelu
Model to klasyczny Machine Learning operujący na tekstowych wektorach TF-IDF (`scikit-learn`). 

Aby zaktualizować/wytrenować model clickbaitów na podstawie zaktualizowanego zestawu danych (`data/train_pl.csv`), uruchom:
```bash
python src/scripts/train_clickbait_model.py
```
Zapisze to świeży pipeline (wektoryzator + logistyczna regresja) jako binarny plik `.pkl` w katalogu `data/`. Uruchomiona aplikacja `app.py` załaduje ten model automatycznie.
