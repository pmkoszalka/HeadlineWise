# WarnerBros Headline Optimizer & Clickbait Analyzer

Zaawansowane narzędzie do scrapowania, analizy i optymalizacji nagłówków artykułów. Łączy moc dużych modeli językowych (LLM - Google Gemini) z lokalnym modelem Machine Learning (Scikit-Learn) do oceny jakości i wiarygodności tekstów.

## 🚀 Główne funkcje

- **Ekstrakcja treści (Scraping):** Pobieranie treści artykułów na podstawie URL lub masowe pobieranie prosto ze strony głównej (TVN24, Eurosport). Omija reklamy, paywalle i duplikaty.
- **Ocena nagłówków w locie:** Pasek boczny do szybkiego wklejenia własnego tekstu i wygenerowania 10 propozycji nagłówków.
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

## 💻 Instalacja

Aplikacja działa w Pythonie 3.11+ i wymaga środowiska wspierającego scikit-learn oraz biblioteki do wizualizacji.

1. **Sklonuj repozytorium**

2. **Przygotuj wirtualne środowisko (opcjonalnie, ale zalecane):**
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```

3. **Zainstaluj zależności:**
   Plik `requirements.txt` instaluje wszystkie pakiety potrzebne dla Streamlit, ML, LLM i scrapowania.
   ```bash
   pip install -r requirements.txt
   ```

4. **Klucz API Gemini:**
   Stwórz plik `.env` w głównym folderze projektu (lub dodaj zmienną systemową) i wpisz swój klucz Google Gemini:
   ```env
   GEMINI_API_KEY=twoj-prywatny-klucz-api
   ```

5. **Ściągnij model języka polskiego spaCy (dla dokładniejszego analizowania SEO):**
   ```bash
   python -m spacy download pl_core_news_sm
   ```

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
- **Zakładka "Przeglądaj portal":** Świeże newsy. Wczytywanie po API zawartości strony głównej portalu (np. Onet, TVP Info). System pokazuje aktualny tytuł na stronie i punktuje jego jakość (Credibility, SEO).

---

## 🧠 Cykl ML & Data Pipeline

Projekt posiada narzędzia do pracy z danymi na wypadek chęci dotrenowania modeli. Wszystko znajduje się w folderze `src/scripts`.

### 1. Przygotowanie danych (Tłumaczenie)
Projekt został przygotowany z użyciem zbiorów angielskich clickbaitów przetłumaczonych na polski. Istnieją tu dwa skrypty:

- `translate_datasets.py`: Tłumaczy zbiory danych za pomocą darmowej biblioteki `googletrans`. Działa wolno ze względu na throttling IP przez serwery Google.
- `fill_translations_gemini.py`: Alternatywny skrypt korzystający z płatnego/szybszego API Gemini (`gemini-3-flash-preview`) do błyskawicznego spolszczenia brakujących próbek.

### 2. Trenowanie modelu
Model to klasyczny Machine Learning operujący na tekstowych wektorach TF-IDF (`scikit-learn`). 

Aby zaktualizować/wytrenować model clickbaitów na podstawie zaktualizowanego zestawu danych (`data/train_pl.csv`), uruchom:
```bash
python src/scripts/train_clickbait_model.py
```
Zapisze to świeży pipeline (wektoryzator + logistyczna regresja) jako binarny plik `.pkl` w katalogu `data/`. Uruchomiona aplikacja `app.py` załaduje ten model automatycznie.
