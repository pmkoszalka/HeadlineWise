"""
fill_translations_gemini.py
────────────────────────────
Fills ONLY the missing text_pl rows in train_pl.csv using Gemini.
Already-translated rows are NEVER touched.

Model: gemini-3-flash-preview (fast, cheap, ~$0 for 2k headlines)
Batch: 30 headlines per prompt → fewer API calls, faster run
Rate:  ~15 RPM free tier → sleep 4.5s between batches (safe margin)
Checkpoint: every 5 batches (150 rows) so Ctrl+C is always safe.
"""

import logging
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 30  # headlines per Gemini call
SLEEP_BETWEEN = 4.5  # seconds between calls (stays under 15 RPM)
CHECKPOINT_EVERY = 5  # save to disk every N batches
MAX_RETRIES = 4
MODEL = "gemini-3-flash-preview"


def is_valid(value) -> bool:
    """Check if a translated string is valid and not empty or NaN."""
    if pd.isna(value):
        return False
    return str(value).strip() not in ("", "None", "nan")


def translate_batch(client, texts: list[str], backoff: float) -> tuple[list, float]:
    """Send one batch to Gemini. Returns (translations, new_backoff)."""
    prompt = (
        "Translate the following news headlines from English to Polish.\n"
        "Return ONLY the translations, one per line, in the same order.\n"
        "No numbering, no bullet points, no extra text.\n\n" + "\n".join(texts)
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
            )
            lines = [
                line.strip()
                for line in response.text.strip().split("\n")
                if line.strip()
            ]

            if len(lines) == len(texts):
                return lines, 4.5  # reset backoff on success

            log.warning(
                "Count mismatch: sent %d, got %d (attempt %d) — retrying",
                len(texts),
                len(lines),
                attempt,
            )
            # On mismatch, try translating one-by-one as fallback
            if attempt == MAX_RETRIES:
                results = []
                for t in texts:
                    try:
                        r = client.models.generate_content(
                            model=MODEL,
                            contents=f"Translate to Polish (return only the translation): {t}",
                        )
                        results.append(r.text.strip())
                        time.sleep(4.5)
                    except Exception:
                        results.append(None)
                return results, 4.5

        except Exception as exc:
            err = str(exc).lower()
            if "429" in err or "resource_exhausted" in err or "quota" in err:
                log.warning(
                    "Rate-limit hit (attempt %d/%d) — backing off %.0fs",
                    attempt,
                    MAX_RETRIES,
                    backoff,
                )
            else:
                log.warning(
                    "Gemini error (attempt %d/%d): %s — backing off %.0fs",
                    attempt,
                    MAX_RETRIES,
                    exc.__class__.__name__,
                    backoff,
                )
            time.sleep(backoff)
            backoff = min(backoff * 2, 120)

    log.error("Batch failed after %d attempts.", MAX_RETRIES)
    return [None] * len(texts), backoff


def main():
    """Main entrypoint for filling missing translations using Gemini API."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY not set in .env")
        return

    client = genai.Client()

    out_path = Path(__file__).parent.parent.parent / "data" / "train_pl.csv"
    if not out_path.exists():
        log.error("train_pl.csv not found: %s", out_path)
        return

    # ── Load with safety: read-only snapshot first ────────────────────────────
    df = pd.read_csv(out_path)
    if "text_pl" not in df.columns:
        df["text_pl"] = None
    df["text_pl"] = df["text_pl"].astype(object)

    already_done = df["text_pl"].apply(is_valid).sum()
    todo_idx = [i for i in df.index if not is_valid(df.at[i, "text_pl"])]

    log.info("Already translated: %d / %d", already_done, len(df))
    log.info("Remaining to fill with Gemini: %d rows", len(todo_idx))

    if not todo_idx:
        log.info("✅ Nothing to do.")
        return

    n_batches = -(-len(todo_idx) // BATCH_SIZE)
    log.info(
        "Model: %s  |  Batch size: %d  |  ~%d API calls  |  ETA: ~%.0f min",
        MODEL,
        BATCH_SIZE,
        n_batches,
        n_batches * SLEEP_BETWEEN / 60,
    )

    backoff = 4.5
    batches_done = 0

    with tqdm(total=len(todo_idx), unit="row", dynamic_ncols=True) as pbar:
        for i in range(0, len(todo_idx), BATCH_SIZE):
            batch_idx = todo_idx[i : i + BATCH_SIZE]
            batch_texts = [str(df.at[j, "text"]) for j in batch_idx]

            translations, backoff = translate_batch(client, batch_texts, backoff)

            # Write results back — ONLY for this batch
            for j, translation in zip(batch_idx, translations):
                df.at[j, "text_pl"] = translation

            batches_done += 1
            pbar.update(len(batch_idx))
            pbar.set_postfix(
                done=already_done + (i + len(batch_idx)),
                missing=len(todo_idx) - i - len(batch_idx),
            )

            # Checkpoint: save to disk periodically
            if batches_done % CHECKPOINT_EVERY == 0 or i + BATCH_SIZE >= len(todo_idx):
                df.to_csv(out_path, index=False)
                still_missing = int(df["text_pl"].isna().sum())
                log.info(
                    "💾 Checkpoint — Gemini batches done: %d/%d  |  missing: %d",
                    batches_done,
                    n_batches,
                    still_missing,
                )

            time.sleep(SLEEP_BETWEEN)

    # Final guaranteed save
    df.to_csv(out_path, index=False)
    missing = int(df["text_pl"].isna().sum())
    log.info("🏁 Done!  Missing: %d / %d", missing, len(df))
    if missing:
        log.info("Re-run to retry %d rows.", missing)


if __name__ == "__main__":
    main()
