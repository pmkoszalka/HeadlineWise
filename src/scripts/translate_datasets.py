"""
translate_datasets.py
─────────────────────
Translates 6,000 English headlines to Polish using deep-translator.

Speed strategy: ThreadPoolExecutor with 5 workers.
Each worker creates its own Translator instance and translates one
headline at a time. This gives ~5x parallelism without triggering
IP bans (each worker throttled individually).

At ~1.5s per headline / 5 workers → ~300 headlines/min → ~20 min total.

Safety:
  • Each worker sleeps 0.5-1.0s after each request
  • On error: exponential back-off per worker (5s → 10s → 20s… cap 120s)
  • Checkpoint to disk every 100 translated rows
  • Idempotent: already-translated rows are skipped on re-run
"""

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MAX_WORKERS = 2  # reduced to avoid IP throttling
WORKER_SLEEP = (1.5, 2.5)  # (min, max) seconds per worker after each request
INITIAL_BACKOFF = 5.0
MAX_BACKOFF = 120.0
MAX_RETRIES = 5
CHECKPOINT_EVERY = 100  # save to disk every N translated rows


def map_label(val) -> int:
    """Map string or numeric labels to a binary integer (1 for clickbait, 0 for news)."""
    if isinstance(val, (int, float)):
        return 1 if val > 0 else 0
    if isinstance(val, str):
        s = val.lower().strip()
        if "clickbait" in s or s == "1":
            return 1
        if "news" in s or s == "0":
            return 0
    return 0


def is_valid(value) -> bool:
    """Check if a translated string is valid and not empty or NaN."""
    if pd.isna(value):
        return False
    return str(value).strip() not in ("", "None", "nan")


def prepare_dataset(data_dir: Path, out_path: Path) -> None:
    """Combine and deduplicate raw CSV chunks into a balanced 6k-row dataset."""
    if out_path.exists() and len(pd.read_csv(out_path)) >= 6000:
        log.info("Dataset already has 6,000 rows — skipping preparation.")
        return

    log.info("Preparing balanced 6,000-sample dataset...")
    df1 = pd.read_csv(data_dir / "train1.csv")
    df2 = pd.read_csv(data_dir / "train2.csv")

    if "headline" in df1.columns:
        df1 = df1.rename(columns={"headline": "text", "clickbait": "label"})
    if "title" in df2.columns:
        df2 = df2.rename(columns={"title": "text"})

    df = (
        pd.concat([df1[["text", "label"]], df2[["text", "label"]]], ignore_index=True)
        .dropna(subset=["text", "label"])
        .drop_duplicates(subset=["text"])
    )
    df["target"] = df["label"].apply(map_label)

    cb = df[df["target"] == 1].sample(n=3000, random_state=42)
    news = df[df["target"] == 0].sample(n=3000, random_state=42)
    sampled = (
        pd.concat([cb, news]).sample(frac=1, random_state=42).reset_index(drop=True)
    )

    sampled["text_pl"] = None
    if out_path.exists():
        old = pd.read_csv(out_path).drop_duplicates(subset=["text"])
        sampled = sampled.merge(old[["text", "text_pl"]], on="text", how="left")

    sampled.to_csv(out_path, index=False)
    log.info("Dataset saved: %s", out_path)


def translate_one(idx: int, text: str) -> tuple[int, str | None]:
    """Translate a single headline. Returns (index, translation_or_None)."""
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="auto", target="pl")
    backoff = INITIAL_BACKOFF

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            translation = translator.translate(text)
            time.sleep(random.uniform(*WORKER_SLEEP))
            return idx, translation
        except Exception as exc:
            err = str(exc).lower()
            if any(
                k in err for k in ("429", "too many", "rate", "exhausted", "blocked")
            ):
                log.warning(
                    "[row %d] Rate-limited (attempt %d/%d) — back off %.0fs",
                    idx,
                    attempt,
                    MAX_RETRIES,
                    backoff,
                )
            else:
                log.debug(
                    "[row %d] Error attempt %d/%d: %s",
                    idx,
                    attempt,
                    MAX_RETRIES,
                    exc.__class__.__name__,
                )
            time.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

    return idx, None


def main():
    """Main entrypoint for dataset translation process mapping English to Polish."""
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    out_path = data_dir / "train_pl.csv"

    prepare_dataset(data_dir, out_path)

    df = pd.read_csv(out_path)
    if "text_pl" not in df.columns:
        df["text_pl"] = None
    df["text_pl"] = df["text_pl"].astype(object)

    todo = [
        (i, str(df.at[i, "text"]))
        for i in df.index
        if not is_valid(df.at[i, "text_pl"])
    ]

    if not todo:
        log.info("✅ All rows already translated.")
        return

    avg_s_per_row = ((WORKER_SLEEP[0] + WORKER_SLEEP[1]) / 2 + 0.5) / MAX_WORKERS
    log.info(
        "Rows to translate: %d  |  workers: %d  |  ETA: ~%.0f min",
        len(todo),
        MAX_WORKERS,
        len(todo) * avg_s_per_row / 60,
    )

    lock = threading.Lock()
    done_count = 0

    with tqdm(total=len(todo), unit="row", dynamic_ncols=True) as pbar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(translate_one, idx, text): idx for idx, text in todo
            }

            for future in as_completed(futures):
                idx, translation = future.result()

                with lock:
                    df.at[idx, "text_pl"] = translation
                    done_count += 1
                    pbar.update(1)
                    still_missing = len(todo) - done_count
                    pbar.set_postfix(done=done_count, missing=still_missing)

                    if done_count % CHECKPOINT_EVERY == 0 or done_count == len(todo):
                        df.to_csv(out_path, index=False)
                        log.info(
                            "💾 Checkpoint — %d / %d done  |  still missing: %d",
                            done_count,
                            len(todo),
                            int(df["text_pl"].isna().sum()),
                        )

    df.to_csv(out_path, index=False)
    missing = int(df["text_pl"].isna().sum())
    log.info("🏁 Done!  Missing: %d / %d", missing, len(df))
    if missing:
        log.info("Re-run to retry %d rows.", missing)


if __name__ == "__main__":
    main()
