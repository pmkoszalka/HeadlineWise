"""
Script to train a local ML model for clickbait detection.
Uses LogisticRegression and TfidfVectorizer.
Writes the output to data/clickbait_model.pkl.
"""

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def train_model() -> None:
    # ── 1. Setup paths ───────────────────────────────────────────────────────
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data"
    train_pl_path = data_dir / "train_pl.csv"
    output_path = data_dir / "clickbait_model.pkl"

    logger.info("Starting model training process...")

    # ── 2. Load data ─────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(train_pl_path)
    except FileNotFoundError as e:
        logger.error(f"Could not find translated dataset: {e}")
        return

    logger.info(f"Loaded dataset from {train_pl_path} with {len(df)} samples")

    # ── 3. Clean schema ──────────────────────────────────────────────────────
    # Drop rows that failed to translate or missing data
    df = df.dropna(subset=["text_pl", "target"])

    logger.info(f"Final dataset size after dropna: {len(df)} samples")
    logger.info(f"Class distribution:\n{df['target'].value_counts()}")

    # ── 4. Split and Train ───────────────────────────────────────────────────
    # Use the Polish translated text
    X = df["text_pl"].astype(str)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        [
            # N-grams up to 3 words capture typical clickbait structures ("zobacz wideo")
            (
                "tfidf",
                TfidfVectorizer(ngram_range=(1, 3), max_features=25000, lowercase=True),
            ),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
        ]
    )

    logger.info("Training pipeline (TF-IDF + LogisticRegression)...")
    pipeline.fit(X_train, y_train)

    # ── 5. Evaluate ──────────────────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["news", "clickbait"])
    logger.info(f"Classification Report:\n{report}")

    # ── 6. Save Model ────────────────────────────────────────────────────────
    logger.info(f"Saving model to {output_path} ...")
    joblib.dump(pipeline, output_path)
    logger.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_model()
