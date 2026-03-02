import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

TELEMETRY_DIR = Path("telemetry")
TELEMETRY_FILE = TELEMETRY_DIR / "generations.jsonl"
EVALUATIONS_FILE = TELEMETRY_DIR / "headline_evaluations.jsonl"
CACHE_FILE = TELEMETRY_DIR / "result_cache.json"
logger = logging.getLogger(__name__)
_CACHE_WRITE_LOCK = threading.Lock()


def log_generation(
    article_length: int,
    model_name: str,
    success: bool,
    latency: float,
    error: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Logs generation metadata to a local JSONL file.
    This supports Approach B (Future Analytics/Feedback).
    """
    TELEMETRY_DIR.mkdir(exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "article_length": article_length,
        "model_name": model_name,
        "success": success,
        "latency_seconds": round(latency, 3),
        "error": error,
        "metadata": metadata or {},
    }

    with open(TELEMETRY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def log_headline_evaluation(
    source_mode: str,
    model_name: str,
    prompt_version: str,
    article_length: int,
    headlines_count: int,
    latency_ms: int,
    success: bool,
    error_type: Optional[str] = None,
    retry_count: int = 0,
    avg_scores: Optional[Dict[str, float]] = None,
    flag_counts: Optional[Dict[str, int]] = None,
) -> None:
    """
    Log headline evaluation metadata (second LLM call) to a separate JSONL file.
    Includes latency, model, prompt version, and optional per-request score aggregates.
    """
    TELEMETRY_DIR.mkdir(exist_ok=True)

    log_entry = {
        "event_type": "headline_evaluation",
        "timestamp": datetime.now().isoformat(),
        "source_mode": source_mode,
        "model_name": model_name,
        "prompt_version": prompt_version,
        "article_length": article_length,
        "headlines_count": headlines_count,
        "latency_ms": latency_ms,
        "success": success,
        "error_type": error_type,
        "retry_count": retry_count,
        "avg_scores": avg_scores or {},
        "flag_counts": flag_counts or {},
    }

    with open(EVALUATIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


def get_generation_logs() -> list[dict]:
    """Read all generation logs from the JSONL file."""
    if not TELEMETRY_FILE.exists():
        return []
    logs = []
    with open(TELEMETRY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return logs


def get_evaluation_logs() -> list[dict]:
    """Read all headline evaluation logs from the JSONL file."""
    if not EVALUATIONS_FILE.exists():
        return []
    logs = []
    with open(EVALUATIONS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return logs


def load_persistent_cache() -> dict:
    """Load the entire result cache from disk."""
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_to_persistent_cache(key: str, data: Any) -> None:
    """Add or update an item in the persistent disk cache."""
    TELEMETRY_DIR.mkdir(exist_ok=True)

    # Load current cache
    cache = load_persistent_cache()

    # Update
    # Robustly convert Pydantic models to dicts even if nested
    def _to_dict(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_dict(item) for item in obj]
        return obj

    cache[key] = _to_dict(data)

    # Save back atomically so partial writes do not corrupt the cache file.
    with _CACHE_WRITE_LOCK:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=TELEMETRY_DIR,
                prefix="result_cache.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                json.dump(cache, tmp, ensure_ascii=False, indent=2)
                tmp.flush()
                os.fsync(tmp.fileno())
                temp_path = tmp.name

            os.replace(temp_path, CACHE_FILE)
        except OSError as exc:
            logger.exception("Failed to persist cache atomically: %s", exc)
            if temp_path:
                try:
                    os.unlink(temp_path)
                except OSError:
                    logger.debug("Failed to remove temporary cache file: %s", temp_path)
