import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional

TELEMETRY_DIR = Path("telemetry")
TELEMETRY_FILE = TELEMETRY_DIR / "generations.jsonl"


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
