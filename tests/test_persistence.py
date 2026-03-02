import json
import threading
from pathlib import Path

from src.utils import telemetry


class DummyModel:
    def __init__(self, value):
        self.value = value

    def model_dump(self):
        return {"value": self.value}


def test_save_and_load_persistent_cache_roundtrip(tmp_path, monkeypatch):
    telemetry_dir = tmp_path / "telemetry"
    cache_file = telemetry_dir / "result_cache.json"

    monkeypatch.setattr(telemetry, "TELEMETRY_DIR", telemetry_dir)
    monkeypatch.setattr(telemetry, "CACHE_FILE", cache_file)

    telemetry.save_to_persistent_cache("k1", {"a": 1})
    loaded = telemetry.load_persistent_cache()

    assert loaded["k1"] == {"a": 1}


def test_save_to_persistent_cache_converts_model_dump(tmp_path, monkeypatch):
    telemetry_dir = tmp_path / "telemetry"
    cache_file = telemetry_dir / "result_cache.json"

    monkeypatch.setattr(telemetry, "TELEMETRY_DIR", telemetry_dir)
    monkeypatch.setattr(telemetry, "CACHE_FILE", cache_file)

    telemetry.save_to_persistent_cache("k2", {"obj": DummyModel(7)})
    loaded = telemetry.load_persistent_cache()

    assert loaded["k2"] == {"obj": {"value": 7}}


def test_atomic_write_failure_logs_and_preserves_previous_cache(tmp_path, monkeypatch, caplog):
    telemetry_dir = tmp_path / "telemetry"
    cache_file = telemetry_dir / "result_cache.json"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({"old": {"x": 1}}), encoding="utf-8")

    monkeypatch.setattr(telemetry, "TELEMETRY_DIR", telemetry_dir)
    monkeypatch.setattr(telemetry, "CACHE_FILE", cache_file)

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("src.utils.telemetry.os.replace", _raise)

    with caplog.at_level("ERROR"):
        telemetry.save_to_persistent_cache("new", {"x": 2})

    loaded = telemetry.load_persistent_cache()
    assert loaded == {"old": {"x": 1}}
    assert "Failed to persist cache atomically" in caplog.text


def test_concurrent_writes_do_not_corrupt_cache_json(tmp_path, monkeypatch):
    telemetry_dir = tmp_path / "telemetry"
    cache_file = telemetry_dir / "result_cache.json"

    monkeypatch.setattr(telemetry, "TELEMETRY_DIR", telemetry_dir)
    monkeypatch.setattr(telemetry, "CACHE_FILE", cache_file)

    def _write(i):
        telemetry.save_to_persistent_cache(f"k{i}", {"v": i})

    threads = [threading.Thread(target=_write, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    loaded = telemetry.load_persistent_cache()
    assert isinstance(loaded, dict)
    assert all(k.startswith("k") for k in loaded.keys())

    # Ensure the file is valid JSON on disk.
    raw = Path(cache_file).read_text(encoding="utf-8")
    json.loads(raw)
