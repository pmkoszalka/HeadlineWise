import os
import sys

# Add src to path
sys.path.append(os.path.abspath("."))

from src.utils.telemetry import (
    save_to_persistent_cache,
    load_persistent_cache,
    CACHE_FILE,
)


def verify_persistence():
    test_key = "test_hash_123"
    test_data = {"result": "success", "content": "Sample article data"}

    print(f"Saving to {CACHE_FILE}...")
    save_to_persistent_cache(test_key, test_data)

    print("Loading from disk...")
    loaded_cache = load_persistent_cache()

    if test_key in loaded_cache and loaded_cache[test_key] == test_data:
        print("✅ VERIFICATION SUCCESS: Data persisted and reloaded correctly!")
        # Cleanup
        if CACHE_FILE.exists():
            os.remove(CACHE_FILE)
            print("Cleaned up test file.")
    else:
        print("❌ VERIFICATION FAILED: Data mismatch or not found.")
        sys.exit(1)


if __name__ == "__main__":
    verify_persistence()
