import re
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts the first JSON-like block from a string using regex.
    Useful if the LLM includes markdown backticks or conversational filler.
    """
    try:
        # Look for content between outermost curly braces
        match = re.search(r"({.*})", text, re.DOTALL)
        if match:
            return json.loads(match.group(1))

        # Fallback: try raw json.loads if no braces found
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse extracted JSON: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during JSON extraction: {e}")
        return None
