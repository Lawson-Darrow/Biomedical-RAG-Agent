"""Small shared helpers."""

from __future__ import annotations

import json
import re


def safe_json(s: str) -> dict:
    """Parse a JSON object from a model response, tolerating stray prose."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {}
