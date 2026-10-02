"""State JSON (de)serialization helpers for CMN-C1-179.

Structured state fields are stored as JSON strings so the LangGraph checkpoint
holds only primitives (msgpack-safe). Producing slots call
``dump_json`` on write; consuming slots/services call ``load_list`` / ``load_dict``
on read. Readers tolerate an already-parsed container (so direct unit tests may
pass raw list/dict) and a missing / malformed value (degrades to the empty
default, never raises). Deterministic, side-effect free.
"""

from __future__ import annotations

import json
from typing import Any, Optional


def dump_json(value: Any) -> Optional[str]:
    """Serialize a state collection to a JSON string. ``None`` stays ``None``."""
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def load_list(value: Any) -> list[Any]:
    """Read a JSON-string (or already-parsed) list state field. Default ``[]``."""
    parsed = _loads_if_str(value)
    return parsed if isinstance(parsed, list) else []


def load_dict(value: Any) -> dict[str, Any]:
    """Read a JSON-string (or already-parsed) dict state field. Default ``{}``."""
    parsed = _loads_if_str(value)
    return parsed if isinstance(parsed, dict) else {}


def _loads_if_str(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return value
