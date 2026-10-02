"""RL log normalization service (Step 1 — .

Normalizes RL agent output logs per RL framework (ART / GRPO / OpenAI RL) using a
per-framework adapter, and applies S-1 input normalization (HTML strip +
per-entry size truncation). S-1 is normalization only — injection *blocking* is
the S-3 gate's job (dev guide §5, the source proposalruling), not asserted here.

Output: list of normalized entries ``[{entry_id, reward, action, observation_digest}]``.

Logic preserved verbatim from the original InputParseNode (the migration moved it
from a node ``execute()`` into this deterministic, testable service function).
"""

from __future__ import annotations

import re
from typing import Any, Optional

# Per-entry observation digest cap (S-1 truncation; deployment may override via
# config "max_observation_chars"). Keeps state bounded and msgpack-light.
DEFAULT_MAX_OBS_CHARS = 2000
_HTML_TAG = re.compile(r"<[^>]+>")
SUPPORTED_FRAMEWORKS = {"art", "grpo", "openai_rl", "generic"}


def parse_logs(raw: list[Any], framework: str, max_obs: int = DEFAULT_MAX_OBS_CHARS) -> list[dict[str, Any]]:
    """Normalize a batch of raw RL log entries onto the canonical schema.

    An empty batch is a *valid* empty result (produces an empty-state report
    downstream), not an error.
    """
    fw = str(framework or "generic").strip().lower()
    if fw not in SUPPORTED_FRAMEWORKS:
        fw = "generic"

    parsed: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        parsed.append(_normalize(fw, idx, entry, max_obs))
    return parsed


def _normalize(framework: str, idx: int, entry: dict[str, Any], max_obs: int) -> dict[str, Any]:
    """Map a framework-specific log entry onto the canonical schema.

    The frameworks differ in field naming for the same concepts:
      ART        : {"reward", "action", "observation"}
      GRPO       : {"group_reward", "policy_action", "obs"}
      OpenAI RL  : {"score", "completion", "prompt"}
      generic    : best-effort across the above keys
    """
    entry_id = entry.get("entry_id") or entry.get("id") or f"e{idx}"

    if framework == "grpo":
        reward = entry.get("group_reward", entry.get("reward"))
        action = entry.get("policy_action", entry.get("action"))
        observation = entry.get("obs", entry.get("observation"))
    elif framework == "openai_rl":
        reward = entry.get("score", entry.get("reward"))
        action = entry.get("completion", entry.get("action"))
        observation = entry.get("prompt", entry.get("observation"))
    else:  # art + generic
        reward = entry.get("reward", entry.get("score"))
        action = entry.get("action", entry.get("policy_action"))
        observation = entry.get("observation", entry.get("obs"))

    return {
        "entry_id": str(entry_id),
        "reward": _as_float(reward),
        "action": _sanitize_text(action, max_obs),
        "observation_digest": _sanitize_text(observation, max_obs),
    }


def _sanitize_text(value: object, max_obs: int) -> str:
    """S-1 normalization: HTML strip + truncate. No injection *blocking*."""
    text = value if isinstance(value, str) else ("" if value is None else str(value))
    text = _HTML_TAG.sub("", text)
    if len(text) > max_obs:
        text = text[:max_obs]
    return text


def _as_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def as_int(value: object, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
