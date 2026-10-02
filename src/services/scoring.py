"""Policy-deviation scoring service (Step 4 — .

Compares deviation scores against config-defined thresholds, assigns a final
per-entry deviation score, and flags entries that exceed the threshold.

Config knobs (config/agent.yaml → graph config → node constructor):
  * ``drift_threshold``  — float; reward-delta z-score above which an entry is
    flagged. Default 3.0 (≈ 3σ).
  * ``semantic_weight``  — float in [0, 1]; weight of the optional semantic
    distance in the combined score. Default 0.0 (semantic layer disabled unless
    a sentence-transformers scorer is injected — see drift).

Output: list ``[{entry_id, score, threshold_exceeded}]``. Logic preserved verbatim.
"""

from __future__ import annotations

from typing import Any, Optional

DEFAULT_DRIFT_THRESHOLD = 3.0
DEFAULT_SEMANTIC_WEIGHT = 0.0


def score_entries(deviations: list[Any], threshold: float, semantic_weight: float) -> list[dict[str, Any]]:
    """Threshold comparison + per-entry final scoring."""
    semantic_weight = _clamp01(semantic_weight)
    scored: list[dict[str, Any]] = []
    for dev in deviations:
        if not isinstance(dev, dict):
            continue
        score = _combine(dev.get("reward_delta"), dev.get("semantic_dist"), semantic_weight)
        scored.append(
            {
                "entry_id": dev.get("entry_id"),
                "score": score,
                "threshold_exceeded": score is not None and score >= threshold,
            }
        )
    return scored


def _combine(reward_delta: object, semantic_dist: object, semantic_weight: float) -> Optional[float]:
    """Blend the reward-delta and (optional) semantic distance.

    When the semantic layer is disabled (weight 0 or dist None) the score is the
    reward delta verbatim, so detection is fully deterministic today.
    """
    rd = _as_float(reward_delta, None)
    if rd is None:
        return None  # no reward signal for this entry
    sd = _as_float(semantic_dist, None)
    if sd is None or semantic_weight <= 0:
        return round(rd, 6)  # semantic layer disabled → deterministic delta only
    # Weighted blend of the reward delta and the semantic distance. Using an
    # explicit weighted sum (NOT an `if rd` shortcut) so a valid on-baseline entry
    # (rd == 0.0) correctly contributes 0 from the reward term rather than being
    # mis-handled as "no signal".
    return round(rd * (1.0 - semantic_weight) + sd * semantic_weight, 6)


def _as_float(value: object, default: float | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
