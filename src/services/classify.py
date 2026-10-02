"""Anomaly classification service (Step 5 — .

Classifies each threshold-exceeding scored entry into one of three RL failure
modes and assigns a severity. Entries that did not exceed the threshold are not
anomalies and are dropped from the output.

3-class taxonomy (deterministic, rule-based — no LLM):
  * ``policy_drift``        — moderate deviation; gradual policy shift.
  * ``reward_hacking``      — high reward delta (reward inflated far above the
    baseline mean), the signature of reward-gaming.
  * ``behavioral_collapse`` — extreme deviation; the policy has degenerated.

The class boundaries are config-tunable. The rules use only fields already in
state, so classification is reproducible and audit-explainable.

Output: list ``[{entry_id, anomaly_type, severity}]``. Logic preserved verbatim.
"""

from __future__ import annotations

from typing import Any

# Multipliers applied to the drift threshold to delimit the three classes.
# score in [1x, hacking) → policy_drift; [hacking, collapse) → reward_hacking;
# >= collapse → behavioral_collapse. (All relative to drift_threshold.)
DEFAULT_DRIFT_THRESHOLD = 3.0
DEFAULT_HACKING_MULT = 1.5
DEFAULT_COLLAPSE_MULT = 2.5


def classify_anomalies(
    scored: list[Any], threshold: float, hacking_mult: float, collapse_mult: float
) -> list[dict[str, Any]]:
    """3-class anomaly classification of flagged entries."""
    hacking_at = threshold * hacking_mult
    collapse_at = threshold * collapse_mult

    anomalies: list[dict[str, Any]] = []
    for entry in scored:
        if not isinstance(entry, dict) or not entry.get("threshold_exceeded"):
            continue
        score = _as_float(entry.get("score"), 0.0)
        anomaly_type, severity = _classify(score, threshold, hacking_at, collapse_at)
        anomalies.append(
            {
                "entry_id": entry.get("entry_id"),
                "anomaly_type": anomaly_type,
                "severity": severity,
            }
        )
    return anomalies


def _classify(score: float, threshold: float, hacking_at: float, collapse_at: float) -> tuple[str, str]:
    """Map a deviation score onto (anomaly_type, severity)."""
    if score >= collapse_at:
        return "behavioral_collapse", "critical"
    if score >= hacking_at:
        return "reward_hacking", "high"
    # >= threshold (guaranteed by threshold_exceeded) but below hacking band.
    return "policy_drift", "medium"


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
