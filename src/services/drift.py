"""Reward-drift compute service (Step 3 — .

Computes the reward-distribution delta between the parsed logs and the baseline,
plus an *optional* semantic-similarity distance.

──────────────────────────────────────────────────────────────────────────────
sentence-transformers dependency : the semantic-similarity layer
depends on the ``sentence-transformers`` package, which is not a declared
dependency of this template. This service ships a **deterministic reward-distribution-delta core**
(always available) and treats semantic similarity as an **optional hook** that
no-ops when no scorer is injected. The package is NOT declared in pyproject.toml;
there is no hard import of it anywhere.

To enable it, install the package yourself and enable the layer by injecting a callable via
``input_context["semantic_scorer"]`` (signature
``(observation: str, baseline_profile: dict) -> float`` returning a distance in
[0, 1]); the service will use it and populate ``semantic_dist``. Without it
``semantic_dist`` is ``None`` and detection runs on the deterministic delta only.
This is documented in docs/02_design.md §3.
──────────────────────────────────────────────────────────────────────────────

Output: list ``[{entry_id, reward_delta, semantic_dist}]``. Logic preserved
verbatim from the original node.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

_logger = logging.getLogger(__name__)


def compute_deviations(
    parsed: list[Any],
    baseline: dict[str, Any],
    scorer: Any | None = None,
    correlation_id: str | None = None,
) -> list[dict[str, Any]]:
    """Compute per-entry reward delta (+ optional semantic distance)."""
    base_mean = float(baseline.get("reward_mean", 0.0))
    base_std = float(baseline.get("reward_std", 0.0))
    action_profile = baseline.get("action_profile") if isinstance(baseline.get("action_profile"), dict) else {}
    scorer = scorer if callable(scorer) else None

    scores: list[dict[str, Any]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        reward = entry.get("reward")
        reward_delta = _reward_delta(reward, base_mean, base_std)
        profile = action_profile if isinstance(action_profile, dict) else {}
        semantic_dist = _semantic_dist(scorer, entry.get("observation_digest"), profile, correlation_id)
        scores.append(
            {
                "entry_id": entry.get("entry_id"),
                "reward_delta": reward_delta,
                "semantic_dist": semantic_dist,
            }
        )
    return scores


def _reward_delta(reward: object, base_mean: float, base_std: float) -> Optional[float]:
    """Standardized reward deviation (|z-score|) vs the baseline.

    Falls back to absolute mean-delta when baseline std is zero. Returns ``None``
    for a non-numeric reward (entry contributes no delta signal).
    """
    try:
        r = float(reward)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if base_std and base_std > 0:
        return round(abs(r - base_mean) / base_std, 6)
    return round(abs(r - base_mean), 6)


def _semantic_dist(
    scorer: Any | None, observation: object, action_profile: dict[str, Any], correlation_id: Any | None = None
) -> Optional[float]:
    """Semantic distance via the injected scorer, or ``None`` if unavailable.

    Never imports sentence-transformers directly. The deployment may inject a
    scorer callable; without one this returns ``None`` and
    the pipeline relies on the deterministic reward delta.

    A misbehaving injected scorer must not break detection — but the failure is
    logged at WARNING (with correlation_id) so a broken scorer in production is
    distinguishable from "scorer absent" (observability).
    """
    if scorer is None or not isinstance(observation, str) or not observation:
        return None
    try:
        dist = float(scorer(observation, action_profile))
    except Exception:  # must not break detection — but surface it
        _logger.warning(
            "semantic_scorer raised; dropping semantic_dist for this entry " "(correlation_id=%s)",
            correlation_id,
        )
        return None
    # Clamp into [0, 1] so downstream scoring is well-bounded.
    return round(max(0.0, min(1.0, dist)), 6)
