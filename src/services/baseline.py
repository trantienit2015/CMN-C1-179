"""Baseline policy load + validation service (Step 2 — .

Validates the baseline policy schema against the expected structure. The baseline
is supplied by the caller via ``input_context["baseline"]`` (a metadata-only object,
no credentials). The template ships with no live baseline store; an absent/invalid
baseline is a fail-fast error because drift cannot be computed without it.

Returns ``(validated_dict, error_message_or_None)``. Logic preserved verbatim from
the original BaselineLoadNode.
"""

from __future__ import annotations

from typing import Any, Optional

# Required keys for a usable baseline. action_profile is optional (semantic
# layer); the reward statistics are mandatory for the deterministic delta.
REQUIRED_KEYS = ("reward_mean", "reward_std")


def validate_baseline(source: object) -> tuple[dict[str, Any], Optional[str]]:
    """Validate the baseline object. Returns ``(validated, error)``.

    ``error`` is a string when validation fails (caller sets AgentStatus.ERROR);
    ``None`` on success.
    """
    baseline = source if isinstance(source, dict) else {}

    if not baseline:
        return {}, "BaselineLoad: no baseline policy supplied (input_context.baseline)"

    missing = [k for k in REQUIRED_KEYS if k not in baseline]
    if missing:
        return {}, f"BaselineLoad: baseline missing required keys: {missing}"

    mean = _as_float(baseline.get("reward_mean"))
    std = _as_float(baseline.get("reward_std"))
    if mean is None or std is None or std < 0:
        return {}, "BaselineLoad: reward_mean/reward_std must be valid numbers (std >= 0)"

    validated = {
        "reward_mean": mean,
        "reward_std": std,
        # action_profile: optional map action->expected frequency (semantic
        # layer input). Kept only if it is a dict.
        "action_profile": baseline.get("action_profile") if isinstance(baseline.get("action_profile"), dict) else {},
    }
    return validated, None


def _as_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
