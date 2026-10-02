"""PreProcessNode — input parse (Step 1) + baseline load/validate (Step 2).

Reads the caller's RL log batch and baseline policy, normalizes the logs per RL
framework (ART / GRPO / OpenAI RL), and validates the baseline. Stores the
results as JSON strings (msgpack-safe state). A missing/invalid baseline is a
fail-fast error (``AgentStatus.ERROR``) since drift cannot be computed without it.

Caller input contract (framework ``invoke(user_input, ctx=, input_context=)``):
  * ``input_context["raw_logs"]``        — list of raw RL log entries (or JSON str)
  * ``input_context["rl_framework"]``    — "art" | "grpo" | "openai_rl" | "generic"
  * ``input_context["baseline"]``        — {reward_mean, reward_std, action_profile?}
  * ``input_context["max_observation_chars"]`` — optional S-1 truncation cap

Node contract: extends FunctionNode; overrides ``execute(self, state) -> dict``
only (no ``config`` param); returns a partial dict with an ``AgentStatus`` enum.
"""

from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from shared.utils.audit_logger import emit_trace_event
from shared.services.events import emitter
from shared.services.events.types import EventType
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.services.baseline import validate_baseline
from src.services.rl_parsing import DEFAULT_MAX_OBS_CHARS, as_int, parse_logs
from src.services.serialization import dump_json, load_list

# S-1: RL output logs are internal operational telemetry — privileged input.
_S1_TRUST: TrustLevel = TrustLevel.INTERNAL


class PreProcessNode(FunctionNode):
    """Step 1 + 2 — per-framework log normalization (S-1) + baseline validation."""

    required_trust_level: ClassVar[TrustLevel] = _S1_TRUST

    def __init__(self, max_observation_chars: int | None = None) -> None:
        self._max_obs_default = max_observation_chars or DEFAULT_MAX_OBS_CHARS

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        # Progress (non-terminal) event for the caller. Outside the Marketplace runtime
        # this resolves to a no-op emitter, so it is safe on every entry path.
        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Checking the request...",
            metadata={"step": "pre_process"},
        )
        input_context = state.get("input_context") or {}

        # ── Step 1: parse / normalize the RL log batch (S-1 normalization) ──
        max_obs = as_int(input_context.get("max_observation_chars"), self._max_obs_default)
        rl_framework = str(input_context.get("rl_framework") or state.get("rl_framework") or "generic")
        raw = load_list(input_context.get("raw_logs"))
        parsed = parse_logs(raw, rl_framework, max_obs)

        # ── Step 2: load + validate the baseline policy ─────────────────────
        validated, error = validate_baseline(input_context.get("baseline"))
        if error is not None:
            return {
                "parsed_logs": dump_json(parsed),
                "rl_framework": rl_framework,
                "error_log": [error],
                "status": AgentStatus.ERROR.value,
            }

        emit_trace_event(
            "rl_log_ingest",
            {"parsed_entries": len(parsed), "rl_framework": rl_framework},
            state,
        )

        return {
            "parsed_logs": dump_json(parsed),
            "baseline": dump_json(validated),
            "rl_framework": rl_framework,
            "status": AgentStatus.SUCCESS.value,
        }
