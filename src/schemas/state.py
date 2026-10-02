"""CMN-C1-179 — state schema.

``RLAnomalyState`` extends the L1 ``AgentState`` (a flat TypedDict). It adds only
agent-specific fields, every one a **primitive** (``str``) or a **JSON-serialized
``str``**, so the state is msgpack-safe for LangGraph checkpointing. All shared fields (user_input, input_context,
status, session_id, correlation_id, node_history, error_log, hitl_*, …) are
inherited from ``AgentState``.

Structured collections (parsed logs, deviation/scored entries, classified
anomalies, the alert report, the audit trail) are stored as **JSON-serialized
``str``** — never as live ``list`` / ``dict`` objects. Each producing slot
``dump_json`` on write and each consuming slot ``load_list`` / ``load_dict`` on
read (helpers in ``src.services.serialization``). A single JSON ``str`` is
unambiguously round-trippable across framework serializer versions.

Hard rules (enforced by usage + PB-2/PB-5):
  * No Pydantic models / dataclasses / arbitrary Python objects in state.
  * No JWTs, API keys, credentials, or raw sensitive log *values* in state
    beyond what the S-3 gate masks before the report is emitted.
  * ``InvocationContext`` is **never** stored here.
"""

from __future__ import annotations

from typing import Optional

from framework.schemas.agent_state import AgentState


class RLAnomalyState(AgentState):
    """Flat state for the RL anomaly detection pipeline.

    Populated stage by stage:
      pre_process   → parsed_logs, baseline, rl_framework
      main          → deviation_scores, scored_entries, classified_anomalies,
                      alert_report
      post_process  → masked_report, audit_log, formatted_output
    """

    # ── Inputs (echoed / derived; raw inputs arrive via input_context) ────
    # RL framework type: "art" | "grpo" | "openai_rl" | "generic".
    rl_framework: Optional[str]

    # ── Stage outputs (JSON-serialized strings — see module docstring) ─────
    # pre_process — InputParse: json.dumps([{entry_id, reward, action, observation_digest}])
    parsed_logs: Optional[str]
    # pre_process — BaselineLoad: json.dumps({reward_mean, reward_std, action_profile})
    baseline: Optional[str]
    # main — RLRewardDriftCompute: json.dumps([{entry_id, reward_delta, semantic_dist}])
    deviation_scores: Optional[str]
    # main — PolicyDeviationScore: json.dumps([{entry_id, score, threshold_exceeded}])
    scored_entries: Optional[str]
    # main — AnomalyClassify: json.dumps([{entry_id, anomaly_type, severity}])
    classified_anomalies: Optional[str]
    # main — AlertGenerate: json.dumps({format, summary, findings, generated_at})
    alert_report: Optional[str]
    # post_process — SecurityGateOutput (S-3): masked rendition of alert_report
    masked_report: Optional[str]
    # post_process — SecurityGateOutput (S-5): json.dumps([{event, ts, ...}]) audit trail
    audit_log: Optional[str]
    # post_process: mirror of masked_report so get_output() surfaces it
    formatted_output: Optional[str]  # type: ignore[misc]
