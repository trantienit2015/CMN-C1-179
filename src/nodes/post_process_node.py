"""PostProcessNode — S-3 output masking + S-5 audit trail (Step 7) + S-4 audit event.

The terminal gate of the pipeline :

  * **S-3 (MANDATORY)** — deterministic masking of sensitive values in the
    serialized alert report (regex, NOT LLM) so any secret/PII embedded in a
    finding's text is redacted before the report leaves the agent. Empty /
    pass-through here would be a CRITICAL violation.
  * **S-5** — emits an ISO 42001-style audit-trail entry (JSON, timestamped) for
    the operation so every run produces governance evidence.
  * **S-4** — ``emit_trace_event`` records the scan on every exit path.

Output state fields: ``masked_report`` (S-3-masked JSON string), ``audit_log``
(JSON-serialized audit entries), ``formatted_output`` (mirror of masked_report so
the framework ``get_output()`` surfaces it).

Node contract: extends FunctionNode; overrides ``execute(self, state) -> dict``
only; returns a partial dict with an ``AgentStatus`` enum.
"""

from __future__ import annotations

from typing import Any, ClassVar

from framework.nodes.function_node import FunctionNode
from shared.services.events import emitter
from shared.services.events.types import EventType
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event

from src.services.masking import contains_sensitive, mask_sensitive
from src.services.serialization import dump_json, load_dict, load_list

_TEMPLATE_ID = "CMN-C1-179"
_S1_TRUST: TrustLevel = TrustLevel.INTERNAL


class PostProcessNode(FunctionNode):
    """Step 7 — S-3 deterministic masking + S-5 audit trail (terminal gate)."""

    required_trust_level: ClassVar[TrustLevel] = _S1_TRUST

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        # Progress (non-terminal) event for the caller. Outside the Marketplace runtime
        # this resolves to a no-op emitter, so it is safe on every entry path.
        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Preparing the result...",
            metadata={"step": "post_process"},
        )
        input_context = state.get("input_context") or {}
        ts = input_context.get("generated_at")

        report = load_dict(state.get("alert_report"))
        anomaly_count = len(load_list(state.get("classified_anomalies")))

        # S-3 MANDATORY: mask the *serialized* report so any sensitive substring
        # embedded anywhere (finding text, observation digests) is redacted.
        serialized = dump_json(report) or "{}"
        had_sensitive = contains_sensitive(serialized)
        masked = mask_sensitive(serialized)

        # S-5: ISO 42001-style audit entry.
        audit = self._audit_entry(state, ts, status="success", anomaly_count=anomaly_count, s3_redactions=had_sensitive)

        # S-4: structured audit event on the success path (no PII values).
        emit_trace_event(
            "rl_anomaly_scan",
            {"anomaly_count": anomaly_count, "s3_redactions_applied": had_sensitive},
            state,
        )

        return {
            "masked_report": masked,
            "formatted_output": masked,
            "audit_log": dump_json([audit]),
            "status": AgentStatus.SUCCESS.value,
        }

    @staticmethod
    def _audit_entry(
        state: dict[str, Any], ts: str | None, *, status: str, anomaly_count: int = 0, s3_redactions: bool = False
    ) -> dict[str, Any]:
        """Build a PII-free ISO 42001-style audit entry (S-5)."""
        return {
            "template_id": _TEMPLATE_ID,
            "event": "rl_anomaly_scan",
            "status": status,
            "session_id": state.get("session_id"),
            "correlation_id": state.get("correlation_id") or state.get("session_id"),
            "rl_framework": state.get("rl_framework"),
            "anomaly_count": anomaly_count,
            "s3_masked": True,
            "s3_redactions_applied": s3_redactions,
            "ts": ts,
        }
