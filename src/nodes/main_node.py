"""MainNode — reward-drift compute (Step 3) → policy-deviation scoring (Step 4)
→ anomaly classification (Step 5) → alert report generation (Step 6).

The deterministic detection core. Each step delegates to a dedicated service so
the original per-node logic is preserved verbatim; the slot only chains them:

    deviation_scores = compute_deviations(parsed_logs, baseline, scorer)
    scored_entries   = score_entries(deviation_scores, drift_threshold, semantic_weight)
    anomalies        = classify_anomalies(scored_entries, threshold, hacking, collapse)
    alert_report     = build_report(anomalies, report_format, generated_at)

Config knobs (drift_threshold / semantic_weight / multipliers / report_format)
are injected via the graph config into this node's constructor (no ``config``
param on ``execute``). The optional semantic scorer and ``generated_at`` are
passed by the caller via ``input_context`` (deterministic, no wall-clock).

Detection (steps 3-5) and the structured report shape (step 6) stay fully
deterministic and are never touched by the LLM step below — only an optional
``narrative`` field is added to the already-complete report. An LLM (Azure
OpenAI) is used to add a natural-language executive summary of the structured
findings, written for the selected governance `report_format` — genuine NL
synthesis over classified findings, not a templated string. Any failure
(missing secret, API error, malformed response, no findings) silently omits
`narrative`; the deterministic report is never blocked on it and `status`
stays `success`.

Node contract: extends FunctionNode; overrides ``execute(self, state) -> dict``
only; returns a partial dict with an ``AgentStatus`` enum.
"""

from __future__ import annotations

from typing import Any, ClassVar, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.invocation_context import InvocationContext
from shared.utils.audit_logger import emit_trace_event
from shared.services.events import emitter
from shared.services.events.types import EventType
from shared.services.llm.azure_openai_client import AzureOpenAIClient
from shared.utils.llm_json import extract_json_object
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.services.classify import (
    DEFAULT_COLLAPSE_MULT,
    DEFAULT_HACKING_MULT,
    classify_anomalies,
)
from src.services.drift import compute_deviations
from src.services.report import DEFAULT_FORMAT, build_report
from src.services.scoring import (
    DEFAULT_DRIFT_THRESHOLD,
    DEFAULT_SEMANTIC_WEIGHT,
    score_entries,
)
from src.services.serialization import dump_json, load_dict, load_list

_S1_TRUST: TrustLevel = TrustLevel.INTERNAL

_MAX_NARRATIVE_CHARS = 2000

_NARRATIVE_SYSTEM_PROMPT = (
    "You are an AI-governance compliance analyst. Given a JSON object describing "
    "RL policy-drift anomaly findings (a governance report format label, counts by "
    "anomaly type and severity, and the individual findings), write a concise "
    "executive narrative (2-4 sentences, plain prose, no markdown) summarizing what "
    "was found and its governance significance for the given report format. If there "
    "are no findings, state plainly that the run is clean. Respond with a single JSON "
    'object only, no prose outside it, no markdown fences: {"narrative": "<text>"}. '
    "Base the narrative only on the counts and findings given — never invent figures, "
    "entry ids, or anomaly types absent from the input."
)


class MainNode(FunctionNode):
    """Steps 3-6 — drift → score → classify → report (deterministic core)."""

    required_trust_level: ClassVar[TrustLevel] = _S1_TRUST

    def __init__(
        self,
        drift_threshold: Optional[float] = None,
        semantic_weight: Optional[float] = None,
        hacking_multiplier: Optional[float] = None,
        collapse_multiplier: Optional[float] = None,
        report_format: Optional[str] = None,
        llm: Any | None = None,
    ) -> None:
        self._drift_threshold = _as_float(drift_threshold, DEFAULT_DRIFT_THRESHOLD)
        self._semantic_weight = _as_float(semantic_weight, DEFAULT_SEMANTIC_WEIGHT)
        self._hacking_multiplier = _as_float(hacking_multiplier, DEFAULT_HACKING_MULT)
        self._collapse_multiplier = _as_float(collapse_multiplier, DEFAULT_COLLAPSE_MULT)
        self._report_format = report_format or DEFAULT_FORMAT
        # Test-double seam only — register_nodes() never passes one in production.
        # The real client is built fresh per invocation in _generate_narrative()
        # from ctx.secrets, never cached on self: node instances are constructed
        # once in register_nodes() (registry LRU cache, shared across every
        # invocation) before any request's secrets are provisioned, so caching a
        # client built from one caller's secrets would leak it to the next caller.
        self._llm = llm

    def execute(self, state: dict[str, Any]) -> dict[str, Any]:
        # Progress (non-terminal) event for the caller. Outside the Marketplace runtime
        # this resolves to a no-op emitter, so it is safe on every entry path.
        emitter().emit_event(
            event_type=EventType.PROGRESS_UPDATE,
            message="Working on the request...",
            metadata={"step": "main"},
        )
        baseline = load_dict(state.get("baseline"))
        if not baseline:
            return {
                "error_log": ["MainNode: baseline missing (pre_process must run first)"],
                "status": AgentStatus.ERROR.value,
            }

        parsed = load_list(state.get("parsed_logs"))
        input_context = state.get("input_context") or {}
        scorer = input_context.get("semantic_scorer")
        generated_at = input_context.get("generated_at")

        # Step 3 — reward-distribution delta (deterministic) + optional semantic dist
        deviations = compute_deviations(parsed, baseline, scorer, state.get("correlation_id"))

        # Step 4 — threshold compare + per-entry final scoring
        scored = score_entries(deviations, self._drift_threshold, self._semantic_weight)

        # Step 5 — 3-class anomaly classification of flagged entries
        anomalies = classify_anomalies(
            scored, self._drift_threshold, self._hacking_multiplier, self._collapse_multiplier
        )

        # Step 6 — structured, config-format alert report
        report = build_report(anomalies, self._report_format, generated_at)

        # Step 6b (optional) — LLM-authored executive narrative over the report
        # already built above. Never changes detection or the structured fields;
        # only adds `narrative` when the call succeeds and validates.
        narrative = self._generate_narrative(anomalies, report, state)
        if narrative is not None:
            report["narrative"] = narrative

        emit_trace_event(
            "rl_anomaly_classification",
            {
                "scored_entries": len(scored),
                "anomalies": len(anomalies),
                "report_format": self._report_format,
                "narrative_generated": narrative is not None,
            },
            state,
        )

        return {
            "deviation_scores": dump_json(deviations),
            "scored_entries": dump_json(scored),
            "classified_anomalies": dump_json(anomalies),
            "alert_report": dump_json(report),
            "status": AgentStatus.SUCCESS.value,
        }

    def _generate_narrative(
        self, anomalies: list[dict[str, Any]], report: dict[str, Any], state: dict[str, Any]
    ) -> Optional[str]:
        """LLM-authored executive narrative over an already-built report.

        Returns None on any failure or when there is nothing to narrate (no
        findings) — caller keeps the structured report exactly as built, with
        no `narrative` key. Never raises.
        """
        if not anomalies:
            return None
        try:
            llm = self._llm
            if llm is None:
                ctx = InvocationContext.from_state(state)
                llm = AzureOpenAIClient(
                    {
                        "api_key": ctx.secrets.require("AZURE_OPENAI_API_KEY"),
                        "azure_endpoint": ctx.secrets.require("AZURE_OPENAI_ENDPOINT"),
                        "azure_deployment": ctx.secrets.require("AZURE_OPENAI_DEPLOYMENT"),
                    }
                )
            payload = {
                "format": report.get("format"),
                "summary": report.get("summary"),
                "findings": report.get("findings"),
            }
            response = llm.complete(
                [
                    {"role": "system", "content": _NARRATIVE_SYSTEM_PROMPT},
                    {"role": "user", "content": dump_json(payload)},
                ]
            )
            parsed = extract_json_object(response.get("content", ""))
            return _validate_narrative(parsed)
        except Exception:
            return None


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _validate_narrative(parsed: object) -> Optional[str]:
    """Accept only a non-empty, length-capped string under the "narrative" key."""
    if not isinstance(parsed, dict):
        return None
    text = parsed.get("narrative")
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    return text[:_MAX_NARRATIVE_CHARS]
