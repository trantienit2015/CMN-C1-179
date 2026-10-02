"""Alert report generation service (Step 6 — .

Generates a structured alert report from classified anomalies. The report
*format* is config-driven (ISO 42001 / NIST AI RMF / EU AI Act / FSA AI MRM) —
there is **no hardcoded governance format** in code (Cat 1 = zero domain
assumption; the framework mapping lives in config, keeping the template
industry-agnostic). The function emits a neutral structured object whose
``format`` field records which governance framework label was selected.

The report is built deterministically from state (no LLM call is required for the
structured report). ``generated_at`` is supplied by the caller (deterministic,
testable); the template never calls a wall-clock so runs stay reproducible.

Output: dict ``{format, summary, findings, generated_at}``. Logic preserved verbatim.
"""

from __future__ import annotations

from typing import Any

DEFAULT_FORMAT = "generic"
KNOWN_FORMATS = {"iso_42001", "nist_ai_rmf", "eu_ai_act", "fsa_ai_mrm", "generic"}


def build_report(anomalies: list[Any], report_format: str, generated_at: Any | None = None) -> dict[str, Any]:
    """Build the structured, config-format alert report."""
    fmt = str(report_format or DEFAULT_FORMAT).strip().lower()
    if fmt not in KNOWN_FORMATS:
        fmt = DEFAULT_FORMAT

    findings = [a for a in anomalies if isinstance(a, dict)]

    by_type: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    for a in findings:
        by_type[a.get("anomaly_type", "unknown")] = by_type.get(a.get("anomaly_type", "unknown"), 0) + 1
        by_severity[a.get("severity", "unknown")] = by_severity.get(a.get("severity", "unknown"), 0) + 1

    return {
        "format": fmt,
        "summary": {
            "total_anomalies": len(findings),
            "by_type": by_type,
            "by_severity": by_severity,
            "status": "anomalies_detected" if findings else "clean",
        },
        "findings": findings,
        "generated_at": generated_at,
    }
