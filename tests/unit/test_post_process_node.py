"""Unit tests — PostProcessNode (Step 7 S-3 mask + S-5 audit + S-4 trace).

Asserts the node contract (status enum) and that S-3 masking redacts embedded
secrets/PII in the serialized report while the S-5 audit entry is always emitted.
"""
import inspect
import json

from framework.schemas.agent_status import AgentStatus
from src.nodes.post_process_node import PostProcessNode
from src.services.serialization import dump_json


def _state(report, anomalies=None, input_context=None):
    return {
        "alert_report": dump_json(report),
        "classified_anomalies": dump_json(anomalies or []),
        "input_context": input_context or {"generated_at": "2026-06-18T00:00:00Z"},
        "session_id": "s-1",
        "correlation_id": "corr-1",
        "rl_framework": "art",
    }


class TestPostProcessNodeContract:
    def test_execute_signature_state_only(self):
        sig = inspect.signature(PostProcessNode.execute)
        params = list(sig.parameters.keys())
        assert params[1] == "state"
        assert "config" not in params
        assert "_invoke_impl" not in PostProcessNode.__dict__


class TestPostProcessNodeMasking:
    def setup_method(self):
        self.node = PostProcessNode()

    def test_clean_report_passes_through(self):
        report = {"format": "generic", "summary": {"status": "clean"}, "findings": []}
        result = self.node.execute(_state(report))
        assert result["status"] == AgentStatus.SUCCESS.value
        masked = json.loads(result["masked_report"])
        assert masked["summary"]["status"] == "clean"
        # masked_report mirrored into formatted_output for get_output()
        assert result["formatted_output"] == result["masked_report"]

    def test_embedded_secret_is_redacted(self):
        report = {
            "format": "generic",
            "findings": [{"entry_id": "e0", "note": "leaked sk-ABCDEFGHIJKLMNOPQRSTUVWX"}],
        }
        result = self.node.execute(_state(report, anomalies=[{"entry_id": "e0"}]))
        assert "sk-ABCDEFGHIJKLMNOPQRSTUVWX" not in result["masked_report"]
        assert "[REDACTED]" in result["masked_report"]
        audit = json.loads(result["audit_log"])
        assert audit[0]["s3_redactions_applied"] is True

    def test_email_pii_is_redacted(self):
        report = {"findings": [{"note": "contact alice@example.com"}]}
        result = self.node.execute(_state(report))
        assert "alice@example.com" not in result["masked_report"]

    def test_audit_entry_always_emitted(self):
        report = {"format": "generic", "findings": []}
        result = self.node.execute(_state(report))
        audit = json.loads(result["audit_log"])
        assert audit[0]["event"] == "rl_anomaly_scan"
        assert audit[0]["template_id"] == "CMN-C1-179"
        assert audit[0]["s3_masked"] is True
