"""Unit tests — MainNode (Step 3 drift → 4 score → 5 classify → 6 report).

Covers the deterministic detection chain: z-score delta, threshold flagging, the
3-class taxonomy boundaries, the config-driven report format, and the
baseline-missing error path. Asserts the node contract (status enum).
"""
import inspect
import json

from framework.schemas.agent_status import AgentStatus
from src.nodes.main_node import MainNode
from src.services.serialization import dump_json

_BASELINE = {"reward_mean": 0.0, "reward_std": 1.0, "action_profile": {}}


def _state(parsed, baseline=_BASELINE, input_context=None):
    return {
        "parsed_logs": dump_json(parsed),
        "baseline": dump_json(baseline),
        "input_context": input_context or {},
        "correlation_id": "corr-1",
    }


class TestMainNodeContract:
    def test_execute_signature_state_only(self):
        sig = inspect.signature(MainNode.execute)
        params = list(sig.parameters.keys())
        assert params[1] == "state"
        assert "config" not in params
        assert "_invoke_impl" not in MainNode.__dict__


class TestMainNodeDetection:
    def test_below_threshold_no_anomaly(self):
        # reward 1.0 vs mean 0 std 1 → z=1.0 < default threshold 3.0
        node = MainNode()
        result = node.execute(_state([{"entry_id": "e0", "reward": 1.0}]))
        assert result["status"] == AgentStatus.SUCCESS.value
        anomalies = json.loads(result["classified_anomalies"])
        assert anomalies == []
        report = json.loads(result["alert_report"])
        assert report["summary"]["status"] == "clean"

    def test_policy_drift_band(self):
        # z=3.0 (== threshold) and < hacking_at 4.5 → policy_drift / medium
        node = MainNode()
        result = node.execute(_state([{"entry_id": "e0", "reward": 3.0}]))
        anomalies = json.loads(result["classified_anomalies"])
        assert len(anomalies) == 1
        assert anomalies[0]["anomaly_type"] == "policy_drift"
        assert anomalies[0]["severity"] == "medium"

    def test_reward_hacking_band(self):
        # z=5.0 >= hacking_at (3*1.5=4.5), < collapse_at (3*2.5=7.5) → reward_hacking
        node = MainNode()
        result = node.execute(_state([{"entry_id": "e0", "reward": 5.0}]))
        anomalies = json.loads(result["classified_anomalies"])
        assert anomalies[0]["anomaly_type"] == "reward_hacking"
        assert anomalies[0]["severity"] == "high"

    def test_behavioral_collapse_band(self):
        # z=8.0 >= collapse_at 7.5 → behavioral_collapse / critical
        node = MainNode()
        result = node.execute(_state([{"entry_id": "e0", "reward": 8.0}]))
        anomalies = json.loads(result["classified_anomalies"])
        assert anomalies[0]["anomaly_type"] == "behavioral_collapse"
        assert anomalies[0]["severity"] == "critical"

    def test_custom_threshold_via_constructor(self):
        # threshold 1.0 → z=2.0 flagged; collapse_at 1*2.5=2.5 so z2 < collapse,
        # hacking_at 1.5 → z2 >= hacking → reward_hacking
        node = MainNode(drift_threshold=1.0)
        result = node.execute(_state([{"entry_id": "e0", "reward": 2.0}]))
        anomalies = json.loads(result["classified_anomalies"])
        assert anomalies[0]["anomaly_type"] == "reward_hacking"

    def test_report_format_from_constructor(self):
        node = MainNode(report_format="iso_42001")
        result = node.execute(_state([{"entry_id": "e0", "reward": 0.0}]))
        report = json.loads(result["alert_report"])
        assert report["format"] == "iso_42001"

    def test_unknown_report_format_falls_back_generic(self):
        node = MainNode(report_format="made_up")
        result = node.execute(_state([{"entry_id": "e0", "reward": 0.0}]))
        report = json.loads(result["alert_report"])
        assert report["format"] == "generic"

    def test_semantic_scorer_blended_when_weight_positive(self):
        node = MainNode(drift_threshold=3.0, semantic_weight=1.0)
        # weight 1.0 → score == semantic_dist (1.0) < threshold → no anomaly,
        # but proves the scorer path is exercised without raising.
        scorer = lambda obs, prof: 1.0  # noqa: E731
        result = node.execute(
            _state(
                [{"entry_id": "e0", "reward": 0.0, "observation_digest": "x"}],
                input_context={"semantic_scorer": scorer},
            )
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        scored = json.loads(result["scored_entries"])
        assert scored[0]["score"] == 1.0


class TestMainNodeErrors:
    def test_missing_baseline_is_error(self):
        node = MainNode()
        state = {"parsed_logs": dump_json([{"entry_id": "e0", "reward": 1.0}]), "input_context": {}}
        result = node.execute(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert result["error_log"]


class _FakeLLM:
    """Test double for AzureOpenAIClient — no network call, matches the ``complete`` shape."""

    def __init__(self, content=None, raises=False):
        self._content = content
        self._raises = raises
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        if self._raises:
            raise RuntimeError("simulated Azure OpenAI API error")
        return {"content": self._content}


class TestMainNodeNarrative:
    """Step 8e — optional LLM-authored executive narrative over the report.

    Detection stays deterministic in every case; only the ``narrative`` key on
    ``alert_report`` varies. No test in this class makes a real network call —
    every LLM interaction goes through ``_FakeLLM``.
    """

    def _anomaly_state(self):
        # z = 8.0 >= collapse_at 7.5 → one behavioral_collapse anomaly, so the
        # report has findings and the narrative path is actually reachable.
        return _state([{"entry_id": "e0", "reward": 8.0}])

    def test_narrative_added_on_valid_json_response(self):
        llm = _FakeLLM(content='{"narrative": "One critical behavioral-collapse finding."}')
        node = MainNode(llm=llm)
        result = node.execute(self._anomaly_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        report = json.loads(result["alert_report"])
        assert report["narrative"] == "One critical behavioral-collapse finding."
        assert llm.calls == 1

    def test_narrative_parses_markdown_fenced_response(self):
        llm = _FakeLLM(content='Sure, here you go:\n```json\n{"narrative": "Collapse detected."}\n```')
        node = MainNode(llm=llm)
        result = node.execute(self._anomaly_state())
        report = json.loads(result["alert_report"])
        assert report["narrative"] == "Collapse detected."

    def test_malformed_response_falls_back_no_narrative(self):
        llm = _FakeLLM(content="not json at all, no braces")
        node = MainNode(llm=llm)
        result = node.execute(self._anomaly_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        report = json.loads(result["alert_report"])
        assert "narrative" not in report

    def test_wrong_shape_json_falls_back_no_narrative(self):
        llm = _FakeLLM(content='{"summary": "wrong key entirely"}')
        node = MainNode(llm=llm)
        result = node.execute(self._anomaly_state())
        report = json.loads(result["alert_report"])
        assert "narrative" not in report

    def test_llm_raising_falls_back_no_narrative(self):
        llm = _FakeLLM(raises=True)
        node = MainNode(llm=llm)
        result = node.execute(self._anomaly_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        report = json.loads(result["alert_report"])
        assert "narrative" not in report

    def test_no_llm_injected_and_no_secret_bound_falls_back(self):
        # Real production shape in any environment without a configured key:
        # InvocationContext.from_state() / ctx.secrets.require() raises MissingSecret,
        # caught internally — never propagates, never sets status=error.
        node = MainNode()
        result = node.execute(self._anomaly_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        report = json.loads(result["alert_report"])
        assert "narrative" not in report

    def test_clean_report_never_calls_llm(self):
        class _AssertNotCalled:
            def complete(self, messages):
                raise AssertionError("LLM must not be called when there are no findings")

        node = MainNode(llm=_AssertNotCalled())
        # z = 1.0 < threshold 3.0 → no anomalies, nothing to narrate.
        result = node.execute(_state([{"entry_id": "e0", "reward": 1.0}]))
        assert result["status"] == AgentStatus.SUCCESS.value
        report = json.loads(result["alert_report"])
        assert "narrative" not in report
