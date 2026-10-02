"""Integration tests — full graph compile() + invoke().

PB-6 full pipeline: Graph().compile() then agent.invoke(user_input, ctx=) runs
initialize → pre_process → main → post_process → finalize and returns a success
result with the masked report in `output`.

PB-1 S-1 trust gate: an under-trusted caller is blocked before the privileged
node's execute() runs (status error).
"""
from uuid import uuid4

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from src.graph.graph import RLAnomalyDetectionAgent

_BASELINE = {"reward_mean": 0.0, "reward_std": 1.0, "action_profile": {}}


def _internal_ctx():
    return InvocationContext(
        correlation_id=str(uuid4()),
        session_id="sess-int",
        caller_trust_level=TrustLevel.INTERNAL,
    )


def _input_context(rewards, fmt="generic"):
    return {
        "rl_framework": "art",
        "raw_logs": [{"entry_id": f"e{i}", "reward": r, "observation": "obs"} for i, r in enumerate(rewards)],
        "baseline": _BASELINE,
        "report_format": fmt,
        "generated_at": "2026-06-18T00:00:00Z",
    }


class TestFullPipeline:
    def test_compile_and_invoke_success(self):
        agent = RLAnomalyDetectionAgent()
        agent.compile()
        result = agent.invoke(
            "rl_anomaly_scan",
            ctx=_internal_ctx(),
            input_context=_input_context([0.0, 8.0]),  # one collapse anomaly
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["output"]  # masked report (formatted_output)
        # node_history runs the fixed 5-node backbone (class names appended by
        # BaseNode.__call__): Initialize → PreProcess → Main → PostProcess → Finalize.
        history = result["node_history"]
        assert len(history) == 5
        assert history[0].startswith("Initialize")
        assert history[-1].startswith("Finalize")
        assert any("PreProcess" in h for h in history)
        assert any("Main" in h for h in history)
        assert any("PostProcess" in h for h in history)

    def test_invoke_clean_when_no_drift(self):
        agent = RLAnomalyDetectionAgent()
        agent.compile()
        result = agent.invoke(
            "rl_anomaly_scan",
            ctx=_internal_ctx(),
            input_context=_input_context([0.1, 0.2]),  # within 3-sigma → clean
        )
        assert result["status"] == AgentStatus.SUCCESS.value
        assert '"clean"' in result["output"]

    def test_missing_baseline_routes_to_error(self):
        agent = RLAnomalyDetectionAgent()
        agent.compile()
        ic = _input_context([1.0])
        ic.pop("baseline")
        result = agent.invoke("rl_anomaly_scan", ctx=_internal_ctx(), input_context=ic)
        # pre_process sets ERROR; main short-circuits; route → finalize.
        assert result["status"] == AgentStatus.ERROR.value


class TestS1TrustGate:
    def test_under_trusted_caller_blocked(self):
        agent = RLAnomalyDetectionAgent()
        agent.compile()
        ctx = InvocationContext(
            correlation_id=str(uuid4()),
            session_id="sess-anon",
            caller_trust_level=TrustLevel.ANONYMOUS,  # below INTERNAL
        )
        result = agent.invoke("rl_anomaly_scan", ctx=ctx, input_context=_input_context([0.0]))
        # The privileged pre_process node's S-1 gate denies before execute().
        assert result["status"] == AgentStatus.ERROR.value
