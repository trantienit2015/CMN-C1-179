"""Unit tests — PreProcessNode (Step 1 InputParse + Step 2 BaselineLoad).

Asserts the node contract (execute(self, state); status is an AgentStatus enum)
and the parse/validate logic across RL frameworks + baseline error paths.
"""
import inspect
import json

from framework.schemas.agent_status import AgentStatus
from src.nodes.pre_process_node import PreProcessNode

_BASELINE = {"reward_mean": 1.0, "reward_std": 0.5, "action_profile": {"a": 0.5}}


class TestPreProcessNodeContract:
    def test_execute_signature_state_only(self):
        sig = inspect.signature(PreProcessNode.execute)
        params = list(sig.parameters.keys())
        assert params[1] == "state"
        # No `config` param; no _invoke_impl override.
        assert "config" not in params
        assert "_invoke_impl" not in PreProcessNode.__dict__


class TestPreProcessNodeParse:
    def setup_method(self):
        self.node = PreProcessNode()

    def test_art_framework_normalization(self):
        state = {
            "input_context": {
                "rl_framework": "art",
                "raw_logs": [{"reward": 2.0, "action": "buy", "observation": "<b>ctx</b>"}],
                "baseline": _BASELINE,
            }
        }
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        parsed = json.loads(result["parsed_logs"])
        assert len(parsed) == 1
        assert parsed[0]["reward"] == 2.0
        assert parsed[0]["action"] == "buy"
        # HTML strip applied (S-1 normalization).
        assert parsed[0]["observation_digest"] == "ctx"
        assert result["rl_framework"] == "art"

    def test_grpo_field_aliases(self):
        state = {
            "input_context": {
                "rl_framework": "grpo",
                "raw_logs": [{"group_reward": 5.0, "policy_action": "x", "obs": "o"}],
                "baseline": _BASELINE,
            }
        }
        result = self.node.execute(state)
        parsed = json.loads(result["parsed_logs"])
        assert parsed[0]["reward"] == 5.0
        assert parsed[0]["action"] == "x"

    def test_unknown_framework_parsed_via_generic_adapter(self):
        # An unknown framework label falls back to the generic adapter internally
        # (the caller's label is echoed back unchanged; parsing still succeeds).
        state = {
            "input_context": {
                "rl_framework": "weird",
                "raw_logs": [{"reward": 1.0}],
                "baseline": _BASELINE,
            }
        }
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        parsed = json.loads(result["parsed_logs"])
        # generic adapter resolved the reward despite the unknown label
        assert parsed[0]["reward"] == 1.0

    def test_empty_batch_is_valid_not_error(self):
        state = {"input_context": {"rl_framework": "art", "raw_logs": [], "baseline": _BASELINE}}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        assert json.loads(result["parsed_logs"]) == []

    def test_observation_truncation_cap(self):
        node = PreProcessNode(max_observation_chars=5)
        state = {
            "input_context": {
                "rl_framework": "art",
                "raw_logs": [{"reward": 1.0, "observation": "abcdefghij"}],
                "baseline": _BASELINE,
            }
        }
        result = node.execute(state)
        parsed = json.loads(result["parsed_logs"])
        assert parsed[0]["observation_digest"] == "abcde"


class TestPreProcessNodeBaselineErrors:
    def setup_method(self):
        self.node = PreProcessNode()

    def test_missing_baseline_is_error(self):
        state = {"input_context": {"rl_framework": "art", "raw_logs": [{"reward": 1.0}]}}
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.ERROR.value
        assert result["error_log"]

    def test_baseline_missing_required_keys_is_error(self):
        state = {
            "input_context": {
                "rl_framework": "art",
                "raw_logs": [{"reward": 1.0}],
                "baseline": {"reward_mean": 1.0},
            }
        }
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.ERROR.value

    def test_negative_std_is_error(self):
        state = {
            "input_context": {
                "rl_framework": "art",
                "raw_logs": [{"reward": 1.0}],
                "baseline": {"reward_mean": 1.0, "reward_std": -1.0},
            }
        }
        result = self.node.execute(state)
        assert result["status"] == AgentStatus.ERROR.value
