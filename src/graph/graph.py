"""Graph composition for CMN-C1-179 (the graph IS the agent).

``RLAnomalyDetectionAgent`` inherits the L1 ``AgentBaseGraph`` directly (2026-05-18
policy — no Level 2 parent). It registers the fixed 5-node backbone via
``super().register_nodes()`` (which injects ``initialize`` + ``finalize``) and then
fills the three domain slots. The framework owns ``add_edges()`` / ``route()`` /
the S-1 trust gate / ``__call__`` — this class does not override them.

Cat 1 backbone (industry-agnostic, single capability — RL policy-drift / anomaly
detection):

    initialize → pre_process → main → {route} → post_process → finalize

The 7 documented domain steps live inside the slots, each delegating to a
deterministic service so no business logic is dropped:
  * pre_process : InputParse (1) + BaselineLoad (2)
  * main        : RLRewardDriftCompute (3) → PolicyDeviationScore (4)
                  → AnomalyClassify (5) → AlertGenerate (6)
  * post_process: SecurityGateOutput (7) — S-3 mask + S-5 audit + S-4 trace

Public entry: ``RLAnomalyDetectionAgent().compile()`` then
``agent.invoke(user_input, ctx=InvocationContext(...), input_context={...})`` —
never ``.run()`` and never a separate agent class driving the graph.

Domain config (thresholds / multipliers / report format / observation cap) is
injected through the graph ``config`` dict into the node constructors — never via
a ``config`` param on ``execute()`` and never stored in state.
"""

from __future__ import annotations

from framework.graph.agent_base_graph import AgentBaseGraph

from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import RLAnomalyState


class RLAnomalyDetectionAgent(AgentBaseGraph):
    """Cat 1 RL policy-drift / anomaly detection agent — L1 direct."""

    @property
    def name(self) -> str:
        return "RLAnomalyDetectionAgent"

    @property
    def state_schema(self) -> type:
        return RLAnomalyState

    def register_nodes(self) -> None:
        super().register_nodes()  # injects default InitializeNode + FinalizeNode

        cfg = self.config if isinstance(self.config, dict) else {}

        self._nodes["pre_process"] = PreProcessNode(
            max_observation_chars=cfg.get("max_observation_chars"),
        )
        self._nodes["main"] = MainNode(
            drift_threshold=cfg.get("drift_threshold"),
            semantic_weight=cfg.get("semantic_weight"),
            hacking_multiplier=cfg.get("hacking_multiplier"),
            collapse_multiplier=cfg.get("collapse_multiplier"),
            report_format=cfg.get("report_format"),
        )
        self._nodes["post_process"] = PostProcessNode()


# Alias so the manifest module path `src.graph` exposes a stable name.
Graph = RLAnomalyDetectionAgent
