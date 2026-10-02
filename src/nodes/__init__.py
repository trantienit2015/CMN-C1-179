"""CMN-C1-179 node package — Cat 1 backbone domain slots.

The fixed 5-node backbone (initialize / pre_process / main / post_process /
finalize) is provided by the L1 framework; this package supplies the three
domain slots. The 7 documented RL-anomaly steps live inside these slots, each
delegating to a deterministic service in ``src.services`` (logic preserved):

  pre_process  → InputParse (1) + BaselineLoad (2)
  main         → RLRewardDriftCompute (3) → PolicyDeviationScore (4)
                 → AnomalyClassify (5) → AlertGenerate (6)
  post_process → SecurityGateOutput (7) — S-3 mask + S-5 audit + S-4 trace
"""

from src.nodes.main_node import MainNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

__all__ = [
    "PreProcessNode",
    "MainNode",
    "PostProcessNode",
]
