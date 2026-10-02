"""Service layer placeholder for CMN-C1-179.

This template requires **no external service integration**: all detection logic
is deterministic and self-contained in the nodes (reward-distribution delta,
threshold scoring, rule-based classification, regex masking). The baseline and
any optional semantic scorer are injected via ``config["configurable"]`` by the
deployment, not fetched here.

The module is kept (per the scaffold §5 layout) as the seam where future external
integrations — e.g. a live baseline/policy store or an audit-log sink — would be
added. It intentionally exposes no behavior today; nodes do not import it.
"""

from __future__ import annotations
