"""S-3 deterministic masking (regex, NOT LLM) for CMN-C1-179.

Sensitive value patterns are masked before any alert report leaves the agent.
These are content-value detectors (NOT field-name detectors): RL output logs may
embed secrets/PII directly in observations. Masking is deterministic and
idempotent — re-masking already-masked text is a no-op.

This is the S-3 enforcement primitive used by the terminal post_process slot.
"""

from __future__ import annotations

import re
from typing import Any

_MASK = "[REDACTED]"
_SENSITIVE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9]{20,}"),  # OpenAI-style key
    re.compile(r"eyJ[A-Za-z0-9._\-]{10,}"),  # JWT
    re.compile(r"AKIA[A-Z0-9]{16}"),  # AWS access key
    re.compile(r"(?i)Bearer\s+[A-Za-z0-9._\-]{20,}"),  # bearer token
    re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"),  # email
    re.compile(r"\b\d{12}\b"),  # my-number (12 digit)
    re.compile(r"\b(?:\d[ \-]?){13,16}\b"),  # card-like PAN
)


def mask_sensitive(text: Any) -> str:
    """Deterministically mask sensitive substrings in ``text``.

    Returns a string with every sensitive match replaced by ``[REDACTED]``.
    Non-string input is coerced via ``str()``. Pure/deterministic — no LLM,
    no network, idempotent. This is the S-3 enforcement primitive.
    """
    out = text if isinstance(text, str) else str(text)
    for pat in _SENSITIVE_PATTERNS:
        out = pat.sub(_MASK, out)
    return out


def contains_sensitive(text: Any) -> bool:
    """True if any sensitive pattern is present in ``text`` (pre-mask check)."""
    s = text if isinstance(text, str) else str(text)
    return any(pat.search(s) for pat in _SENSITIVE_PATTERNS)
