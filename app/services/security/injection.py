"""Lightweight prompt-injection heuristic scanner (docs/security.md
"Prompt injection defense"). Retrieved document/graph content is always
treated as DATA in the generation prompt (see context_builder.py's
explicit delimiters and system instructions) — this scanner doesn't gate
whether content is used as evidence (spec: still usable, just flagged),
it only flags/logs suspicious chunks so the fact that a document tried to
inject instructions is visible in audit logs rather than silently invisible.
"""

import re
from dataclasses import dataclass

# Deliberately simple pattern matching, not ML-based classification — this
# is a first line of defense (raise visibility), not the actual defense.
# The actual defense is that retrieved text is never given instruction
# authority in the prompt regardless of what it contains (see
# context_builder.py).
_SUSPICIOUS_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"system\s*prompt", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?(system|hidden|confidential)", re.IGNORECASE),
    re.compile(r"act\s+as\s+if\s+you\s+(are|were)", re.IGNORECASE),
    re.compile(r"</?(system|instructions?)>", re.IGNORECASE),
]


@dataclass
class InjectionScanResult:
    is_suspicious: bool
    matched_patterns: list[str]


def scan_for_injection(text: str) -> InjectionScanResult:
    matched = [p.pattern for p in _SUSPICIOUS_PATTERNS if p.search(text)]
    return InjectionScanResult(is_suspicious=bool(matched), matched_patterns=matched)
