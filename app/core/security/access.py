"""Clearance <-> document access_level comparison. See docs/security.md
"Retrieval-time enforcement (not generation-time)": authorization is
computed here and passed into the retrieval query itself (Qdrant payload
filter / BM25 metadata filter / SQL WHERE) — never applied by asking the
LLM to withhold content it was already shown.
"""

from app.models.document import AccessLevel
from app.models.user import ClearanceLevel

_ORDER = [
    ClearanceLevel.PUBLIC,
    ClearanceLevel.INTERNAL,
    ClearanceLevel.DEPARTMENT,
    ClearanceLevel.CONFIDENTIAL,
    ClearanceLevel.RESTRICTED,
]
_RANK = {level: i for i, level in enumerate(_ORDER)}


def authorized_access_levels(clearance: ClearanceLevel) -> list[str]:
    """All AccessLevel values a user with this clearance may see — a
    user's clearance is a ceiling, not an exact match (RESTRICTED
    clearance can still see PUBLIC content)."""
    max_rank = _RANK[ClearanceLevel(clearance)]
    return [
        level.value
        for level in AccessLevel
        if _RANK.get(ClearanceLevel(level.value), -1) <= max_rank
    ]


def can_access(clearance: ClearanceLevel, access_level: AccessLevel) -> bool:
    return (
        _RANK[ClearanceLevel(clearance)] >= _RANK[ClearanceLevel(AccessLevel(access_level).value)]
    )
