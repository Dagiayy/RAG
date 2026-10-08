"""Conflict detection (spec section 19): finds pairs of chunks that are
topically related (similar embedding) but state DIFFERENT numeric values
for what looks like the same kind of quantity (same unit) — e.g. one
document saying a speed limit is 60 km/h and another saying 40 km/h.

Deliberately narrow and detection-only, per the explicit design note this
project has carried since Phase 8: "a rushed version risks producing
misleading 'conflicts' that are actually false positives, which is worse
than not having the feature." This implements the numeric-heuristic
approach spec section 19 names as the alternative to LLM-based claim
extraction — no LLM call, no hallucination risk, fully deterministic and
auditable: every flagged conflict names the exact two numbers, their
shared unit, and the chunks they came from, so a human can verify it in
seconds rather than trusting a model's judgment. The real end-to-end
verification of what this is FOR — the model correctly surfacing these as
"conflicting_evidence" at answer-generation time — is covered separately
in app/services/generation/answer_generation.py; this module only decides
which chunks are worth flagging as related-but-disagreeing in the first
place.

Reuses the SAME embedding-similarity search `duplicates.py` already uses
for near-duplicate detection, at a lower "topically related" threshold
(near-duplicate detection only looks above 0.92; this also looks below
that). Near-duplicate and conflict detection are independent,
complementary signals, not mutually exclusive categories — a pair can
legitimately be BOTH near-identically worded AND numerically conflicting
(e.g. "the limit is 60 units per hour" vs. "the limit is 40 units per
hour" are extremely similar in wording but disagree on the one number
that matters). An earlier version of this module excluded anything above
the near-duplicate threshold from conflict consideration, reasoning that
high similarity meant "the same statement, not a conflict" — that was
wrong, caught via a real integration test: two sentences differing only
in their numeric claim scored above 0.92 (sentence embeddings aren't very
sensitive to a single digit token), so the exclusion was hiding exactly
the clearest kind of conflict instead of catching it.

Still detection only: never blocks ingestion, never auto-resolves which
value is correct, never discards either document. Findings are recorded
in the ingestion audit detail, same place near-duplicate findings go.
"""

import asyncio
import re
import uuid
from dataclasses import dataclass

from app.services.retrieval.vector import search as vector_search

# NUMBER UNIT, e.g. "60 km/h", "$420,000", "40 units per hour", "90 days",
# "3 days per week", "18 months". Deliberately conservative: a number is
# only treated as a comparable "claim" when immediately preceded by "$" or
# followed by one of these unit words/symbols — a bare number (a page
# reference, a year mentioned in prose, a document code) is too ambiguous
# to compare safely and is never extracted. Two alternatives rather than
# one shared pattern: a trailing `\b` after a symbol like "%" doesn't
# reliably match when the next character is also non-word (e.g. a space),
# and "$420,000" has no unit word to require after it at all.
_CLAIM_PATTERN = re.compile(
    r"\$(?P<currency_number>\d[\d,]*(?:\.\d+)?)"
    r"|"
    r"\b(?P<number>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>%|percent|km/h|kph|mph|units?(?:\s+per\s+\w+)?|"
    r"days?(?:\s+per\s+\w+)?|months?|years?|hours?|minutes?|"
    r"meters?|metres?|centimeters?|centimetres?|kg|lbs?|pounds?)",
    re.IGNORECASE,
)


@dataclass
class NumericClaim:
    number: float
    unit: str  # normalized (lowercased, trailing "s" stripped) for comparison
    raw_text: str


def extract_numeric_claims(text: str) -> list[NumericClaim]:
    claims = []
    for m in _CLAIM_PATTERN.finditer(text):
        if m.group("currency_number") is not None:
            number = float(m.group("currency_number").replace(",", ""))
            unit = "dollars"
        else:
            number = float(m.group("number").replace(",", ""))
            unit = m.group("unit").lower().rstrip("s")
        claims.append(NumericClaim(number=number, unit=unit, raw_text=m.group(0)))
    return claims


@dataclass
class ConflictCandidate:
    new_chunk_id: str
    existing_chunk_id: str
    existing_document_id: str
    similarity: float
    new_claim: NumericClaim
    existing_claim: NumericClaim


async def find_conflicting_claims(
    new_chunk_vectors: list[tuple[str, list[float]]],
    new_chunk_text_by_id: dict[str, str],
    exclude_document_id: uuid.UUID,
    related_threshold: float = 0.55,
) -> list[ConflictCandidate]:
    """For each new chunk with at least one numeric claim, searches for
    topically related existing chunks (similarity >= related_threshold —
    no upper bound; see module docstring for why). Flags a conflict only
    when both chunks contain a numeric claim with the SAME unit but a
    DIFFERENT number — chunks that share no comparable unit, or agree on
    the number, are never flagged.
    """
    candidates: list[ConflictCandidate] = []
    exclude_str = str(exclude_document_id)

    for chunk_id, vector in new_chunk_vectors:
        new_claims = extract_numeric_claims(new_chunk_text_by_id.get(chunk_id, ""))
        if not new_claims:
            continue

        results = await asyncio.to_thread(vector_search, vector, 10, None, related_threshold)
        for result in results:
            if result.document_id == exclude_str:
                continue

            existing_claims = extract_numeric_claims(result.payload.get("text", ""))

            for new_claim in new_claims:
                for existing_claim in existing_claims:
                    if new_claim.unit != existing_claim.unit:
                        continue
                    if new_claim.number == existing_claim.number:
                        continue
                    candidates.append(
                        ConflictCandidate(
                            new_chunk_id=chunk_id,
                            existing_chunk_id=result.chunk_id,
                            existing_document_id=result.document_id,
                            similarity=result.score,
                            new_claim=new_claim,
                            existing_claim=existing_claim,
                        )
                    )

    return candidates
