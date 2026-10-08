"""Generation-quality metrics (docs/evaluation.md), computed from benchmark
ground truth rather than a second model's opinion — see `llm_judge.py` for
the LLM-as-judge dimensions (correctness/relevance/groundedness/citation
quality) that these simpler, deterministic checks don't cover.

All functions are pure and take plain id lists/sets rather than the live
`GeneratedAnswer`/`EvidenceItem` dataclasses, so they're usable for either
the document-RAG path (ids = `document_id`) or the graph path (ids =
`pg_id`) without any dependency on retrieval/generation internals —
`scripts/evaluate.py` does the id extraction from the real pipeline output.
"""


def answer_correctness(answer: str, expected_contains: list[str]) -> bool:
    """True if every expected substring appears in the answer text,
    case-insensitive. A simple, honest baseline (not semantic similarity —
    no second model call, no threshold to tune); `llm_judge.py` covers the
    more nuanced cases this can't (paraphrase, partial correctness)."""
    if not expected_contains:
        return True
    lowered = answer.lower()
    return all(fragment.lower() in lowered for fragment in expected_contains)


def context_precision(evidence_ids: list[str], relevant_ids: set[str]) -> float:
    """Fraction of retrieved evidence that is actually ground-truth
    relevant — a pure retrieval-quality-as-seen-by-generation signal,
    independent of what the model chose to cite. Returns 0.0 for no
    evidence (nothing to be precise about)."""
    if not evidence_ids:
        return 0.0
    hits = sum(1 for eid in evidence_ids if eid in relevant_ids)
    return hits / len(evidence_ids)


def citation_precision(cited_ids: list[str], relevant_ids: set[str]) -> float:
    """Of the sources the model actually cited, what fraction are
    ground-truth relevant — catches citing real-but-irrelevant evidence.
    Returns 0.0 if nothing was cited."""
    if not cited_ids:
        return 0.0
    hits = sum(1 for cid in cited_ids if cid in relevant_ids)
    return hits / len(cited_ids)


def citation_recall(cited_ids: list[str], relevant_ids: set[str]) -> float:
    """Of the ground-truth relevant sources, what fraction got cited —
    catches under-citing / ignoring relevant evidence that was retrieved."""
    if not relevant_ids:
        raise ValueError("relevant_ids must be non-empty")
    hits = sum(1 for rid in relevant_ids if rid in cited_ids)
    return hits / len(relevant_ids)
