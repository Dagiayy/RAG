"""Seeds a small, permanent synthetic document corpus — the Phase 9 gap
flagged since Phase 6 ("Full Phase 9 synthetic dataset (conflicting
versions, duplicates, restricted docs) is still separate/later"). Unlike
`evaluation/fixtures/docs/*.txt` (ingested and deleted within a single
`scripts/evaluate.py` run — see its module docstring for why), these
documents are meant to stay in the corpus permanently: a realistic demo
corpus for manual exploration and for richer evaluation-benchmark cases,
parallel to `scripts/seed_demo_data.py`'s permanent employee/project graph.

Three scenarios, all exercising already-built detection/enforcement
mechanisms with real ingested content instead of only ad-hoc test
fixtures:

- **Conflicting facts**: two independent documents (not a supersedes
  chain — see Phase 8's document versioning, which is for the SAME
  document's history) that give different numeric answers to the same
  real-world question. Exercises `generate_answer()`'s
  "conflicting_evidence" path end-to-end against real retrieval, not just
  the mocked-evidence unit test in tests/unit/test_answer_generation.py.
- **Near-duplicates**: two documents with substantially similar, reworded
  (not byte-identical) content — triggers
  `find_near_duplicate_chunks()` (Phase 8) for real.
- **Access-level ladder**: one document at each of the 5 `AccessLevel`
  values, so clearance-based retrieval filtering (Phase 8) has real seeded
  content at every level, not just the single PUBLIC/RESTRICTED pair in
  tests/security/test_retrieval_leakage.py.

Every company/policy/figure here is fictional, consistent with
scripts/seed_demo_data.py's own disclaimer.

Run: python scripts/seed_synthetic_documents.py
Idempotent: ingest_document()'s checksum-based exact-duplicate rejection
means re-running this script is a safe no-op for documents already
present (caught and reported, not treated as an error).
"""

import asyncio

from app.models.document import AccessLevel
from app.repositories.db import get_session_factory
from app.services.ingestion.pipeline import IngestionError, ingest_document

DOCUMENTS = [
    # --- Conflicting facts (two independent documents, same real-world
    # question, different answers) ---
    dict(
        filename="vehicle_safety_policy_2023.txt",
        title="Vehicle Safety Policy (2023)",
        content=(
            b"Vehicle Safety Policy, effective 2023. Field vehicles must not "
            b"exceed 60 km/h on unpaved site access roads. All drivers must "
            b"complete defensive driving certification before operating a "
            b"field vehicle."
        ),
        department="Field Operations",
        access_level=AccessLevel.INTERNAL,
    ),
    dict(
        filename="site_safety_addendum_2024.txt",
        title="Site Safety Addendum (2024)",
        content=(
            b"Site Safety Addendum, effective 2024. Per updated client site "
            b"requirements, field vehicles on unpaved site access roads must "
            b"not exceed 40 km/h. This addendum applies to all active field "
            b"sites until further notice."
        ),
        department="Field Operations",
        access_level=AccessLevel.INTERNAL,
    ),
    # --- Near-duplicates (substantially similar, reworded) ---
    dict(
        filename="equipment_checkout_procedure.txt",
        title="Equipment Checkout Procedure",
        content=(
            b"All field equipment must be checked out through the warehouse "
            b"logbook before leaving site. Equipment must be returned within "
            b"48 hours or a loss report filed with the warehouse supervisor."
        ),
        department="Field Operations",
        access_level=AccessLevel.INTERNAL,
    ),
    dict(
        filename="equipment_checkout_procedure_v2_draft.txt",
        title="Equipment Checkout Procedure (Draft v2)",
        content=(
            b"All field equipment has to be checked out via the warehouse "
            b"logbook prior to leaving the site. Equipment needs to be "
            b"returned within 48 hours, or else a loss report should be filed "
            b"with the warehouse supervisor."
        ),
        department="Field Operations",
        access_level=AccessLevel.INTERNAL,
    ),
    # --- Access-level ladder (one document per AccessLevel) ---
    dict(
        filename="company_holiday_calendar.txt",
        title="Company Holiday Calendar",
        content=(
            b"Meridian Field Data & Monitoring Services observes the "
            b"following public holidays: New Year's Day, Labour Day, "
            b"Independence Day, and the last week of December."
        ),
        department=None,
        access_level=AccessLevel.PUBLIC,
    ),
    dict(
        filename="employee_handbook_excerpt.txt",
        title="Employee Handbook Excerpt — Training Requirements",
        content=(
            b"All employees must complete annual safety refresher training "
            b"by March 31 each year. Failure to complete training suspends "
            b"field deployment eligibility until the requirement is met."
        ),
        department=None,
        access_level=AccessLevel.INTERNAL,
    ),
    dict(
        filename="department_budget_summary_field_ops.txt",
        title="Field Operations — Q1 Budget Summary",
        content=(
            b"The Field Operations department's Q1 budget allocation is "
            b"$420,000, covering vehicle maintenance, fuel, and per-diem "
            b"costs for active field deployments."
        ),
        department="Field Operations",
        access_level=AccessLevel.DEPARTMENT,
    ),
    dict(
        filename="client_contract_terms_agririse.txt",
        title="AgriRise Foundation — Contract Terms Summary",
        content=(
            b"The AgriRise Foundation contract includes a termination "
            b"clause requiring 90 days written notice and a liquidated "
            b"damages provision of $50,000 for early termination without "
            b"cause."
        ),
        department=None,
        access_level=AccessLevel.CONFIDENTIAL,
    ),
    dict(
        filename="executive_compensation_summary.txt",
        title="Executive Compensation Summary (FY2024)",
        content=(
            b"Executive leadership compensation for FY2024 totals $1.2M "
            b"across 3 roles. Individual figures are confidential to the "
            b"board and the executive leadership team."
        ),
        department=None,
        access_level=AccessLevel.RESTRICTED,
    ),
]


async def main() -> None:
    factory = get_session_factory()
    ingested = 0
    skipped = 0

    async with factory() as session:
        for spec in DOCUMENTS:
            try:
                result = await ingest_document(
                    session,
                    spec["filename"],
                    spec["content"],
                    title=spec["title"],
                    department=spec["department"],
                    access_level=spec["access_level"],
                    chunk_size=60,
                    chunk_overlap=0,
                )
            except IngestionError as exc:
                print(f"  SKIP {spec['filename']}: {exc}")
                skipped += 1
                continue
            print(
                f"  OK   {spec['filename']} -> document_id={result.document_id} "
                f"near_duplicate_count={result.near_duplicate_count}"
            )
            ingested += 1

    print(f"\nDone: {ingested} ingested, {skipped} already present (skipped).")


if __name__ == "__main__":
    asyncio.run(main())
