from app.services.retrieval.bm25 import BM25ChunkMetadata, build_bm25_index, tokenize


def _meta(text: str, document_id: str = "doc-1") -> BM25ChunkMetadata:
    return BM25ChunkMetadata(
        document_id=document_id,
        text=text,
        source_filename="test.txt",
        page_number=None,
        section=None,
        access_level="internal",
        department=None,
        document_status="active",
    )


def test_tokenize_lowercases_and_splits_on_non_alphanumeric():
    assert tokenize("Siemens PLC-4471, SCADA!") == ["siemens", "plc", "4471", "scada"]


def test_empty_index_search_returns_empty():
    index = build_bm25_index([])
    assert index.is_empty
    assert index.search("anything") == []


def test_search_finds_exact_term_match():
    index = build_bm25_index(
        [
            ("c1", _meta("Employee has Siemens PLC and SCADA experience")),
            ("c2", _meta("Quarterly financial reporting summary")),
            ("c3", _meta("Cement plant safety procedure for confined spaces")),
        ]
    )

    results = index.search("SCADA")
    assert results[0][0] == "c1"


def test_search_ranks_more_term_overlap_higher():
    # BM25's IDF term goes to ~0 or negative when a term appears in most of
    # a tiny corpus, so this needs enough unrelated documents to keep IDF
    # meaningful for "PLC"/"SCADA" — a 2-doc corpus where both share a term
    # isn't representative of real BM25 behavior.
    index = build_bm25_index(
        [
            ("c1", _meta("PLC SCADA PLC SCADA automation systems")),
            ("c2", _meta("PLC mentioned once in passing")),
            ("c3", _meta("Quarterly financial reporting summary")),
            ("c4", _meta("Cement plant safety procedure for confined spaces")),
            ("c5", _meta("Employee certification renewal schedule")),
        ]
    )

    results = index.search("PLC SCADA")
    ids = [chunk_id for chunk_id, _ in results]
    assert ids[0] == "c1"


def test_search_excludes_zero_score_documents():
    index = build_bm25_index(
        [
            ("c1", _meta("Siemens PLC experience")),
            ("c2", _meta("Completely unrelated financial content")),
        ]
    )

    results = index.search("Siemens")
    ids = [chunk_id for chunk_id, _ in results]
    assert "c2" not in ids


def test_search_respects_top_k():
    rows = [(f"c{i}", _meta(f"document number {i} about PLC systems")) for i in range(10)]
    index = build_bm25_index(rows)

    results = index.search("PLC", top_k=3)
    assert len(results) == 3


def test_metadata_is_queryable_by_chunk_id():
    index = build_bm25_index([("c1", _meta("some text", document_id="doc-42"))])
    assert index.metadata["c1"].document_id == "doc-42"
