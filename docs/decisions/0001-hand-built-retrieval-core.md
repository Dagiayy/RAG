# ADR 0001: Hand-build the retrieval core instead of LangChain/LlamaIndex

## Status
Accepted

## Context
Frameworks like LangChain/LlamaIndex can implement RAG quickly but hide
chunking, fusion, and context-construction decisions behind abstractions,
making the system's actual behavior hard to inspect, test, and explain.

## Decision
Chunking, embedding orchestration, BM25, fusion (RRF), reranking, graph
query construction, and context assembly are implemented directly in
`app/services/*`. Libraries are used for narrow, well-bounded jobs
(SQLAlchemy for ORM, sentence-transformers for embedding models,
rank-bm25 for lexical scoring, qdrant-client for the vector DB driver) —
never as an end-to-end RAG framework.

## Consequences
More code to write and test ourselves; in exchange, every stage is a plain
Python function/class with unit tests and can be explained and modified
without fighting framework abstractions.
