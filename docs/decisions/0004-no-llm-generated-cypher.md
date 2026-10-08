# ADR 0004: No free-form LLM-generated Cypher against the live graph

## Status
Accepted

## Context
Letting an LLM generate arbitrary Cypher from natural language is a known
injection and correctness risk (the model can generate destructive or
unbounded queries, or subtly wrong traversals that look plausible).

## Decision
Graph retrieval uses parameterized Cypher templates
(`app/services/graph/queries.py`) selected and parameterized from the
validated `QueryIntent` schema. New question shapes are added as new
templates, not by giving the model raw query-generation capability. All
graph queries run against a read-only Neo4j role from the application.

## Consequences
Multi-hop questions the templates don't yet cover fall back to
vector+BM25 RAG over documents describing those relationships, which is
less structured but safe. Template coverage grows as real question shapes
are observed in evaluation.
