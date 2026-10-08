"""Uniqueness constraints on `pg_id` for every node label — makes MERGE
lookups in sync.py use an index instead of a label scan, and prevents
duplicate nodes for the same PostgreSQL row. Neo4j is otherwise
schema-optional (see docs/data-model.md: new labels/relationship types can
be added without migrating existing ones); this is the one piece of schema
we do declare upfront.
"""

from app.services.graph.client import get_neo4j_driver

_NODE_LABELS = [
    "Employee",
    "Company",
    "Department",
    "Industry",
    "Skill",
    "Certification",
    "JobRole",
    "Project",
]


def ensure_graph_schema() -> None:
    driver = get_neo4j_driver()
    with driver.session() as session:
        for label in _NODE_LABELS:
            session.run(
                f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.pg_id IS UNIQUE"
            )
