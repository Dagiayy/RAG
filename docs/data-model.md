# Data Model

Two stores hold entities and their relationships:

- **PostgreSQL** — system of record for structured attributes (dates, numbers,
  free text, access control, audit). Source of truth.
- **Neo4j** — relationship/traversal index over the *same* entities, keyed by
  the PostgreSQL primary key (`id` as `pg_id` property on every node). Built
  from PostgreSQL via the graph sync step in ingestion; never a second source
  of truth for attributes, only for relationships and traversal.

Vector chunks in Qdrant reference `document_id` + `chunk_id`, which map back
to PostgreSQL `documents` / `document_chunks` rows.

## Entities (PostgreSQL tables, simplified)

- `employees(id, employee_code, full_name, email, hire_date, years_experience, department_id, ...)`
- `departments(id, name, parent_department_id)`
- `companies(id, name, is_own_org)` — own org + client companies
- `projects(id, name, company_id, client_id, industry_id, start_date, end_date, description)`
- `industries(id, name)`
- `skills(id, name, category)`
- `employee_skills(employee_id, skill_id, years_experience, proficiency)`
- `certifications(id, name, issuing_body)`
- `employee_certifications(employee_id, certification_id, issued_date, expiry_date)`
- `job_roles(id, title)`
- `employee_roles(employee_id, job_role_id, start_date, end_date)`
- `employee_projects(employee_id, project_id, role_on_project)`
- `locations(id, name, country, region)`
- `products(id, name, vendor)` / `equipment(id, name, category)` / `technologies(id, name)`
- `documents(id, document_uid, title, source_filename, source_type, version, status, effective_date, expiry_date, supersedes_id, department_id, author, access_level, checksum, source_uri, created_at)`
- `document_chunks(id, document_id, chunk_index, page_number, section, text, checksum)`
- `policies(id, document_id, ...)` / `procedures(id, document_id, ...)` / `contracts(id, document_id, ...)`
- `events(id, name, event_date, related_project_id)`
- `users(id, username, role, department_id, clearance_level)` — for AuthZ
- `audit_log(id, user_id, action, resource_type, resource_id, timestamp, detail)`

Full DDL lives in `migrations/` (Alembic) once Phase 1 lands; this file is
the conceptual model, not the DDL.

## Graph schema (Neo4j)

Node labels mirror the tables above: `(:Employee)`, `(:Department)`,
`(:Company)`, `(:Project)`, `(:Industry)`, `(:Skill)`, `(:Certification)`,
`(:JobRole)`, `(:Document)`, `(:Client)`, `(:Location)`, `(:Product)`,
`(:Equipment)`, `(:Technology)`.

Every node carries `pg_id` (matches the PostgreSQL row id) so a graph hit can
be joined back to full structured attributes and access-control metadata.

Relationships:

```
(:Employee)-[:WORKS_FOR]->(:Company)
(:Employee)-[:WORKS_IN]->(:Department)
(:Employee)-[:HAS_SKILL {years_experience, proficiency}]->(:Skill)
(:Employee)-[:HAS_CERTIFICATION {issued_date, expiry_date}]->(:Certification)
(:Employee)-[:WORKED_ON {role}]->(:Project)
(:Employee)-[:HAS_ROLE {start_date, end_date}]->(:JobRole)
(:Project)-[:BELONGS_TO]->(:Industry)
(:Project)-[:FOR_CLIENT]->(:Company)
(:Project)-[:USES]->(:Product|:Equipment|:Technology)
(:Document)-[:DESCRIBES]->(:Project)
(:Document)-[:REFERENCES]->(:Employee)
(:Document)-[:SUPERSEDES]->(:Document)
(:Document)-[:BELONGS_TO]->(:Department)
```

The schema is additive: new node labels/relationship types can be introduced
without migrating existing ones (Neo4j is schema-optional). New entity types
require: (1) a PostgreSQL table, (2) a graph-sync mapping in
`app/services/graph/sync.py`, (3) optional Cypher query helpers in
`app/services/graph/queries.py`.

## Chunk metadata (Qdrant payload + BM25 doc store)

Every indexed chunk carries:
`document_id, document_version, source_filename, source_type, page_number,
section, paragraph, chunk_id, department, author, created_at, effective_date,
expiry_date, access_level, document_status, source_uri, checksum,
embedding_model, embedding_model_version`.

This payload is what retrieval filters on (access control, effective dates,
department) *before* results ever reach the LLM.
