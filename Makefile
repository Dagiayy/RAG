.PHONY: install dev test lint format typecheck ingest evaluate docker-up docker-down

install:
	pip install -e ".[dev,ingestion,retrieval,generation,documents,ui]"

dev:
	uvicorn app.main:app --reload --port 8010

test:
	pytest tests/unit tests/integration -v

lint:
	ruff check .

format:
	black .
	ruff check --fix .

typecheck:
	mypy app

ingest:
	python scripts/ingest.py

evaluate:
	python scripts/evaluate.py

docker-up:
	docker compose up -d

docker-down:
	docker compose down
