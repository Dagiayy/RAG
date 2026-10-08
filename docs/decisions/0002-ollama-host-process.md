# ADR 0002: Ollama runs as a host process, not in Docker Compose

## Status
Accepted

## Context
The dev environment is Windows 11 with Docker Desktop. GPU passthrough to
containers for LLM inference is unreliable on this combination, and the
project spec explicitly allows Ollama to run on the host.

## Decision
Ollama is installed and run on the host; the app reaches it via
`OPENAI_BASE_URL` (default `http://host.docker.internal:11434/v1` from a
container — see ADR 0005 for why it's an OpenAI-compatible URL rather than
Ollama's native one), configurable per environment. Stage 2+ deployment
(Linux servers) may containerize Ollama or replace it with vLLM/TGI — see
`docs/deployment.md`.

## Consequences
Local setup requires a manual `ollama pull <model>` step outside Compose;
documented in the README. The app never assumes Ollama is reachable at a
Compose service name.
