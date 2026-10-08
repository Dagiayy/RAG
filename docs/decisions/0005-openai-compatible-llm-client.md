# ADR 0005: Talk to the LLM via the OpenAI-compatible API, not Ollama's native API

## Status
Accepted

## Context
Phase 7 initially used the native `ollama` Python package
(`ollama.AsyncClient`), calling Ollama's own `/api/chat` endpoint with its
`format=<json schema>` structured-output parameter. This worked well, but
ties the whole generation layer to one server implementation's SDK and
wire format — switching to any other model server later (real OpenAI, a
hosted API, vLLM, TGI — see docs/deployment.md stage 3 "model serving")
would mean rewriting the client and the request shape, not just changing a
URL.

Ollama also exposes an OpenAI-compatible endpoint (`/v1/chat/completions`)
that supports the same structured-output capability via the standard
`response_format={"type": "json_schema", ...}` parameter.

## Decision
`app/services/generation/llm_client.py` uses `openai.AsyncOpenAI` pointed
at `Settings.openai_base_url` (default Ollama's `/v1` endpoint). Every
piece of code that calls the LLM (`query_understanding.py` today, Phase
9's answer generation next) goes through this one client and the standard
OpenAI request/response shapes. Swapping providers — real OpenAI, another
local server, a hosted endpoint — is a three-setting config change
(`openai_base_url`/`openai_api_key`/`openai_model`), never a code change.

Verified empirically (not assumed): re-ran Phase 7's full structured-
extraction test suite against the new client and got identical correct
results on every case, including the multi-hop skill+industry+experience-
filter extraction.

The native Ollama API is kept for exactly one purpose: the `/ready`
liveness probe (`Settings.ollama_native_base_url`), since `/api/tags` is a
simpler, cheaper reachability check than a chat completion round trip.

## Consequences
`app/services/generation/ollama_client.py` was deleted; the `ollama`
PyPI package is no longer a dependency (replaced by `openai` in the
`generation` extra). Two settings for one backend (`openai_base_url` +
`ollama_native_base_url`, both currently pointing at the same Ollama
instance) is a small amount of duplication, traded for a readiness check
that doesn't depend on chat-completion latency/availability.
