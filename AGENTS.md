# AGENTS.md — docgrep

Single contract for every coding agent in this repository (Codex, Claude Code, others). `CLAUDE.md` only imports this file. Read it fully at the start of every session.

## What docgrep is

A local, cloud-free document corpus search service: passage-level, ranked, verbatim retrieval with source metadata and deep-links. It returns *findings*, not answers.

Authoritative specs (read before implementing anything they cover):

- `specs/docgrep_vision.md`: scope, non-goals, architectural constraints, NFRs
- `specs/uc-*.md`: functional requirements (Cockburn). UC-01 and UC-03 drive the current slice
- `docs/adr/*.md`: closed architectural decisions (public)

`specs/` is **private**: it is git-ignored here and versioned in a separate private repository. See the confidentiality rule below.

If a spec is ambiguous, pick the simplest reading consistent with the Vision, implement it, and state the choice in the commit body. Never edit files under `specs/`, `docs/`, `AGENTS.md` or `CLAUDE.md`.

## Absolute constraints

1. **No LLM inference in core.** No `llama-cpp-python`, `vllm`, generative `transformers` pipelines. Embedding and cross-encoder models are allowed.
2. **No Obsidian in domain.** Wikilinks and `obsidian://` deep-links live only in adapters.
3. **No telemetry, no remote calls at runtime.** Downloading model weights on first use is the only network access, and it is allowed.
4. **Verbatim passages only.** Nothing returned to the user is paraphrased, summarised or generated.
5. **No `print()`** in `src/`. Use `logging` (structured, via the standard library).
6. **Specs are confidential.** Never copy text from `specs/` into code, comments, docstrings, tests, commit messages or PR descriptions, and never paraphrase a spec section at length. Refer to requirements by id only (e.g. `UC-03 ext. 6.3b`). Never `git add -f` anything under `specs/`.
7. **English only** in the repository: code, comments, docstrings, docs, commit messages and PR descriptions. This is a public repository. Non-English text is allowed only as test data (e.g. Russian queries and passages).

## Layering (strict)

`src/docgrep/{domain,usecases,adapters,api}` — dependency direction `api → usecases → domain ← adapters`.

- `domain` imports only the stdlib: frozen dataclasses, value types, `typing.Protocol`s, domain errors.
- `usecases` import `domain` only and depend on Protocols, never on concrete adapters.
- `adapters` implement domain Protocols (Qdrant, SQLite, parsers, models, filesystem).
- `api` (FastAPI, CLI) holds the single composition root (`api/container.py`).
- A layer violation is a bug: refactor immediately, in the same change. A test enforces this.

## Closed decisions (see `docs/adr/`; do not revisit without a new ADR)

| Topic | Decision |
|---|---|
| Language / packaging | Python 3.12, `uv`, lockfile committed, single package, `src/` layout |
| Vector store | Qdrant **local mode**; collection accessed via alias; alias swap for full reindex |
| Manifest + OCR registry | SQLite (separate table for OCR sibling delegation) |
| Hashing | BLAKE3 |
| Structural cache | Text files `cache/sources/<source_id>.txt` |
| Browse aggregation | Query-time Qdrant count, not pre-computed |
| Embeddings (GPU) | `BAAI/bge-m3` |
| Reranker (GPU) | `BAAI/bge-reranker-v2-m3` |
| LLM companion | `docgrep-llm-bridge`, separate repository (polyrepo) |
| HTTP | FastAPI; `POST` for queries, `GET` only for parameterless reads; bind `127.0.0.1`, no auth |
| Config | `Settings(BaseSettings)`, env prefix `DOCGREP_` |

## Toolchain and quality gate

- `ruff` (lint + format), `mypy --strict`, `pytest`.
- Gate: `scripts/check.sh` = `uv run ruff check . && uv run ruff format --check . && uv run mypy --strict src && uv run pytest -m "not gpu" -q`.
- Never weaken the gate: do not delete, skip, xfail or loosen existing tests; no `# type: ignore` / `noqa` without a one-line justification comment.

## Testing

- `tests/unit`, `tests/integration` (real SQLite / local-mode Qdrant in `tmp_path`), `tests/e2e`.
- Mock only at Protocol boundaries. No `unittest.mock.patch` on internals.
- ML-free deterministic embedder and reranker live in `src/docgrep/adapters/models/deterministic.py` (also used by `backend='fake'`); in-memory store and manifest fakes live in `tests/fakes/`.
- Tests needing real models or a GPU are marked `@pytest.mark.gpu` and excluded from the gate.
- Test fixtures (PDFs, corpora) are generated in tests, not committed as binaries.

## Determinism and errors

- Same corpus + config + index → identical results.
- Passage ids derive from `(source_id, chunk offset)` via a stable hash (never Python `hash()`).
- Tie-break: `(reranker_score DESC, dense_score DESC, passage_id ASC)`.
- Failures raise domain errors carrying `category` and `details`; never silent partial success.

## Commits and releases

- Conventional Commits: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `ci`, `build`, `chore`. Scopes: `domain`, `usecases`, `adapters`, `api`, `cli`, `settings`, `ci`, `deps`.
- `feat` → minor, `fix`/`perf` → patch, `BREAKING CHANGE:` footer → major (via `python-semantic-release`).
- One story per commit; put the story id (e.g. `US-007`) in the commit body.

## Plugin entry points (Vision §11.2)

Groups `docgrep.parsers` (`DocumentParser`), `docgrep.embedders` (`EmbeddingProvider`), `docgrep.citation_formatters` (`CitationFormatter`), `docgrep.rerankers` (`Reranker`). Built-in adapters register through the same groups once discovery is implemented; not required in the current slice.

## Layout

```
src/docgrep/
  domain/      models.py, protocols.py, errors.py
  usecases/    build_index.py, find_passages.py
  adapters/    corpus/, parsers/, chunking/, manifest/, qdrant/, models/
  api/         http.py, cli.py, container.py
  settings.py
tests/  unit/ integration/ e2e/ fakes/
scripts/check.sh
docs/   adr/ (public)
specs/  docgrep_vision.md, uc-*.md (private, git-ignored)
```
