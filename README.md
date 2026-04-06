# ScholarLens (CS6493 Topic 3)

LlamaIndex-based academic paper QA with **manifest-driven metadata** and **persisted** indices.

## Prerequisites

1. [Ollama](https://ollama.com) running, with models pulled:
   - `ollama pull nomic-embed-text`
   - `ollama pull mistral` (or another chat model you pass via flags)
2. Python 3.10+ and venv:
   - `python -m venv .venv`
   - Windows: `.venv\Scripts\activate`
   - `pip install -r requirements.txt`

**Run all scripts from the repository root**, e.g. `python scripts/build_index.py`. On Windows, if you skip `activate`, use `.venv\Scripts\python` instead of `python`.

`configs/default.example.yaml` documents suggested parameters; **Phase 2 scripts currently use CLI defaults and flags only** (the YAML is not auto-loaded). Copy values into your command line or extend the scripts if you want single-file config.

## Phase 1 — Smoke test (no manifest required)

Put at least one `.pdf` under `data/papers/`, then:

```bash
python scripts/minimal_rag_smoke.py
```

Quick diagnostics (no LLM calls): `python scripts/check_env.py`

## Phase 2 — Build persisted index from `manifest.csv`

1. List each paper in [`data/papers/manifest.csv`](data/papers/manifest.csv) (`paper_id`, `title`, `year`, `file_name`, `source_url`).
2. Place matching PDFs in `data/papers/`.
3. From repo root:

```bash
python scripts/build_index.py
```

This writes to `storage/index/` (gitignored). Missing PDFs are skipped with a warning.

### Query the persisted index

```bash
python scripts/query_index.py "What is retrieval-augmented generation?"
python scripts/query_index.py --paper-id lewis2020_rag "Summarize the method in one paragraph."
```

Options: `--top-k`, `--persist-dir`, `--llm-model`, `--embed-model`, `--base-url` (see scripts).

## Layout

- `scholarlens/` — package: [`manifest.py`](scholarlens/manifest.py), [`indexing.py`](scholarlens/indexing.py), [`ollama_config.py`](scholarlens/ollama_config.py)
- `scripts/build_index.py`, `scripts/query_index.py` — Phase 2 CLI
- `scripts/minimal_rag_smoke.py` — in-memory smoke test
- `docs/contract.md` — metadata / eval JSON contract
- `configs/default.example.yaml` — reference parameters

## Branching

Use short-lived `feature/*` branches; merge only after smoke or build+query passes in CI/review.

## Handoff checklist (for teammates)

1. Pull repo; create `.venv` and `pip install -r requirements.txt`.
2. Start Ollama; `ollama pull nomic-embed-text` and a chat model (e.g. `mistral`).
3. Ensure at least one PDF listed in `data/papers/manifest.csv` exists under `data/papers/`.
4. `python scripts/check_env.py` — should print "All checks passed."
5. `python scripts/build_index.py` — creates `storage/index/` (folder is gitignored; **rebuild after clone**).
6. `python scripts/query_index.py "..."` and optionally `--paper-id <id>`.

Sample successful console output is kept under [`docs/evidence/sample_run_log.txt`](docs/evidence/sample_run_log.txt). **Do not delete** team-provided log or evidence files used for progress reports.
