# ScholarLens (CS6493 Topic 3)

LlamaIndex-based academic paper QA with **manifest-driven metadata**, **secondary data sources**, and **persisted** indices.

## Prerequisites

1. [Ollama](https://ollama.com) running, with models pulled:
   - `ollama pull nomic-embed-text`
   - `ollama pull mistral` (or another chat model you pass via flags)
2. Python 3.10+ and venv:
   - `python -m venv .venv`
   - Windows: `.venv\Scripts\activate`
   - `pip install -r requirements.txt`

**Run all scripts from the repository root**, e.g. `python scripts/build_index.py`. On Windows, if you skip `activate`, use `.venv\Scripts\python` instead of `python`.

`configs/default.example.yaml` documents suggested parameters; **Phase 2/3 scripts currently use CLI defaults and flags only** (the YAML is not auto-loaded).

## Phase 1 — Smoke test (no manifest required)

Put at least one `.pdf` under `data/papers/`, then:

```bash
python scripts/minimal_rag_smoke.py
```

Quick diagnostics (no LLM calls): `python scripts/check_env.py`

## Phase 2 — Build persisted index (Papers + Course Materials)

1. List each paper in [`data/papers/manifest.csv`](data/papers/manifest.csv) (`paper_id`, `title`, `year`, `file_name`, `source_url`).
2. Place matching PDFs in `data/papers/`.
3. **Secondary Sources**: Place any `.md` or `.txt` files in `data/course_materials/`.
4. Build the integrated index:

```bash
python scripts/build_index.py
```
*Note: The script now automatically detects and indexes materials in the `course_materials` directory alongside the research papers.*

### Query the persisted index

```bash
python scripts/query_index.py "What is retrieval-augmented generation?"
python scripts/query_index.py --paper-id lewis2020_rag "Summarize the method in one paragraph."
```

Options: `--top-k`, `--persist-dir`, `--llm-model`, `--embed-model`, `--base-url` (see scripts).

## Phase 3 — Advanced Features (Comparison & Evaluation)

### Cross-Paper Comparison
Compare methodologies or results between two specific papers using stratified retrieval:

```bash
python scripts/compare_papers.py --id1 lewis2020_rag --id2 asai2024_selfrag --query "Compare the retrieval strategies of RAG and Self-RAG."
```

### Automated Evaluation
Run the batch evaluation pipeline using the questions defined in `data/eval/questions.json`:

```bash
python scripts/run_eval.py --questions data/eval/questions.json --output data/eval/results.json
```

## Layout

- `scholarlens/` — package: [`manifest.py`](scholarlens/manifest.py), [`indexing.py`](scholarlens/indexing.py), [`ollama_config.py`](scholarlens/ollama_config.py)
- `scripts/` — CLI tools: `build_index.py`, `query_index.py`, `compare_papers.py`, `run_eval.py`, `check_env.py`
- `data/` — data storage:
  - `papers/`: PDF files and `manifest.csv`
  - `course_materials/`: `.md` and `.txt` secondary sources
  - `eval/`: `questions.json` and generated `results.json`
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
6. Run `python scripts/run_eval.py` to verify the evaluation pipeline.

Sample successful console output is kept under [`docs/evidence/sample_run_log.txt`](docs/evidence/sample_run_log.txt). **Do not delete** team-provided log or evidence files used for progress reports.
