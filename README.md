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
4. **Web Sources** (optional): Add URLs to [`data/urls.txt`](data/urls.txt) (one per line, `#` for comments).
5. Build the integrated index:

```bash
python scripts/build_index.py --chunk-size 512 --persist-dir storage/index_512
```

The script automatically indexes:
- Papers from `manifest.csv`
- Course materials from `data/course_materials/`
- Web pages from `data/urls.txt` (using BeautifulSoupWebReader)

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

### Interactive Chat Agent (Multi-turn Conversation)
Start an interactive chat session with conversation memory:

```bash
python scripts/chat_agent.py
python scripts/chat_agent.py --persist-dir storage/index_512 --top-k 5
```

Commands during chat:
- `exit` / `quit` — End the conversation
- `clear` — Clear conversation history
- `history` — View conversation history

The chat agent uses `ChatMemoryBuffer` to maintain context across multiple turns, allowing follow-up questions like:
- "What is RAG?" → "Can you explain its limitations?" → "How does Self-RAG address them?"

### Automated Evaluation
Run the batch evaluation pipeline using the questions defined in `data/eval/questions.json`:

```bash
python scripts/run_eval.py --persist-dir storage/index_512 --output data/eval/results_512.json
```

### LLM-as-a-Judge Scoring
Score generated answers against gold answers using an LLM judge:

```bash
python scripts/score_eval.py --input data/eval/results_512.json --output data/eval/scores_512.json
```

This produces quantified metrics:
- Overall average score (1-5 scale)
- Scores breakdown by question type (factual, cross_paper, reasoning)
- Individual question scores with explanations

### Web UI Demo
Launch the interactive Streamlit interface for a visual demonstration:

```bash
streamlit run scripts/app_ui.py
```

Open http://localhost:8501 in your browser. Features:
- Multi-turn chat with conversation history
- Index selection (256/512 chunk sizes)
- Source citation display (shows retrieved passages)
- Adjustable Top-K retrieval

## Layout

- `scholarlens/` — package: [`manifest.py`](scholarlens/manifest.py), [`indexing.py`](scholarlens/indexing.py), [`ollama_config.py`](scholarlens/ollama_config.py)
- `scripts/` — CLI tools: `build_index.py`, `query_index.py`, `chat_agent.py`, `compare_papers.py`, `run_eval.py`, `score_eval.py`, `check_env.py`, `app_ui.py`
- `data/` — data storage:
  - `papers/`: PDF files and `manifest.csv`
  - `course_materials/`: `.md` and `.txt` secondary sources
  - `urls.txt`: Web URLs to crawl (tech blogs, documentation)
  - `eval/`: `questions.json` (30 test questions) and generated results/scores
- `docs/contract.md` — metadata / eval JSON contract
- `configs/default.example.yaml` — reference parameters

## Experiment Results

Chunk size comparison on 30 evaluation questions (LLM-as-a-Judge scoring):

| Configuration | Avg Score | Factual | Reasoning | Cross-paper |
|---------------|-----------|---------|-----------|-------------|
| Chunk 256, Top-K 5 | 3.03 / 5 | 3.11 | 3.00 | 2.80 |
| **Chunk 512, Top-K 5** | **3.47 / 5** | **3.53** | **3.67** | **3.00** |

**Recommendation**: Use `--chunk-size 512` for academic paper QA tasks.

## Branching

Use short-lived `feature/*` branches; merge only after smoke or build+query passes in CI/review.

## Handoff checklist (for teammates)

1. Pull repo; create `.venv` and `pip install -r requirements.txt`.
2. Start Ollama; `ollama pull nomic-embed-text` and a chat model (e.g. `mistral`).
3. Ensure at least one PDF listed in `data/papers/manifest.csv` exists under `data/papers/`.
4. `python scripts/check_env.py` — should print "All checks passed."
5. `python scripts/build_index.py --chunk-size 512 --persist-dir storage/index_512` — creates optimized index (folder is gitignored; **rebuild after clone**).
6. Run `python scripts/run_eval.py --persist-dir storage/index_512` to verify the evaluation pipeline.
7. Try `python scripts/chat_agent.py` for interactive multi-turn Q&A.

Sample successful console output is kept under [`docs/evidence/sample_run_log.txt`](docs/evidence/sample_run_log.txt). **Do not delete** team-provided log or evidence files used for progress reports.
