# Interface contract (v0): lock these early; change only with PR + reviewer ack.

## Document metadata (per paper)

| Field       | Type   | Required | Example        |
|------------|--------|----------|----------------|
| paper_id   | string | yes      | `lewis2020_rag`|
| title      | string | yes      | `RAG ...`      |
| year       | int    | optional | 2020           |

## Eval question JSON (eval/questions.json)

Per item:

- `id`: string
- `question`: string
- `type`: `"factual" | "cross_paper" | "reasoning" | "conversational"`
- `gold_answer`: string (short reference answer)
- `source_paper_ids`: string[] (which papers may support the answer)

Schema changes: bump version in this file and notify all pairs.
