"""
Automated evaluation pipeline for ScholarLens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from scholarlens.ollama_config import apply_ollama_settings

    parser = argparse.ArgumentParser(description="Run batch evaluation queries.")
    parser.add_argument("--questions", type=Path, default=root / "data" / "eval" / "questions.json")
    parser.add_argument("--output", type=Path, default=root / "data" / "eval" / "results.json")
    parser.add_argument("--persist-dir", type=Path, default=root / "storage" / "index")
    parser.add_argument("--llm-model", default="mistral")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    apply_ollama_settings(llm_model=args.llm_model)

    print(f"[*] Loading index from {args.persist_dir}...")
    storage_context = StorageContext.from_defaults(persist_dir=str(args.persist_dir))
    index = load_index_from_storage(storage_context)

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    print(f"[*] Running {len(questions)} evaluation questions...")

    for q in questions:
        print(f"  -> Q: {q['question']}")

        # Apply filter if target paper is specified
        filters = None
        if q.get("target_paper_id"):
            filters = MetadataFilters(
                filters=[ExactMatchFilter(key="paper_id", value=q["target_paper_id"])]
            )

        query_engine = index.as_query_engine(
            similarity_top_k=args.top_k,
            filters=filters
        )

        try:
            response = query_engine.query(q['question'])
            answer = str(response)
        except Exception as e:
            answer = f"ERROR: {str(e)}"

        results.append({
            "question_id": q["question_id"],
            "type": q.get("type", "factual"),
            "question": q["question"],
            "gold_answer": q.get("gold_answer", ""),
            "generated_answer": answer
        })

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[*] Evaluation complete. Results saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())