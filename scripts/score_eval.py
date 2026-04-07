"""
LLM-as-a-Judge: Score generated answers against gold answers.

Run from repo root:
  .venv\\Scripts\\python scripts/score_eval.py
  .venv\\Scripts\\python scripts/score_eval.py --input data/eval/results_512.json --output data/eval/scores_512.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


JUDGE_PROMPT = """You are an expert evaluator for a question-answering system about NLP research papers.

Your task is to compare a generated answer against a reference (gold) answer and assign a score from 1 to 5.

Scoring criteria:
- 5 (Excellent): The generated answer is accurate, complete, and captures all key points from the gold answer.
- 4 (Good): The answer is mostly correct with minor omissions or slight inaccuracies.
- 3 (Acceptable): The answer is partially correct but misses some important information or contains some errors.
- 2 (Poor): The answer has significant errors or misses most of the key points.
- 1 (Very Poor): The answer is incorrect, irrelevant, or says "I don't know" / "Empty Response".

Question: {question}

Gold Answer: {gold_answer}

Generated Answer: {generated_answer}

Provide your evaluation in the following format:
SCORE: [1-5]
REASON: [Brief explanation of your scoring decision]
"""


def extract_score(response_text: str) -> int:
    """Extract numeric score from LLM response."""
    match = re.search(r"SCORE:\s*(\d)", response_text)
    if match:
        score = int(match.group(1))
        return max(1, min(5, score))
    match = re.search(r"\b([1-5])\b", response_text)
    if match:
        return int(match.group(1))
    return 3


def main() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from llama_index.llms.ollama import Ollama

    parser = argparse.ArgumentParser(description="Score evaluation results using LLM-as-a-Judge.")
    parser.add_argument("--input", type=Path, default=root / "data" / "eval" / "results_512.json")
    parser.add_argument("--output", type=Path, default=root / "data" / "eval" / "scores_512.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--judge-model", default="mistral", help="Model to use as judge")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        print("Run: python scripts/run_eval.py first", file=sys.stderr)
        return 1

    with open(args.input, "r", encoding="utf-8") as f:
        results = json.load(f)

    print(f"[*] Loading judge model: {args.judge_model}")
    judge = Ollama(model=args.judge_model, base_url=args.base_url, request_timeout=120.0)

    scored_results = []
    type_scores = defaultdict(list)

    print(f"[*] Scoring {len(results)} answers...")
    for i, item in enumerate(results, 1):
        question = item["question"]
        gold = item.get("gold_answer", "")
        generated = item.get("generated_answer", "")
        q_type = item.get("type", "factual")

        prompt = JUDGE_PROMPT.format(
            question=question,
            gold_answer=gold,
            generated_answer=generated
        )

        print(f"  [{i}/{len(results)}] Scoring {item['question_id']}...", end=" ", flush=True)

        try:
            response = judge.complete(prompt)
            response_text = str(response)
            score = extract_score(response_text)
            reason_match = re.search(r"REASON:\s*(.+)", response_text, re.DOTALL)
            reason = reason_match.group(1).strip()[:200] if reason_match else "No reason provided"
        except Exception as e:
            score = 0
            reason = f"Error: {str(e)}"

        print(f"Score: {score}")

        scored_results.append({
            "question_id": item["question_id"],
            "type": q_type,
            "question": question,
            "gold_answer": gold,
            "generated_answer": generated,
            "score": score,
            "reason": reason
        })

        if score > 0:
            type_scores[q_type].append(score)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(scored_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    all_scores = [r["score"] for r in scored_results if r["score"] > 0]
    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        print(f"\nOverall Average Score: {avg_score:.2f} / 5.00")
        print(f"Total Questions: {len(all_scores)}")

    print("\nScores by Question Type:")
    print("-" * 40)
    for q_type, scores in sorted(type_scores.items()):
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {q_type:15} : {avg:.2f} / 5.00  (n={len(scores)})")

    print("\nIndividual Scores:")
    print("-" * 40)
    for r in scored_results:
        status = "[OK]" if r["score"] >= 4 else "[--]" if r["score"] >= 3 else "[X]"
        print(f"  {status} {r['question_id']:5} [{r['type']:12}] : {r['score']}/5")

    print(f"\n[*] Detailed results saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
