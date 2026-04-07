"""
Compare two academic papers using stratified retrieval to prevent Top-K dominance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llama_index.core import PromptTemplate, StorageContext, load_index_from_storage
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.response_synthesizers import get_response_synthesizer

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def main() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from scholarlens.ollama_config import apply_ollama_settings

    parser = argparse.ArgumentParser(description="Compare two papers using ScholarLens.")
    parser.add_argument("--id1", type=str, required=True, help="Paper ID for the first paper.")
    parser.add_argument("--id2", type=str, required=True, help="Paper ID for the second paper.")
    parser.add_argument("--query", type=str, required=True, help="Comparison question.")
    parser.add_argument("--persist-dir", type=Path, default=root / "storage" / "index")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--llm-model", default="mistral")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--top-k-per-paper", type=int, default=3)
    args = parser.parse_args()

    apply_ollama_settings(
        base_url=args.base_url,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
    )

    print(f"[*] Loading index from {args.persist_dir}...")
    storage_context = StorageContext.from_defaults(persist_dir=str(args.persist_dir))
    index = load_index_from_storage(storage_context)

    # 1. Force retrieve K chunks specifically from Paper 1
    print(f"[*] Retrieving top {args.top_k_per_paper} chunks from {args.id1}...")
    retriever1 = index.as_retriever(
        similarity_top_k=args.top_k_per_paper,
        filters=MetadataFilters(filters=[ExactMatchFilter(key="paper_id", value=args.id1)])
    )
    nodes1 = retriever1.retrieve(args.query)

    # 2. Force retrieve K chunks specifically from Paper 2
    print(f"[*] Retrieving top {args.top_k_per_paper} chunks from {args.id2}...")
    retriever2 = index.as_retriever(
        similarity_top_k=args.top_k_per_paper,
        filters=MetadataFilters(filters=[ExactMatchFilter(key="paper_id", value=args.id2)])
    )
    nodes2 = retriever2.retrieve(args.query)

    # Combine the guaranteed nodes
    all_nodes = nodes1 + nodes2

    compare_prompt_str = (
        "You are an expert academic AI assistant. Your task is to compare two academic papers.\n"
        "Context information from BOTH papers is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information, explicitly compare the two papers based on the query.\n"
        "Structure your response clearly. Highlight key differences and similarities.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    compare_prompt = PromptTemplate(compare_prompt_str)

    print("[*] Synthesizing comparison from both papers...")
    synthesizer = get_response_synthesizer(text_qa_template=compare_prompt)
    response = synthesizer.synthesize(args.query, nodes=all_nodes)

    print("\n" + "=" * 50)
    print("COMPARISON RESULT")
    print("=" * 50)
    print(response)
    print("=" * 50 + "\n")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())