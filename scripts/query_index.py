"""
Query a persisted index under storage/index/.

Run from repo root:
  .venv\\Scripts\\python scripts/query_index.py "Your question here"
  .venv\\Scripts\\python scripts/query_index.py --paper-id lewis2020_rag "What is RAG?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    root = _repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from llama_index.core import StorageContext, load_index_from_storage
    from llama_index.core.vector_stores.types import MetadataFilter, MetadataFilters

    from scholarlens.ollama_config import apply_ollama_settings

    parser = argparse.ArgumentParser(description="Query persisted ScholarLens index.")
    parser.add_argument("question", nargs="?", default="What is retrieval-augmented generation?")
    parser.add_argument("--persist-dir", type=Path, default=root / "storage" / "index")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--paper-id", default=None, help="Restrict retrieval to one paper_id")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--llm-model", default="mistral")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=25)
    args = parser.parse_args()

    if not args.persist_dir.is_dir():
        print(f"Persist dir not found: {args.persist_dir}", file=sys.stderr)
        print("Run: python scripts/build_index.py", file=sys.stderr)
        return 1

    apply_ollama_settings(
        base_url=args.base_url,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    storage_context = StorageContext.from_defaults(persist_dir=str(args.persist_dir))
    index = load_index_from_storage(storage_context)

    filters = None
    if args.paper_id:
        filters = MetadataFilters(
            filters=[MetadataFilter(key="paper_id", value=args.paper_id)]
        )

    engine = index.as_query_engine(similarity_top_k=args.top_k, filters=filters)

    print("--- Query ---")
    print(args.question)
    print("--- Answer ---")
    resp = engine.query(args.question)
    print(str(resp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
