"""
Build a persisted VectorStoreIndex from data/papers/manifest.csv.

Run from repo root:
  .venv\\Scripts\\python scripts/build_index.py
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

    from scholarlens.indexing import build_and_persist
    from scholarlens.ollama_config import apply_ollama_settings

    parser = argparse.ArgumentParser(description="Build and persist RAG index from manifest and materials.")
    parser.add_argument("--manifest", type=Path, default=root / "data" / "papers" / "manifest.csv")
    parser.add_argument("--papers-dir", type=Path, default=root / "data" / "papers")
    parser.add_argument("--materials-dir", type=Path, default=root / "data" / "course_materials")
    parser.add_argument("--persist-dir", type=Path, default=root / "storage" / "index")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--llm-model", default="mistral")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=25)
    args = parser.parse_args()

    apply_ollama_settings(
        base_url=args.base_url,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"[*] Building index from {args.manifest} and {args.materials_dir} -> {args.persist_dir} ...")
    build_and_persist(
        manifest_path=args.manifest,
        papers_dir=args.papers_dir,
        persist_dir=args.persist_dir,
        materials_dir=args.materials_dir,
    )
    print("[*] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())