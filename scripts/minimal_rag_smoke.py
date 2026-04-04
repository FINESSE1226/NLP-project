"""
Smoke test: one folder of PDFs -> in-memory index -> one query via Ollama.

Prerequisites:
  1. Ollama running: https://ollama.com
  2. Pull models (examples):
       ollama pull mistral
       ollama pull nomic-embed-text
  3. Put at least one PDF under data/papers/

Run from repo root:
  python scripts/minimal_rag_smoke.py
Optional:
  python scripts/minimal_rag_smoke.py "What is the main contribution of this paper?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "question",
        nargs="?",
        default="Summarize the main contribution of this work in 3 bullet points.",
    )
    parser.add_argument("--data-dir", default="data/papers")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--llm-model", default="mistral")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--chunk-overlap", type=int, default=25)
    args = parser.parse_args()

    root = repo_root()
    data_dir = root / args.data_dir
    if not data_dir.is_dir():
        print(f"Missing data directory: {data_dir}", file=sys.stderr)
        return 1

    pdfs = list(data_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found under {data_dir}. Add at least one .pdf.", file=sys.stderr)
        return 1

    Settings.llm = Ollama(
        model=args.llm_model,
        base_url=args.base_url,
        request_timeout=120.0,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=args.embed_model,
        base_url=args.base_url,
    )
    Settings.node_parser = SentenceSplitter.from_defaults(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    documents = SimpleDirectoryReader(input_dir=str(data_dir)).load_data()
    index = VectorStoreIndex.from_documents(documents)
    engine = index.as_query_engine(similarity_top_k=5)

    print("--- Query ---")
    print(args.question)
    print("--- Answer ---")
    response = engine.query(args.question)
    print(str(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
