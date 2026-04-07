"""
Interactive chat agent with conversation memory.

Run from repo root:
  .venv\\Scripts\\python scripts/chat_agent.py
  .venv\\Scripts\\python scripts/chat_agent.py --persist-dir storage/index_512

Commands during chat:
  exit / quit  - End the conversation
  clear        - Clear conversation history
  history      - Show conversation history
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
    from llama_index.core.memory import ChatMemoryBuffer

    from scholarlens.ollama_config import apply_ollama_settings

    parser = argparse.ArgumentParser(description="Interactive ScholarLens chat agent.")
    parser.add_argument("--persist-dir", type=Path, default=root / "storage" / "index_512")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--llm-model", default="mistral")
    parser.add_argument("--embed-model", default="nomic-embed-text")
    parser.add_argument("--memory-token-limit", type=int, default=3000,
                        help="Max tokens to keep in conversation memory")
    args = parser.parse_args()

    if not args.persist_dir.is_dir():
        print(f"Persist dir not found: {args.persist_dir}", file=sys.stderr)
        print("Run: python scripts/build_index.py --persist-dir storage/index_512", file=sys.stderr)
        return 1

    apply_ollama_settings(
        base_url=args.base_url,
        llm_model=args.llm_model,
        embed_model=args.embed_model,
        chunk_size=512,
        chunk_overlap=50,
    )

    print(f"[*] Loading index from {args.persist_dir}...")
    storage_context = StorageContext.from_defaults(persist_dir=str(args.persist_dir))
    index = load_index_from_storage(storage_context)

    memory = ChatMemoryBuffer.from_defaults(token_limit=args.memory_token_limit)

    chat_engine = index.as_chat_engine(
        chat_mode="condense_plus_context",
        memory=memory,
        similarity_top_k=args.top_k,
        system_prompt=(
            "You are ScholarLens, an expert academic AI assistant specialized in NLP research papers. "
            "Answer questions based on the indexed papers and course materials. "
            "When comparing papers, clearly structure your response with similarities and differences. "
            "If the context doesn't contain enough information, say so honestly."
        ),
    )

    print("\n" + "=" * 60)
    print("ScholarLens Chat Agent")
    print("=" * 60)
    print("Commands: 'exit'/'quit' to end, 'clear' to reset, 'history' to view")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[*] Goodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ("exit", "quit"):
            print("[*] Goodbye!")
            break
        elif cmd == "clear":
            memory.reset()
            print("[*] Conversation history cleared.\n")
            continue
        elif cmd == "history":
            messages = memory.get_all()
            if not messages:
                print("[*] No conversation history yet.\n")
            else:
                print("\n--- Conversation History ---")
                for msg in messages:
                    role = msg.role.upper()
                    content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
                    print(f"[{role}] {content}")
                print("----------------------------\n")
            continue

        print("\nScholarLens: ", end="", flush=True)
        try:
            response = chat_engine.chat(user_input)
            print(str(response))
        except Exception as e:
            print(f"[Error] {e}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
