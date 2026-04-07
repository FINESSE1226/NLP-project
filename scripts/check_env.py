"""
Fast environment diagnostics (no LLM calls). Run from repo root:

  .venv\\Scripts\\python scripts\\check_env.py

Exits 0 if all *required for smoke* checks pass, else 1.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def check_ollama(base_url: str, timeout_s: float = 3.0) -> tuple[bool, str]:
    url = base_url.rstrip("/") + "/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode())
        names = [m.get("name", "") for m in body.get("models", [])]
        if not names:
            return False, "Ollama reachable but no models installed (run: ollama pull mistral && ollama pull nomic-embed-text)"
        return True, f"Ollama OK; models: {', '.join(names[:8])}{'...' if len(names) > 8 else ''}"
    except urllib.error.URLError as e:
        return False, f"Ollama not reachable at {base_url}: {e}"
    except TimeoutError:
        return False, f"Ollama request timed out ({timeout_s}s) - is the service running?"


def main() -> int:
    root = repo_root()
    issues: list[str] = []

    py = sys.version.split()[0]
    print(f"Python: {py}")

    try:
        import llama_index.core  # noqa: F401

        print("llama_index.core: import OK")
    except ImportError as e:
        issues.append(f"llama_index import failed: {e}")

    try:
        from llama_index.llms.ollama import Ollama  # noqa: F401
        from llama_index.embeddings.ollama import OllamaEmbedding  # noqa: F401

        print("Ollama LLM/embedding integrations: import OK")
    except ImportError as e:
        issues.append(f"Ollama integrations import failed: {e}")

    base = "http://127.0.0.1:11434"
    ok, msg = check_ollama(base)
    print(msg)
    if not ok:
        issues.append(msg)

    papers = root / "data" / "papers"
    pdfs = list(papers.glob("*.pdf")) if papers.is_dir() else []
    print(f"PDFs under data/papers: {len(pdfs)}")
    if not pdfs:
        issues.append("No PDFs in data/papers - add at least one .pdf for smoke test")

    manifest = papers / "manifest.csv"
    if manifest.is_file():
        print(f"manifest.csv: present ({manifest.stat().st_size} bytes)")
    else:
        print("manifest.csv: missing (optional for smoke; required for metadata build)")

    if issues:
        print("\n--- Issues ---", file=sys.stderr)
        for i in issues:
            print(i, file=sys.stderr)
        return 1

    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
