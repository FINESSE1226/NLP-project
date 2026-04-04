"""Configure global LlamaIndex Settings for Ollama LLM + embeddings."""

from __future__ import annotations

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def apply_ollama_settings(
    *,
    base_url: str = "http://127.0.0.1:11434",
    llm_model: str = "mistral",
    embed_model: str = "nomic-embed-text",
    chunk_size: int = 256,
    chunk_overlap: int = 25,
    request_timeout: float = 120.0,
) -> None:
    Settings.llm = Ollama(
        model=llm_model,
        base_url=base_url,
        request_timeout=request_timeout,
    )
    Settings.embed_model = OllamaEmbedding(
        model_name=embed_model,
        base_url=base_url,
    )
    Settings.node_parser = SentenceSplitter.from_defaults(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
