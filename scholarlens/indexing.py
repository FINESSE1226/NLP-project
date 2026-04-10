"""Build VectorStoreIndex from manifest + persist to disk."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex

from scholarlens.manifest import PaperRecord, load_manifest, resolve_paper_paths


_PDF_BINARY_PATTERNS = (
    r"/Filter",
    r"/FlateDecode",
    r"\bstream\b",
    r"\bendstream\b",
    r"\bxref\b",
    r"\btrailer\b",
    r"%PDF-",
)


def _looks_like_binary_pdf_text(text: str) -> bool:
    """Heuristic detection for raw PDF object-stream leakage."""
    if not text:
        return True
    joined_patterns = "|".join(_PDF_BINARY_PATTERNS)
    if re.search(joined_patterns, text, flags=re.IGNORECASE):
        return True

    symbol_count = sum(1 for ch in text if not ch.isalnum() and not ch.isspace())
    symbol_ratio = symbol_count / max(len(text), 1)
    return symbol_ratio > 0.45


def _clean_loaded_text(text: str) -> str:
    """Normalize whitespace and remove control characters."""
    text = text.replace("\uFFFD", " ").replace("\x00", " ")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_urls_from_file(urls_file: Path) -> list[str]:
    """Load URLs from a text file, ignoring comments and blank lines."""
    if not urls_file.is_file():
        return []
    urls = []
    with open(urls_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def documents_from_urls(urls: list[str]) -> tuple[list[Document], list[str]]:
    """Load documents from web URLs using BeautifulSoupWebReader."""
    if not urls:
        return [], []

    warnings: list[str] = []
    documents: list[Document] = []

    try:
        from llama_index.readers.web import BeautifulSoupWebReader
        loader = BeautifulSoupWebReader()

        for url in urls:
            try:
                docs = loader.load_data(urls=[url])
                for doc in docs:
                    doc.metadata["source_type"] = "web"
                    doc.metadata["source_url"] = url
                documents.extend(docs)
            except Exception as e:
                warnings.append(f"Failed to load {url}: {e}")

    except ImportError:
        warnings.append("llama-index-readers-web not installed. Run: pip install llama-index-readers-web")

    return documents, warnings


def documents_from_manifest(
    manifest_path: Path,
    papers_dir: Path,
) -> tuple[list[Document], list[str]]:
    """Load PDFs listed in manifest; attach paper metadata. Skips missing files."""
    records = load_manifest(manifest_path)
    resolved = resolve_paper_paths(records, papers_dir)
    warnings: list[str] = []

    for rec in records:
        if not (papers_dir / rec.file_name).is_file():
            warnings.append(f"Skipping missing file: {rec.file_name} ({rec.paper_id})")

    documents: list[Document] = []
    for rec, pdf_path in resolved:
        docs: list[Document] = []
        try:
            from llama_index.readers.file import PDFReader

            docs = PDFReader().load_data(file=pdf_path)
        except Exception:
            # Fallback if PDFReader is unavailable.
            docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()

        cleaned_docs: list[Document] = []
        skipped_for_noise = 0
        for doc in docs:
            raw_text = getattr(doc, "text", "")
            clean_text = _clean_loaded_text(raw_text)
            if len(clean_text) < 30 or _looks_like_binary_pdf_text(clean_text):
                skipped_for_noise += 1
                continue

            metadata = dict(getattr(doc, "metadata", {}) or {})
            metadata.update(
                {
                    "paper_id": rec.paper_id,
                    "title": rec.title,
                    "year": rec.year if rec.year is not None else "",
                    "source_url": rec.source_url,
                    "file_name": rec.file_name,
                }
            )
            cleaned_docs.append(Document(text=clean_text, metadata=metadata))

        if skipped_for_noise:
            warnings.append(
                f"Skipped {skipped_for_noise} noisy chunks while reading {rec.file_name}"
            )
        documents.extend(cleaned_docs)

    if not documents:
        warnings.append("No paper documents loaded from manifest.")

    return documents, warnings


def build_and_persist(
    manifest_path: Path,
    papers_dir: Path,
    persist_dir: Path,
    materials_dir: Optional[Path] = None,
    urls_file: Optional[Path] = None,
) -> VectorStoreIndex:
    documents, warnings = documents_from_manifest(manifest_path, papers_dir)
    print(f"[*] Loaded {len(documents)} document chunks from papers.")
    for w in warnings:
        print(f"[!] {w}")

    if materials_dir and materials_dir.is_dir() and any(materials_dir.iterdir()):
        print(f"[*] Reading secondary data source from {materials_dir}...")
        materials_docs = SimpleDirectoryReader(
            input_dir=str(materials_dir),
            required_exts=[".txt", ".md"]
        ).load_data()

        for doc in materials_docs:
            doc.metadata["source_type"] = "course_material"
            doc.metadata["file_name"] = doc.metadata.get("file_name", "unknown")

        documents.extend(materials_docs)
        print(f"[*] Loaded {len(materials_docs)} document chunks from course materials.")
    else:
        print(f"[*] No secondary materials found in {materials_dir}. Skipping.")

    if urls_file:
        urls = load_urls_from_file(urls_file)
        if urls:
            print(f"[*] Loading {len(urls)} web URLs from {urls_file}...")
            web_docs, web_warnings = documents_from_urls(urls)
            for w in web_warnings:
                print(f"[!] {w}")
            if web_docs:
                documents.extend(web_docs)
                print(f"[*] Loaded {len(web_docs)} document chunks from web sources.")
        else:
            print(f"[*] No URLs found in {urls_file}. Skipping web sources.")

    if not documents:
        raise RuntimeError(
            "No documents loaded. Add PDFs matching manifest, text files to materials directory, or URLs to urls.txt."
        )

    print("[*] Building vector store index...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    return index