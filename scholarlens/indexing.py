"""Build VectorStoreIndex from manifest + persist to disk."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from llama_index.core import Document, SimpleDirectoryReader, VectorStoreIndex

from scholarlens.manifest import PaperRecord, load_manifest, resolve_paper_paths


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
        docs = SimpleDirectoryReader(input_files=[str(pdf_path)]).load_data()
        for doc in docs:
            doc.metadata.update(
                {
                    "paper_id": rec.paper_id,
                    "title": rec.title,
                    "year": rec.year if rec.year is not None else "",
                    "source_url": rec.source_url,
                    "file_name": rec.file_name,
                }
            )
        documents.extend(docs)

    if not documents:
        warnings.append("No paper documents loaded from manifest.")

    return documents, warnings


def build_and_persist(
    manifest_path: Path,
    papers_dir: Path,
    persist_dir: Path,
    materials_dir: Optional[Path] = None,
) -> VectorStoreIndex:
    documents, warnings = documents_from_manifest(manifest_path, papers_dir)
    print(f"[*] Loaded {len(documents)} document chunks from papers. ")
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

    if not documents:
        raise RuntimeError(
            "No documents loaded. Add PDFs matching manifest or add text files to materials directory."
        )

    print("[*] Building vector store index...")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    return index