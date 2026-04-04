"""Load and validate data/papers/manifest.csv."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str
    title: str
    year: int | None
    file_name: str
    source_url: str


def load_manifest(path: Path) -> list[PaperRecord]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

    rows: list[PaperRecord] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"paper_id", "title", "file_name", "source_url"}
        if reader.fieldnames is None:
            raise ValueError("manifest.csv has no header row")
        missing = required - {h.strip() for h in reader.fieldnames}
        if missing:
            raise ValueError(f"manifest.csv missing columns: {sorted(missing)}")

        for raw in reader:
            paper_id = (raw.get("paper_id") or "").strip()
            title = (raw.get("title") or "").strip()
            file_name = (raw.get("file_name") or "").strip()
            source_url = (raw.get("source_url") or "").strip()
            year_raw = (raw.get("year") or "").strip()
            year: int | None = None
            if year_raw:
                try:
                    year = int(year_raw)
                except ValueError:
                    year = None

            if not paper_id or not file_name:
                continue

            rows.append(
                PaperRecord(
                    paper_id=paper_id,
                    title=title,
                    year=year,
                    file_name=file_name,
                    source_url=source_url,
                )
            )

    if not rows:
        raise ValueError("manifest.csv contains no valid rows")
    return rows


def resolve_paper_paths(
    records: list[PaperRecord], papers_dir: Path
) -> list[tuple[PaperRecord, Path]]:
    out: list[tuple[PaperRecord, Path]] = []
    for rec in records:
        pdf_path = papers_dir / rec.file_name
        if pdf_path.is_file():
            out.append((rec, pdf_path))
    return out
