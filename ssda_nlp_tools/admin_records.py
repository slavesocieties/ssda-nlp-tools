"""Deterministic preparation for Archivault administrative-record dossiers.

Unlike sacramental registers, these sources are already grouped into archival
items.  Keep that grouping and every source transcription intact; do not apply
the sacramental entry splitter.
"""
from __future__ import annotations

import re
from typing import Any


def _marker(item: dict[str, Any]) -> bool:
    """True for synthetic START/END pages added around an export."""
    images = item.get("images") or []
    if len(images) != 1:
        return False
    text = str(images[0].get("transcription") or "").strip().upper()
    return text in {"START", "END"}


def to_documents(items: list[dict[str, Any]], *, source: str = "") -> dict[str, Any]:
    """Convert Archivault items to stable, provenance-preserving dossiers."""
    documents = []
    markers = 0
    for index, item in enumerate(items, 1):
        if _marker(item):
            markers += 1
            continue
        images = item.get("images") or []
        pages = []
        faithful_parts = []
        for image in images:
            filename = str(image.get("file") or "")
            text = str(image.get("transcription") or "")
            pages.append({"file": filename, "transcription": text})
            faithful_parts.append(f"[source image: {filename}]\n{text}".strip())
        documents.append({
            "id": f"doc-{index:03d}",
            "title": str(item.get("title") or "Untitled archival item"),
            "source_images": [p["file"] for p in pages],
            "pages": pages,
            "faithful_text": "\n\n".join(faithful_parts),
            "metadata": item.get("metadata") or {},
        })
    return {
        "schema": "ssda-administrative-dossier-v1",
        "source": source,
        "documents": documents,
        "stats": {
            "source_items": len(items),
            "synthetic_markers_omitted": markers,
            "documents": len(documents),
            "pages": sum(len(d["pages"]) for d in documents),
        },
    }


_DATE = re.compile(r"\b(?:\d{1,2}\s+de\s+)?(?:enero|febrero|marzo|abril|mayo|junio|"
                   r"julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(?:de\s+)?\d{2,4}\b"
                   r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE)
_NAME = re.compile(r"\b(?:[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+|$)){2,4}")


def to_page_index(dossiers: dict[str, Any]) -> dict[str, Any]:
    """Create a deterministic, provenance-preserving administrative page index.

    Names and dates are explicit regex *candidates*, never asserted entities.
    This supplies free QA and routing signals while the original transcription
    remains attached to every page.
    """
    pages = []
    for document in dossiers.get("documents", []):
        for number, page in enumerate(document.get("pages", []), 1):
            text = str(page.get("transcription") or "")
            names = [match.group(0).strip() for match in _NAME.finditer(text)]
            pages.append({
                "id": f"{document['id']}--p{number:02d}",
                "document_id": document["id"],
                "source_image": str(page.get("file") or ""),
                "faithful_text": text,
                "text_characters": len(text),
                "candidate_dates": _DATE.findall(text),
                "candidate_proper_names": names,
            })
    return {"schema": "ssda-administrative-page-index-v1", "pages": pages,
            "stats": {"pages": len(pages), "candidate_dates": sum(len(p["candidate_dates"]) for p in pages),
                      "candidate_proper_names": sum(len(p["candidate_proper_names"]) for p in pages)}}
