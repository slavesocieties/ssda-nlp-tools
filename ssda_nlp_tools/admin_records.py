"""Deterministic preparation for Archivault administrative-record dossiers.

Unlike sacramental registers, these sources are already grouped into archival
items.  Keep that grouping and every source transcription intact; do not apply
the sacramental entry splitter.
"""
from __future__ import annotations

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
