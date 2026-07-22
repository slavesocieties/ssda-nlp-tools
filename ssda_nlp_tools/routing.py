"""Deterministic, provenance-preserving routing for heterogeneous SSDA volumes.

Routing chooses a pre-approved processing profile; it never calls an LLM or
silently changes a budget.  Ambiguous material is deliberately routed to QA.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

from .admin_records import to_documents, to_page_index
from .segment import detect_page_type, segment_volume


_ADMIN_CUES = re.compile(
    r"\b(?:cofrad[ií]a|hermandad|petici[oó]n|solicita|expediente|cabildo|"
    r"mayordomo|constituci[oó]n|licencia|reglamento)\b", re.IGNORECASE)
_SACRAMENTAL_CUES = re.compile(
    r"\b(?:bautiz|baptis|matrim[oô]ni|casament|obit|[oó]bitos|sepult|"
    r"enterr|fallec|defunt|burial|parish\s+(?:burial|baptism|marriage)|"
    r"livro\s+de\s+(?:obitos|[oó]bitos|baptismos))\b", re.IGNORECASE)


def infer_source_kind(items: list[dict[str, Any]], pages: list[tuple[str, str]]) -> dict[str, Any]:
    """Return a conservative source-kind decision and its reproducible evidence."""
    text = "\n".join(text for _, text in pages)
    admin_cues = len(_ADMIN_CUES.findall(text))
    sacramental_cues = len(_SACRAMENTAL_CUES.findall(text))
    multi_image_items = sum(1 for item in items if len(item.get("images") or []) > 1)
    page_types = Counter(detect_page_type(text) for _, text in pages)
    registers = page_types["register"]
    if admin_cues >= 2 and multi_image_items:
        return {"kind": "administrative", "confidence": 0.95,
                "evidence": {"administrative_cues": admin_cues,
                             "sacramental_cues": sacramental_cues,
                             "multi_image_items": multi_image_items,
                             "detected_page_types": dict(page_types)}}
    # Large registers may contain cover, index and table pages.  Require both
    # some register-shaped text and corpus-scale sacramental evidence rather
    # than treating a table-heavy volume as unknown.
    strong_sacramental_evidence = sacramental_cues >= max(15, len(pages) * 0.05)
    if pages and registers / len(pages) >= 0.2 and strong_sacramental_evidence:
        return {"kind": "sacramental", "confidence": 0.90,
                "evidence": {"administrative_cues": admin_cues,
                             "sacramental_cues": sacramental_cues,
                             "multi_image_items": multi_image_items,
                             "detected_page_types": dict(page_types)}}
    return {"kind": "unknown", "confidence": 0.0,
            "evidence": {"administrative_cues": admin_cues, "sacramental_cues": sacramental_cues,
                         "multi_image_items": multi_image_items,
                         "detected_page_types": dict(page_types)}}


def route_sacramental(pages: list[tuple[str, str]], *, source: str = "") -> dict[str, Any]:
    """Run the existing free segmenter and route only uncertain pages onward."""
    segmented = segment_volume(pages)
    source_text = {image: text for image, text in pages}
    routes = []
    for page in segmented["per_image"]:
        page_type = page.get("page_type") or detect_page_type(source_text[page["image"]])
        if page_type == "error":
            route, reason = "retranscribe", "source transcription error"
        elif page_type == "index":
            route, reason = "skip-index", "not a record page"
        elif page["confidence"] < 0.7:
            route, reason = "luna-sacramental-fallback", "low deterministic confidence"
        else:
            route, reason = "deterministic-sacramental", "confidence at or above 0.70"
        routes.append({"source_image": page["image"], "route": route,
                       "reason": reason, "confidence": page["confidence"],
                       "page_type": page_type})
    return {"schema": "ssda-routing-manifest-v1", "source": source,
            "source_kind": "sacramental", "requires_review": False,
            "routes": routes, "segmentation": segmented["stats"],
            "summary": dict(Counter(row["route"] for row in routes))}


def route_administrative(items: list[dict[str, Any]], *, source: str = "") -> dict[str, Any]:
    """Build the local administrative index and nominate only bounded pages."""
    dossiers = to_documents(items, source=source)
    index = to_page_index(dossiers)
    routes = []
    for page in index["pages"]:
        density = len(page["candidate_dates"]) + len(page["candidate_proper_names"])
        if not page["faithful_text"].strip():
            route, reason, review = "qa-empty-page", "empty transcription", True
        elif page["text_characters"] > 1600 or density > 14:
            route, reason, review = "qa-admin-dense", "too dense for compact page schema", True
        else:
            # The compact profile is intentionally not auto-promoted: its
            # schema needs a successful stratified validation before it can be
            # used outside an explicitly approved pilot.
            route, reason, review = "qa-admin-compact-pilot", "bounded candidate; profile not production-validated", True
        routes.append({"page_id": page["id"], "document_id": page["document_id"],
                       "source_image": page["source_image"], "route": route,
                       "reason": reason, "requires_review": review,
                       "text_characters": page["text_characters"],
                       "candidate_dates": len(page["candidate_dates"]),
                       "candidate_proper_names": len(page["candidate_proper_names"])})
    return {"schema": "ssda-routing-manifest-v1", "source": source,
            "source_kind": "administrative", "requires_review": any(r["requires_review"] for r in routes),
            "routes": routes, "dossiers": dossiers, "page_index": index,
            "summary": dict(Counter(row["route"] for row in routes))}


def route_volume(items: list[dict[str, Any]], pages: list[tuple[str, str]], *,
                 source: str = "", source_kind: str = "auto") -> dict[str, Any]:
    """Route an Archivault volume. ``auto`` refuses uncertain genre decisions."""
    decision = infer_source_kind(items, pages) if source_kind == "auto" else {
        "kind": source_kind, "confidence": 1.0, "evidence": {"override": True}}
    if decision["kind"] == "sacramental":
        result = route_sacramental(pages, source=source)
    elif decision["kind"] == "administrative":
        result = route_administrative(items, source=source)
    else:
        result = {"schema": "ssda-routing-manifest-v1", "source": source,
                  "source_kind": "unknown", "requires_review": True, "routes": [],
                  "summary": {"qa-source-kind": 1}}
    result["classification"] = decision
    return result
