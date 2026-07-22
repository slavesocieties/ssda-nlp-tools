import run_admin_luna_pilot as pilot


def test_admin_schema_validation_requires_all_lists_and_matching_id():
    value = {"document_id": "doc-002", "document_types": [], "organizations": [],
             "people": [], "places": [], "dates": [], "actions": [], "uncertainties": []}
    assert pilot._valid(value, "doc-002")
    assert not pilot._valid({"document_id": "doc-002"}, "doc-002")
    assert not pilot._valid(value, "doc-003")


def test_page_chunking_preserves_provenance_and_text():
    doc = {"id": "doc-003", "title": "Dossier", "metadata": {},
           "pages": [{"file": "a.jpg", "transcription": "A"},
                     {"file": "b.jpg", "transcription": "B"},
                     {"file": "c.jpg", "transcription": "C"}],
           "source_images": ["a.jpg", "b.jpg", "c.jpg"], "faithful_text": "old"}
    chunks = pilot._chunk(doc, 2)
    assert [c["id"] for c in chunks] == ["doc-003--p01-02", "doc-003--p03-03"]
    assert chunks[0]["parent_document_id"] == "doc-003"
    assert "[source image: b.jpg]" in chunks[0]["faithful_text"]


def test_request_body_uses_no_reasoning_when_requested():
    body = pilot._request_body([], 2000, "none")
    assert body["reasoning_effort"] == "none"
    assert body["max_completion_tokens"] == 2000


def test_completed_page_chunks_can_be_excluded_by_id():
    doc = {"id": "doc-003", "title": "Dossier", "metadata": {},
           "pages": [{"file": "a.jpg", "transcription": "A"},
                     {"file": "b.jpg", "transcription": "B"}],
           "source_images": ["a.jpg", "b.jpg"], "faithful_text": "old"}
    chunks = pilot._chunk(doc, 1)
    remaining = [chunk for chunk in chunks if chunk["id"] not in {"doc-003--p01-01"}]
    assert [chunk["id"] for chunk in remaining] == ["doc-003--p02-02"]


def test_compact_profile_has_a_smaller_no_evidence_prompt():
    doc = {"id": "doc-003--p01-01", "title": "Page", "metadata": {}, "faithful_text": "Text"}
    compact = pilot._messages(doc, "compact-index")[0]["content"]
    full = pilot._messages(doc, "full")[0]["content"]
    assert "roles/evidence" in compact
    assert len(compact) < len(full)
