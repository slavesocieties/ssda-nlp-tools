from ssda_nlp_tools.admin_records import to_documents, to_page_index


def test_to_documents_preserves_pages_and_omits_export_markers():
    items = [
        {"title": "Start", "images": [{"file": "0001.jpg", "transcription": "START"}]},
        {"title": "Petition", "images": [
            {"file": "0002.jpg", "transcription": "first page"},
            {"file": "0003.jpg", "transcription": "second page"},
        ], "metadata": {"date": "1819"}},
        {"title": "End", "images": [{"file": "0004.jpg", "transcription": "END"}]},
    ]
    result = to_documents(items, source="sample.json")
    assert result["stats"] == {"source_items": 3, "synthetic_markers_omitted": 2,
                               "documents": 1, "pages": 2}
    doc = result["documents"][0]
    assert doc["source_images"] == ["0002.jpg", "0003.jpg"]
    assert "[source image: 0003.jpg]" in doc["faithful_text"]
    assert doc["metadata"]["date"] == "1819"


def test_page_index_retains_text_and_labels_regex_matches_as_candidates():
    dossiers = to_documents([{"title": "Petition", "images": [
        {"file": "0002.jpg", "transcription": "Juan Pérez llegó el 3 de mayo de 1819."}]}])
    index = to_page_index(dossiers)
    page = index["pages"][0]
    assert page["id"] == "doc-001--p01"
    assert page["faithful_text"].startswith("Juan Pérez")
    assert page["candidate_dates"] == ["3 de mayo de 1819"]
    assert "Juan Pérez" in page["candidate_proper_names"]
