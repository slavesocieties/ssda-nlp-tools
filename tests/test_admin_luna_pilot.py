import run_admin_luna_pilot as pilot


def test_admin_schema_validation_requires_all_lists_and_matching_id():
    value = {"document_id": "doc-002", "document_types": [], "organizations": [],
             "people": [], "places": [], "dates": [], "actions": [], "uncertainties": []}
    assert pilot._valid(value, "doc-002")
    assert not pilot._valid({"document_id": "doc-002"}, "doc-002")
    assert not pilot._valid(value, "doc-003")
