from ssda_nlp_tools.routing import infer_source_kind, route_administrative, route_sacramental


def test_administrative_router_preserves_local_provenance_and_flags_dense_pages():
    items = [{"title": "Cofradía", "images": [
        {"file": "a.jpg", "transcription": "Cofradía petición de Juan Pérez."},
        {"file": "b.jpg", "transcription": "Petición " + "Juan Pérez “" * 900},
    ]}]
    result = route_administrative(items)
    assert result["page_index"]["pages"][0]["faithful_text"].startswith("Cofradía")
    assert result["routes"][0]["route"] == "luna-admin-compact-index"
    assert result["routes"][1]["route"] == "qa-admin-dense"
    assert result["requires_review"]


def test_source_classifier_refuses_ambiguous_material():
    decision = infer_source_kind([{"images": [{"file": "a.jpg", "transcription": "miscellaneous text"}]}],
                                 [("a.jpg", "miscellaneous text")])
    assert decision["kind"] == "unknown"


def test_sacramental_router_sends_only_low_confidence_pages_to_model_fallback():
    pages = [("a.jpg", "En veinte de Junio de mil setecientos.\nBauticé a Ana.\nlo firmé."),
             ("b.jpg", "[transcription failed: max retries reached]")]
    result = route_sacramental(pages)
    routes = {row["source_image"]: row["route"] for row in result["routes"]}
    assert routes["b.jpg"] == "retranscribe"
    assert all(route != "luna-sacramental-fallback" or image == "a.jpg" for image, route in routes.items())
