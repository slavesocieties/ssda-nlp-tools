from ssda_nlp_tools.routing import infer_source_kind, route_administrative, route_sacramental


def test_administrative_router_preserves_local_provenance_and_flags_dense_pages():
    items = [{"title": "Cofradía", "images": [
        {"file": "a.jpg", "transcription": "Cofradía petición de Juan Pérez."},
        {"file": "b.jpg", "transcription": "Petición " + "Juan Pérez “" * 900},
    ]}]
    result = route_administrative(items)
    assert result["page_index"]["pages"][0]["faithful_text"].startswith("Cofradía")
    assert result["routes"][0]["route"] == "qa-admin-compact-pilot"
    assert result["routes"][1]["route"] == "qa-admin-dense"
    assert result["requires_review"]


def test_source_classifier_refuses_ambiguous_material():
    decision = infer_source_kind([{"images": [{"file": "a.jpg", "transcription": "miscellaneous text"}]}],
                                 [("a.jpg", "miscellaneous text")])
    assert decision["kind"] == "unknown"


def test_source_classifier_accepts_a_multi_image_parish_register():
    text = "Livro de óbitos. Aos dez dias se sepultou o defunto. " * 10
    decision = infer_source_kind([{"images": [{"file": "a.jpg", "transcription": text},
                                                 {"file": "b.jpg", "transcription": text}]}],
                                 [("a.jpg", text), ("b.jpg", text)])
    assert decision["kind"] == "sacramental"


def test_source_classifier_accepts_a_table_heavy_parish_register():
    prose = "Livro de óbitos. Aos dez dias se sepultou o defunto. " * 10
    table = "| pessoa | obitos |\n| Ana | sepultou |"
    pages = [(f"p{i}.jpg", table if i < 6 else prose) for i in range(10)]
    decision = infer_source_kind([{"images": [{"file": image, "transcription": text}
                                                 for image, text in pages]}], pages)
    assert decision["kind"] == "sacramental"


def test_sacramental_router_sends_only_low_confidence_pages_to_model_fallback():
    pages = [("a.jpg", "En veinte de Junio de mil setecientos.\nBauticé a Ana.\nlo firmé."),
             ("b.jpg", "[transcription failed: max retries reached]")]
    result = route_sacramental(pages)
    routes = {row["source_image"]: row["route"] for row in result["routes"]}
    assert routes["b.jpg"] == "retranscribe"
    assert all(route != "luna-sacramental-fallback" or image == "a.jpg" for image, route in routes.items())
