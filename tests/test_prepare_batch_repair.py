import json

import pytest

from prepare_batch_repair import select_rejected_rows


def _rows():
    return [
        {"custom_id": "701157-b0000", "tail_message": {"content": "a"}},
        {"custom_id": "701157-b0001", "tail_message": {"content": "b"}},
        {"custom_id": "701157-b0002", "tail_message": {"content": "c"}},
    ]


def test_selects_only_rejected_requests_in_source_order():
    validation = {"rejected_custom_ids": {
        "701157-b0002": "missing entries",
        "701157-b0000": "bad schema",
    }}
    selected = select_rejected_rows(_rows(), validation)
    assert [row["custom_id"] for row in selected] == [
        "701157-b0000", "701157-b0002"]


def test_maps_namespaced_request_ids_through_receipt():
    validation = {"rejected_custom_ids": {"v3-701157-b0001": "missing"}}
    receipt = {"source_custom_ids": {
        "v3-701157-b0001": "701157-b0001"}}
    selected = select_rejected_rows(_rows(), validation, receipt)
    assert [row["custom_id"] for row in selected] == ["701157-b0001"]


def test_refuses_unknown_rejected_id():
    with pytest.raises(ValueError, match="absent from compact source"):
        select_rejected_rows(
            _rows(), {"rejected_custom_ids": {"701157-b9999": "missing"}})
