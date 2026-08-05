"""Volume geography, and the coordinate defect it has to survive.

Daniel, 2026-08-05: foundational location "is absolutely critical to
disambiguation". The data supports that -- but only after repair.
"""
import json
import os

import pytest

from ssda_nlp_tools.volume_geo import VolumeGeo


def _fixture(tmp_path, rows):
    p = os.path.join(str(tmp_path), "volumes.json")
    json.dump([{"id": vid, "fields": f} for vid, f in rows],
              open(p, "w", encoding="utf-8"))
    return VolumeGeo(p)


def test_positive_longitude_is_repaired_not_trusted(tmp_path):
    """42 of 397 Cuban volumes carry a positive longitude -- the minus sign is
    missing. Raw, that puts Guanabacoa in the Indian Ocean and makes our largest
    volume 14,613 km from a parish 6 km away."""
    g = _fixture(tmp_path, [
        ("201991", {"country": "Cuba", "city": "Havana",
                    "institution": "Guanabacoa", "coords": "23.09992, 82.31738"}),
        ("29597", {"country": "Cuba", "city": "Havana",
                   "institution": "Santo Angel", "coords": "23.14171, -82.35594"}),
    ])
    assert len(g.repaired) == 1
    assert g.km_between("201991", "29597") < 20


def test_negative_longitude_is_left_alone(tmp_path):
    g = _fixture(tmp_path, [
        ("a", {"country": "Brazil", "coords": "-22.8, -43.0"}),
        ("b", {"country": "Brazil", "coords": "-22.7, -42.8"}),
    ])
    assert g.repaired == []
    assert g.km_between("a", "b") < 40


def test_same_place_prefers_structured_fields_over_coordinates(tmp_path):
    """city/state/country cannot carry the sign defect, so they decide."""
    g = _fixture(tmp_path, [
        ("a", {"country": "Cuba", "state": "La Habana", "city": "Havana",
               "institution": "X", "coords": "23.0, 82.0"}),
        ("b", {"country": "Cuba", "state": "La Habana", "city": "Havana",
               "institution": "Y", "coords": "23.1, -82.3"}),
    ])
    assert g.same_place("a", "b") == "city"
    assert g.same_institution("a", "b") is False


def test_different_continents_share_nothing(tmp_path):
    g = _fixture(tmp_path, [
        ("cu", {"country": "Cuba", "city": "Havana", "coords": "23.1, -82.3"}),
        ("br", {"country": "Brazil", "city": "Niteroi", "coords": "-22.9, -43.1"}),
    ])
    assert g.same_place("cu", "br") == "none"
    assert g.km_between("cu", "br") > 5000


def test_non_overlapping_volume_dates_are_detected(tmp_path):
    g = _fixture(tmp_path, [
        ("old", {"country": "Cuba", "start_date": "1770-07-01T00:00:00Z",
                 "end_date": "1792-10-01T00:00:00Z"}),
        ("new", {"country": "Cuba", "start_date": "1839-11-23T00:00:00Z",
                 "end_date": "1852-09-27T00:00:00Z"}),
    ])
    assert g.overlapping_years("old", "new") is False
    assert g.overlapping_years("old", "old") is True


def test_missing_or_unknown_volumes_return_None_not_a_guess(tmp_path):
    g = _fixture(tmp_path, [("a", {"country": "Cuba"})])
    assert g.km_between("a", "nope") is None
    assert g.same_place("a", "nope") is None
    assert g.overlapping_years("a", "nope") is None
