"""Stratified collection sampling, and the ceiling Daniel asked for.

"don't literally scrape the whole bucket (or even download 10ks of images)"
-- 2026-08-03. That limit is enforced in code rather than left to whoever runs
it, because the whole bucket is 750,527 images and a typo in --images is all it
would take.
"""
import pytest

from sample_images import MAX_IMAGES, era, n_images, plan, stratum


def _vol(vid, images=100, country="Brazil", language="Portuguese",
         typ="Ecclesiastical records", start="1850-01-01T00:00:00Z"):
    return {"id": vid, "fields": {"images": images, "country": country,
                                  "language": language, "type": typ,
                                  "start_date": start}}


def test_the_ceiling_is_a_constant_not_a_suggestion():
    assert MAX_IMAGES <= 1000


def test_sample_never_exceeds_its_budget():
    vols = [_vol(str(i), images=500) for i in range(200)]
    picked = plan(vols, 200, 4, seed=1, exclude=set())
    assert sum(len(p["keys"]) for p in picked) == 200


def test_per_volume_cap_is_respected():
    vols = [_vol(str(i), images=5000) for i in range(50)]
    picked = plan(vols, 100, 4, seed=1, exclude=set())
    assert all(len(p["keys"]) <= 4 for p in picked)
    assert len(picked) >= 25, "a per-volume cap must spread across volumes"


def test_it_spreads_across_strata_rather_than_taking_the_biggest_books():
    """The collection is 72% Brazilian. Sampling proportionally would tell us
    almost nothing about the Cuban and Colombian material, which is where every
    number we have was measured."""
    vols = ([_vol(f"br{i}", country="Brazil") for i in range(500)]
            + [_vol(f"cu{i}", country="Cuba", language="Spanish") for i in range(20)])
    picked = plan(vols, 40, 4, seed=1, exclude=set())
    countries = {p["stratum"].split("|")[0] for p in picked}
    assert countries == {"Brazil", "Cuba"}


def test_delivered_volumes_can_be_excluded():
    vols = [_vol("201991"), _vol("999999")]
    picked = plan(vols, 8, 4, seed=1, exclude={"201991"})
    assert {p["volume"] for p in picked} == {"999999"}


def test_page_numbers_stay_inside_the_volume():
    picked = plan([_vol("123", images=7)], 4, 4, seed=1, exclude=set())
    pages = [int(k.split("-")[1][:4]) for k in picked[0]["keys"]]
    assert all(1 <= p <= 7 for p in pages)


def test_a_volume_with_no_images_is_skipped():
    assert plan([_vol("123", images=0)], 10, 4, seed=1, exclude=set()) == []


def test_seed_makes_the_sample_reproducible():
    vols = [_vol(str(i)) for i in range(80)]
    a = plan(vols, 40, 4, seed=7, exclude=set())
    b = plan(vols, 40, 4, seed=7, exclude=set())
    assert a == b


def test_era_buckets_by_half_century():
    assert era({"start_date": "1848-09-14T00:00:00Z"}) == "1800s"
    assert era({"start_date": "1884-05-18T00:00:00Z"}) == "1850s"
    assert era({}) == "undated"


def test_list_valued_metadata_does_not_become_a_stratum_of_its_own():
    """`subject` and friends are lists in volumes.json; a stringified list would
    make every volume its own stratum and defeat the spreading."""
    s = stratum({"country": ["Cuba"], "language": ["Spanish"],
                 "type": ["Ecclesiastical records"], "start_date": "1850-01-01"})
    assert s.startswith("Cuba|Spanish|Ecclesiastical records|")


def test_image_count_accepts_int_or_digit_string():
    assert n_images({"images": 154}) == 154
    assert n_images({"images": "154"}) == 154
    assert n_images({"images": "many"}) == 0
