"""Comparing two transcriptions of handwriting we cannot read.

There is no ground truth: no human-verified text exists for these registers, so
character error rate is not computable and "A differs from B" says nothing about
which is right. These tests pin the substitute -- measuring whether the text
supports the downstream pipeline -- and the two ways it went wrong.
"""
import json

import pytest

from run_transcription_bakeoff import main as bakeoff_main
from ssda_nlp_tools.transcription_bakeoff import (MATERIAL, compare,
                                                  divergent_pages,
                                                  score_transcription)


def _vol(texts, partials=0, low_conf=(), pages=2):
    entries = [{"id": f"V-{i:04d}", "text": t, "source_images": [f"V-{i%pages:04d}.jpg"],
                "partial": i < partials} for i, t in enumerate(texts)]
    return {"volume": "V", "entries": entries, "stats": {"pages": pages},
            "low_confidence_pages": list(low_conf)}


GOOD = ["En la Villa de Guanabacoa en cinco de Enero de mil ochocientos cuarenta "
        "años se dio sepultura al cadaver de Jose de nacion Congo esclavo de Don "
        "Rafael Santalla"] * 6
MANGLED = ["En1a Vi11a de Guanabac0a en cinc0 de Ener0 de mi1 0ch0cient0s cuarenta "
           "an0s se di0 sepu1tura a1 cadauer de J0se de naci0n C0ng0 esc1au0"] * 6


def test_identical_transcriptions_produce_no_winner():
    """The instrument must not invent a difference."""
    s = score_transcription(_vol(GOOD))
    res = compare(s, s, "a", "b")
    assert res["wins"]["a"] == res["wins"]["b"] == 0
    assert "no measurable difference" in res["verdict"]


def test_a_rounding_difference_is_a_tie_not_a_win():
    """This is the bug the tool shipped with, caught by running it: on a pair
    where one side had 150x the dangling-entry rate, the tally came out 3-3,
    because a 0.09% difference in vocabulary hits scored as a win exactly equal
    to the catastrophe. An unweighted tally is an implicit EQUAL weighting."""
    a = {"entries_per_page": 3.054, "partial_rate": 0.2103,
         "vocab_hits_per_1k_words": 1.089, "median_entry_chars": 380}
    b = {"entries_per_page": 3.417, "partial_rate": 0.0014,
         "vocab_hits_per_1k_words": 1.088, "median_entry_chars": 375}
    res = compare(a, b, "old", "new")
    by = {r["metric"]: r["better"] for r in res["rows"]}
    assert by["vocab_hits_per_1k_words"] == "tie"     # 0.1% apart
    assert by["median_entry_chars"] == "tie"          # 1.3% apart
    assert by["partial_rate"] == "new"                # 99.3% apart
    assert res["wins"]["new"] == 2 and res["wins"]["old"] == 0


def test_dangling_entries_dominate_because_they_break_stitching():
    clean = score_transcription(_vol(GOOD, partials=0))
    broken = score_transcription(_vol(GOOD, partials=5))
    assert broken["partial_rate"] > clean["partial_rate"]
    assert compare(clean, broken, "clean", "broken")["wins"]["clean"] >= 1


def test_mangled_text_loses_formulae_and_vocabulary():
    """A transcription that garbles characters drops out of the controlled
    vocabulary and stops matching the register's opening formulae. Both are
    countable without knowing what the page actually says."""
    good = score_transcription(_vol(GOOD))
    bad = score_transcription(_vol(MANGLED))
    assert good["formula_rate"] > bad["formula_rate"]
    assert good["vocab_hits"] > bad["vocab_hits"]


def test_embedded_api_failures_are_counted_not_ignored():
    vol = _vol(GOOD[:3] + ["[transcription error] unable to transcribe"] * 2)
    assert score_transcription(vol)["error_marks_in_text"] >= 1


def test_raw_archivault_pages_are_rejected_until_segmented():
    """A raw page list has text, but no record boundaries to score."""
    try:
        score_transcription([{"file": "V-0001.jpg", "transcription": "Aos..."}])
    except ValueError as exc:
        assert "segment" in str(exc)
    else:
        raise AssertionError("raw Archivault pages must never score as a tie")


def test_divergent_pages_rank_disagreement_first():
    """The reviewer's time should go where the models disagree, and long texts
    must not be silently mis-scored -- SequenceMatcher's autojunk disables
    matching above 200 characters, which is every page here."""
    a = _vol(GOOD, pages=2)
    b = _vol([GOOD[0]] * 3 + MANGLED[:3], pages=2)
    rows = divergent_pages(a, b, top=2)
    assert rows and rows[0]["similarity"] <= rows[-1]["similarity"]
    assert all(0.0 <= r["similarity"] <= 1.0 for r in rows)


def test_material_threshold_is_relative_not_absolute():
    """Metrics live on wildly different scales -- a rate in [0,1] and a word
    count in the tens of thousands. An absolute epsilon would make every rate a
    tie and every count a win."""
    small = compare({"partial_rate": 0.001}, {"partial_rate": 0.002}, "a", "b")
    big = compare({"words": 56954}, {"words": 56976}, "a", "b")
    assert small["rows"][0]["better"] == "a"       # 50% apart, material
    assert big["rows"][0]["better"] == "tie"       # 0.04% apart, noise
    assert MATERIAL == 0.05


def test_report_html_escapes_and_marks_differences(tmp_path):
    from ssda_nlp_tools.bakeoff_html import render_bakeoff_html
    div = [{"image": "<img onerror=1>.jpg", "similarity": 0.5,
            "a": "En la Villa de Guanabacoa", "b": "En la Vi11a de Guanabac0a"}]
    out = str(tmp_path / "b.html")
    render_bakeoff_html(div, out, "gemini", "luna")
    html = open(out, encoding="utf-8").read()
    assert "<img onerror=1>" not in html          # escaped
    assert "&lt;img onerror=1&gt;.jpg" in html
    assert "<del>" in html and "<ins>" in html    # word-level diff rendered


def test_paid_dry_runs_need_no_credentials_and_never_print_inputs(tmp_path, monkeypatch, capsys):
    """A dry-run is a planning operation, not a weaker paid operation."""
    (tmp_path / "submit_job.py").write_text("# dry-run placeholder\n", encoding="utf-8")
    key_file = tmp_path / "one_key.txt"
    key_file.write_text("sensitive/source/object.jpg\n", encoding="utf-8")
    images = tmp_path / "images"; images.mkdir()
    (images / "page.jpg").write_bytes(b"not read in dry-run")
    divergent = tmp_path / "divergent.json"
    divergent.write_text(json.dumps([{"image": "page.jpg", "similarity": 0.5,
                                      "a": "A", "b": "B"}]), encoding="utf-8")
    monkeypatch.delenv("ARCHIVAULT_PASSWORD", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert bakeoff_main(["probe", "--archivault", str(tmp_path),
                         "--email", "test@example.org", "--keys-file", str(key_file)]) == 0
    assert bakeoff_main(["judge", str(divergent), "--images", str(images),
                         "--model", "test-model", "--reservation-per-page", "0.05",
                         "--max-usd", "0.10"]) == 0
    printed = capsys.readouterr().out
    assert "sensitive/source/object.jpg" not in printed
    assert "one S3 key" in printed


def test_confirmed_probe_requires_a_persistent_hard_cap(tmp_path, monkeypatch, capsys):
    """A prior one-page probe is not a precedent for uncapped upstream spend."""
    (tmp_path / "submit_job.py").write_text("# placeholder\n", encoding="utf-8")
    image = tmp_path / "page.jpg"; image.write_bytes(b"one page")
    monkeypatch.setenv("ARCHIVAULT_PASSWORD", "process-only-test-secret")
    assert bakeoff_main(["probe", "--archivault", str(tmp_path),
                         "--email", "test@example.org", "--local-image", str(image),
                         "--outdir", str(tmp_path / "out"), "--confirm"]) == 2
    assert "requires both --reservation-usd and --max-usd" in capsys.readouterr().out


def test_confirmed_probe_reserves_before_running_and_never_exceeds_cap(
        tmp_path, monkeypatch):
    """Unknown Archivault billing remains held, so a rerun cannot overspend."""
    import run_transcription_bakeoff as runner
    (tmp_path / "submit_job.py").write_text("# placeholder\n", encoding="utf-8")
    image = tmp_path / "page.jpg"; image.write_bytes(b"one page")
    ledger = tmp_path / "ledger.json"
    monkeypatch.setenv("ARCHIVAULT_PASSWORD", "process-only-test-secret")
    monkeypatch.setattr(runner.runpy, "run_path", lambda *a, **k: {})
    args = ["probe", "--archivault", str(tmp_path), "--email", "test@example.org",
            "--local-image", str(image), "--outdir", str(tmp_path / "out"),
            "--reservation-usd", "0.03", "--max-usd", "0.05",
            "--ledger", str(ledger), "--confirm"]
    assert bakeoff_main(args) == 0
    saved = json.loads(ledger.read_text(encoding="utf-8"))
    assert saved["reserved_usd"] == 0.03
    assert saved["reservations"][0]["status"] == "submitted_billing_pending"
    assert bakeoff_main(args) == 2


def test_judge_refuses_declared_reservation_above_cap(tmp_path):
    images = tmp_path / "images"; images.mkdir()
    (images / "page.jpg").write_bytes(b"not read in dry-run")
    divergent = tmp_path / "divergent.json"
    divergent.write_text(json.dumps([{"image": "page.jpg", "similarity": 0.5,
                                      "a": "A", "b": "B"}]), encoding="utf-8")
    assert bakeoff_main(["judge", str(divergent), "--images", str(images),
                         "--model", "test-model", "--reservation-per-page", "0.05",
                         "--max-usd", "0.04"]) == 2


def test_repair_burden_measures_what_the_extractor_had_to_fix():
    """The most direct evidence that transcription quality reaches the end of
    the pipeline: the gap between the faithful text handed to the extractor and
    the normalised text it returns. Baseline on the delivered corpus is 0.909
    median with 11.2% heavily rewritten."""
    from ssda_nlp_tools.transcription_bakeoff import repair_burden
    clean = {"entries": [{"text_faithful": t, "normalized": t} for t in GOOD]}
    assert repair_burden(clean)["median_similarity"] == 1.0
    assert repair_burden(clean)["heavily_rewritten"] == 0

    repaired = {"entries": [{"text_faithful": a, "normalized": b}
                            for a, b in zip(MANGLED, GOOD)]}
    r = repair_burden(repaired)
    assert r["median_similarity"] < 1.0
    assert r["entries_compared"] == len(GOOD)


def test_repair_burden_ignores_entries_too_short_to_score():
    """A 20-character fragment produces an unstable ratio; counting it would
    make the metric noisy in exactly the volumes with the worst transcription."""
    from ssda_nlp_tools.transcription_bakeoff import repair_burden
    vol = {"entries": [{"text_faithful": "corto", "normalized": "corto"}]}
    assert repair_burden(vol)["entries_compared"] == 0
    assert repair_burden(vol)["median_similarity"] is None


def test_an_empty_volume_is_not_the_same_as_an_unsegmented_one():
    """`entries or records` short-circuits on an EMPTY list, so a volume that
    legitimately segmented to zero entries -- blank pages, a failed
    transcription -- raised "run segment first", telling the caller to redo a
    step they had already done. Absent and empty are different answers."""
    from ssda_nlp_tools.transcription_bakeoff import repair_burden
    s = score_transcription({"entries": []})
    assert s["entries"] == 0 and s["partial_rate"] is None
    assert repair_burden({"entries": []})["entries_compared"] == 0
    # raw Archivault page JSON still refuses, which is the guard that matters
    with pytest.raises(ValueError, match="run .*segment"):
        score_transcription({"volume": "V", "pages": [{"file": "x.jpg"}]})
    # and a malformed type says so specifically rather than blaming the caller
    with pytest.raises(ValueError, match="not a list"):
        score_transcription({"entries": {"nope": 1}})
