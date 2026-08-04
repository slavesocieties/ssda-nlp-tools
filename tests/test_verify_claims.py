import re
"""The claim matcher, and why it has to accept prose.

verify_claims exists because I twice quoted a figure computed once and never
rechecked. Its first run flagged four figures and three were correct prose --
"7.3 million" for 7,305,667, "nine" for 9. A checker that cries wolf gets
ignored, so accepting the forms a person actually writes is not a nicety.
"""
from verify_claims import _states_value


def test_exact_and_comma_grouped():
    assert _states_value("we delivered 6794 records", 6794)
    assert _states_value("we delivered 6,794 records", 6794)


def test_spelled_out_small_counts():
    """DM_OPENAI_REPO_BUGS says "all nine files" and contains no digit 9."""
    assert _states_value("all nine files parse cleanly", 9)
    assert _states_value("three volumes overlap", 3)


def test_rounded_millions():
    assert _states_value("scored 7.3 million pairs", 7_305_667)
    assert _states_value("scored 7,305,667 pairs", 7_305_667)


def test_a_wrong_figure_is_still_caught():
    """The point is not to accept everything."""
    assert not _states_value("we delivered 5,228 records", 6794)
    assert not _states_value("all eight files parse", 9)


def test_word_match_is_bounded():
    """"nine" must not match inside another word."""
    assert not _states_value("the ninety-third entry", 9)


def test_floats_match_at_sensible_precision():
    assert _states_value("median similarity 0.891", 0.89081)
    assert _states_value("substitution 6.31%", 6.31)


def test_none_never_matches():
    assert not _states_value("anything at all", None)


def test_no_stray_control_characters_in_the_source():
    """A heredoc turned every \b in this file into a literal backspace byte,
    so the word-boundary regexes silently matched nothing and the checker
    reported false mismatches it had itself created."""
    import verify_claims
    src = open(verify_claims.__file__, "rb").read()
    assert bytes([8]) not in src, "literal backspace byte in source"


# --- added 2026-08-04: the checker must not pass vacuously -------------------
# DM_DISAMBIG_REPLY.md triggered ZERO tracked claims and the report said "every
# tracked figure in every DM matches", which reads as verified. It contained an
# unverified pair score at the time.

def test_a_dm_stating_no_tracked_figure_is_reported_as_unchecked(tmp_path, capsys):
    import verify_claims as V
    d = tmp_path / "dms"
    d.mkdir()
    (d / "DM_NOTHING.md").write_text("Prose with no tracked quantity in it.",
                                     encoding="utf-8")
    V.main(["--dms", str(d), "--root", ".",
            "--transcriptions", str(tmp_path), "--manual", str(tmp_path)])
    out = capsys.readouterr().out
    assert "NOT CHECKED" in out
    assert "DM_NOTHING.md" in out


def test_agreement_trigger_ignores_agreeing_about_something_else():
    """'how much we agree with that model' is about extraction, not labels."""
    import verify_claims as V
    text = ("measuring our extraction against another model's extraction tells "
            "us how much we agree with that model, not how accurate we are.")
    pats = V.CLAIMS["label_agreement_pct"]
    assert not any(re.search(p, text, re.I) for p in pats)


def test_agreement_trigger_fires_on_a_real_agreement_claim():
    import verify_claims as V
    text = "Of the 24 you were certain about, we now agree on 21, or 88%:"
    pats = V.CLAIMS["label_agreement_pct"]
    assert any(re.search(p, text, re.I) for p in pats)


def test_merge_claims_survive_missing_artifacts(tmp_path):
    """A fresh clone has no merge outputs; recompute must not explode."""
    import verify_claims as V
    assert V._merge_claims(str(tmp_path), {}) == {}
