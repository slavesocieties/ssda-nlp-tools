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
