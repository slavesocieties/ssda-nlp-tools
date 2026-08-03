"""The A/B comparator refuses comparisons it cannot interpret.

Two comparisons went wrong in one day and both looked fine on screen. The
comparator exists to make those failures loud rather than plausible.
"""
import json
import os

import pytest

from compare_merge_runs import main


def _write(tmp, tag, mentions, identities, lifespan_blocks, volumes=("a", "b")):
    json.dump({"mentions": mentions, "identities": identities,
               "merged_identities": 100, "auto_merges": 500,
               "config": {"volumes": list(volumes)},
               "merges_blocked_by_surname_tier": {
                   "exact": 1000, "blocked-lifespan": lifespan_blocks}},
              open(os.path.join(tmp, f"{tag}.stats.json"), "w", encoding="utf-8"))


def test_a_valid_comparison_is_reported(tmp_path, capsys):
    d = str(tmp_path)
    _write(d, "ctrl", 1000, 900, 0)
    _write(d, "treat", 1000, 910, 40)
    assert main(["ctrl", "treat", "--outdir", d]) == 0
    assert "+10" in capsys.readouterr().out


def test_different_corpora_are_refused(tmp_path, capsys):
    """v7 vs v8: mentions went 27,875 -> 39,697 because the corpus grew from 5
    volumes to 7. Nothing in that comparison was attributable."""
    d = str(tmp_path)
    _write(d, "ctrl", 1000, 900, 0)
    _write(d, "treat", 2000, 1800, 40)
    assert main(["ctrl", "treat", "--outdir", d]) == 1
    assert "not the same" in capsys.readouterr().out


def test_a_control_that_is_not_a_control_is_refused(tmp_path, capsys):
    """The dangerous one. --no-lifespan set a flag nothing consumed, so both
    runs had the guard ON and the delta was exactly zero everywhere -- which
    reads as "the change does nothing"."""
    d = str(tmp_path)
    _write(d, "ctrl", 1000, 900, 1416)
    _write(d, "treat", 1000, 900, 1416)
    assert main(["ctrl", "treat", "--outdir", d]) == 1
    out = capsys.readouterr().out
    assert "not a control" in out


def test_differing_volume_lists_are_refused(tmp_path, capsys):
    d = str(tmp_path)
    _write(d, "ctrl", 1000, 900, 0, volumes=("a", "b"))
    _write(d, "treat", 1000, 910, 40, volumes=("a", "b", "c"))
    assert main(["ctrl", "treat", "--outdir", d]) == 1
    assert "volume lists" in capsys.readouterr().out


def test_force_prints_but_says_the_figures_are_not_attributable(tmp_path, capsys):
    d = str(tmp_path)
    _write(d, "ctrl", 1000, 900, 1416)
    _write(d, "treat", 1000, 900, 1416)
    main(["ctrl", "treat", "--outdir", d, "--force"])
    assert "NOT attributable" in capsys.readouterr().out
