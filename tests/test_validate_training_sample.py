"""The sampler validator must be able to FAIL, and its fixtures must be real.

Its first water-fill fixture varied only the volume id, which stratum_of barely
keys on, so it produced a single stratum -- and "spread 0 across one cell" passes
an evenness assertion while proving nothing.
"""
from validate_training_sample import (check_reservoir_uniformity,
                                      check_water_fill, check_weights)


def test_the_real_reservoir_is_uniform():
    assert check_reservoir_uniformity(trials=400, stream=120, per_cell=10)


def test_a_biased_reservoir_is_CAUGHT(monkeypatch):
    """Guard against a validator that cannot fail: a head-biased reservoir that
    never evicts must be reported as biased."""
    import validate_training_sample as V

    class NeverEvict(V.StratifiedReservoir):
        def append(self, pair):
            key = "only"
            self.seen[key] += 1
            self.total += 1
            if len(self.cells[key]) < self.per_cell:
                self.cells[key].append(pair)      # keeps the HEAD forever

    monkeypatch.setattr(V, "StratifiedReservoir", NeverEvict)
    assert not check_reservoir_uniformity(trials=200, stream=120, per_cell=10)


def test_water_fill_fixture_makes_more_than_one_stratum():
    """The assert inside check_water_fill is the real subject here."""
    assert check_water_fill()


def test_weights_missing_file_is_skipped_not_passed_silently(capsys):
    assert check_weights("no_such_labels.json") is True
    assert "skipped" in capsys.readouterr().out
