"""Stratified pair sampling for the disambiguation training set.

Daniel, 2026-07-29: "what would be most useful now is a ~10% sample *of pairs*
containing as wide a variety of cases as possible. If/when that exists, I will
do the manual review (let's say on a 0%/25%/50%/75%/100% likelihood of sameness
basis) and that data can be used to train said model."

Two things follow from that, and they pull in opposite directions.

**Variety, not proportion.** A uniform 10% draw is ~97% easy cases, because the
population is. Every rare case type — a surname-blocked chain, a cross-volume
match, a name agreement contradicted by attributes — would appear a handful of
times or not at all, which is precisely the material the model needs most. So
pairs are binned by case type and the budget is spread across bins by
water-filling: every bin fills to the same depth until it runs out of members or
the budget runs out. Rare bins are therefore over-represented on purpose.

**Which does bias the sample.** That is recoverable, not ignorable: every
sampled pair carries its `stratum` and that stratum's true `stratum_population`,
so an inverse-probability weight is `stratum_population / stratum_sampled`. A
calibrated model can be trained from this sample; an uncalibrated one that just
learns the decision boundary can ignore the weights.

**The range has to span the decision, not just the doubt.** The review queue
holds only [review_threshold, auto_threshold) — pairs the algorithm was unsure
about. Labels drawn from it alone would cluster at 25/50/75 and never teach the
model what a confident merge or a confident non-merge looks like. So the sample
is drawn from the full scored population (via `disambiguate_volume(pair_log=...)`),
including auto-merged pairs and sub-threshold pairs. Auto-merges are where a
wrong label is most expensive, since nothing downstream questions them.
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .disambiguate import _namesets_overlap, _surname_of

# Score bands. The three narrow ones straddle the review window because that is
# where the label carries the most information; the wide outer ones exist so the
# 0% and 100% ends of Daniel's scale are actually populated.
_BANDS: Tuple[Tuple[float, float, str], ...] = (
    (0.00, 0.55, "very-low"),
    (0.55, 0.70, "low"),
    (0.70, 0.78, "review-low"),
    (0.78, 0.86, "review-high"),
    (0.86, 0.95, "auto"),
    (0.95, 1.01, "auto-strong"),
)


def band_of(score: float) -> str:
    for lo, hi, name in _BANDS:
        if lo <= score < hi:
            return name
    return "auto-strong"


def volume_of(entry_id: Any) -> str:
    """`29597-0003-01` -> `29597`. Cross-volume pairs are a distinct case type:
    different register, often different scribe, sometimes different language."""
    m = re.match(r"(\d+)", str(entry_id or ""))
    return m.group(1) if m else "?"


def surname_relation(a_name: Optional[str], b_name: Optional[str]) -> str:
    """How the two surnames relate — the axis the merge guard turns on."""
    sa, sb = _surname_of(a_name), _surname_of(b_name)
    if not sa or not sb:
        return "one-missing"
    if sa == sb:
        return "identical"
    return "variant" if _namesets_overlap({sa}, {sb}) else "different"


def signal_of(reasons: Iterable[str]) -> str:
    """The dominant evidence type, collapsed to one label so the cross-product
    of strata stays small enough to fill."""
    r = " ".join(reasons or ())
    if "blocked" in r:
        return "blocked"
    if "conflict" in r:
        return "attribute-conflict"
    if "third party" in r or "third-party" in r:
        return "third-party"
    if "context" in r or "shared" in r:
        return "shared-context"
    if "agree" in r:
        return "attribute-agree"
    return "name-only"


def stratum_of(pair: Dict[str, Any]) -> str:
    a, b = pair["a"], pair["b"]
    return "|".join((
        band_of(pair["score"]),
        pair.get("disposition", "?"),
        surname_relation(a.get("name"), b.get("name")),
        "same-vol" if volume_of(a["entry"]) == volume_of(b["entry"]) else "cross-vol",
        signal_of(pair.get("reasons")),
    ))


class StratifiedReservoir:
    """Bounded, seeded reservoir sampling, one reservoir per stratum.

    Duck-types as the `pair_log` list `disambiguate_volume` appends to, so the
    scoring pass never has to materialise its tens of millions of pairs. Each
    stratum keeps at most `per_cell` members by Algorithm R, so what survives is
    a uniform random draw from that stratum however large it grew, and the true
    population size is still counted exactly.
    """

    def __init__(self, per_cell: int = 120, seed: int = 20260729):
        self.per_cell = per_cell
        self._rng = random.Random(seed)
        self.cells: Dict[str, List[dict]] = defaultdict(list)
        self.seen: Counter = Counter()
        self.total = 0

    def append(self, pair: dict) -> None:
        key = stratum_of(pair)
        self.seen[key] += 1
        self.total += 1
        cell = self.cells[key]
        if len(cell) < self.per_cell:
            cell.append(pair)
            return
        # Algorithm R: the n-th member replaces a uniformly chosen incumbent
        # with probability per_cell/n, which keeps the reservoir uniform.
        j = self._rng.randrange(self.seen[key])
        if j < self.per_cell:
            cell[j] = pair

    # -- drawing ---------------------------------------------------------- #

    def draw(self, budget: int) -> List[dict]:
        """Water-fill the budget across strata: every stratum gets the same
        depth until it is exhausted or the budget is. Maximises the number of
        distinct case types represented, which is the stated goal."""
        keys = sorted(self.cells)
        taken: Dict[str, int] = {k: 0 for k in keys}
        remaining = budget
        while remaining > 0:
            open_cells = [k for k in keys if taken[k] < len(self.cells[k])]
            if not open_cells:
                break
            # one pass of depth 1 across every cell that still has members
            for k in open_cells:
                if remaining == 0:
                    break
                taken[k] += 1
                remaining -= 1

        out: List[dict] = []
        for k in keys:
            n = taken[k]
            if not n:
                continue
            cell = self.cells[k]
            picks = cell if n >= len(cell) else self._rng.sample(cell, n)
            for p in picks:
                q = dict(p)
                q["stratum"] = k
                q["stratum_population"] = self.seen[k]
                q["stratum_sampled"] = n
                # inverse-probability weight, so a model that needs the real
                # distribution can recover it from this deliberately skewed draw
                q["weight"] = round(self.seen[k] / n, 4)
                out.append(q)
        out.sort(key=lambda p: (-p["score"], p["a"]["entry"], p["b"]["entry"]))
        return out

    def coverage(self, drawn: List[dict]) -> Dict[str, Any]:
        got = Counter(p["stratum"] for p in drawn)
        return {
            "pairs_scored": self.total,
            "strata_present": len(self.cells),
            "strata_represented": len(got),
            "strata_missed": sorted(set(self.cells) - set(got)),
            "sampled": len(drawn),
            "by_band": dict(Counter(band_of(p["score"]) for p in drawn)),
            "by_disposition": dict(Counter(p.get("disposition") for p in drawn)),
            "by_surname_relation": dict(Counter(p["stratum"].split("|")[2] for p in drawn)),
            "by_scope": dict(Counter(p["stratum"].split("|")[3] for p in drawn)),
            "by_signal": dict(Counter(p["stratum"].split("|")[4] for p in drawn)),
            "population_by_band": {
                b: sum(c for k, c in self.seen.items() if k.split("|")[0] == b)
                for b in {k.split("|")[0] for k in self.seen}},
        }


def attach_entry_text(pairs: List[dict], texts: Dict[str, str],
                      limit: int = 700) -> List[dict]:
    """Give each side its register entry, truncated. Without the source text a
    reviewer is judging a name against a name, which is not the judgement being
    asked for."""
    for p in pairs:
        for side in ("a", "b"):
            t = texts.get(str(p[side]["entry"]))
            if t:
                p[side]["text"] = t[:limit] + ("…" if len(t) > limit else "")
    return pairs
