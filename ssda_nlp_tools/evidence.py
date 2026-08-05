"""evidence.py — weight-of-evidence scoring, replacing the binary gates.

Daniel, 2026-08-05: "I'm less convinced of the effectiveness of hard binary
gating and would be more convinced by a fundamentally probabilistic approach
that aggregates the weight of the available pieces of evidence."

    log_odds(same person) = log_prior_odds + SUM of log-likelihood ratios

Each piece of evidence contributes an LLR: how much more likely we are to see it
if the two mentions ARE one person than if they are not. Positive pulls together,
negative pushes apart, and NOTHING is decided by a single term crossing a
threshold. The output carries its own itemisation, so a merge can be read as
"name +4.9, shared enslaver +6.1, same parish +1.4, 20-year gap -0.7".

WHAT REMAINS A VETO, AND WHY IT IS NOT A GATE
---------------------------------------------
Three things are not weak evidence, they are impossibilities, and giving them
finite weight would let a pile of weak agreement outvote a fact:

    same entry              the extractor already separated these two people
    lifespan impossible     no one is born 130 years before they are buried
    both sacrament principals   you are baptised once

Everything else that used to be a gate -- surname tiers, the N-corroborating-
signals bar, cluster surname compatibility -- becomes a weight.

WHERE THE WEIGHTS COME FROM, HONESTLY
-------------------------------------
The name and network terms are DERIVED, not guessed. If a name has corpus
frequency p, then two mentions sharing it is expected with probability ~p when
they are the same person and ~p^2 when they are not, so the evidence is
-ln(p) nats. That is why sharing "Maria" (p=0.008) is worth 4.8 and sharing
"Custodio Jose Vieira da Silva" is worth 10.6: a factor of e^5.8, about 330x.

The remaining weights (location, dates, attributes) are PRIORS I chose, stated
here so they can be argued with, and they are the ones Daniel's graded labels
should calibrate. 25 labels constrain a handful of weights; they do not fit all
of them. Nothing here is claimed to be trained.

CLERGY ARE EXCLUDED FROM THE FREQUENCY MODEL. A priest signing 400 entries makes
his own name look common, and the whole point of the name term is that a common
name is weak evidence. Using raw frequency would penalise exactly the recurrence
that identifies him. Daniel: "Clergy are a ~special case and can be merged very
aggressively."
"""
from __future__ import annotations

import collections
import math
from typing import Any, Dict, List, Optional, Tuple

from .textmatch import name_similarity, name_tokens, normalize_name

# --------------------------------------------------------------------------- #
# priors -- stated, arguable, and NOT fitted
# --------------------------------------------------------------------------- #

# Two mentions drawn from the candidate pool are rarely the same person. The
# blocked pool ran ~1.8M pairs to ~6.9k merges, so the prior odds are ~1:260.
LOG_PRIOR_ODDS = math.log(1 / 260)

# Location, from the institution that produced the record. Daniel: "absolutely
# critical ... should be considered heavily".
# MEASURED where the measurement means something, CONSTRAINED where it does not.
#
# LLR = ln( P(feature|same) / P(feature|different) ), with P(.|different) from
# 60,000 sampled CANDIDATE pairs and P(.|same) from the 14 confirmed positives.
#
# `institution` is the one level with real support: 56.7% of candidates and 13 of
# 14 positives, giving +0.50. It is small because BLOCKING HAS ALREADY SELECTED
# FOR IT -- over half of all candidate pairs already share an institution, so it
# can barely discriminate. My first guess was +2.00.
#
# THE FINER LEVELS ARE NOT MEASURABLE IN THIS CORPUS AND MUST NOT BE FITTED.
# With seven volumes and no two from the same institution, a "place level" is not
# an independent observation, it is a lookup on a VOLUME PAIR. There are 21 such
# pairs, and `city` corresponds to exactly ONE of them: 201991 / 29597,
# Guanabacoa and Santo Angel. Fitting it gave +1.45 -- higher than same-parish,
# which is backwards on its face -- from a single positive pair. Worse, those two
# registers NEVER COEXIST (1839-1852 against 1770-1792), so the one pair driving
# the weight is one that should be discouraged, not rewarded.
#
# So the levels are held MONOTONE by construction: co-location cannot become
# stronger evidence as it gets coarser. institution >= city >= state >= country
# >= different-country. Only `institution` and `none` are measured; the middle is
# interpolated, and honestly labelled as such.
W_PLACE = {"institution": 0.50,   # measured, 13/14 positives
           "city": 0.40,          # interpolated -- see above, NOT the fitted 1.45
           "state": 0.20,         # interpolated
           "country": 0.0,
           "none": -0.43}         # measured, but on ONE cross-country positive

W_VOLUMES_NEVER_COEXIST = -1.5     # still a prior; not separately measurable here
W_ATTR_AGREE = 0.25                # per agreeing hard attribute, deliberately small
W_ATTR_CONFLICT = -2.5             # per conflicting one
W_YEAR_CLOSE = 0.27                # measured; 70.9% of candidates already qualify
W_YEAR_FAR = -0.32                 # measured
W_CLERGY_BOTH = 2.5                # Daniel: merge clergy aggressively

AUTO_MERGE_LOG_ODDS = 3.0          # ~95% posterior
REVIEW_LOG_ODDS = 0.0              # ~50%

# A shared associate is capped so one very rare name cannot carry a merge alone.
MAX_LLR_PER_ASSOCIATE = 7.0

# THE NAME CAP IS DERIVED FROM DANIEL'S RULING, NOT CHOSEN.
#
# Daniel, 2026-07-29: "No people should be merged strictly based on name
# correspondence; it should depend on a combination of date overlap, same-named
# relation, same/similar qualities."
#
# That is a constraint on this number. For a name alone never to auto-merge:
#     LOG_PRIOR_ODDS + MAX_NAME_LLR  <  AUTO_MERGE_LOG_ODDS      -> < 8.56
# and for a matching name to stay a live candidate rather than be dismissed:
#     LOG_PRIOR_ODDS + MAX_NAME_LLR  >= REVIEW_LOG_ODDS          -> >= 5.56
# so anything in [5.56, 8.56) satisfies him.
#
# 7.0 WAS STILL TOO HIGH, and the corpus A/B is what found it. The constraint has
# to bind on the name PLUS every circumstantial term that can accompany it for
# free, not on the name by itself:
#     prior -5.56 + name 7.00 + same-city 1.45 + close-date 0.27 = +3.16 -> MERGED
# on no relationship evidence whatsoever. So the cap is set from the stronger
# requirement that name + ALL non-discriminative evidence stays below the bar,
# which leaves the threshold to be crossed only by a shared associate (+3.06
# measured) or an unusually rare name. That is Daniel's "same-named relation"
# arriving as arithmetic rather than as a gate.
#
# The first version used 9.0, taken from the raw -ln(p) of an unseen name, and
# that auto-merged on the name by itself -- exactly the thing he ruled out. It is
# also why the scorer's mean probability on the synthetic set was 0.97 against
# his mean grade of 0.57: every synthetic name is absent from the corpus, hits
# the rarity floor, and collected maximum evidence for being unknown.
MAX_NAME_LLR = 5.5


class NameStats:
    """Corpus name frequencies, computed over LAY mentions only."""

    def __init__(self, mentions, is_clergy=None):
        self.counts: collections.Counter = collections.Counter()
        for m in mentions:
            if is_clergy and is_clergy(m):
                continue
            n = normalize_name(m.get("name") or "")
            if n:
                self.counts[n] += 1
        self.total = max(sum(self.counts.values()), 1)
        # An unseen name is at least as rare as a once-seen one.
        self._floor = 1.0 / (self.total + 1)

    def p(self, name: Optional[str]) -> float:
        n = normalize_name(name or "")
        if not n:
            return 1.0                      # no name: no evidence either way
        return max(self.counts.get(n, 0) / self.total, self._floor)

    def llr(self, name: Optional[str]) -> float:
        """Evidence in nats from two mentions sharing this exact name."""
        n = normalize_name(name or "")
        if not n:
            return 0.0
        return min(-math.log(self.p(n)), MAX_NAME_LLR)


def _clergy(m) -> bool:
    o = str(m.get("occupation") or "").lower()
    t = " ".join(str(x) for x in (m.get("titles") or [])).lower()
    return ("cleric" in o or "cura" in o or "presb" in o or "priest" in o
            or any(k in t for k in ("pbro", "presb", "padre", "reveren", "cura")))


def _assoc_names(m) -> Dict[str, set]:
    """{role: {names}} from either a mention's `_ctx` or a synthetic `relations`."""
    out: Dict[str, set] = collections.defaultdict(set)
    for t, n in (m.get("_ctx") or ()):
        if n:
            out[str(t)].add(normalize_name(n))
    for r in (m.get("relations") or ()):
        if isinstance(r, dict) and r.get("name"):
            out[str(r.get("type"))].add(normalize_name(r["name"]))
    return out


def network_llr(a, b, stats: NameStats) -> Tuple[float, List[str]]:
    """Evidence from the surrounding social network.

    Daniel: people "appear embedded in a social network of some density. This is
    also critical to disambiguation."

    SHARING an associate is strong and its weight is the associate's own name
    rarity, for the same reason the name term is: two records naming the same
    "Maria" is weak, two naming the same "Custodio Jose Vieira" is not.

    NOT sharing anyone is evidence of DIFFERENCE, but only when both sides are
    densely embedded. One record with six named relatives and another with six
    entirely different ones is two families; a record with six and a record with
    one that happens to miss is nothing at all. That asymmetry is the whole
    reason a flat "no shared context" rule was wrong.
    """
    na, nb = _assoc_names(a), _assoc_names(b)
    all_a = set().union(*na.values()) if na else set()
    all_b = set().union(*nb.values()) if nb else set()
    if not all_a or not all_b:
        return 0.0, []                       # absence of evidence, not evidence

    shared = all_a & all_b
    reasons, total = [], 0.0
    for nm in sorted(shared):
        w = min(stats.llr(nm), MAX_LLR_PER_ASSOCIATE)
        # Same person in the SAME role is stronger than a role that shifted.
        roles_a = {r for r, s in na.items() if nm in s}
        roles_b = {r for r, s in nb.items() if nm in s}
        if not (roles_a & roles_b):
            w *= 0.6                         # godparent here, parent there
        total += w
        reasons.append(f"shared:{nm}(+{w:.1f})")

    if not shared:
        # Expected overlap grows with both densities; seeing none is telling
        # only when there was room for it.
        density = min(len(all_a), len(all_b))
        if density >= 2:
            pen = -0.9 * min(density, 5)
            total += pen
            reasons.append(f"disjoint-networks({len(all_a)}v{len(all_b)}){pen:.1f}")
    return total, reasons


def score(a: dict, b: dict, stats: NameStats, geo=None,
          vol_of=None) -> Dict[str, Any]:
    """Return {log_odds, probability, vetoed, terms:[(label, llr)]}."""
    terms: List[Tuple[str, float]] = []

    # ---- vetoes: impossibilities, not weights ----------------------------
    if a.get("_entry") and a.get("_entry") == b.get("_entry"):
        return _out(terms, veto="same-entry")
    if a.get("_unique_sacrament") and b.get("_unique_sacrament"):
        return _out(terms, veto="both-sacrament-principals")

    # ---- name ------------------------------------------------------------
    sim = name_similarity(a.get("name"), b.get("name"))
    if sim <= 0.0:
        return _out(terms, veto="different-names")
    rarity = min(stats.llr(a.get("name")), stats.llr(b.get("name")))
    terms.append((f"name~{sim:.2f} rarity", sim * rarity))

    # ---- social network --------------------------------------------------
    n_llr, n_why = network_llr(a, b, stats)
    if n_llr or n_why:
        terms.append(("network:" + ",".join(n_why) if n_why else "network", n_llr))

    # ---- clergy ----------------------------------------------------------
    if _clergy(a) and _clergy(b):
        terms.append(("both-clergy", W_CLERGY_BOTH))

    # ---- location --------------------------------------------------------
    if geo is not None and vol_of is not None:
        va, vb = vol_of(a), vol_of(b)
        lvl = geo.same_place(va, vb)
        if lvl is not None:
            terms.append((f"place:{lvl}", W_PLACE.get(lvl, 0.0)))
        if geo.overlapping_years(va, vb) is False:
            terms.append(("volumes-never-coexist", W_VOLUMES_NEVER_COEXIST))

    # ---- dates -----------------------------------------------------------
    ya, yb = a.get("_year") or a.get("year"), b.get("_year") or b.get("year")
    if ya and yb:
        gap = abs(int(ya) - int(yb))
        if gap <= 20:
            terms.append((f"gap{gap}y", W_YEAR_CLOSE))
        elif gap > 40:
            terms.append((f"gap{gap}y", W_YEAR_FAR))

    # ---- attributes ------------------------------------------------------
    agree = conflict = 0
    for k in ("phenotype", "free", "ethnicity", "origin", "occupation", "legitimate"):
        x, y = a.get(k), b.get(k)
        if x is None or y is None:
            continue
        if str(x).strip().lower() == str(y).strip().lower():
            agree += 1
        else:
            conflict += 1
    if agree:
        terms.append((f"attrs-agree x{agree}", W_ATTR_AGREE * agree))
    if conflict:
        terms.append((f"attrs-conflict x{conflict}", W_ATTR_CONFLICT * conflict))

    return _out(terms)


def _out(terms, veto=None) -> Dict[str, Any]:
    if veto:
        return {"log_odds": float("-inf"), "probability": 0.0,
                "vetoed": veto, "terms": terms, "decision": "refuse"}
    lo = LOG_PRIOR_ODDS + sum(w for _, w in terms)
    p = 1.0 / (1.0 + math.exp(-lo)) if lo > -700 else 0.0
    return {"log_odds": lo, "probability": p, "vetoed": None, "terms": terms,
            "decision": ("merge" if lo >= AUTO_MERGE_LOG_ODDS
                         else "review" if lo >= REVIEW_LOG_ODDS else "refuse")}


def explain(res: Dict[str, Any]) -> str:
    if res["vetoed"]:
        return f"REFUSE (impossible: {res['vetoed']})"
    rows = "\n".join(f"    {lbl:<44s} {w:+6.2f}" for lbl, w in res["terms"])
    return (f"{res['decision'].upper()}  p={res['probability']:.3f}  "
            f"log-odds {res['log_odds']:+.2f}\n"
            f"    {'prior':<44s} {LOG_PRIOR_ODDS:+6.2f}\n{rows}")
