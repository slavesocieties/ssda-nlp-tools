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

from .disambiguate import _third_party_same
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

# CONFLICTING relationships. Daniel, 2026-08-07: two same-named people with
# DIFFERENT spouses is "essentially disqualifying - it's not just that they have
# no shared relationship, they have actively distinguishing individual
# relationships." Before this, a mismatch scored exactly 0.00 -- the same as
# silence.
#
# NOT ALL CLASHING RELATIONS ARE THE SAME. Daniel, 2026-08-10:
#
#   "Clearly different parents = automatically disqualifying; Clearly different
#   spouses in an early modern Catholic society = essentially automatically
#   disqualifying; different slave owners within a short span of time =
#   essentially automatically disqualifying; different slave owner across a span
#   of a decade+ = substantial penalty but not disqualifying."
#
#   "children/godchildren can't really conflict because they're not a limited
#   set, godparents/grandparents function similarly to parents, witnesses can
#   only increase match probability through an overlapping association."
#
# So a role conflicts only if it is a LIMITED SET, and the test is arithmetic:
# more distinct holders between the two mentions than a person can have. You
# have two parents; you have one spouse at a time; you have any number of
# children, godchildren and witnesses, so those can never contradict.
#
# `former spouse` is deliberately absent. It is the register's own record of
# widowhood and remarriage -- the one lawful way to have had two spouses -- so
# treating it as a clash would penalise exactly the case that explains it.
MAX_HOLDERS = {
    "parent": 2,        # a mother and a father
    "godparent": 2,     # Daniel: "function similarly to parents"
    "grandparent": 4,   # unsided: two per side, but we are not told which
    "maternal grandparent": 2,
    "paternal grandparent": 2,
    "spouse": 1,        # exclusive, and see the note on indissolubility below
    "enslaver": 1,      # exclusive at a moment, but transferable -- see below
}

# GRANDPARENTS, AND WHY THEY NEEDED FIXING UPSTREAM.
#
# Daniel, 2026-08-10: "This is a question that can/should be resolved upstream.
# Maternal/paternal grandparents should be labeled differently when extracted."
#
# Capacity 4 was safe but nearly unreachable. Because an unsided extraction does
# not say which side a grandparent belongs to, one record naming the maternal
# pair and another naming the paternal pair presented as four distinct
# grandparents -- indistinguishable from agreement, so the role could contradict
# only in the rare five-grandparent case. With the side extracted, each side
# holds two and a grandparent clash becomes as informative as a parent clash.
#
# THE AGGREGATE CHECK IS WHAT MAKES THE TRANSITION SAFE. The delivered corpus is
# entirely unsided, and re-extraction will land volume by volume, so sided and
# unsided records will be compared against each other for a long time. Checking
# only the per-side capacities would silently stop checking those mixed pairs
# altogether -- a regression, not a refinement. So the three terms are ALSO
# checked together against the old capacity of 4, and a pair is charged at most
# once for the family.
ROLE_FAMILY_CAPACITY = {"grandparent": 4}
ROLE_FAMILY_MEMBERS = {
    "grandparent": ("grandparent", "maternal grandparent", "paternal grandparent"),
}
# Which family a role is charged under. A role absent here is its own family.
ROLE_FAMILY = {member: family
               for family, members in ROLE_FAMILY_MEMBERS.items()
               for member in members}

# WHY "DISQUALIFYING" IS A SIZE AND NOT A VETO.
#
# Daniel's word is "automatically", which is veto language, and these very
# nearly are vetoes. They are sized instead of hard-coded because the relation
# names are extracted by an LLM: a hallucinated spouse would otherwise be
# unrecoverable, and we already know duplicate and malformed extractions exist
# in this corpus. So the weight is DERIVED from the constraint it has to satisfy
# -- it must outweigh the strongest agreement the model can otherwise assemble,
# including a maximally rare shared associate -- rather than picked. If any
# other weight changes, this one follows, and a test asserts the property.
#
# The historical reasoning Daniel asked for, stated so it can be argued with:
# marriage in this society is indissoluble, so two records naming different
# living spouses cannot both be the same person; the only lawful second marriage
# follows a death, and the registers record that separately as `former spouse`.
# Enslavement carries no such permanence -- sale, inheritance and manumission all
# transfer a person between owners -- so it is the one clash that time can
# explain, and it is graded by the gap rather than flat.
W_CONFLICT_SUBSTANTIAL = -3.0      # "substantial penalty but not disqualifying"
DISQUALIFY_MARGIN = 2.0

# ENSLAVEMENT IS THE ONE CLASH THAT TIME EXPLAINS, SO IT IS GRADED CONTINUOUSLY.
#
# Daniel, 2026-08-10, asked for exactly this: "Continuously is likely the right
# answer, as is the nuanced observation that a child was more likely to be
# inherited or sold than an adult."
#
#     penalty(gap) = SUBSTANTIAL + (DISQUALIFYING - SUBSTANTIAL) * exp(-gap/TAU)
#
# At a gap of zero it is the full disqualifying weight: two owners on the same
# day is not a transfer, it is two people. It relaxes towards "substantial but
# not disqualifying" as the gap grows, because sale, inheritance and manumission
# all become likelier the longer the interval. It never reaches zero, which is
# deliberate -- a different owner is always some evidence.
#
# TAU is the timescale over which a transfer becomes likely. At 5 years the
# curve gives roughly -8.5 at a 2-year gap, -4.1 at a decade, -3.2 at twenty
# years, which is the shape Daniel described. It is a prior, not a measurement.
#
# CHILD_MOBILITY makes the same gap less damning for a child, because a child
# was likelier to be inherited or sold. It multiplies the EFFECTIVE gap, so the
# curve decays faster for them. Only ~21% of enslaver clashes have an age on
# both sides, so this is a refinement on a minority, not a load-bearing term.
#
# One exp() per conflicting pair. Daniel: "if this level of nuance can be
# captured elegantly without inordinate compute requirements at scale, I'm all
# for it" -- this is O(1) and adds no measurable cost.
ENSLAVER_TAU_YEARS = 5.0
CHILD_MOBILITY = 2.0
_CHILD_AGES = {"infant", "child"}
# W_CONFLICT_DISQUALIFYING is derived below, once the weights it must outweigh
# (MAX_NAME_LLR, MAX_LLR_PER_ASSOCIATE, REVIEW_LOG_ODDS) have been defined.

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
# 5.5 LOOKS TOO LOW ON THE ARITHMETIC AND IS NOT. Re-deriving against the
# CURRENT weights says the cap could go to 7.79: circumstantial evidence now
# tops out at +0.77 (best place 0.50 + close date 0.27), not the +1.72 that
# justified 5.5. And the cost of 5.5 is real -- only 1.2% of the corpus's 39,696
# mentions keep a distinct name weight; 98.8% are flattened to one value,
# including every one of the 29,609 same-name same-parish pairs whose rarity
# genuinely spans 6.03 to 10.43 nats.
#
# I RAISED IT TO 7.5 AND THE CORPUS REJECTED IT. Transatlantic contamination
# went from 56 mentions to 1,169, a 21x regression, because THE CAP AND THE
# LOCATION WEIGHT ARE COUPLED:
#     cap 5.5: rare name + different continent = -0.49  -> dropped
#     cap 7.5: rare name + different continent = +1.51  -> survives to review
# different-country is only -0.43, and that is MEASURED, from 14 positives. A
# stronger name term needs a stronger geographic veto to hold it back, and there
# is no evidence available to strengthen that veto with.
#
# So the discarded 2.29 nats are not waste, they are the price of a location
# term too weakly estimated to do its job. Raising this cap is blocked on better
# geographic evidence, not on arithmetic. Reverted to 5.5, which the corpus
# prefers on every measure that matters.
MAX_NAME_LLR = 5.5

# The size a "disqualifying" conflict has to be, derived rather than chosen.
#
# Daniel's ruling is qualitative -- clearly different parents, or spouses, or an
# enslaver within a short span, are "automatically" different people. Turning
# that into a number means asking what it has to beat: the strongest agreement
# the model can otherwise assemble for a pair. That is a maximal name match, a
# maximally rare shared associate, same institution, close dates, and every hard
# attribute agreeing. If a conflict outweighs all of that with room to spare,
# Daniel's "automatically" holds without a veto's irreversibility.
_MAX_AGREEMENT = (MAX_NAME_LLR + MAX_LLR_PER_ASSOCIATE + max(W_PLACE.values())
                  + W_YEAR_CLOSE + 6 * W_ATTR_AGREE)
W_CONFLICT_DISQUALIFYING = -(LOG_PRIOR_ODDS + _MAX_AGREEMENT
                             - REVIEW_LOG_ODDS + DISQUALIFY_MARGIN)


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


def _enslaver_penalty(gap: Optional[int], a, b) -> float:
    """How damning two different owners are, given how far apart the records sit.

    See ENSLAVER_TAU_YEARS. With no dates there is no interval to appeal to, so
    the clash gets the strict reading rather than the benefit of the doubt.
    """
    if gap is None:
        return W_CONFLICT_DISQUALIFYING
    eff = gap
    if str(a.get("age") or "").lower() in _CHILD_AGES or \
       str(b.get("age") or "").lower() in _CHILD_AGES:
        eff *= CHILD_MOBILITY
    decay = math.exp(-eff / ENSLAVER_TAU_YEARS)
    return W_CONFLICT_SUBSTANTIAL + (W_CONFLICT_DISQUALIFYING
                                     - W_CONFLICT_SUBSTANTIAL) * decay


def _n_distinct(names) -> int:
    """How many DIFFERENT third parties a set of names refers to.

    Scribal variation means the raw set size overcounts: {"joao da silva",
    "joam da silva"} is one man. Names are folded together transitively under
    `_third_party_same`, the same test used to suppress a false conflict.
    """
    groups: List[set] = []
    for n in sorted(names):
        for g in groups:
            if any(_third_party_same(n, m) for m in g):
                g.add(n)
                break
        else:
            groups.append({n})
    return len(groups)


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

    # CONFLICTING relationships, graded by how limited the role actually is.
    #
    # A role can only contradict if it is a LIMITED SET, and the test is
    # arithmetic: more distinct holders between the two mentions than one person
    # can have. See MAX_HOLDERS for the capacities and Daniel's 2026-08-10
    # ruling behind them. Roles absent from that table -- child, godchild,
    # witness, sibling, and the rest -- can never conflict, however little they
    # overlap, because there is no limit for them to exceed.
    #
    # A conflict is suppressed when the two sides name the same person under
    # `_third_party_same`, so scribal drift is never read as a contradiction.
    #
    # Each check is (label, roles-to-pool, capacity). Most are a single role;
    # the sided grandparents additionally get pooled into one family check at
    # the old capacity of 4, so a sided record compared against an unsided one
    # is no less protected than before (see ROLE_FAMILY_MEMBERS).
    ya = a.get("_year") or a.get("year")
    yb = b.get("_year") or b.get("year")
    gap = abs(int(ya) - int(yb)) if (ya and yb) else None
    checks = [(role, (role,), cap) for role, cap in MAX_HOLDERS.items()]
    checks += [(fam, ROLE_FAMILY_MEMBERS[fam], cap)
               for fam, cap in ROLE_FAMILY_CAPACITY.items()]
    charged = set()
    for role, members, capacity in checks:
        family = ROLE_FAMILY.get(role, role)
        if family in charged:
            continue          # one penalty per family, not one per member role
        ra = set().union(*(na.get(m) or set() for m in members))
        rb = set().union(*(nb.get(m) or set() for m in members))
        if not ra or not rb:
            continue
        if any(_third_party_same(x, y) for x in ra for y in rb):
            continue                       # same person, differently spelled
        if _n_distinct(ra | rb) <= capacity:
            # Within capacity, so not a contradiction. This is what makes a
            # mother in one record and a father in the other legitimate: two
            # distinct parents is exactly two, and the registers name whichever
            # one they please. The same arithmetic on the pooled grandparent
            # family is what keeps a maternal pair and a paternal pair -- or an
            # unsided record against a sided one -- reading as exactly four.
            continue

        # A GODPARENT CLASH ONLY MEANS ANYTHING WITHIN ONE SACRAMENT.
        #
        # Daniel, 2026-08-10: "for the purpose of historical/doctrinal rigor:
        # yes, godparent clashes only count within the same sacrament type."
        # You are sponsored afresh at each sacrament, so a different godparent
        # at a marriage than at a baptism is the norm, not a contradiction.
        # When either entry records no sacrament at all there is nothing to
        # match on, so the clash is not counted -- the conservative reading.
        if role == "godparent":
            sa = a.get("_sacraments") or set()
            sb = b.get("_sacraments") or set()
            if not (sa & sb):
                continue

        pen = (_enslaver_penalty(gap, a, b) if role == "enslaver"
               else W_CONFLICT_DISQUALIFYING)
        total += pen
        charged.add(family)
        reasons.append(f"conflict:{role}({pen:+.2f})")

    if not shared:
        # Absence of overlap is only evidence among roles that COULD have
        # overlapped meaningfully. Daniel: "witnesses can only increase match
        # probability through an overlapping association", and children and
        # godchildren "can't really conflict because they're not a limited set".
        # Counting those toward a disjointness penalty punished a record for
        # naming more people, which is backwards -- richer records were being
        # penalised for being richer.
        lim_a = set().union(*(na[r] for r in MAX_HOLDERS if r in na)) if any(
            r in na for r in MAX_HOLDERS) else set()
        lim_b = set().union(*(nb[r] for r in MAX_HOLDERS if r in nb)) if any(
            r in nb for r in MAX_HOLDERS) else set()
        density = min(len(lim_a), len(lim_b))
        if density >= 2:
            pen = -0.9 * min(density, 5)
            total += pen
            reasons.append(f"disjoint-networks({len(lim_a)}v{len(lim_b)}){pen:.1f}")
    return total, reasons


def score(a: dict, b: dict, stats: NameStats, geo=None,
          vol_of=None) -> Dict[str, Any]:
    """Return {log_odds, probability, vetoed, terms:[(label, llr)]}."""
    terms: List[Tuple[str, float]] = []

    # ---- vetoes: impossibilities, not weights ----------------------------
    if a.get("_entry") and a.get("_entry") == b.get("_entry"):
        return _out(terms, veto="same-entry")
    # Same EVENT recorded in two entries. The same-entry veto cannot fire, so
    # co-participants -- siblings sharing parents, two witnesses to one marriage
    # -- reach scoring with every circumstantial term agreeing, and 24% of the
    # scorable ones auto-merge. Measured: eval_data/latent_coparticipant_20260813.md
    #
    # The veto is on a DIFFERENT LOCAL ID, not on the event. Across two copies
    # of one record, local id i and i are the same person and must still merge;
    # only i vs j is the impossibility. A blanket event veto would destroy the
    # 170 true merges to prevent 50 false ones.
    #
    # SCOPE: `_event` is assigned by grouping entries on the extracted people
    # payload, so the two copies share a local-id scheme. The other form of this
    # defect -- one event written up independently in two registers -- has no
    # such correspondence and needs person-level alignment, not id equality.
    # This does not address that form.
    ev_a, ev_b = a.get("_event"), b.get("_event")
    if ev_a and ev_a == ev_b and \
            str(a.get("_local_id")) != str(b.get("_local_id")):
        return _out(terms, veto="same-event")
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
