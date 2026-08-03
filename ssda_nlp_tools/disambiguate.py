"""Automated, confidence-scored cross-entry person disambiguation.

Replaces utility.py's manual ``input("y/n")`` merge hook with a scalable
pipeline: score every candidate mention pair, auto-merge the confident ones,
and route the borderline ones to a ranked review queue instead of blocking on a
human for all of them.

Pairwise score = name similarity, adjusted by attribute *compatibility* (hard
conflicts like free vs enslaved or different phenotype push apart; agreement
pulls together) and shared-relationship context (two people enslaved by the same
named enslaver are likely linkable). Mentions from the *same* entry are never
merged — the extractor already separated them.

Design choices called out honestly:
  * Blocking by first name-token keeps this near-linear; a phonetic key
    (double-metaphone) is the natural upgrade for spelling drift across blocks.
  * Auto-merge uses union-find, so links are transitive. We record each
    cluster's weakest internal edge and flag clusters whose cohesion dips below
    `auto_threshold` for review, rather than trusting the chain blindly.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .textmatch import name_similarity, name_tokens, normalize_name, phonetic_key

# Attributes that, when both present and different, are evidence of DIFFERENT people.
HARD_ATTRS = ["phenotype", "free", "ethnicity", "origin", "occupation", "legitimate", "rank"]


class _UnionFind:
    def __init__(self, n): self.p = list(range(n))
    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def _val(p, k):
    v = p.get(k)
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v).strip().lower()


# Relationship types whose counterpart identifies a person strongly: an enslaved
# "Juan" is distinguished by WHO enslaves him; a spouse by whom they married.
DISCRIMINATIVE_CTX = ("enslaver", "spouse", "parent")


def _ctx_by_type(ctx):
    d: Dict[str, set] = defaultdict(set)
    for t, n in ctx or ():
        if t in DISCRIMINATIVE_CTX:
            d[t].add(n)
    return d


def _third_party_same(x: str, y: str) -> bool:
    """Are two third-party names (a spouse/enslaver/parent) the same person?

    Estate surnames are shared by everyone attached to the estate ("hanna
    macqueen" vs "rachael macqueen" are DIFFERENT wives), so the GIVEN name must
    agree; but short forms contain long forms ("rachael" is "rachael macqueen").

    THE SAME ARGUMENT RUNS THE OTHER WAY and used not to. Given names in these
    registers are drawn from a tiny pool -- Francisco, Juan, Maria, Jose -- so
    two people sharing one are barely evidence at all. "francisco pulgason" and
    "francisco challi" scored 0.65 whole-name similarity and were treated as one
    enslaver, which handed a merge two spurious corroborating signals
    (same-named-relation and discriminative-relation).

    That produced a merge Daniel labelled 0: an adult Juana attached to an 1840
    burial in Guanabacoa, joined to an infant Juana baptised in 1878 in another
    volume, on the strength of two different men both called Francisco.

    So when BOTH names carry a surname, the surnames must be compatible. This is
    exactly symmetric with the estate rule above; only one half of it existed.
    """
    from difflib import SequenceMatcher
    from .textmatch import phonetic_fold
    tx, ty = name_tokens(x), name_tokens(y)
    if not tx or not ty:
        return False
    if set(tx) <= set(ty) or set(ty) <= set(tx):     # containment / short form
        return True
    given = SequenceMatcher(None, tx[0], ty[0]).ratio()
    if not (given >= 0.75 or phonetic_fold(tx[0]) == phonetic_fold(ty[0])):
        return False
    sx, sy = _surname_of(x), _surname_of(y)
    if sx and sy and not _namesets_overlap({sx}, {sy}):
        return False                                 # same given, different family
    return name_similarity(x, y) >= 0.6


def _namesets_overlap(sa: set, sb: set) -> bool:
    for x in sa:
        for y in sb:
            if _third_party_same(x, y):
                return True
    return False


def _entry_year(events) -> Optional[int]:
    """Earliest 4-digit year across an entry's events, or None."""
    years = []
    for ev in events or []:
        d = str((ev or {}).get("date") or "")
        m = re.match(r"\s*(\d{4})", d)
        if m:
            years.append(int(m.group(1)))
    return min(years) if years else None


def _shares_context(a: dict, b: dict, year_window: int) -> bool:
    """Daniel, 2026-07-27 (Q6): filter before scoring.

    FAILS OPEN. This returns False (skip the pair) only on positive evidence
    that the two cannot plausibly be one person: different registers, no person
    named in both entries, and dated further apart than a lifetime. Absent
    metadata is never treated as evidence of separation, because an entry with
    no date would otherwise be excluded from every comparison and its genuine
    merges would vanish silently.

    Requiring the same register would be wrong for the same reason in reverse:
    cross-register links are the most valuable output, so a shared enslaver or a
    compatible date is enough to keep a pair in play.
    """
    if a.get("_register") and a.get("_register") == b.get("_register"):
        return True                       # same register
    ac, bc = a.get("_ctx") or set(), b.get("_ctx") or set()
    if ac and bc and {n for _, n in ac} & {n for _, n in bc}:
        return True                       # a person named in both entries
    ya, yb = a.get("_year"), b.get("_year")
    if ya is None or yb is None:
        return True                       # undated: cannot rule the pair out
    return abs(ya - yb) <= year_window


def _surname_of(name: Optional[str]) -> Optional[str]:
    """Last name-token, or None for a single-token ('Juan', 'Maria') name."""
    toks = name_tokens(name)
    return toks[-1] if len(toks) > 1 else None


# Daniel, 2026-07-29, ruling on the Llopiz cluster: "Llopiz/Llopis is something
# that I'd want to merge assuming context is reasonable, and likely Llepiz as
# well. Llepico less certain unless context is very clear."
#
# That is not a yes/no rule, it is a sliding evidential bar: the further the
# spelling drifts, the more corroboration a merge needs. The existing phonetic
# fold already separates his three cases cleanly (Llopiz and Llopis both fold to
# 'iopis', Llepiz scores 0.80 against it, Llepico 0.55), so the tiers below are
# his sentence with numbers attached rather than a new heuristic.
SURNAME_TIERS = (
    # (min affinity, corroborating signals required, label)
    (1.00, 2, "exact"),          # same surname  -- STILL needs corroboration
    (0.90, 2, "orthographic"),   # Llopiz/Llopis -- reasonable context
    (0.65, 3, "near"),           # Llepiz        -- wants real corroboration
    (0.00, 4, "distant"),        # Llepico       -- only if context is very clear
)

# Daniel, 2026-07-29: "No people should be merged strictly based on name
# correspondence; it should depend on a combination of date overlap, same-named
# relation, same/similar qualities."
#
# That removes the exemption an exact surname used to get, and with it the need
# for the epithet and placeholder special cases -- those were patches around a
# rule that let a name alone carry a merge. They are kept below only because
# reading "N." (nomen nescio) as a surname is simply wrong, not because they are
# still load-bearing.
#
# The operative word in his sentence is "combination". A single weak signal is
# what a score threshold lets through: tuned to 0.30, shared register plus a
# rough date cleared it, and every "Maria de la Concepcion" in a parish has both.
# So corroboration is COUNTED, not scored -- independent signals, each of which
# says something about this pair rather than about the volume:
#
#   1. date overlap
#   2. a same-named relation (a person named in both entries)
#   3. same/similar qualities (attributes agree, none conflict)
#   4. a discriminative relation specifically (enslaver / spouse / parent),
#      which is how these registers identify someone in the first place
#
# Shared register is deliberately NOT a signal. Everyone in a volume shares it.
MIN_SIGNALS_FOR_ANY_MERGE = 2


# Attribute values that cannot both describe one person at one moment. These are
# NOT the same as the softer "conflict" counting below: a scribe may write
# "morena" once and "parda" another time for the same woman, and origins are
# recorded loosely. Being simultaneously an infant and an adult is not scribal
# variation, it is two people.
#
# Found by validating the social graph. The largest hubs in the corpus are
# exactly the identities carrying these contradictions: "María de la Cruz" holds
# 82 mentions and 208 edges while being both free and enslaved, both infant and
# adult. 22 of 1,042 merged identities are in that state, and because they are
# the hubs they distort the network out of all proportion to their number.
#
# This is the one over-merge signal available WITHOUT Daniel's labels, because
# it needs no judgement: no threshold, no name reasoning, just a fact about
# people. It is a hard block rather than a score adjustment for the same reason.
_MUTUALLY_EXCLUSIVE: Tuple[Tuple[str, Tuple[frozenset, frozenset]], ...] = (
    ("free", (frozenset({"true", "libre", "free"}),
              frozenset({"false", "esclavo", "esclava", "slave"}))),
    ("age", (frozenset({"infant", "párvulo", "parvulo", "child"}),
             frozenset({"adult", "elderly"}))),
)


# Below this many years apart, a change of age band or legal status is not a
# life event but a contradiction. Above it, people grow up and are manumitted.
_STATUS_STABLE_YEARS = 3


def attributes_contradict(a: dict, b: dict) -> Optional[str]:
    """Two mentions that cannot be the same person AT THE SAME TIME.

    Returns the contradicting field, or None.

    TIME MATTERS HERE AND THE FIRST VERSION IGNORED IT, which building Daniel's
    synthetic corner cases exposed before he had labelled a single one. Every
    `lifespan_edge` pair came back blocked on "age", including an infant in 1811
    and an adult in 1825 -- who is simply the same person at fourteen.

    An infant becomes an adult; an enslaved person is manumitted. Neither is a
    contradiction across a gap, only within one. So these fields are checked
    only when the two mentions sit within a few years of each other, and the
    time-aware work is done by `lifespan_conflict`, which reasons about implied
    birth years rather than about labels.

    The original rule still catches what it was built for -- the 21 impossible
    identities were single clusters holding "adult, infant" and "free, enslaved"
    simultaneously, and those mentions are years apart in both directions, so
    the birth-window test refuses them on better grounds.
    """
    ya, yb = a.get("_year"), b.get("_year")
    if ya is not None and yb is not None and abs(ya - yb) > _STATUS_STABLE_YEARS:
        return None
    for field, (side_a, side_b) in _MUTUALLY_EXCLUSIVE:
        x, y = _val(a, field), _val(b, field)
        if x is None or y is None:
            continue
        x, y = str(x).strip().lower(), str(y).strip().lower()
        if (x in side_a and y in side_b) or (x in side_b and y in side_a):
            return field
    return None


def corroborating_signals(a: dict, b: dict) -> List[str]:
    """Which independent things support these being one person."""
    out: List[str] = []
    ya, yb = a.get("_year"), b.get("_year")
    # Proximity only counts when the two mentions could belong to one life. An
    # 1840 adult and an 1878 infant are 38 years apart, which used to read as
    # corroboration; it is the opposite.
    if (ya is not None and yb is not None and abs(ya - yb) <= 40
            and not lifespan_conflict(a, b)):
        out.append("date-overlap")
    if _shares_third_party(a, b):
        out.append("same-named-relation")
        ad, bd = _ctx_by_type(a.get("_ctx")), _ctx_by_type(b.get("_ctx"))
        if any(_namesets_overlap(ad.get(k, set()), bd.get(k, set()))
               for k in DISCRIMINATIVE_CTX):
            out.append("discriminative-relation")
    agree = conflict = 0
    for k in ("occupation", "ethnicity", "phenotype", "free", "legitimate",
              "rank", "origin"):
        x, y = _val(a, k), _val(b, k)
        if x is None or y is None:
            continue
        if x == y:
            agree += 1
        else:
            conflict += 1
    if agree and not conflict:
        out.append("matching-qualities")
    return out


_EPITHETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              os.pardir, "name_epithets.json")
_epithets_cache: Optional[Tuple[set, float]] = None


def _load_epithets() -> Tuple[set, float, set]:
    """Name elements that occupy the surname slot without naming a family.
    See name_epithets.json for the reasoning and the term lists."""
    global _epithets_cache
    if _epithets_cache is None:
        terms: set = set()
        holders: set = set()
        need = 0.60
        try:
            with open(_EPITHETS_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            terms = {str(t).lower() for t in (raw.get("devotional") or [])}
            terms |= {str(t).lower() for t in (raw.get("ambiguous") or {})}
            holders = {str(t).lower() for t in (raw.get("placeholders") or [])}
            need = float(raw.get("context_required", 0.60))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass                      # absent or malformed -> previous behaviour
        _epithets_cache = (terms, need, holders)
    return _epithets_cache


def _norm_surname(surname: Optional[str]) -> str:
    if not surname:
        return ""
    from .textmatch import normalize_name
    return normalize_name(surname).strip().lower()


def is_devotional_epithet(surname: Optional[str]) -> bool:
    return bool(surname) and _norm_surname(surname) in _load_epithets()[0]


def is_placeholder_surname(surname: Optional[str]) -> bool:
    """Does this surname slot say "not recorded" rather than name a family?

    `N.` is nomen nescio and a bare initial is the same thing abbreviated, so
    any single character counts. Without this, "Francisco N." matched "Francisco
    N." EXACTLY and two people merged on the strength of both being unnamed:
    one such identity reached 37 mentions across volumes.
    """
    s = _norm_surname(surname)
    if not s:
        return True                   # no surname at all
    return len(s) <= 1 or s in _load_epithets()[2]


def surname_affinity(a_name: Optional[str], b_name: Optional[str]) -> float:
    """How close two surnames are, in [0,1], on the phonetic form.

    Compared after `phonetic_fold` so the scribal alternations that dominate
    these registers (z/s, ll/y, b/v, silent h, doubled letters) cost nothing,
    and only genuine vowel or consonant divergence does.
    """
    sa, sb = _surname_of(a_name), _surname_of(b_name)
    if not sa or not sb:
        return 1.0                        # no surname to disagree about
    if sa == sb:
        return 1.0
    from difflib import SequenceMatcher
    from .textmatch import phonetic_fold
    fa, fb = phonetic_fold(sa), phonetic_fold(sb)
    if fa == fb:
        return 1.0                        # Llopiz / Llopis
    return SequenceMatcher(None, fa, fb, autojunk=False).ratio()


CLERGY_ENTRY_WINDOW = 12          # pages apart, i.e. a few folios


def _entry_seq(entry_id: Optional[str]) -> Optional[Tuple[int, int]]:
    """(page, index) for "consecutive records".

    Takes the LAST TWO numbers in the id rather than matching a fixed shape:
    entry ids come in at least `201991-0279-A-02` and `0013-00` forms, and a
    pattern written for the first silently returns None for the second, which
    disabled the clergy rule on exactly the recurring-priest fixtures it exists
    to handle.
    """
    nums = re.findall(r"\d+", str(entry_id or ""))
    return (int(nums[-2]), int(nums[-1])) if len(nums) >= 2 else None


def _is_clergy(m: dict) -> bool:
    if str(_val(m, "occupation") or "").startswith("cleric"):
        return True
    titles = " ".join(str(t).lower() for t in (m.get("titles") or []))
    return any(k in titles for k in ("cura", "presb", "vicar", "padre", "pbro",
                                     "coadjutor", "capellan", "vig"))


def _is_recurring_clergy(a: dict, b: dict) -> bool:
    """The officiant of consecutive entries in one register.

    Daniel's sanctioned exception to "never merge on the name alone", and it is
    narrow on purpose: same register, clergy on both sides, near-identical name,
    and entries close together in the book. A priest signing folio after folio is
    the one pattern where the sequence itself is the corroboration.
    """
    if not (_is_clergy(a) and _is_clergy(b)):
        return False
    # Deliberately NOT keyed on `_register`. That field is the entry-id prefix,
    # which is the volume for `201991-0279-A-02` but the page for `0001-01`, so
    # requiring it to match disabled this rule on the very fixtures of a
    # recurring priest. The sequence window below carries the "consecutive
    # records" requirement, and `_shares_context` has already screened the pair.
    if name_similarity(a.get("name"), b.get("name")) < 0.92:
        return False
    sa, sb = _entry_seq(a.get("_entry")), _entry_seq(b.get("_entry"))
    if not sa or not sb:
        return False
    return abs(sa[0] - sb[0]) <= CLERGY_ENTRY_WINDOW


def _shares_third_party(a: dict, b: dict) -> bool:
    """Is some person named in BOTH entries? Matched by `_third_party_same`, so
    short forms and spelling drift count but a merely shared estate surname does
    not."""
    ac = {n for _, n in (a.get("_ctx") or set())}
    bc = {n for _, n in (b.get("_ctx") or set())}
    return any(_third_party_same(x, y) for x in ac for y in bc)


def context_strength(a: dict, b: dict) -> float:
    """How much non-name evidence supports these being one person, in [0,1].

    Deliberately coarse. It exists to distinguish "reasonable", "real" and
    "very clear" corroboration, which is the granularity Daniel's ruling asks
    for, not to be a second similarity score.
    """
    s = 0.0
    if a.get("_register") and a.get("_register") == b.get("_register"):
        # A weak prior, not corroboration. Everyone in a volume shares this, so
        # at its old weight of 0.35 it cleared bars by itself and every bare
        # "Maria" in a parish looked mutually supported.
        s += 0.15
    ac = {n for _, n in (a.get("_ctx") or set())}
    bc = {n for _, n in (b.get("_ctx") or set())}
    shared = sum(1 for x in ac for y in bc if _third_party_same(x, y))
    if shared:
        s += min(0.55, 0.30 * shared)     # a person named in both entries
    ya, yb = a.get("_year"), b.get("_year")
    if ya is not None and yb is not None:
        gap = abs(ya - yb)
        s += 0.20 if gap <= 20 else (0.10 if gap <= 40 else 0.0)
    agree = conflict = 0
    for k in ("occupation", "ethnicity", "phenotype", "free", "legitimate"):
        x, y = _val(a, k), _val(b, k)
        if x is None or y is None:
            continue
        if x == y:
            agree += 1
        else:
            conflict += 1
    s += min(0.20, 0.08 * agree) - 0.30 * conflict
    return max(0.0, min(1.0, s))


def surname_tier_allows(a: dict, b: dict) -> Tuple[bool, str]:
    """Apply the tiered bar. Returns (allowed, tier label).

    The bar applies ONLY to surnames that actually differ. Daniel's ruling is
    about how far a spelling may drift before a merge needs corroboration; an
    exact surname match has not drifted, and demanding extra evidence for it
    would block the ordinary same-name merges this stage exists to make.
    """
    # Checked before everything else, including the clergy shortcut, because it
    # is a fact about people rather than a judgement about names. Nothing later
    # in this function can rescue a pair where one mention is an infant and the
    # other an adult.
    field = attributes_contradict(a, b)
    if field:
        return False, f"blocked-contradiction-{field}"

    # Chronology, for the same reason: it is a fact about people rather than a
    # judgement about names, so nothing later can rescue it.
    if lifespan_conflict(a, b):
        return False, "blocked-lifespan"

    if _is_recurring_clergy(a, b):
        # Daniel's one sanctioned rules-based shortcut: "obvious merges like the
        # clergy that appear in many consecutive records". A priest signing
        # consecutive entries of his own register is the one case where the name
        # really does carry it, because the role and the sequence corroborate.
        return True, "clergy-consecutive"
    sa, sb = _surname_of(a.get("name")), _surname_of(b.get("name"))
    if is_placeholder_surname(sa) or is_placeholder_surname(sb):
        # No surname, or one that says "not recorded". This used to be exempt on
        # the reasoning that there was nothing to disagree about -- but absence
        # of contradiction is not evidence of identity, and treating it as a free
        # pass merged 41 women recorded only as "Maria" into one person across
        # two registers.
        #
        # The requirement is a PERSON NAMED IN BOTH ENTRIES, not a general
        # context score. That distinction is the whole fix. Sharing a register
        # and a rough date is what every bare "Maria" in a parish has in common,
        # so scoring those was what let them chain together; a shared enslaver,
        # spouse or parent is how these registers actually identify someone with
        # no surname, and it still links them.
        return _shares_third_party(a, b), "uninformative"
    aff = surname_affinity(a.get("name"), b.get("name"))
    n = len(corroborating_signals(a, b))
    for min_aff, need, label in SURNAME_TIERS:
        if aff >= min_aff:
            return n >= need, label
    return False, "distant"


# Nobody lives this long, so two mentions further apart than this are different
# people whatever else agrees. Deliberately generous: the point is to exclude the
# impossible, not to model mortality.
MAX_LIFESPAN_YEARS = 100
# Youngest plausible parent. Used only to bound a birth year from ABOVE, so a
# generous value stays safe.
MIN_PARENT_AGE = 14
# "infant" in these registers means a baptism in the first months of life; the
# slack absorbs a late baptism and a transcription slip in the year.
_INFANT_SLACK = 3
_CHILD_MAX_AGE = 14
_ADULT_MIN_AGE = 15


def birth_window(mention: dict) -> Optional[Tuple[int, int]]:
    """Earliest and latest year this person could have been born, or None.

    Derived only from things the registers state plainly:

      age "infant" in year Y   -> born about Y          (Y - 3 .. Y + 1)
      age "child"  in year Y   -> born Y - 14 .. Y
      age "adult"  in year Y   -> born at most Y - 15
      is a PARENT in year Y    -> born at most Y - 14

    Being someone's parent is the strongest of these and needs no age field at
    all, which matters because age is present on well under half of mentions.
    """
    year = mention.get("_year")
    if year is None:
        return None
    lo, hi = None, None

    age = _val(mention, "age")
    if age == "infant":
        lo, hi = year - _INFANT_SLACK, year + 1
    elif age == "child":
        lo, hi = year - _CHILD_MAX_AGE, year
    elif age in ("adult", "elderly"):
        hi = year - _ADULT_MIN_AGE

    # A parent must predate the child. `_ctx` holds (role, other-name) pairs from
    # THIS person's point of view, so a "child" edge means they are the parent.
    if any(role == "child" for role, _ in (mention.get("_ctx") or ())):
        cap = year - MIN_PARENT_AGE
        hi = cap if hi is None else min(hi, cap)

    if lo is None and hi is None:
        return None
    return (lo if lo is not None else hi - MAX_LIFESPAN_YEARS,
            hi if hi is not None else lo + MAX_LIFESPAN_YEARS)


def lifespan_conflict(a: dict, b: dict) -> Optional[str]:
    """Why these two mentions cannot be one person, on chronology alone.

    Daniel, 2026-08-03: "I'm seeing some nonsensical pairings like children born
    after a same-name adult died... someone can't die twice - there likely should
    be some larger contextual knowledge built into the algorithm somehow."

    He was right, and the cause was subtler than dates being ignored. Dates ARE
    used, but only as PROXIMITY: `date-overlap` fires whenever two mentions sit
    within 40 years, so an adult attached to an 1840 burial and an infant
    baptised in 1878 counted as corroborating rather than contradictory. Nearness
    in time is not compatibility.

    This derives what each mention implies about a birth year and refuses the
    pair when those implications cannot both hold.
    """
    ya, yb = a.get("_year"), b.get("_year")
    if ya is not None and yb is not None and abs(ya - yb) > MAX_LIFESPAN_YEARS:
        return f"events {abs(ya - yb)} years apart"

    wa, wb = birth_window(a), birth_window(b)
    if wa and wb and (wa[1] < wb[0] or wb[1] < wa[0]):
        return (f"birth year cannot be both {wa[0]}-{wa[1]} and {wb[0]}-{wb[1]}")
    return None


def _exclusive_sides(mention: dict) -> Dict[str, int]:
    """Which side of each mutually-exclusive opposition this mention sits on."""
    out: Dict[str, int] = {}
    for field, (side_a, side_b) in _MUTUALLY_EXCLUSIVE:
        v = _val(mention, field)
        if v is None:
            continue
        v = str(v).strip().lower()
        if v in side_a:
            out[field] = 0
        elif v in side_b:
            out[field] = 1
    return out


# How far to look for an ancestry path before giving up. Descent chains in a
# sacramental register are shallow -- parent, grandparent, occasionally a third
# generation -- so 4 covers the real ones, and an unbounded search over a graph
# this size would dominate the merge loop.
_ANCESTRY_DEPTH = 4


def _would_close_ancestry_cycle(uf, i: int, j: int,
                                cluster_parents: Dict[int, set],
                                depth: int = _ANCESTRY_DEPTH) -> bool:
    """Would merging these two clusters make somebody their own ancestor?

    If a descent path already runs from one cluster to the other, merging the
    endpoints closes that path into a loop. Nobody is their own grandmother, so
    a merge that produces one is wrong however well the names match.

    This is what the attribute guard structurally cannot see. Ramona Bernal is
    recorded as the PARENT of Rosalía Bernal on folio 0017 and as her CHILD on
    folio 0195. Both women are parda, both from Trinidad, two mentions each,
    and no attribute conflicts anywhere -- the contradiction lives entirely in
    the relationship direction. 178 folios apart, the likely reading is two
    different mother/daughter pairs sharing a surname, collapsed on name
    similarity, which is exactly the failure Daniel's ruling is aimed at.

    Searches both directions because either one closes a loop.
    """
    ri, rj = uf.find(i), uf.find(j)
    if ri == rj:
        return False
    for start, target in ((ri, rj), (rj, ri)):
        seen = {start}
        frontier = [start]
        for _ in range(depth):
            nxt = []
            for node in frontier:
                for child in cluster_parents.get(node, ()):
                    c = uf.find(child)
                    if c == target:
                        return True
                    if c not in seen:
                        seen.add(c)
                        nxt.append(c)
            if not nxt:
                break
            frontier = nxt
    return False


def _clusters_share_an_entry(uf, i: int, j: int,
                             cluster_entries: Dict[int, set]) -> bool:
    """Would this merge put two people from ONE record into one identity?

    The module has always refused to merge two mentions from the same entry --
    "the extractor already separated them" -- but only PAIRWISE, and union-find
    routes straight around that: A from entry E merges with X from entry F, then
    B from entry E merges with X too, and A and B end up in one cluster without
    ever being compared. 35 delivered identities were in that state.

    It is the direct cause of the role contradictions left in the graph. Merging
    two different people who appear in one record collapses their separate
    relationships to a third person, so "Jose Antonio" comes out as both the
    parent and the godparent of Jose Maria Almada in entry 201991-0141-B-01 --
    a contradiction that does NOT exist in the extraction and is manufactured
    entirely by the merge.

    This is the THIRD guard in this file defeated by transitivity, after the
    surname chain and the attribute contradiction. A pairwise rule is not a
    rule about clusters, and every rule here is really about clusters.
    """
    return bool(cluster_entries.get(uf.find(i), set())
                & cluster_entries.get(uf.find(j), set()))


def _clusters_attributes_compatible(uf, i: int, j: int,
                                    cluster_sides: Dict[int, Dict[str, set]]) -> bool:
    """May these clusters be joined without making an impossible person?

    Cluster-to-cluster for the same reason as the surname guard, and this is
    where it actually bites. A pairwise check blocked only 15 merges in the
    whole corpus, because the contradiction is TRANSITIVE: an infant merges with
    a plausible adult, that adult merges with another adult, and nobody ever
    compares the infant to the last one. The cluster carrying "María de la Cruz"
    reached 82 mentions and 208 edges that way while being both free and
    enslaved, both infant and adult.
    """
    sa = cluster_sides.get(uf.find(i), {})
    sb = cluster_sides.get(uf.find(j), {})
    for field in set(sa) & set(sb):
        if sa[field] | sb[field] == {0, 1}:
            return False
    return True


def _clusters_surname_compatible(uf, i: int, j: int,
                                 cluster_surnames: Dict[int, set]) -> bool:
    """May the clusters containing i and j be joined?

    Yes when either side carries no surname at all (single-token names are
    routinely merged on context, and blocking them would undo the existing
    behaviour), or when some surname on one side is compatible with some surname
    on the other. No when both sides have surnames and none of them match.

    Compared cluster-to-cluster rather than pair-to-pair, because the failure is
    transitive: Llopiz~Llopis and Llopis~Lopez are each defensible, but the
    resulting cluster holds Llopiz and Valdes.
    """
    sa = cluster_surnames.get(uf.find(i), set())
    sb = cluster_surnames.get(uf.find(j), set())
    if not sa or not sb:
        return True
    return _namesets_overlap(sa, sb)


def pair_score(a: dict, b: dict, a_rel_ctx=None, b_rel_ctx=None) -> Tuple[float, List[str]]:
    """Return (score in [0,1], reasons). Higher = more likely the same person."""
    reasons = []
    nsim = name_similarity(a.get("name"), b.get("name"))
    if nsim <= 0.0:
        return 0.0, ["different names"]

    score = nsim
    reasons.append(f"name~{nsim:.2f}")

    # Attribute agreement weighted by informativeness: phenotype/free agree for
    # nearly everyone in an enslaved-population register, so they barely count;
    # occupation/rank are rarer and mean more.
    AGREE_W = {"phenotype": 0.02, "free": 0.02, "legitimate": 0.02,
               "origin": 0.04, "ethnicity": 0.04, "occupation": 0.08, "rank": 0.08}
    agree = conflict = 0
    for k in HARD_ATTRS:
        va, vb = _val(a, k), _val(b, k)
        if va is not None and vb is not None:
            if va == vb:
                agree += 1
                score += AGREE_W.get(k, 0.03)
            else:
                conflict += 1
                reasons.append(f"conflict:{k}({va}!={vb})")
    score -= 0.25 * conflict

    # shared relationship context: same-typed edge to a same-named third party
    ctx_overlap = False
    if a_rel_ctx and b_rel_ctx:
        shared = a_rel_ctx & b_rel_ctx
        if shared:
            ctx_overlap = True
            score += min(0.15, 0.05 * len(shared))
            reasons.append(f"shared_rel({len(shared)})")
        # typed comparison: same rel type on both sides but pointing at
        # non-overlapping third parties = near-definitive evidence of DIFFERENT
        # people in this domain (your enslaver / spouse / parents don't vary by
        # entry), so it must outweigh any pile of weak attribute agreements.
        ca, cb = _ctx_by_type(a_rel_ctx), _ctx_by_type(b_rel_ctx)
        for t in set(ca) & set(cb):
            if _namesets_overlap(ca[t], cb[t]):
                ctx_overlap = True
            else:
                score -= 0.35
                reasons.append(f"ctx-conflict:{t}({'/'.join(sorted(ca[t])[:1])}"
                               f"!={'/'.join(sorted(cb[t])[:1])})")

    # single bare-name guard: "Juan" ~ "Juan" carries little identity on its own,
    # and shared phenotype/free/origin is population-universal here, not personal.
    # Require CONTEXT corroboration (same spouse/enslaver/parents) to auto-merge;
    # otherwise cap below auto threshold so the pair lands in review. This also
    # stops context-empty mentions from acting as transitive union-find bridges.
    single = len(name_tokens(a.get("name"))) == 1 and len(name_tokens(b.get("name"))) == 1
    if single and not ctx_overlap:
        if score > 0.82:
            score = 0.82
            reasons.append("capped: single-token name, no context corroboration")

    return max(0.0, min(1.0, score)), reasons


def _mentions_from_volume(volume: dict) -> List[dict]:
    """Flatten entries -> mentions, attaching a relationship context set per mention."""
    entries = volume.get("entries") or volume.get("examples") or []
    mentions = []
    for e in entries:
        # examples-format rows carry the per-entry id in "entry" ("0013-01") and the
        # *volume* id in "id" (239746); volume-record rows carry the entry id in "id".
        # Prefer "entry" so we don't collapse a whole volume into one pseudo-entry.
        eid = str(e.get("entry") or e.get("id") or "")
        data = e.get("data") or {}
        people = data.get("people", []) or []
        events = data.get("events", []) or []
        id_to_name = {str(p.get("id")): normalize_name(p.get("name")) for p in people}
        # person ids that are principals of a once-in-a-lifetime sacrament: you are
        # baptized/born/buried exactly once, so two such mentions in DIFFERENT
        # entries cannot be the same person (a strong precision constraint).
        unique_sacrament_pids = set()
        for ev in events:
            if str(ev.get("type", "")).lower() in ("baptism", "birth", "burial"):
                for pid in ev.get("principals", []) or []:
                    unique_sacrament_pids.add(str(pid))
        for p in people:
            # context = set of (rel_type, related-person-normalized-name)
            ctx = set()
            for r in p.get("relationships", []) or []:
                if isinstance(r, dict):
                    rn = id_to_name.get(str(r.get("related_person")))
                    rt = r.get("relationship_type")
                    if rn and rt:
                        ctx.add((str(rt).lower(), rn))
            # Local ids of this person's children/grandchildren within the
            # entry, in BOTH directions of expression: registers write "parent
            # of X" and "child of Y" interchangeably, so a descent edge has to
            # be read off whichever way the scribe put it.
            descendants = set()
            for r in p.get("relationships", []) or []:
                if not isinstance(r, dict):
                    continue
                rt = str(r.get("relationship_type") or "").lower()
                other = str(r.get("related_person") or "")
                if not other or other == "None":
                    continue
                if rt in ("child", "grandchild"):
                    descendants.add(other)
            m = dict(p)
            m["_entry"] = eid
            m["_local_id"] = str(p.get("id"))
            m["_descendants"] = descendants
            m["_ctx"] = ctx
            m["_unique_sacrament"] = str(p.get("id")) in unique_sacrament_pids
            # blocking signals (see _shares_context): the register this entry
            # belongs to, and the year of its earliest dated event.
            m["_register"] = str(eid).split("-")[0] if eid else ""
            m["_year"] = _entry_year(events)
            mentions.append(m)
    return mentions


def _merge_attributes(members: List[dict]) -> Dict[str, Any]:
    """Merge attribute values across a cluster, recording conflicts as lists."""
    merged: Dict[str, Any] = {}
    conflicts: Dict[str, list] = {}
    for k in HARD_ATTRS + ["age"]:
        vals = []
        for m in members:
            v = m.get(k)
            if v is not None and not (isinstance(v, str) and not v.strip()):
                vals.append(v)
        uniq = []
        for v in vals:
            if v not in uniq:
                uniq.append(v)
        if len(uniq) == 1:
            merged[k] = uniq[0]
        elif len(uniq) > 1:
            merged[k] = uniq          # keep all; downstream/human resolves
            conflicts[k] = uniq
    # titles = union
    titles = []
    for m in members:
        for t in (m.get("titles") or []):
            if t not in titles:
                titles.append(t)
    if titles:
        merged["titles"] = titles
    return merged, conflicts


def _snapshot(m: dict) -> dict:
    """Compact evidence card for a mention, for human review."""
    snap = {k: m.get(k) for k in ("occupation", "phenotype", "free", "origin",
                                  "ethnicity", "age", "legitimate")
            if m.get(k) is not None}
    if m.get("titles"):
        snap["titles"] = m["titles"]
    ctx = sorted(m.get("_ctx") or ())
    if ctx:
        snap["context"] = [f"{t}: {n}" for t, n in ctx]
    return snap


def disambiguate_volume(
    volume: Any,
    auto_threshold: float = 0.86,
    review_threshold: float = 0.70,
    volume_tag: Optional[str] = None,
    constraints: Optional[dict] = None,
    block_context: bool = True,
    year_window: int = 60,
    pair_log: Optional[List[dict]] = None,
    pair_log_floor: float = 0.0,
    surname_tiers: bool = True,
    collect_review: bool = True,
) -> Dict[str, Any]:
    """Cluster person mentions into identities.

    Returns {identities, review_queue, stats}. Pairs at or above auto_threshold
    are merged; pairs in [review_threshold, auto_threshold) become review items.

    `constraints` carries human review decisions back in:
        {"must":   [[{"entry","id"}, {"entry","id"}], ...],   # same person
         "cannot": [[{"entry","id"}, {"entry","id"}], ...]}   # different people
    Must-links are unioned outright; cannot-links are excluded from auto-merge
    and any cluster that still joins them transitively is flagged for review.

    `pair_log`, when given a list, receives one row per scored pair at or above
    `pair_log_floor` with the disposition the algorithm chose (auto / review /
    blocked-* / below-threshold). The review queue alone spans only
    [review_threshold, auto_threshold), so it cannot show what confident merges
    or confident non-merges look like — which is exactly what a training sample
    for the merge model needs. Logging is off by default and changes nothing.
    """
    if isinstance(volume, str):
        with open(volume, "r", encoding="utf-8") as f:
            volume = json.load(f)
    tag = volume_tag or str(volume.get("id", "V"))

    mentions = _mentions_from_volume(volume)
    n = len(mentions)
    uf = _UnionFind(n)

    key_to_idx = {(m["_entry"], m["_local_id"]): i for i, m in enumerate(mentions)}

    def _pair_idx(pair):
        a = key_to_idx.get((str(pair[0]["entry"]), str(pair[0]["id"])))
        b = key_to_idx.get((str(pair[1]["entry"]), str(pair[1]["id"])))
        return (a, b) if a is not None and b is not None else None

    must_pairs = [p for p in ((constraints or {}).get("must") or []) if _pair_idx(p)]
    cannot_set = set()
    for p in (constraints or {}).get("cannot") or []:
        idx = _pair_idx(p)
        if idx:
            cannot_set.add(frozenset(idx))

    # block by PHONETIC key of the first name-token to avoid O(n^2) while still
    # grouping scribal variants (Gonzalez/Gonzales) into the same candidate block
    blocks: Dict[str, List[int]] = defaultdict(list)
    for i, m in enumerate(mentions):
        blocks[phonetic_key(m.get("name"))].append(i)

    review_queue = []
    blocked = 0
    chain_blocked = 0
    tier_blocked: Counter = Counter()
    review_dropped = 0

    def _enqueue(item: dict) -> None:
        """Review items are counted always, but only MATERIALISED when someone
        will read them. The corpus produces ~1.1M of these with evidence cards
        attached; building that list to throw it away is several GB for nothing,
        which is what the training-sample pass was doing before it ran out of
        memory."""
        nonlocal review_dropped
        if collect_review:
            review_queue.append(item)
        else:
            review_dropped += 1

    # surnames seen in each union-find cluster, for the transitive-chain guard
    cluster_surnames: Dict[int, set] = {}
    # Per-cluster record of which side of each mutually-exclusive opposition the
    # cluster's members occupy. Maintained exactly like cluster_surnames.
    cluster_sides: Dict[int, Dict[str, set]] = {}
    # Cluster-level descent edges: root -> set of roots that are its children or
    # grandchildren. Built from the mention-level `_descendants` and rewired on
    # every union, exactly like cluster_surnames.
    cluster_parents: Dict[int, set] = defaultdict(set)
    # Every entry a cluster already draws a mention from. Two people from one
    # record are two people, however alike they look.
    cluster_entries: Dict[int, set] = {}
    for _i, _m in enumerate(mentions):
        _s = _surname_of(_m.get("name"))
        cluster_surnames[_i] = {_s} if _s else set()
        cluster_sides[_i] = {f: {v} for f, v in _exclusive_sides(_m).items()}
        cluster_entries[_i] = {_m["_entry"]}
    # index (entry, local id) -> mention index so descent edges resolve
    _by_local = {(m["_entry"], m["_local_id"]): k for k, m in enumerate(mentions)}
    for _i, _m in enumerate(mentions):
        for _d in _m.get("_descendants") or ():
            _k = _by_local.get((_m["_entry"], str(_d)))
            if _k is not None and _k != _i:
                cluster_parents[_i].add(_k)
    auto_edges: List[Tuple[int, int, float]] = []

    def _log(disposition: str, i: int, j: int, s: float, reasons: List[str]) -> None:
        if pair_log is None or s < pair_log_floor:
            return
        pair_log.append({
            "score": round(s, 3),
            "disposition": disposition,
            "reasons": list(reasons),
            "a": {"entry": mentions[i]["_entry"], "id": mentions[i]["_local_id"],
                  "name": mentions[i].get("name"), "detail": _snapshot(mentions[i])},
            "b": {"entry": mentions[j]["_entry"], "id": mentions[j]["_local_id"],
                  "name": mentions[j].get("name"), "detail": _snapshot(mentions[j])},
        })

    for _, idxs in blocks.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if mentions[i]["_entry"] == mentions[j]["_entry"]:
                    continue  # never merge two people from the same entry
                if block_context and not _shares_context(mentions[i], mentions[j],
                                                          year_window):
                    blocked += 1
                    continue  # no register, related person, or date in common
                s, reasons = pair_score(mentions[i], mentions[j],
                                        mentions[i]["_ctx"], mentions[j]["_ctx"])
                # once-in-a-lifetime sacrament guard: two baptism/birth/burial
                # principals from different entries are different people (you are
                # baptized once). Block auto-merge; keep very-similar pairs visible
                # in the review queue (could be a double-recorded entry).
                if (mentions[i].get("_unique_sacrament") and mentions[j].get("_unique_sacrament")
                        and s >= review_threshold):
                    _enqueue({
                        "score": round(min(s, auto_threshold - 0.01), 3),
                        "reasons": reasons + ["blocked: both sacrament principals"],
                        "a": {"entry": mentions[i]["_entry"], "id": mentions[i]["_local_id"],
                              "name": mentions[i].get("name"), "detail": _snapshot(mentions[i])},
                        "b": {"entry": mentions[j]["_entry"], "id": mentions[j]["_local_id"],
                              "name": mentions[j].get("name"), "detail": _snapshot(mentions[j])},
                    })
                    _log("blocked-sacrament-principal", i, j, s, reasons)
                    continue
                if s >= auto_threshold:
                    if frozenset((i, j)) in cannot_set:
                        _log("human-cannot", i, j, s, reasons)
                        continue          # human already ruled: different people
                    # Transitive-chain guard. Each link in Llopiz~Llopis~Lopez is
                    # individually strong, but union-find joins the endpoints, and
                    # one weak link collapses hundreds of people into a node with
                    # 209 surnames. Compare the SURNAMES ALREADY IN BOTH CLUSTERS,
                    # not just this pair: a merge is refused when both sides carry
                    # surnames and none of them are compatible.
                    if _clusters_share_an_entry(uf, i, j, cluster_entries):
                        _enqueue({
                            "score": round(min(s, auto_threshold - 0.01), 3),
                            "reasons": reasons + ["blocked: clusters already share "
                                                  "an entry (two people in one record)"],
                            "a": {"entry": mentions[i]["_entry"], "id": mentions[i]["_local_id"],
                                  "name": mentions[i].get("name"), "detail": _snapshot(mentions[i])},
                            "b": {"entry": mentions[j]["_entry"], "id": mentions[j]["_local_id"],
                                  "name": mentions[j].get("name"), "detail": _snapshot(mentions[j])},
                        })
                        _log("blocked-cluster-same-entry", i, j, s, reasons)
                        continue
                    if not _clusters_surname_compatible(uf, i, j, cluster_surnames):
                        _enqueue({
                            "score": round(min(s, auto_threshold - 0.01), 3),
                            "reasons": reasons + ["blocked: incompatible cluster surnames"],
                            "a": {"entry": mentions[i]["_entry"], "id": mentions[i]["_local_id"],
                                  "name": mentions[i].get("name"), "detail": _snapshot(mentions[i])},
                            "b": {"entry": mentions[j]["_entry"], "id": mentions[j]["_local_id"],
                                  "name": mentions[j].get("name"), "detail": _snapshot(mentions[j])},
                        })
                        _log("blocked-cluster-surname", i, j, s, reasons)
                        chain_blocked += 1
                        continue
                    # No person is both an infant and an adult, or both free and
                    # enslaved. Checked cluster-to-cluster because the
                    # contradiction is transitive; see
                    # _clusters_attributes_compatible.
                    if not _clusters_attributes_compatible(uf, i, j, cluster_sides):
                        _enqueue({
                            "score": round(min(s, auto_threshold - 0.01), 3),
                            "reasons": reasons + ["blocked: cluster holds mutually "
                                                  "exclusive attributes"],
                            "a": {"entry": mentions[i]["_entry"], "id": mentions[i]["_local_id"],
                                  "name": mentions[i].get("name"), "detail": _snapshot(mentions[i])},
                            "b": {"entry": mentions[j]["_entry"], "id": mentions[j]["_local_id"],
                                  "name": mentions[j].get("name"), "detail": _snapshot(mentions[j])},
                        })
                        _log("blocked-cluster-contradiction", i, j, s, reasons)
                        continue
                    # Nobody is their own grandmother. If a descent path already
                    # runs between these clusters, merging closes it into a loop.
                    if _would_close_ancestry_cycle(uf, i, j, cluster_parents):
                        _enqueue({
                            "score": round(min(s, auto_threshold - 0.01), 3),
                            "reasons": reasons + ["blocked: merge would make a "
                                                  "person their own ancestor"],
                            "a": {"entry": mentions[i]["_entry"], "id": mentions[i]["_local_id"],
                                  "name": mentions[i].get("name"), "detail": _snapshot(mentions[i])},
                            "b": {"entry": mentions[j]["_entry"], "id": mentions[j]["_local_id"],
                                  "name": mentions[j].get("name"), "detail": _snapshot(mentions[j])},
                        })
                        _log("blocked-ancestry-cycle", i, j, s, reasons)
                        continue
                    # tiered spelling bar: the further the surname has drifted,
                    # the more corroboration the merge needs
                    if surname_tiers:
                        allowed, tier = surname_tier_allows(mentions[i], mentions[j])
                        if not allowed:
                            _enqueue({
                                "score": round(min(s, auto_threshold - 0.01), 3),
                                "reasons": reasons + [f"blocked: {tier} surname variant "
                                                      f"without enough context"],
                                "a": {"entry": mentions[i]["_entry"], "id": mentions[i]["_local_id"],
                                      "name": mentions[i].get("name"), "detail": _snapshot(mentions[i])},
                                "b": {"entry": mentions[j]["_entry"], "id": mentions[j]["_local_id"],
                                      "name": mentions[j].get("name"), "detail": _snapshot(mentions[j])},
                            })
                            _log(f"blocked-surname-tier-{tier}", i, j, s, reasons)
                            tier_blocked[tier] += 1
                            continue
                    _log("auto", i, j, s, reasons)
                    ra, rb = uf.find(i), uf.find(j)
                    uf.union(i, j)
                    root = uf.find(i)
                    merged = cluster_surnames.get(ra, set()) | cluster_surnames.get(rb, set())
                    cluster_surnames[root] = merged
                    sides: Dict[str, set] = {}
                    for src in (cluster_sides.get(ra, {}), cluster_sides.get(rb, {})):
                        for f, v in src.items():
                            sides.setdefault(f, set()).update(v)
                    cluster_sides[root] = sides
                    kids = cluster_parents.pop(ra, set()) | cluster_parents.pop(rb, set())
                    cluster_parents[root] = kids
                    cluster_entries[root] = (cluster_entries.pop(ra, set())
                                             | cluster_entries.pop(rb, set()))
                    auto_edges.append((i, j, s))
                elif s >= review_threshold:
                    _enqueue({
                        "score": round(s, 3),
                        "reasons": reasons,
                        "a": {"entry": mentions[i]["_entry"], "id": mentions[i]["_local_id"],
                              "name": mentions[i].get("name"), "detail": _snapshot(mentions[i])},
                        "b": {"entry": mentions[j]["_entry"], "id": mentions[j]["_local_id"],
                              "name": mentions[j].get("name"), "detail": _snapshot(mentions[j])},
                    })
                    _log("review", i, j, s, reasons)
                else:
                    _log("below-threshold", i, j, s, reasons)

    # human decisions: must-links union outright and settle their review items
    for p in must_pairs:
        a, b = _pair_idx(p)
        uf.union(a, b)
        auto_edges.append((a, b, 1.0))
    decided = {frozenset(_pair_idx(p)) for p in must_pairs} | set(cannot_set)
    if decided:
        review_queue = [r for r in review_queue
                        if frozenset((key_to_idx.get((r["a"]["entry"], r["a"]["id"])),
                                      key_to_idx.get((r["b"]["entry"], r["b"]["id"])))) not in decided]

    # gather clusters
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        clusters[uf.find(i)].append(i)

    # a cannot-link that still ended up in one cluster (via a transitive chain)
    # is a conflict a human must untangle — flag the cluster, don't hide it
    violated_roots = set()
    for pair in cannot_set:
        a, b = tuple(pair)
        if uf.find(a) == uf.find(b):
            violated_roots.add(uf.find(a))

    # cluster cohesion: weakest internal auto edge (for flagging chained merges)
    cohesion: Dict[int, float] = {}
    for i, j, s in auto_edges:
        root = uf.find(i)
        cohesion[root] = min(cohesion.get(root, 1.0), s)

    identities = []
    flagged = 0
    for k, (root, idxs) in enumerate(sorted(clusters.items()), 1):
        members = [mentions[i] for i in idxs]
        merged_attrs, conflicts = _merge_attributes(members)
        # Canonical name = the form the scribes actually used most, with the
        # longest spelling breaking ties.
        #
        # "Longest wins" alone picks the least representative variant: a cluster
        # of 31 "Maria" and one stray "Maria Maria" was labelled "Maria Maria",
        # and 36 "Francisco" plus one "Francisco N." became "Francisco N." --
        # promoting a transcription artefact over the actual name in both cases.
        _names = Counter(m.get("name") or "" for m in members)
        canonical = max(_names, key=lambda n: (_names[n], len(n)))
        coh = cohesion.get(root, 1.0) if len(idxs) > 1 else 1.0
        needs_review = (len(idxs) > 1 and coh < auto_threshold) or root in violated_roots
        if needs_review:
            flagged += 1
        identities.append({
            "person_id": f"{tag}-{k:04d}",
            "canonical_name": canonical,
            "n_mentions": len(idxs),
            "mentions": [{"entry": m["_entry"], "id": m["_local_id"], "name": m.get("name")}
                         for m in members],
            "attributes": merged_attrs,
            "attribute_conflicts": conflicts,
            "cluster_cohesion": round(coh, 3),
            "needs_review": needs_review,
        })

    multi = [i for i in identities if i["n_mentions"] > 1]
    return {
        "identities": identities,
        "review_queue": sorted(review_queue, key=lambda x: -x["score"]),
        "stats": {
            "mentions": n,
            "identities": len(identities),
            "merged_identities": len(multi),
            "auto_merges": len(auto_edges),
            "review_pairs": len(review_queue) + review_dropped,
            "pairs_blocked_by_context": blocked,
            "merges_blocked_by_surname": chain_blocked,
            "merges_blocked_by_surname_tier": dict(tier_blocked),
            "flagged_clusters": flagged,
            "reduction": round(1 - len(identities) / n, 4) if n else 0.0,
        },
    }


def format_disambiguation(result: Dict[str, Any], top: int = 12) -> str:
    s = result["stats"]
    lines = ["=" * 60, "Person disambiguation", "=" * 60,
             f"mentions:            {s['mentions']}",
             f"distinct identities: {s['identities']}  "
             f"(mention->identity reduction {s['reduction']*100:.1f}%)",
             f"merged identities:   {s['merged_identities']}  "
             f"(from {s['auto_merges']} auto-merge links)",
             f"review queue:        {s['review_pairs']} borderline pairs",
             f"flagged clusters:    {s['flagged_clusters']} (weak internal link)",
             ""]
    multi = [i for i in result["identities"] if i["n_mentions"] > 1]
    if multi:
        lines.append("merged identities (top by mention count):")
        for idn in sorted(multi, key=lambda x: -x["n_mentions"])[:top]:
            flag = "  ⚑REVIEW" if idn["needs_review"] else ""
            where = ", ".join(f"{m['entry']}:{m['id']}" for m in idn["mentions"])
            lines.append(f"  {idn['person_id']}  {idn['canonical_name']!r}  "
                         f"x{idn['n_mentions']} [{where}]{flag}")
            if idn["attribute_conflicts"]:
                lines.append(f"        conflicts: {idn['attribute_conflicts']}")
    if result["review_queue"]:
        lines.append("")
        lines.append(f"review queue (top {min(top, len(result['review_queue']))} by score):")
        for r in result["review_queue"][:top]:
            lines.append(f"  {r['score']:.2f}  {r['a']['name']!r}({r['a']['entry']}) ~ "
                         f"{r['b']['name']!r}({r['b']['entry']})  {r['reasons']}")
    lines.append("=" * 60)
    return "\n".join(lines)
