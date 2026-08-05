"""Synthetic person pairs built around the decision boundaries, for labelling.

Daniel, 2026-08-03, after working through the live sample:

    "Most of these are very obvious based on chronology and geography. I still
    think some manual guidance makes sense, but rather than depend on this
    limited set of live data, perhaps a better option would be to create an
    artificial set of people records with more nuanced (but realistic) sets of
    characteristics. That way, I can inform intended behavior on true corner
    cases rather than fairly obvious 0/100s."

So the generator is not a random-record factory. Every pair is constructed to
sit ON a boundary the algorithm currently has to guess at, and each carries the
question it is meant to answer.

WHY THE LIVE SAMPLE PRODUCED OBVIOUS CASES
------------------------------------------
It was stratified by score band, surname relation, scope, evidence type and
information richness -- all properties of OUR pipeline. None of those is the
same as "hard for a historian". A pair can sit in the most uncertain score band
and still be trivially resolved by a date, which is exactly what he found.

Difficulty here is defined the other way round: a case is hard when the evidence
genuinely underdetermines the answer, so that a reasonable person could go
either way and the tie-break is a policy decision rather than a fact.

THE FAMILIES, and what each one is asking
-----------------------------------------
Each family varies ONE thing against a fixed background, so a label is
attributable. Mixing several would tell us a pair is hard without saying why.

  name_variant      How far may a name drift before it is a different person,
                    when everything else matches? (Daniel's Llopiz ruling made
                    this a sliding bar; this asks where the bar sits.)
  shared_given      Two people share a common given name and differ in surname.
                    Currently the cause of a confirmed false merge.
  lifespan_edge     Ages and dates that are compatible only just, or only just
                    not. Where is the line?
  single_signal     Exactly one corroborating signal, varied by TYPE. Is a
                    shared enslaver worth more than a shared date?
  clergy_recurrence The one case Daniel sanctioned merging on name; how far does
                    that licence extend?
  same_household    Two enslaved people, same enslaver, same estate, similar
                    names. Genuinely ambiguous and very common.
  attribute_drift   Scribal variation (morena/parda) against real change (free
                    status). Which differences are evidence?
  temporal_gap      Same name, same place, growing gap, no age evidence.
  placeholder_name  "N.", "no consta", unnamed infants -- names that identify
                    nobody.
  transcription     One side is plainly a mistranscription of the other.

WHAT THIS IS NOT
----------------
It generates no ground truth. Every pair ships with `expected: null`; the point
is Daniel's judgement, and pre-filling a guess would anchor it. The `question`
field is there so a label can be read back as a rule rather than a data point.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

# Drawn from what the corpus actually contains, so the pairs read like records
# rather than like test fixtures.
GIVEN_F = ["María", "Juana", "Francisca", "Antonia", "Josefa", "Rosalía",
           "Dolores", "Isabel", "Petrona", "Manuela", "Caridad", "Ramona"]
GIVEN_M = ["José", "Juan", "Francisco", "Antonio", "Manuel", "Pedro",
           "Miguel", "Domingo", "Lorenzo", "Rafael"]
DEVOTIONAL = ["de la Cruz", "de la Concepción", "del Rosario", "de Jesús",
              "de los Dolores", "de las Nieves"]
SURNAMES = ["Rodríguez", "Hernández", "Valdés", "Llopiz", "Bernal", "Carabalí",
            "Angulo", "Pulgarón", "Almada", "Zaldo", "Corrales", "Fleites"]
ETHNICITY = ["conga", "carabalí", "mandinga", "lucumí", "gangá", "criolla"]
PHENOTYPE = ["morena", "parda", "negra", "mulata"]
ORIGIN = ["Trinidad", "Guanabacoa", "Cienfuegos", "Matanzas", "La Habana",
          "Cumanayagua", "Regla"]
OCCUPATION = ["cleric", "field labourer", "domestic worker", "carpenter"]


def _person(rng, **over) -> Dict[str, Any]:
    p = {"name": None, "phenotype": None, "free": None, "ethnicity": None,
         "origin": None, "age": None, "occupation": None, "titles": None,
         "year": None, "relations": []}
    p.update(over)
    return p


def _pair(family, question, a, b, note=None, seed_id=0) -> Dict[str, Any]:
    return {"id": f"{family}-{seed_id:03d}", "family": family,
            "question": question, "a": a, "b": b, "note": note,
            "expected": None}


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #

def _name_variant(rng, i):
    """Everything matches; only the spelling drifts. Where is the bar?"""
    base = rng.choice(SURNAMES)
    drift = [(base, base, "identical"),
             (base, base.replace("z", "s").replace("Z", "S"), "z/s swap"),
             (base, base.replace("ll", "y").replace("Ll", "Y"), "yeísmo ll/y"),
             (base, base[:-1] + "os" if not base.endswith("s") else base,
              "suffix drift"),
             (base, base[0] + "e" + base[2:], "one-vowel change")]
    a_s, b_s, how = drift[i % len(drift)]
    given = rng.choice(GIVEN_F)
    year = rng.randint(1820, 1880)
    common = dict(phenotype="parda", free=False, origin=rng.choice(ORIGIN))
    return _pair(
        "name_variant",
        f"Surname differs by {how}. Everything else agrees. Same person?",
        _person(rng, name=f"{given} {a_s}", year=year, **common),
        _person(rng, name=f"{given} {b_s}", year=year + rng.randint(1, 6), **common),
        note=f"drift: {how}", seed_id=i)


def _shared_given(rng, i):
    """A very common given name, different families. Currently over-merged."""
    given = rng.choice(GIVEN_M)
    sa, sb = rng.sample(SURNAMES, 2)
    year = rng.randint(1830, 1870)
    # The two PEOPLE must share a name, or the pair is trivially different and
    # tests nothing. The question is entirely about whether two enslavers who
    # share a common given name count as the same man.
    person = rng.choice(GIVEN_F)
    return _pair(
        "shared_given",
        (f"Same name, and both are enslaved by a man called {given} -- but "
         f"{given} {sa} and {given} {sb}. Does that shared given name "
         f"corroborate, or is it just a common name?"),
        _person(rng, name=person, year=year, free=False, phenotype="morena",
                relations=[("enslaver", f"{given} {sa}")]),
        _person(rng, name=person, year=year + rng.randint(0, 8),
                free=False, phenotype="morena",
                relations=[("enslaver", f"{given} {sb}")]),
        note="mirror of a confirmed false merge (francisco pulgason / challi)",
        seed_id=i)


def _lifespan_edge(rng, i):
    """Compatible only just, or only just not."""
    birth = rng.randint(1800, 1850)
    gap = [14, 16, 45, 62, 78, 95][i % 6]
    # SAME name on both sides -- otherwise the pair is trivially different and
    # tests nothing about chronology, which is the whole point of the family.
    name = f"{rng.choice(GIVEN_F)} {rng.choice(DEVOTIONAL)}"
    owner = f"{rng.choice(GIVEN_M)} {rng.choice(SURNAMES)}"
    return _pair(
        "lifespan_edge",
        (f"An infant baptised in {birth}, and an adult of the same name "
         f"{gap} years later, same enslaver. Same person?"),
        _person(rng, name=name, age="infant", year=birth, free=False,
                relations=[("enslaver", owner)]),
        _person(rng, name=name, age="adult", year=birth + gap, free=False,
                relations=[("enslaver", owner)]),
        note=f"implied age at the later event: {gap}", seed_id=i)


def _single_signal(rng, i):
    """Exactly one corroborating signal, varied by type."""
    kinds = ["date only", "shared enslaver only", "shared parent only",
             "matching attributes only", "shared godparent only"]
    kind = kinds[i % len(kinds)]
    name = f"{rng.choice(GIVEN_F)} {rng.choice(DEVOTIONAL)}"
    y = rng.randint(1830, 1875)
    a = _person(rng, name=name, year=y)
    b = _person(rng, name=name, year=y + (2 if kind == "date only" else 20))
    if kind == "shared enslaver only":
        who = f"{rng.choice(GIVEN_M)} {rng.choice(SURNAMES)}"
        a["relations"] = [("enslaver", who)]
        b["relations"] = [("enslaver", who)]
    elif kind == "shared parent only":
        who = f"{rng.choice(GIVEN_F)} {rng.choice(SURNAMES)}"
        a["relations"] = [("parent", who)]
        b["relations"] = [("parent", who)]
    elif kind == "shared godparent only":
        who = f"{rng.choice(GIVEN_M)} {rng.choice(SURNAMES)}"
        a["relations"] = [("godparent", who)]
        b["relations"] = [("godparent", who)]
    elif kind == "matching attributes only":
        for p in (a, b):
            p["phenotype"], p["free"] = "morena", False
            p["ethnicity"] = "conga"
    return _pair(
        "single_signal",
        f"Same name; the ONLY corroboration is: {kind}. Enough to merge?",
        a, b, note=f"signal type: {kind}", seed_id=i)


def _clergy_recurrence(rng, i):
    """Daniel sanctioned merging clergy on name. How far does that go?"""
    name = f"{rng.choice(GIVEN_M)} {rng.choice(SURNAMES)}"
    span = [1, 3, 12, 28, 41][i % 5]
    y = rng.randint(1820, 1850)
    return _pair(
        "clergy_recurrence",
        (f"The same priest's name signing entries {span} years apart, with no "
         f"other shared detail. One career, or two men?"),
        _person(rng, name=name, occupation="cleric", titles=["Don"], year=y),
        _person(rng, name=name, occupation="cleric", titles=["Don"], year=y + span),
        note=f"career span implied: {span} years", seed_id=i)


def _same_household(rng, i):
    """Two enslaved people on one estate with similar names. Very common."""
    owner = f"{rng.choice(GIVEN_M)} {rng.choice(SURNAMES)}"
    given = rng.choice(GIVEN_F)
    variants = [(given, given), (given, given + " " + rng.choice(DEVOTIONAL)),
                (given, rng.choice([g for g in GIVEN_F if g != given]))]
    a_n, b_n = variants[i % len(variants)]
    y = rng.randint(1830, 1870)
    return _pair(
        "same_household",
        ("Two enslaved people of the same enslaver, names as shown. The estate "
         "is shared by everyone on it. Same person or two of the household?"),
        _person(rng, name=a_n, free=False, year=y, phenotype="morena",
                relations=[("enslaver", owner)]),
        _person(rng, name=b_n, free=False, year=y + rng.randint(1, 9),
                phenotype="morena", relations=[("enslaver", owner)]),
        note="shared enslaver is population-wide within an estate", seed_id=i)


def _attribute_drift(rng, i):
    """Scribal variation against real change."""
    cases = [("phenotype", "morena", "parda", "scribal variation"),
             ("ethnicity", "conga", "carabalí", "different stated origin"),
             ("free", False, True, "manumission, or two people?"),
             ("origin", "Trinidad", "Cienfuegos", "moved, or two people?"),
             ("age", "infant", "adult", "24 years apart")]
    field, va, vb, why = cases[i % len(cases)]
    name = f"{rng.choice(GIVEN_F)} {rng.choice(SURNAMES)}"
    y = rng.randint(1830, 1860)
    gap = 24 if field == "age" else rng.randint(2, 10)
    # Give the pair enough baseline corroboration (a shared enslaver, which is
    # a discriminative relation) that the ONLY thing in question is the
    # differing attribute. Without this the pair fails the two-signal minimum
    # for reasons unrelated to the attribute, and the family tests nothing.
    owner = f"{rng.choice(GIVEN_M)} {rng.choice(SURNAMES)}"
    a = _person(rng, name=name, year=y, relations=[("enslaver", owner)])
    b = _person(rng, name=name, year=y + gap, relations=[("enslaver", owner)])
    a[field], b[field] = va, vb
    return _pair(
        "attribute_drift",
        f"Same name, {gap} years apart, but {field} differs ({va} vs {vb}). "
        f"Is that evidence of two people, or {why}?",
        a, b, note=why, seed_id=i)


def _temporal_gap(rng, i):
    """Same name and place, growing gap, no age evidence at all."""
    gap = [2, 9, 21, 34, 55][i % 5]
    name = f"{rng.choice(GIVEN_F)} {rng.choice(DEVOTIONAL)}"
    place = rng.choice(ORIGIN)
    y = rng.randint(1820, 1860)
    return _pair(
        "temporal_gap",
        (f"Identical name, same parish, {gap} years apart, nothing else known. "
         f"At what gap does this stop being one person?"),
        _person(rng, name=name, origin=place, year=y),
        _person(rng, name=name, origin=place, year=y + gap),
        note=f"gap: {gap} years, no age or relations", seed_id=i)


def _placeholder_name(rng, i):
    """Names that identify nobody."""
    kinds = [("N.", "N."), ("no consta", "no consta"),
             ("párvulo no nombrado", "párvula no nombrada"),
             ("N.", rng.choice(GIVEN_F))]
    a_n, b_n = kinds[i % len(kinds)]
    y = rng.randint(1830, 1870)
    owner = f"{rng.choice(GIVEN_M)} {rng.choice(SURNAMES)}"
    return _pair(
        "placeholder_name",
        ("Both names are placeholders rather than names. Should a placeholder "
         "ever match another placeholder?"),
        _person(rng, name=a_n, free=False, year=y,
                relations=[("enslaver", owner)]),
        _person(rng, name=b_n, free=False, year=y + rng.randint(0, 4),
                relations=[("enslaver", owner)]),
        note="nomen nescio", seed_id=i)


def _transcription(rng, i):
    """One side is plainly a misreading of the other."""
    base = rng.choice(SURNAMES)
    corrupt = [base.replace("r", "n"), base.replace("l", "i"),
               base.replace("a", "o"), base + "z", base[:-2]]
    given = rng.choice(GIVEN_F)
    y = rng.randint(1830, 1870)
    common = dict(free=False, phenotype="parda", origin=rng.choice(ORIGIN))
    return _pair(
        "transcription",
        ("One surname looks like a misreading of the other and everything else "
         "matches. Merge, and if so which spelling should the record keep?"),
        _person(rng, name=f"{given} {base}", year=y, **common),
        _person(rng, name=f"{given} {corrupt[i % len(corrupt)]}",
                year=y + rng.randint(0, 3), **common),
        note="tests Daniel's name-variant retention question", seed_id=i)


# --------------------------------------------------------------------------- #
# social networks -- Daniel, 2026-08-05
#
# "in most cases people will appear embedded in a social network of some
# density. This is also critical to disambiguation, and so social networks of
# varying size, complexity, and overlap should be included."
#
# The first ten families vary ONE attribute against a fixed background and give
# each side at most a single relation, so they cannot ask the question that
# actually decides most real merges: how much of the surrounding network is
# shared, and how much of it CONFLICTS. These four vary size (1 to 6 associates),
# overlap (none to complete), and role structure independently.
#
# The hard case is deliberate: identical names with DENSE, DISJOINT networks.
# Every string-level signal says merge and the social evidence says two different
# people, which is the shape a weight-of-evidence model should get right and a
# name-similarity threshold cannot.
# --------------------------------------------------------------------------- #

_REL_ROLES = ("parent", "godparent", "spouse", "enslaver", "sibling")


def _some_name(rng):
    given = rng.choice(GIVEN_F + GIVEN_M)
    return f"{given} {rng.choice(SURNAMES)}"


def _assoc(rng, n, exclude=None):
    """n DISTINCT named associates, each in a different role where possible.

    Distinctness matters: two associates who happen to share a generated name
    would make a network look more overlapping than it is, which is the very
    thing these families exist to measure.
    """
    out, used = [], set(exclude or ())
    guard = 0
    while len(out) < n and guard < 200:
        guard += 1
        nm = _some_name(rng)
        if nm in used:
            continue
        used.add(nm)
        out.append({"type": _REL_ROLES[len(out) % len(_REL_ROLES)], "name": nm})
    return out


def _network_overlap(rng, i):
    """Same name, networks of the SAME size, sharing k of them."""
    size = rng.choice((2, 3, 4, 6))
    share = rng.choice((0, 1, size // 2, size))
    nm = _some_name(rng)
    common = _assoc(rng, share)
    seen = {r["name"] for r in common}
    a_only = _assoc(rng, size - share, exclude=seen)
    seen |= {r["name"] for r in a_only}
    b_only = _assoc(rng, size - share, exclude=seen)
    y = rng.randint(1790, 1880)
    a = _person(rng, name=nm, year=y, relations=common + a_only)
    b = _person(rng, name=nm, year=y + rng.randint(1, 6),
                relations=common + b_only)
    return _pair("network_overlap",
                 f"Identical names. Each is embedded in {size} relationships, "
                 f"of which {share} name the same person. Same person?",
                 a, b, note=f"size {size}, shared {share}", seed_id=i)


def _network_asymmetric(rng, i):
    """One side densely embedded, the other barely -- the common real case,
    where absence of overlap may only mean absence of evidence."""
    nm = _some_name(rng)
    dense = _assoc(rng, rng.choice((4, 5, 6)))
    thin = (dense[:1] if rng.random() < 0.5
            else _assoc(rng, 1, exclude={r["name"] for r in dense}))
    y = rng.randint(1790, 1880)
    a = _person(rng, name=nm, year=y, relations=dense)
    b = _person(rng, name=nm, year=y + rng.randint(1, 8), relations=thin)
    shared = len({r["name"] for r in dense} & {r["name"] for r in thin})
    return _pair("network_asymmetric",
                 f"Identical names. One appears with {len(dense)} relations, the "
                 f"other with {len(thin)}, sharing {shared}. Same person?",
                 a, b, note=f"dense {len(dense)} vs thin {len(thin)}, "
                            f"shared {shared}", seed_id=i)


def _network_conflict(rng, i):
    """Identical names, DENSE networks, ZERO overlap. Two different people who
    look identical to every string comparison."""
    nm = _some_name(rng)
    n = rng.choice((3, 4, 5))
    y = rng.randint(1790, 1880)
    ra = _assoc(rng, n)
    # Drawn with the first side EXCLUDED. Two independent draws from a small
    # name pool collide by chance, and a "zero overlap" family that quietly
    # overlaps measures nothing -- which is what the test caught.
    rb = _assoc(rng, n, exclude={r["name"] for r in ra})
    a = _person(rng, name=nm, year=y, relations=ra)
    b = _person(rng, name=nm, year=y + rng.randint(0, 4), relations=rb)
    return _pair("network_conflict",
                 f"Identical names and dates. Each is embedded in {n} "
                 f"relationships and they share NONE of them. Same person?",
                 a, b, note=f"dense {n} each, zero overlap", seed_id=i)


def _network_role_shift(rng, i):
    """The same associates in DIFFERENT roles -- a woman who is a godparent in
    one record and a parent in another, which is ordinary, versus role pairs
    that cannot both hold."""
    nm = _some_name(rng)
    who = _assoc(rng, 2)
    y = rng.randint(1790, 1880)
    ra, rb = rng.choice((("godparent", "parent"), ("parent", "child"),
                         ("spouse", "parent"), ("godparent", "godparent")))
    a = _person(rng, name=nm, year=y,
                relations=[{"type": ra, "name": who[0]["name"]}, who[1]])
    b = _person(rng, name=nm, year=y + rng.randint(1, 10),
                relations=[{"type": rb, "name": who[0]["name"]}, who[1]])
    return _pair("network_role_shift",
                 f"Identical names sharing two associates, but {who[0]['name']} "
                 f"is their {ra} in one record and their {rb} in the other. "
                 f"Same person?",
                 a, b, note=f"role {ra} -> {rb}", seed_id=i)


FAMILIES = {
    "name_variant": _name_variant,
    "shared_given": _shared_given,
    "lifespan_edge": _lifespan_edge,
    "single_signal": _single_signal,
    "clergy_recurrence": _clergy_recurrence,
    "same_household": _same_household,
    "attribute_drift": _attribute_drift,
    "temporal_gap": _temporal_gap,
    "placeholder_name": _placeholder_name,
    "transcription": _transcription,
    "network_overlap": _network_overlap,
    "network_asymmetric": _network_asymmetric,
    "network_conflict": _network_conflict,
    "network_role_shift": _network_role_shift,
}


def generate(n: int = 300, seed: int = 20260803,
             families: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Round-robin across families so every one is represented evenly."""
    rng = random.Random(seed)
    names = [f for f in (families or list(FAMILIES)) if f in FAMILIES]
    out: List[Dict[str, Any]] = []
    i = 0
    while len(out) < n:
        out.append(FAMILIES[names[i % len(names)]](rng, i // len(names)))
        i += 1
    for k, p in enumerate(out):
        p["id"] = f"{p['family']}-{k:03d}"
    return out
