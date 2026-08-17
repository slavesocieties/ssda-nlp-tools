"""Daniel's 2026-08-13 ruling, measured before anyone implements it.

    "they could all be ruled out by dates, and don't really require any
     contextual knowledge to inspect. The only case in which a pairing >50 years
     apart, or frankly even half of that, should even be 'looked at twice'
     (i.e. passed to more complex algorithmic processes) is if there is potential
     overlap in social networks or complete alignment in
     name/phenotype/ethnonym etc."

That is a rule about BLOCKING, not about scoring. `_shares_context` currently
keeps a pair if it satisfies ANY of: same register, a person named in both, or
dated within 60 years. The third clause is the permissive one -- a pair can
survive on the year window alone, with no register in common and nobody shared.

His rule says such a pair is worth nothing beyond ~25-50 years unless the
networks overlap or the identity attributes align completely.

So: how many candidate pairs are kept ONLY by the year window, how far apart are
they, and how many of those would his exceptions rescue? Measure before changing
anything -- 60 is the current window and it is load-bearing.
"""
import collections, glob, json, sys

sys.path.insert(0, r"C:\Users\mahajar\Downloads\sample images\ssda-wt-verification")
import ssda_nlp_tools.blocking as BL
import ssda_nlp_tools.disambiguate as D
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.evidence import NameStats
from ssda_nlp_tools.textmatch import name_similarity

CORPUS = "production/luna_v3/assembled_deduped_sided"
entries = []
for p in sorted(glob.glob(f"{CORPUS}/*.materialized.json")):
    entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
M = D._mentions_from_volume({"id": "corpus", "entries": entries})
stats = NameStats(M, is_clergy=E._clergy)

ATTRS = ("phenotype", "free", "ethnicity", "origin")


def attrs_align(x, y):
    """'complete alignment in name/phenotype/ethnonym': every attribute both
    records state must agree, and at least one must actually be stated."""
    stated = 0
    for k in ATTRS:
        a, b = x.get(k), y.get(k)
        if a is None or b is None:
            continue
        stated += 1
        if str(a).strip().lower() != str(b).strip().lower():
            return False
    return stated > 0


tab = collections.Counter()
rescued = collections.Counter()
for i, j in BL.candidate_pairs(M):
    x, y = M[i], M[j]
    same_reg = x.get("_register") and x.get("_register") == y.get("_register")
    if same_reg:
        tab["same register (his rule does not touch these)"] += 1
        continue
    ax = {n for _, n in (x.get("_ctx") or ())}
    ay = {n for _, n in (y.get("_ctx") or ())}
    shared = bool(ax & ay)
    if shared:
        tab["cross-register, shares a person"] += 1
        continue
    ya, yb = x.get("_year"), y.get("_year")
    if not (ya and yb):
        tab["cross-register, undated"] += 1
        continue
    gap = abs(ya - yb)
    band = ("<=25y" if gap <= 25 else "26-50y" if gap <= 50 else ">50y")
    tab[f"cross-register, YEAR WINDOW ONLY, {band}"] += 1
    if gap > 25:
        nm = name_similarity(x.get("name"), y.get("name"))
        if nm >= 0.999 and attrs_align(x, y):
            rescued[band] += 1

tot = sum(tab.values())
print(f"candidate pairs: {tot:,}\n")
for k, n in sorted(tab.items(), key=lambda kv: -kv[1]):
    print(f"  {k:<48}{n:>12,}{100*n/tot:>8.2f}%")

only = sum(n for k, n in tab.items() if "YEAR WINDOW ONLY" in k)
over25 = sum(n for k, n in tab.items() if "26-50y" in k or ">50y" in k)
over50 = sum(n for k, n in tab.items() if ">50y" in k)
print(f"\nkept ONLY by the year window        : {only:,} ({100*only/tot:.1f}% of candidates)")
print(f"  of those, more than 25 years apart: {over25:,}")
print(f"  of those, more than 50 years apart: {over50:,}")
print(f"\nhis exceptions rescue (exact name AND every stated attribute agreeing):")
for b in ("26-50y", ">50y"):
    n = sum(v for k, v in tab.items() if b in k)
    print(f"  {b:<8} {rescued[b]:,} of {n:,}")
print(f"\nSo his rule would drop {over25 - sum(rescued.values()):,} candidate pairs")
print("that currently reach the scorer on the year window alone.")
