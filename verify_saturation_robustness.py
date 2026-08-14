"""Try to break the name-saturation finding before reporting it to Daniel.

THE CLAIM UNDER ATTACK
    MAX_NAME_LLR = 5.5 binds for essentially every name, so the name term is a
    constant, and the resulting grid is inverted: first-name-only pairs within
    5 years score HIGHER (0.170) than full-name pairs (0.122), which is the
    opposite of Daniel's ruling.

THE MOST LIKELY WAY IT IS WRONG
    "full name" was defined as `len(name_tokens(n)) > 1`, and that is not the
    same thing as carrying a surname. "Maria de la Concepcion" is four tokens of
    devotional epithet and no family name at all; disambiguate.py says as much
    in `surname_affinity`, which declines to guess the paternal surname for
    exactly this reason. If the epithet names are what drive the inversion, the
    finding is an artifact of my definition and must be withdrawn.

So the grid is recomputed under progressively stricter definitions of "carries a
surname", plus within-volume only (to rule out pooling registers with different
naming conventions), and with mean as well as median. A finding that survives
all of them is worth Daniel's time; one that does not, is not.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import statistics

import ssda_nlp_tools.disambiguate as D
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.evidence import NameStats, score
from ssda_nlp_tools.textmatch import name_tokens, phonetic_key
from ssda_nlp_tools.volume_geo import load as load_geo

# Tokens that are grammar or devotion, never a family name.
PARTICLES = {"de", "del", "de la", "la", "las", "los", "el", "da", "das", "do",
             "dos", "y", "e", "di", "van", "der"}
DEVOTIONAL = {"jesus", "maria", "jose", "josef", "joseph", "concepcion",
              "carmen", "rosario", "dolores", "mercedes", "pilar", "socorro",
              "santisima", "trinidad", "cruz", "santa", "san", "espiritu",
              "sacramento", "asuncion", "encarnacion", "natividad", "angeles",
              "remedios", "candelaria", "regla", "belen", "loreto", "nieves",
              "purificacion", "presentacion", "visitacion", "gracia", "luz",
              "paz", "consolacion", "amparo", "patrocinio", "salud"}


def content_tokens(name):
    toks = [t for t in (name_tokens(name) or [])]
    return [t for t in toks if t not in PARTICLES]


def has_surname_loose(name):        # the original definition
    return len(name_tokens(name) or []) > 1


def has_surname_strict(name):
    """At least one token after the first that is neither particle nor devotion."""
    ct = content_tokens(name)
    return len(ct) > 1 and any(t not in DEVOTIONAL for t in ct[1:])


def has_surname_verystrict(name):
    """As above, and the LAST content token itself is a plausible family name."""
    ct = content_tokens(name)
    return len(ct) > 1 and ct[-1] not in DEVOTIONAL and ct[-1] != ct[0]


DEFS = [("loose  (len(tokens)>1, the original)", has_surname_loose),
        ("strict (non-devotional token after the first)", has_surname_strict),
        ("very strict (last content token is a family name)", has_surname_verystrict)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--assembled", default="production/luna_v3/assembled_deduped_sided")
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    a = ap.parse_args(argv)

    entries = []
    for p in sorted(glob.glob(f"{a.assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    stats = NameStats(M, is_clergy=E._clergy)
    geo = load_geo(a.volumes)
    vol_of = lambda m: str(m.get("_entry", "")).split("-")[0]

    # How much does the definition actually change the population?
    print("share of mentions counted as carrying a surname:")
    for label, fn in DEFS:
        n = sum(1 for m in M if fn(m.get("name")))
        print(f"    {label:52s} {n:6,}/{len(M):,} ({100*n/len(M):.1f}%)")

    for cap in (60, 200):
        blocks = collections.defaultdict(list)
        for m in M:
            blocks[phonetic_key(m.get("name"))].append(m)
        rows = []
        for key, g in blocks.items():
            if not key or len(g) < 2 or len(g) > cap:
                continue
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    x, y = g[i], g[j]
                    if x["_entry"] == y["_entry"] or D.lifespan_conflict(x, y):
                        continue
                    yx, yy = x.get("_year"), y.get("_year")
                    if not (yx and yy):
                        continue
                    r = score(x, y, stats, geo=geo, vol_of=vol_of)
                    if r["vetoed"]:
                        continue
                    rows.append((x.get("name"), y.get("name"), abs(yx - yy),
                                 r["probability"],
                                 vol_of(x) == vol_of(y)))
        print(f"\n{'='*76}\nblock cap {cap}: {len(rows):,} scorable pairs\n{'='*76}")
        for label, fn in DEFS:
            for scope in ("all volumes", "same volume only"):
                cells = collections.defaultdict(list)
                for nx, ny, gap, p, samevol in rows:
                    if scope == "same volume only" and not samevol:
                        continue
                    full = fn(nx) and fn(ny)
                    tm = "close" if gap <= 5 else ("far" if gap >= 20 else "mid")
                    cells[("full" if full else "first", tm)].append(p)
                fc, ic = cells[("full", "close")], cells[("first", "close")]
                ff = cells[("full", "far")]
                if not (fc and ic and ff):
                    print(f"  {label:52s} {scope:17s} EMPTY CELL"); continue
                med_ok = statistics.median(fc) > statistics.median(ic)
                mean_ok = statistics.fmean(fc) > statistics.fmean(ic)
                verdict = ("ordering HOLDS" if med_ok and mean_ok else
                           "INVERTED" if not med_ok and not mean_ok else
                           "MIXED (median/mean disagree)")
                print(f"  {label:52s} {scope:17s}\n"
                      f"      full-close med {statistics.median(fc):.3f} "
                      f"mean {statistics.fmean(fc):.3f} (n={len(fc):,})   "
                      f"first-close med {statistics.median(ic):.3f} "
                      f"mean {statistics.fmean(ic):.3f} (n={len(ic):,})   -> {verdict}")

    print("""
READ THIS BEFORE QUOTING ANY OF IT
  "INVERTED" means the model ranks a bare first name above a full name at equal
  temporal distance, contradicting Daniel's ruling. If the stricter definitions
  do NOT invert, then the original finding was an artifact of counting
  devotional epithets as surnames, and only the SATURATION half of the claim
  (that the cap binds for nearly every name) survives -- that half is measured
  directly from the frequency table and does not depend on this definition.""")


if __name__ == "__main__":
    main()
