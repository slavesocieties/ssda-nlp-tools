"""Which weights and guards in the scorer actually fire on the delivered corpus?

THE GENERAL SHAPE THIS LOOKS FOR
--------------------------------
On 2026-08-13 the maternal/paternal grandparent capacities turned out to be
unreachable: implemented in `MAX_HOLDERS`, recorded as done, and absent from
every delivered volume, so the rule they encode was silently not in force. No
error, no warning, a normal-looking merge result.

That is a class, not an incident. Any term in a weight-of-evidence model can be
inert for the same reason -- the data never presents the condition -- and the
model still returns a confident number. So this enumerates every term and veto
the scorer can emit and counts how often each one actually fires.

Three findings are possible and all three matter:

  NEVER FIRES     the weight is not in force. Whatever it encodes -- a ruling, a
                  penalty, a special case -- is currently decoration.
  ALWAYS BINDS    a cap that binds on every pair is not a cap, it is a constant,
                  and the quantity it caps has no remaining variance. That is
                  already known for MAX_NAME_LLR (HANDOVER §4) and is worth
                  checking for every other bound.
  FIRES RARELY    real but negligible. Worth knowing before anyone tunes it,
                  because a weight that fires 12 times cannot move a corpus and
                  time spent calibrating it is wasted.

Offline, $0. Samples candidate pairs from the real blocking keys rather than
random pairs, because the population a weight fires on is the CANDIDATE pool, not
the cartesian product -- scoring random pairs would understate every term that
blocking already selects for.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import random
import re

import ssda_nlp_tools.blocking as BL
import ssda_nlp_tools.disambiguate as D
import ssda_nlp_tools.evidence as E
from ssda_nlp_tools.evidence import NameStats, score
from ssda_nlp_tools.volume_geo import load as load_geo


def _uncapped(stats, name):
    """-ln(p) with no cap. `NameStats.llr` applies MAX_NAME_LLR, so asking it
    whether the cap binds is circular and always answers no."""
    p = stats.p(name)
    return -math.log(p) if p and p > 0 else float("inf")


def _norm(label):
    """Collapse a term label to the weight that produced it."""
    lab = label.split("(")[0]
    if lab.startswith("name~"):
        return "name (rarity x similarity)"
    # The label carries the associate's NAME, so an unnormalised tally produces
    # one row per person in the corpus and buries the finding under itself.
    if lab.startswith("shared:") or lab.startswith("network:shared"):
        return "network: shared associate"
    if re.match(r"^gap\d+y$", lab):
        return "year gap"
    return lab.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled_deduped_sided")
    ap.add_argument("--volumes", default="../ssda-openai/volumes.json")
    ap.add_argument("--sample", type=int, default=400_000,
                    help="candidate pairs to score; 0 = all (slow)")
    ap.add_argument("--seed", type=int, default=20260813)
    a = ap.parse_args(argv)

    entries = []
    for p in sorted(glob.glob(f"{a.assembled}/*.materialized.json")):
        entries.extend(json.load(open(p, encoding="utf-8"))["entries"])
    if not entries:
        raise SystemExit(f"no entries under {a.assembled!r}")
    M = D._mentions_from_volume({"id": "corpus", "entries": entries})
    stats = NameStats(M, is_clergy=E._clergy)
    geo = load_geo(a.volumes)
    vol_of = lambda m: str(m.get("_entry", "")).split("-")[0]

    pairs = list(BL.candidate_pairs(M))
    rng = random.Random(a.seed)
    total_pairs = len(pairs)
    if a.sample and a.sample < total_pairs:
        pairs = rng.sample(pairs, a.sample)
    print(f"corpus  : {len(entries):,} entries, {len(M):,} mentions")
    print(f"scoring : {len(pairs):,} of {total_pairs:,} candidate pairs "
          f"({100*len(pairs)/total_pairs:.1f}%)\n")

    fires = collections.Counter()
    vetoes = collections.Counter()
    name_capped = name_seen = 0
    assoc_capped = assoc_seen = 0
    for i, j in pairs:
        x, y = M[i], M[j]
        r = score(x, y, stats, geo=geo, vol_of=vol_of)
        if r["vetoed"]:
            vetoes[r["vetoed"]] += 1
            continue
        for lab, _llr in r["terms"]:
            fires[_norm(lab)] += 1
            # MEASURE THE CAP AGAINST THE MODEL, NOT AGAINST THE LABEL.
            #
            # A first version parsed "rarity ([\d.]+)" out of the label. The
            # label is literally "name~0.55 rarity" -- the number lives in the
            # tuple's second element, not the string -- so the regex never
            # matched and the tool reported the cap binding on 0.0% of 246,517
            # name terms. That is a clean, confident, completely wrong answer of
            # exactly the shape self_check.py exists to catch, and it survived
            # only until it contradicted HANDOVER §4's measured 100%.
            #
            # `NameStats.llr` already applies the cap, so asking it is circular.
            # Recompute the UNCAPPED rarity and compare.
            if lab.startswith("name~"):
                name_seen += 1
                if min(_uncapped(stats, x.get("name")),
                       _uncapped(stats, y.get("name"))) >= E.MAX_NAME_LLR - 1e-9:
                    name_capped += 1
            elif _norm(lab) == "network: shared associate":
                assoc_seen += 1
                if _llr >= E.MAX_LLR_PER_ASSOCIATE - 1e-9:
                    assoc_capped += 1

    scored = len(pairs) - sum(vetoes.values())
    print(f"{'TERM':<34}{'fires':>10}{'% of scored':>13}")
    for lab, n in fires.most_common():
        print(f"  {lab:<32}{n:>10,}{100*n/max(1,scored):>12.2f}%")
    print(f"\n{'VETO':<34}{'fires':>10}")
    for lab, n in vetoes.most_common():
        print(f"  {lab:<32}{n:>10,}")

    # ---- what the code can emit but the corpus never produced --------------
    # DERIVED FROM THE CODE, NOT INVENTED.
    #
    # A first version listed labels from memory and reported "enslaver-gap NEVER
    # FIRED". There is no such label: ENSLAVER_TAU_YEARS decays the penalty
    # INSIDE `conflict:enslaver`, which fires normally. A hand-written vocabulary
    # turns a name I made up into a finding about the model -- an inert-weight
    # report that was itself inert.
    #
    # These are the literal label strings `evidence.py` constructs.
    CAN_EMIT = {
        "name (rarity x similarity)": "name term",
        "network: shared associate": "shared-associate LLR",
        "year gap": "W_YEAR_CLOSE / W_YEAR_FAR",
        "both-clergy": "W_CLERGY_BOTH",
        "attrs-agree": "W_ATTR_AGREE",
        "attrs-conflict": "W_ATTR_CONFLICT",
        "place:": "W_PLACE (all levels)",
        "volumes-never-coexist": "W_VOLUMES_NEVER_COEXIST",
        "disjoint-networks": "the density penalty",
        "conflict:": "MAX_HOLDERS capacities, incl. the enslaver decay",
    }
    print("\nNEVER FIRED (a weight not in force is decoration, not a safeguard)")
    # SUBSTRING, not prefix. A first version matched on prefix and declared
    # "conflict" dead while `network:conflict:enslaver` was firing 261 times --
    # a false inert report inside the tool built to find inert weights. The
    # label vocabulary is namespaced (`network:conflict:enslaver`), so a
    # prefix test asks whether a term STARTS with the concept rather than
    # whether it uses it.
    dead = []
    for needle, what in sorted(CAN_EMIT.items()):
        if not any(needle.split()[0] in k for k in fires):
            dead.append(f"{needle}  ({what})")
    for d in dead:
        print(f"  {d}")
    if not dead:
        print("  (every term in the audited vocabulary fired at least once)")

    print("\nCAPS -- a bound that always binds is a constant, not a cap")
    if name_seen:
        print(f"  MAX_NAME_LLR={E.MAX_NAME_LLR}: binds on {name_capped:,} of "
              f"{name_seen:,} name terms ({100*name_capped/name_seen:.1f}%)")
    if assoc_seen:
        print(f"  MAX_LLR_PER_ASSOCIATE={E.MAX_LLR_PER_ASSOCIATE}: binds on "
              f"{assoc_capped:,} of {assoc_seen:,} shared associates "
              f"({100*assoc_capped/assoc_seen:.1f}%)")
    else:
        print("  MAX_LLR_PER_ASSOCIATE: no shared associate fired in this sample")

    print("\nRead this the right way: 'never fired' on a SAMPLE is not proof of")
    print("inert -- rerun with --sample 0 before acting on a zero. A term that")
    print("fires rarely is real; one that cannot fire is a rule not in force.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
