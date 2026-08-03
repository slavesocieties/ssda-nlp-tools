#!/usr/bin/env python3
"""audit_corpus.py — every known defect in the delivered corpus, triaged.

Offline, $0, no network, no key.

    python audit_corpus.py
    python audit_corpus.py --stage production/repair_20260801

`qa.py` reports 1,048 issues across the delivered corpus and nobody has ever
triaged them. Raw, that number is useless in both directions: it looks alarming,
and most of the largest classes are the checker being conservative rather than
the data being wrong. This separates them, so what remains is a work list.

THE TRIAGE, measured rather than assumed
----------------------------------------
  REAL, deterministically repairable
    duplicate_entry (159)     Every one is confirmed: >=0.77 text similarity AND
                              the same sacrament principal. The window-repair
                              pass double-reported them. They inflate every
                              count we have given Daniel, so they matter more
                              than their share suggests.
    null_relationship (68)    A relationship whose `related_person` is null. Not
                              a pointer to a missing person -- a pointer to
                              nothing. Nothing downstream can use it.

  REAL, needs re-extraction (the source text is fine; we misread it)
    no_people_real (57)       A record whose text plainly describes a sacrament
                              and which carries no people at all. Same shape as
                              the NO_EVENT class found on 2026-07-31, and found
                              the same way.
    event_shape (23)          19 baptisms with 0 principals, 4 sacraments with
                              the wrong count. A baptism with nobody baptised is
                              not a record.
    dangling_relationship     A relationship pointing at a person id that does
      (178)                   not exist in its entry. Breaks the social graph
                              silently: the edge is simply absent downstream.

  MOSTLY NOT DEFECTS, and this is the important half
    no_people_admin (242)     Cover pages, pastoral-visit certificates, oficios.
                              They have no people because they are not records.
    chronology_break (298)    Registers are only roughly chronological, and 61
                              of the flagged entries carry more than one event
                              so the "primary" date is a choice. Verified
                              separately: only 8 of 6,844 dated events sit in a
                              different century from the one spelled out in
                              their own text, and 5 of those 8 are marginal
                              annotations recording a LATER marriage, which are
                              correct. Dates are not the problem chronology
                              order made them look.
    impossible_date (16)      Same cause. A baptism register ending in 1889 that
                              carries a 1914 marriage note is right, not wrong.

WHAT THIS DOES NOT DO
---------------------
It repairs nothing on its own. Deduplication and null-edge removal are staged as
a plan for `--stage`, not applied, because both change delivered counts and that
is Daniel's call. Re-extraction costs money and is his call twice over.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import Counter, defaultdict

from ssda_nlp_tools.qa import qa_volume

SACRAMENT = re.compile(
    r"\b(bau?ti[csz]|baptiz|batiz|sepult|enterr|cad[aá]ver|muri[oó]|falleci|"
    r"fall?eceu|matrimoni|velad|despos|recebe?r[aã]o em matrim)", re.I)

# Spelled-out century -> the century the extracted year must fall in.
_CENTURIES = [(re.compile(p, re.I), c) for p, c in [
    (r"mil\s+quinientos|quinhentos", 15),
    (r"mil\s*seis\s?cientos|seiscentos", 16),
    (r"mil\s*(?:sie|se)te\s?[cs]ientos|setecentos|sete\s?centos", 17),
    (r"mil\s*ocho\s?[cs]ientos|oitocentos|oito\s?centos", 18),
    (r"mil\s*nove\s?[cs]ientos|novecentos", 19),
]]
# A later marginal note ("Contrajo matrimonio el dia 10 de Abril de 1911")
# legitimately carries a date outside the register's own century.
_MARGINAL = re.compile(r"contrajo matrimoni|cas[oó] con|\bNota\s*[=:]|al margen", re.I)


def century_check(entries, field="text_faithful"):
    """Events whose year contradicts the century spelled out in the register.

    The strongest date check available without images: the scribe wrote the
    century in words, so a digit year in a different century is our error. Only
    entries with exactly ONE unambiguous century marker are checked.

    Run over BOTH fields, because they fail differently and the pair of results
    tells you which stage is wrong:

      text_faithful disagrees, normalized agrees -> the DATE is wrong
      normalized disagrees, text agrees          -> the NORMALIZATION is wrong

    That second case is real and would otherwise be invisible. 29597-0257-A-01
    reads "en tres de Febrero de mil nov.ta y dos" -- 1792 abbreviated -- and
    normalizes to "mil novecientos noventa y dos", i.e. 1992, in an
    18th-century Havana marriage register. The extracted date is correctly
    1792, so every date-based check passes while the text a researcher actually
    reads is off by two hundred years.
    """
    out, checked = [], 0
    for e in entries:
        text = e.get(field) or ""
        cents = {c for rx, c in _CENTURIES if rx.search(text)}
        if len(cents) != 1:
            continue
        want = cents.pop()
        years = [m.group(1) for m in
                 (re.match(r"(\d{4})", str(ev.get("date") or ""))
                  for ev in (e.get("data") or {}).get("events") or []) if m]
        if not years:
            continue
        checked += 1
        if want not in {int(y) // 100 for y in years}:
            both = (e.get("text_faithful") or "") + " " + (e.get("normalized") or "")
            out.append({"id": e["id"], "field": field,
                        "expected_century": f"{want}xx",
                        "event_years": sorted(set(years)),
                        # A later marginal note legitimately carries a date from
                        # outside the register's own century.
                        "marginal_note": bool(_MARGINAL.search(both))})
    return out, checked


def audit(paths):
    entries, issues = {}, []
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        for e in d.get("entries") or []:
            entries[e["id"]] = e
        issues.extend(qa_volume(d)["issues"])

    buckets = defaultdict(list)
    for i in issues:
        t, eid = i.get("type"), i.get("entry")
        detail = i.get("detail") or ""
        text = (entries.get(eid, {}) or {}).get("text_faithful") or ""
        if t == "no_people":
            buckets["no_people_real" if SACRAMENT.search(text)
                    else "no_people_admin"].append(i)
        elif t == "dangling_relationship":
            buckets["null_relationship" if "missing None" in detail
                    else "dangling_relationship"].append(i)
        else:
            buckets[t].append(i)

    faithful, n_faithful = century_check(entries.values(), "text_faithful")
    normal, n_normal = century_check(entries.values(), "normalized")
    cent = faithful + normal
    real_cent = [c for c in cent if not c["marginal_note"]]
    return (entries, buckets, cent, real_cent,
            {"text_faithful": n_faithful, "normalized": n_normal})


REPAIRABLE = ("duplicate_entry", "null_relationship")
REEXTRACT = ("no_people_real", "event_shape", "dangling_relationship",
             "dangling_principal", "vocab_violation")
BENIGN = ("no_people_admin", "chronology_break", "impossible_date")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--assembled", default="production/luna_v3/assembled")
    ap.add_argument("--stage", metavar="DIR",
                    help="write the repair plan and re-extraction id list")
    args = ap.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.assembled, "*.materialized.json")))
    if not paths:
        ap.error(f"no *.materialized.json under {args.assembled}")
    entries, buckets, cent, real_cent, cent_checked = audit(paths)
    n_checked = max(cent_checked.values())

    total = sum(len(v) for v in buckets.values())
    print(f"{len(entries):,} delivered records, {total:,} raw QA issues\n")

    def block(title, keys, note):
        n = sum(len(buckets[k]) for k in keys)
        print(f"=== {title}: {n:,}")
        for k in keys:
            if buckets[k]:
                print(f"      {len(buckets[k]):5d}  {k}")
        print(f"    {note}\n")
        return n

    n_fix = block("REAL, repairable without spending anything", REPAIRABLE,
                  "Both change delivered counts, so staged rather than applied.")
    n_ext = block("REAL, needs re-extraction", REEXTRACT,
                  "Source text is fine; the extractor misread it. PAID.")
    n_ok = block("NOT defects", BENIGN,
                 "Cover pages, and registers that are only roughly chronological.")

    print(f"=== dates and normalization, against the century spelled out in words")
    for f, n in cent_checked.items():
        print(f"    {n:,} records checked via {f}")
    print(f"    {len(cent)} disagree, of which {len(cent) - len(real_cent)} are "
          f"marginal notes recording a LATER event (correct)")
    print(f"    {len(real_cent)} genuine "
          f"({100*len(real_cent)/max(n_checked,1):.3f}%)")
    # Which side is wrong cannot be decided from the flag alone, and asserting
    # it produced an inverted conclusion once already. 29597-0056-B-02
    # normalizes correctly to "mil setecientos setenta y siete" (1777) while the
    # DATE says 1677 -- there the date is at fault; 29597-0257-A-01 has the
    # right date (1792) and a normalized text reading 1992. Both are flagged the
    # same way because the faithful text abbreviates the century ("mil nov.ta",
    # "mil Set.s Set.ta") and no pattern can read it. So report the conflict and
    # name both candidates rather than guessing.
    for c in real_cent:
        print(f"      {c['id']}  events {c['event_years']} vs "
              f"{c['expected_century']} spelled in {c['field']}")
        print(f"          the date or that text is wrong; the faithful text "
              f"abbreviates the century, so a human decides which")

    print(f"\nSUMMARY  real defects {n_fix + n_ext + len(real_cent):,} "
          f"of {len(entries):,} records "
          f"({100*(n_fix+n_ext+len(real_cent))/len(entries):.1f}%); "
          f"{n_ok:,} raw issues are not defects")

    if args.stage:
        os.makedirs(args.stage, exist_ok=True)
        # Some issues are volume-level and carry no entry id.
        dupes = [i["entry"] for i in buckets["duplicate_entry"] if i.get("entry")]
        reextract = sorted({i["entry"] for k in REEXTRACT
                            for i in buckets[k] if i.get("entry")})
        plan = {
            "generated_from": args.assembled,
            "records": len(entries),
            "deterministic": {
                "withdraw_duplicates": sorted(set(dupes)),
                "strip_null_relationships": sorted(
                    {i["entry"] for i in buckets["null_relationship"]
                     if i.get("entry")}),
            },
            "reextract": reextract,
            "century_errors": real_cent,
            "not_defects": {k: len(buckets[k]) for k in BENIGN},
        }
        out = os.path.join(args.stage, "corpus_audit.json")
        json.dump(plan, open(out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=1)
        print(f"\n-> {out}")
        print(f"   {len(set(dupes))} duplicates to withdraw, "
              f"{len(reextract)} records to re-extract")
        print("   Nothing applied. Deduplication changes every count Daniel has "
              "been given, and re-extraction costs money.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
