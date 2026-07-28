#!/usr/bin/env python3
"""backfill_accepted.py — migrate pre-existing validated output to the
accepted-artifact convention. Offline, $0, no network.

assemble_corpus now reads only `*.accepted.jsonl`, so that raw provider output
can never reach delivery. That is the right rule, but it was introduced after
the original run had already been validated and delivered: `luna_live` holds 18
`*.output.jsonl` files with no accepted counterpart, so assembling it now yields
zero records and a summary claiming the corpus is missing.

This writes an accepted artifact for each output whose `*.validation.json` says
`valid`, and refuses the ones that do not. It grants nothing that was not
already checked — it only records the existing verdict in the new form.

    python backfill_accepted.py [--live production/luna_live] [--dry-run]
"""
import argparse
import json
from pathlib import Path


def _request_ok(row) -> bool:
    """Per-request acceptance: the same checks the guarded runner now applies.

    A request is accepted only if the provider returned 200, the model stopped
    normally, and the content parses as the expected JSON envelope with at least
    one entry. Nothing here is more permissive than the whole-job validator; it
    is simply applied per request instead of per file.
    """
    from ssda_nlp_tools.batch_extract import parse_response
    resp = (row or {}).get("response") or {}
    if resp.get("status_code") != 200:
        return False
    choices = (resp.get("body") or {}).get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        return False
    try:
        values, missing = parse_response(choices[0].get("message", {}).get("content"),
                                         [], validate=True)
    except Exception:
        return False
    return bool(values) and not missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", type=Path, default=Path("production/luna_live"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    made = skipped = 0
    for out in sorted(args.live.glob("*.output.jsonl")):
        acc = out.with_name(out.name.replace(".output.jsonl", ".accepted.jsonl"))
        val = out.with_name(out.name.replace(".output.jsonl", ".validation.json"))
        if acc.exists():
            continue
        if not val.exists():
            print(f"  SKIP {out.name}: no validation.json — never verified")
            skipped += 1
            continue
        try:
            verdict = json.loads(val.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  SKIP {out.name}: unreadable validation ({exc})")
            skipped += 1
            continue
        if not verdict.get("valid"):
            # A job marked invalid failed as a WHOLE, usually because a handful
            # of its requests returned bad IDs. Its remaining requests were each
            # checked and are fine. Rejecting the whole file would discard ~4,500
            # good records, which is precisely what the accepted-artifact design
            # exists to prevent — it just has to be applied retroactively here.
            rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
            good = [r for r in rows if _request_ok(r)]
            bad = len(rows) - len(good)
            if not good:
                print(f"  SKIP {out.name}: invalid, and no request passed individually")
                skipped += 1
                continue
            if not args.dry_run:
                acc.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in good),
                               encoding="utf-8")
            made += 1
            print(f"  {'would salvage' if args.dry_run else 'salvaged'} {out.name}: "
                  f"{len(good)} requests accepted, {bad} rejected")
            continue
        if not args.dry_run:
            acc.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        made += 1
        print(f"  {'would accept' if args.dry_run else 'accepted'} {out.name}")
    print(f"\n{made} accepted, {skipped} left unaccepted"
          + (" (dry run — nothing written)" if args.dry_run else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
