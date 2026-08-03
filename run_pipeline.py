#!/usr/bin/env python3
"""run_pipeline.py — the one-command post-extraction pipeline.

    python run_pipeline.py VOLUME.json [MORE_CHUNKS.json ...] --outdir out_v239746
                           [--tag V239746] [--auto 0.86] [--review 0.70] [--dup 0.75]

Runs everything this package does, in order, and writes all artifacts:

    qa_report.json      data-quality report per input (duplicates, chronology, ...)
    resolved.json       volume with a global_id on every person mention
    person_index.json   one row per identity (mentions, attributes, conflicts)
    network.graphml     the social graph (Gephi / networkx / Cytoscape)
    nodes.csv           one row per person identity (opens in Excel)
    edges.csv           one row per typed relationship, with both labels
    network.json        graph summary (hubs, components, relationship counts)
    review.html         self-contained page for deciding borderline merges
    pipeline_stats.json machine-readable counts, including the full review-pair
                        count and the number displayed in review.html
    summary.txt         human-readable rollup of all of the above

With several inputs, they are first combined and people are linked ACROSS them
(see link.py). After deciding pairs in review.html, fold the decisions back in
with run_review.py apply. No LLM calls, no network, no API keys.
"""
import argparse
import json
import os

from ssda_nlp_tools.link import link_volumes, format_link
from ssda_nlp_tools.network import to_graphml, to_csv, format_network
from ssda_nlp_tools.qa import qa_volume, format_qa
from ssda_nlp_tools.review_html import render_review_html


def machine_stats(link_result, displayed_review_pairs: int) -> dict:
    """Persist full counts without serializing the potentially huge pair queue."""
    stats = dict(link_result["stats"])
    stats["review_pairs_displayed"] = int(displayed_review_pairs)
    stats["review_pairs_persisted_in_full"] = False
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--tag", default="VOL")
    ap.add_argument("--tags", nargs="*", default=None, help="short labels per input file")
    ap.add_argument("--auto", type=float, default=0.86)
    ap.add_argument("--review", type=float, default=0.70)
    ap.add_argument("--dup", type=float, default=0.75)
    ap.add_argument("--review-limit", type=int, default=5000,
                    help="rows written to review.html, highest-scoring first. The "
                         "corpus review queue is ~750k pairs and rendering all of "
                         "them produced a 230 MB page no browser will open. 0 = no "
                         "cap. The full pair COUNT is stored in pipeline_stats.json; "
                         "the unbounded queue itself is not persisted by default.")
    args = ap.parse_args(argv)

    os.makedirs(args.outdir, exist_ok=True)
    out = lambda name: os.path.join(args.outdir, name)
    sections = []

    # 1. QA per input
    qa_reports = []
    for f in args.files:
        rep = qa_volume(f, dup_threshold=args.dup)
        rep["file"] = f
        qa_reports.append(rep)
        sections.append(format_qa(rep))
    with open(out("qa_report.json"), "w", encoding="utf-8") as fh:
        json.dump(qa_reports, fh, ensure_ascii=False, indent=2)

    # 2. link (single input degrades gracefully to plain resolve)
    res = link_volumes(args.files, tags=args.tags, volume_tag=args.tag,
                       auto_threshold=args.auto, review_threshold=args.review)
    sections.append(format_link(res, top=12))

    with open(out("resolved.json"), "w", encoding="utf-8") as fh:
        json.dump(res["resolved"]["volume"], fh, ensure_ascii=False, indent=2)
    with open(out("person_index.json"), "w", encoding="utf-8") as fh:
        json.dump(res["registry"], fh, ensure_ascii=False, indent=2)
    to_graphml(res["network"], out("network.graphml"))
    to_csv(res["network"], out("nodes.csv"), out("edges.csv"))
    with open(out("network.json"), "w", encoding="utf-8") as fh:
        json.dump(res["network"], fh, ensure_ascii=False, indent=2)
    queue = res["review_queue"]
    shown = queue if not args.review_limit else sorted(
        queue, key=lambda p: -p.get("score", 0))[:args.review_limit]
    render_review_html(shown, out("review.html"), tag=args.tag)
    with open(out("pipeline_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(machine_stats(res, len(shown)), fh, ensure_ascii=False, indent=2)
    if len(shown) < len(queue):
        # say what was dropped: a silently truncated page reads as "this is the
        # whole queue" when it is 0.7% of it
        sections.append(f"review.html: {len(shown):,} of {len(queue):,} pairs "
                        f"(highest-scoring first; full count in pipeline_stats.json; "
                        f"remaining pair rows are not persisted)")

    with open(out("summary.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(sections))

    print("\n\n".join(sections))
    print(f"\nartifacts -> {args.outdir}{os.sep}"
          f"{{qa_report.json, resolved.json, person_index.json, "
          f"network.graphml, nodes.csv, edges.csv, network.json, "
          f"review.html, pipeline_stats.json, summary.txt}}")
    print(f"next: open {out('review.html')} to decide {len(shown)} displayed "
          f"borderline pairs ({len(res['review_queue'])} total candidates), then "
          f"run_review.py apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
