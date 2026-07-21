# End-to-end pipeline validation on real model output (2026-07-20)

Answers "does the whole pipeline Zekai/Daniel wanted actually run and produce
sound output?" — not on synthetic data, but on **GPT-5.6 Luna's real extractions**
of the three San Agustín reference volumes (0013–0023, 0024–0034, 0035–0044).
`run_pipeline.py`, $0, no network.

## What ran, and what it produced

Input: 78 extracted entries (Luna), 432 person-mentions.

| Stage | Result |
|---|---|
| QA (`qa.py`) | 0 false-positive duplicates; real issues surfaced (5 dangling relationships, 1 event-shape violation) |
| Identity resolution (`disambiguate.py`) | 432 mentions → **215 identities** (50.2% reduction) |
| Cross-chunk linking (`link.py`) | **8 people linked across volumes** |
| Social graph (`network.py`) | 215 nodes, 606 unique typed edges, largest component 69 people |

Cross-volume links are the meaningful ones and they are correct: priest
**Miguel O'Reilly ×39** (0013+0035), priest **Thomas Hassett ×34** (all three),
enslaver **Juan Macqueen ×33** (all three), **Thomas Sterling ×30**. These are
the recurring officiants/enslavers that *should* collapse to one identity across
volumes, and they do — while same-named infants and enslaved people stay
distinct (the domain-guarded scorer, memory: identity resolution).

Relationship types recovered: child 114 / parent 114 / godparent 92 /
godchild 92 / enslaver 71 / slave 71 / spouse 52 — a coherent kinship +
enslavement graph, which is the archive's end goal.

## The duplicate-handling story is now closed end to end

The 544367 segmentation over-split (an Archivault page-boundary re-transcription)
is handled correctly by the layered design, *without* any risky segmenter
heuristic:

1. **Segmenter** keeps both copies, flags `partial` — safe over-inclusion, never
   loses a record.
2. **QA** (`qa.py` duplicate check) is built for exactly this ("LLM window-repair
   double-reports a record… fuzzy-match"). It flags near-duplicate text **only
   after checking the sacrament principal**: same principal → confirmed
   duplicate; different principals → two real records (kept); principal unknown
   (as in 544367's truncated partial) → flagged **unconfirmed for review**, not
   dropped. This is the discriminator the segmenter lacks — it needs the
   *extracted* principal, which only exists post-extraction.
3. Proof it doesn't over-flag: **0 duplicates flagged across 78 genuinely
   distinct but formulaic records.** The guard that would have wrongly merged
   65858's distinct 21-May record at the text level correctly keeps records
   apart here because their principals differ.

## What QA actually flagged (the review worklist)

Across 78 entries, 6 structural defects — all in relationship/event linkage,
which corroborates the 0.84 relationship F1 (the weak dimension), and none in
people/events themselves:

- **5 dangling relationships** — a person points at a `related_person` id the
  model never emitted as a person (0035: P03→P05; 0013: P02/P03→P04 ×2 each).
- **1 event-shape violation** — a baptism with 0 principals (the baptized
  person wasn't linked to the event).

These are machine-detectable with zero gold labels, so at corpus scale they
become a per-volume worklist rather than silent database pollution — which is
the point of the QA stage.

## Conclusion

The pipeline is **complete and sound end-to-end** on real output for the
extraction → QA → identity → graph path. Residual quality (relationships, fine
attributes) is surfaced by QA and routed to human review by design, not silently
accepted. The dominant human cost is identity-merge review (268 borderline pairs
for 3 volumes), not extraction correctness (6 structural defects).
