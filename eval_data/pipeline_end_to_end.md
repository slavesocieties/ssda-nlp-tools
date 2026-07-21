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

Manual spot checks found plausible cross-volume links: priest
**Miguel O'Reilly ×39** (0013+0035), priest **Thomas Hassett ×34** (all three),
enslaver **Juan Macqueen ×33** (all three), **Thomas Sterling ×30**. These are
the recurring officiants/enslavers that *should* collapse to one identity across
volumes, and they do — while same-named infants and enslaved people stay
distinct (the domain-guarded scorer, memory: identity resolution).

Relationship types recovered: child 114 / parent 114 / godparent 92 /
godchild 92 / enslaver 71 / slave 71 / spouse 52 — a coherent kinship +
enslavement graph, which is the archive's end goal.

## 544367 correction: the sixth segment is a real record

Reinspection of the source fixture disproved the earlier duplicate diagnosis.
Page 0107's No. 543 closes at the top of page 0108; page 0108 then contains Nos.
544 and 545 and begins a distinct **No. 546, Juan Alberto**. The supplied
five-entry reference omits No. 546 because it runs off the final image.

The segmenter correctly emits six records and marks No. 546 `partial`. A new
integration regression runs the actual two-page fixture through segmentation,
adds representative extracted principals, and passes it through principal-aware
QA. QA preserves all six and emits no duplicate flag. This validates the safe
behavior for this example; it does not claim that every future duplicate can be
resolved automatically.

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

## Does the model choice change the final graph?

Ran the same pipeline on **all three models'** extractions of the three volumes:

| Model | identities | cross-volume links | edges | dangling refs | event-shape |
|---|---|---|---|---|---|
| GPT-5.6 Luna | 215 | 8 | 606 | 5 | 1 |
| GPT-5.4 mini | 230 | 7 | 711 | 0 | 0 |
| Claude Haiku 4.5 | 222 | 7 | 642 | 1 | 1 |

The broad graph shape is similar, but the spread is material: identities differ
by up to 7%, cross-volume link counts by 14%, and edge counts by 17%. Luna's
end-to-end relationship F1 against the GPT-4o-generated references is 0.789,
versus 0.738 for mini and 0.718 for Haiku. Caveat, stated honestly: Luna
shows a few more dangling references than mini/Haiku (5 vs 0) — an internal-
consistency slip that QA flags and review fixes, not a gold-measured error. The
corrected comparison favors Luna for relationships and mini for event accuracy
and coverage. Either yields a broadly comparable graph, with different review
tradeoffs.

## Conclusion

The extraction → QA → identity → graph path runs end to end on real output.
This is an integration validation, not proof of production correctness:
relationship and fine-attribute quality still require reference evaluation and
human review. The dominant generated worklist is identity-merge review (268
borderline pairs for 3 volumes), alongside 6 machine-detected structural defects.
