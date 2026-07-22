# Deterministic production build — six Drive volumes (2026-07-22)

The free ($0, no network) production output for the six Drive exports.
Reproduce with `python run_production.py`, then price the paid step with
`run_corpus_prompts.py`. Nothing here calls a provider.

## Result

`run_production.py` segments every sacramental volume, tags each record by the
routing disposition of the pages it spans, and emits production record sets.

| Volume | Material | Pages | Records | Production | Needs fallback |
|---|---|---:|---:|---:|---:|
| 176899 | Cienfuegos baptism/marriage | 495 | 1087 | 1087 | 0 |
| 201991 | Guanabacoa burial/parish | 779 | 2085 | 2021 | 64 |
| 29597 | Havana marriage | 521 | 813 | 781 | 32 |
| 375062 | Limonar baptism/pastoral | 466 | 1137 | 1134 | 3 |
| 701054 | Portuguese burial | 105 | 221 | 212 | 9 |
| 3952 | Cofradía administrative | — | — | — | — (separate admin path) |
| **Total** | | **2366** | **5343** | **5235** | **108** |

- **5,235 production-ready records** — every page they span routed
  `deterministic-sacramental`. Each carries faithful text, `source_images`
  provenance, and a `partial` flag (556 are truthfully partial page-boundary
  records, kept and tagged, never dropped).
- **108 records touch a fallback page** and are withheld from the free output
  for the separately-approved capped Luna run. A record inheriting a harder
  disposition (worst-page-wins) never leaks into production.
- 3952 (administrative) stays on its own dossier path, QA/pilot status.

## Paid step — staged and priced, NOT sent

`run_corpus_prompts.py --corpus production/corpus --model gpt-5.6-luna` prepares
ready-to-send extraction batches and prices them with zero API calls:

- 5,235 entries → **527 LLM calls** (batch 10, 15-shot, 14,350-token cacheable prefix).
- Projected extraction: **~$30.21 interactive-with-cache, ~$15.07 via Batch API.**

Sending is a deliberate, separately-approved step (dry-run → approval → `--confirm`),
per the paid-run safety rules. The recommended model is GPT-5.6 Luna (leads
people + relationships cross-volume; see `entity_f1_bakeoff.md`).

## Artifacts (gitignored — regenerable in ~seconds)

```
production/segmented/<vol>.json   every segmented record + stats
production/records/<vol>.json     records tagged by disposition
production/corpus/<vol>.segmented.json   production records only (for staging)
production/batches/<vol>.batches.jsonl   priced, ready-to-send Luna batches
production/production_summary.json       this rollup, machine-readable
```

## What "finished" means here

The deterministic build is complete: 5,235 sacramental records with faithful
text and provenance are produced for $0, and the paid extraction is staged and
priced one approval away. Downstream QA → identity → graph
(`run_pipeline.py`) runs on the extracted output once the Luna step is approved
and run.
