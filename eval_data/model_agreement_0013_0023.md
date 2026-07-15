# Model/prompt agreement report — pages 0013–0023, volume 239746

Generated offline by `ssda_nlp_tools.evaluate` (no LLM calls). All variants are
scored against `Sample_output/Generated_0013_0023_4o_prompt_V2.json` as the
reference, so numbers are **agreement with the reference run**, not absolute
accuracy — there is no hand-checked gold for these pages yet. Low agreement
still localizes exactly *where* runs diverge, which is what a bake-off needs.

| variant | aligned/pred entries | people F1 | events F1 | rels F1 | date acc | phenotype | ethnicity | age |
|---|---|---|---|---|---|---|---|---|
| 4o_merged       | 27/27 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| less_ex_4o      | 25/29 | 0.851 | 0.972 | 0.730 | 0.771 | 0.292 | 0.133 | 0.833 |
| less_ex_5.4     | 27/39 | 0.786 | 0.947 | 0.254 | 0.889 | 0.844 | 0.000 | 0.778 |
| reduce_skip_4o  | 26/31 | 0.846 | 0.987 | 0.708 | 0.892 | 0.575 | 0.000 | 0.632 |
| reduce_skip_5.4 | 25/34 | 0.798 | 0.973 | 0.520 | 0.889 | 0.604 | 0.167 | 0.647 |
| merge_4o        | 27/30 | 0.932 | 0.987 | 0.733 | 0.949 | 0.698 | 0.067 | 0.368 |
| merge_5.4mini   | 27/35 | 0.856 | 0.974 | 0.280 | 0.919 | 0.235 | 0.077 | 0.333 |

## Readings

1. **Sanity check passed** — `4o_merged` is the reference data post-merge and
   scores a perfect 1.000 on every dimension, confirming the harness measures
   what it claims on real data.
2. **Relationships are the fragile dimension.** Every 5.4-family variant
   collapses on relationship agreement (0.25–0.52) while 4o variants hold
   0.71–0.73. If relationships matter (they are SSDA's core product), the
   model/prompt choice is not interchangeable, and "events look fine" (0.95+
   everywhere) is a misleading health signal.
3. **Entry inflation tracks the duplicate problem.** The reference has 27
   entries; 5.4 variants produce 34–39 on the same pages. The QA tool
   independently confirmed ~17% of entries in these outputs are window-overlap
   duplicates the repair step's first-500-chars dedup misses.
4. **`ethnicity` is broken everywhere** (0.00–0.17 agreement, high
   hallucination). Models cannot decide whether values like "north america" /
   "negro" belong to ethnicity, origin, or phenotype. This is a schema/prompt
   defect, not a model defect: the three fields need crisper definitions and
   disjoint vocabularies, or ethnicity should be dropped/merged.
5. **Prompt reduction is not free.** Both `less_ex` and `reduce_skip` shed
   people-agreement (~0.80–0.85 vs 0.93 for the merge prompt) — the few-shot
   examples are doing real work.

## Suggested next actions upstream

* Fix the window-overlap dedup (use principal-identity + fuzzy text, as in
  `ssda_nlp_tools/qa.py`) before any model comparison — entry inflation
  contaminates every downstream number.
* Tighten ethnicity/origin/phenotype definitions in `instructions.json`.
* Re-run this table after each prompt change: `python run_eval.py <ref> <variant>`.
* To convert agreement into true accuracy, hand-label these 27 entries as gold
  (they already exist in every variant, making them the cheapest gold to make).
