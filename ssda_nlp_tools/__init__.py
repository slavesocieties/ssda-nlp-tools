"""ssda_nlp_tools — local, dependency-light tooling for the SSDA NLP stage.

All runnable offline (no API keys); the full chain is
extract.py output -> QA -> disambiguate -> resolve -> link -> network,
with a human review loop (review.html -> decisions.json -> constraints):
  * textmatch    — name normalization, phonetic folding, similarity, alignment
  * evaluate     — a gold-set precision/recall/F1 harness for extracted data
  * disambiguate — automated, confidence-scored cross-entry person resolution
                   (domain guards: sacrament principals, discriminative context,
                    estate surnames, bare-name cap; human must/cannot constraints)
  * resolve      — apply disambiguation back onto a volume (global_id per mention)
  * link         — combine chunks/volumes and link people across them
  * network      — build the person/relationship social graph; export GraphML
  * qa           — per-volume data-quality report (duplicates, chronology, drift)
  * review_html  — self-contained merge-review page + decisions->constraints
  * cost         — token/cost model + optimizer to hit a $/image target
  * batch_extract— batched, cache-ordered, single-pass extractor (the cost recipe)
  * fixes        — corrected parse_date / complete_date / fix_relationships

Nothing here calls an LLM; everything scores or transforms JSON that the
existing extract.py / driver produced.
"""
__all__ = ["textmatch", "evaluate", "disambiguate", "resolve", "link",
           "network", "qa", "review_html", "cost", "batch_extract", "fixes"]
