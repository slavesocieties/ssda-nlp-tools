# Administrative-record pilot: Drive volume 3952

The Drive volume was inspected and prepared locally without any LLM call.
Its Archivault export has five top-level items and 27 transcribed images:

* two synthetic export markers (`START` and `END`), omitted deterministically;
* three actual archival dossiers spanning 25 pages.

The existing sacramental-register segmenter is deliberately not used for this
corpus. It produced 14 artificial entries, 10 low-confidence pages, and only
11/27 agreement with its sacramental margin-number heuristic. That is expected:
the source is administrative cofradía material, not a sequence of baptism or
marriage entries.

`run_admin_pilot.py` and `ssda_nlp_tools.admin_records.to_documents()` preserve
each dossier's page grouping, raw transcription, source image filenames, and
metadata. The next model-assisted stage must use an administrative-document
schema (organizations, petitions/actions, people, offices, places, dates and
uncertainties) rather than the sacramental people/events/relationships schema.

## Luna schema pilot

With explicit approval, Luna received the three prepared dossiers under a
$0.25 total cap. The first dossier (seven pages) returned valid JSON with all
required arrays and cost $0.027876 (2,700 input, 4,196 output tokens). The
second dossier (17 pages) hit the 5,000-token response limit after $0.036106
(6,106 input, 5,000 output tokens), yielding no parseable JSON. The runner
stopped immediately, so the one-page third dossier was not sent.

Confirmed spend is $0.063982 with no outstanding reservation. The next attempt
must chunk long dossiers by page range and merge the chunk-level, provenance
tagged output locally; raising a whole-dossier output cap would be a less
reliable and more expensive solution.

## Chunked retry

The initial four-page chunk retry also stopped after one call: it consumed the
2,000-token limit and returned no JSON, at a cost of $0.013914 (1,914 input,
2,000 output tokens).  The pilot then used explicit `reasoning_effort: none`,
two-page chunks, and a 1,800-token response limit.  This produced one valid
chunk (`doc-003--p01-02`) for $0.011270 (1,322 input, 1,658 output tokens).
The next chunk (`doc-003--p03-04`) still reached the output limit, cost
$0.012136 (1,336 input, 1,800 output tokens), and was not parseable; the
runner stopped and sent no further chunks.

Confirmed Luna spend is now $0.101302 with no outstanding reservation.  The
valid result demonstrates that disabling reasoning is appropriate for this
bounded metadata task, but a complete administrative extraction requires a
smaller extraction unit or a narrower page-level schema before another paid
attempt.

## Page-level retry

A final one-page attempt on the next unprocessed page (`doc-003--p03-03`) used
the bounded schema, `reasoning_effort: none`, and a 700-token response limit.
It reached that limit without valid JSON at $0.005248 (1,048 input, 700 output
tokens), so the runner immediately stopped before sending the other 15 pages.
Confirmed spend is $0.106550 with no reservation.  This establishes that even
a page can contain more entities/actions than the current all-fields schema
can encode in 700 tokens; any future paid continuation must split extraction
by field group or introduce a compact, page-specific schema.
