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
