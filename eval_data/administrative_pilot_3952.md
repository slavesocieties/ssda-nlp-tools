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
