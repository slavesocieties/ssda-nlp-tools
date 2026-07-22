# Drive volume inventory — 2026-07-21

Read-only structural inventory of the six JSON exports in the supplied Drive
volume folder. Counts are taken from Archivault JSON; no images or model calls
were used.

| Volume | Top-level items | Pages | Failed transcriptions | Table-like pages | Primary material |
|---|---:|---:|---:|---:|---|
| 29597 | 1 | 521 | 3 | 0 | Havana marriage register |
| 176899 | 1 | 495 | 0 | 0 | Cienfuegos baptism/marriage register |
| 3952 | 5 | 27 | 0 | 3 | Cofradía administrative dossiers |
| 375062 | 1 | 466 | 1 | 1 | Limonar baptism/pastoral register |
| 201991 | 4 | 779 | 6 | 2 | Guanabacoa burial/parish register |
| 701054 | 1 | 105 | 6 | 55 | Jurujuba Portuguese burial register |

## Routing consequences

- Item grouping is not genre: 701054 is one 105-page item but is a parish
  register, whereas 3952 contains multiple administrative dossiers.
- Table-like pages are common in 701054, so the source classifier accepts a
  table-heavy register only when corpus-scale sacramental evidence also exists.
- Failed transcriptions route to re-transcription, never to extraction.
- The router produces deterministic output for all volume types but keeps the
  administrative compact model profile in QA/pilot status until it completes a
  stratified validation. It never sends a model request by itself.
