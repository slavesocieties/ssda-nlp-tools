# Drive routing sweep — 2026-07-22

Ran `run_route_volume.py --source-kind auto` locally against the six JSON
exports that match the supplied Drive copies. No model calls were made. The
full per-page manifests remain local under `drive_pilots/routing_manifests/`.

| Volume | Genre | Deterministic sacramental | Model-fallback candidates | Re-transcribe | Skip index | Administrative QA/pilot |
|---|---|---:|---:|---:|---:|---:|
| 29597 | marriage register | 486 | 32 | 3 | 0 | 0 |
| 176899 | baptism/marriage register | 495 | 0 | 0 | 0 | 0 |
| 3952 | administrative dossiers | 0 | 0 | 0 | 0 | 25 |
| 375062 | baptism/pastoral register | 461 | 3 | 1 | 1 | 0 |
| 201991 | burial/parish register | 707 | 64 | 6 | 2 | 0 |
| 701054 | Portuguese burial register | 43 | 2 | 6 | 54 | 0 |
| **Total** |  | **2,192** | **101** | **16** | **57** | **25** |

Two synthetic `START`/`END` export markers in 3952 are omitted before routing;
they account for the difference between the 2,393 exported pages and the 2,391
routed units.

## Production consequences

1. Route the 2,192 deterministic sacramental pages through the existing local
   segmenter with no provider spend.
2. Re-transcribe the 16 source-error pages before any extraction.
3. Skip the 57 index pages for record extraction while retaining their source
   provenance.
4. Do not automatically send the 101 fallback candidates. They require a
   separate capped extraction approval and should be batched by source genre.
5. Keep the 25 administrative pages in compact-profile QA/pilot status until
   an administrative schema has a successful stratified validation.
