# ADR-001: Deterministic routing for heterogeneous historical volumes

**Status:** Accepted  
**Date:** 2026-07-21  
**Decider:** SSDA project

## Context

The corpus contains sacramental registers and administrative dossiers. Their
boundaries, schemas and failure modes differ. A generic LLM should not select
its own schema, chunk size or budget, and the faithful source transcription
must remain available for every output.

## Decision

Use a deterministic router that emits a versioned manifest before any model
call. Sacramental pages run through the existing free state-machine segmenter;
only low-confidence pages are nominated for the sacramental fallback.
Administrative material is grouped deterministically into dossiers and pages,
then indexed locally. Bounded administrative pages are nominated only for a
QA-approved compact-index pilot until that profile has completed stratified
validation; dense or empty pages also require QA.

`auto` classification acts only with strong, recorded evidence. Otherwise the
manifest is `unknown` and requires review. A caller must separately approve a
model run and its hard spend cap.

## Consequences

- Source genre, route, confidence, evidence and QA state are auditable.
- Faithful text and source images remain local and attached to every unit.
- Adding a new source genre requires an explicit route and tests, rather than
  changing a prompt in place.
- The router itself has zero API cost and makes no network calls.
