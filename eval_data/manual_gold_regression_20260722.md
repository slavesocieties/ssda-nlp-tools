# Manual segmentation-gold regression — 2026-07-22

Re-ran every available manually prepared input/output fixture after the
heterogeneous-volume routing work. This is an offline deterministic check: no
model calls and no source text changes.

| Volume | Gold records | Result | Coverage represented |
|---|---:|---|---|
| 65858 | 10 | pass | Portuguese register boundaries |
| 420550 | 8 | pass | Colombian Spanish boundary formula |
| 260950 | 13 | pass | Portuguese 1910 baptism records |
| 544367 | 5 | pass | trailing partial record retained |
| 740018 | 11 | pass | `demil` year form and cross-page behavior |

**Result: 47/47 manual-gold records located.**

The supplied `breaking fix examples` ZIP contains these same five fixture
pairs; no additional manual examples were available locally or in the current
remote branch at verification time. Further manual labeling is not needed for
the existing sacramental segmentation route unless it targets an unrepresented
genre, layout, language, or a specific observed failure.
