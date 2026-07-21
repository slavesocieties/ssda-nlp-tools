# Breaking-fix examples (2026-07-20) — segmentation result & diagnosis

Five paired gold examples from Daniel (`_sample.json` input → `_sample_output.json`
gold). Three seen before (65858, 420550, 740018), two new (**260950, 544367**).

## Result: perfect recall; two supplied references omit trailing partials

| stem | gold | pred | records matched | note |
|---|---|---|---|---|
| 65858 | 10 | **11** | 10/10 | +1 trailing partial (real 21-May record gold omits) |
| 260950 | 13 | 13 | 13/13 | ✓ exact (Portuguese, 1910) |
| 420550 | 8 | 8 | 8/8 | ✓ exact |
| 544367 | 5 | **6** | 5/5 | +1 real trailing partial (No. 546, Juan Alberto; reference omits it) |
| 740018 | 11 | 11 | 11/11 | ✓ exact |

**Every reference record boundary is found (47/47 recall).** In both 65858 and
544367, the apparent extra segment is a genuine record beginning at the bottom
of the final supplied page. The reference stops before that trailing partial.

- **544367**: page 0107's No. 543 continues and closes at the top of page 0108.
  Page 0108 then contains Nos. 544 and 545 and visibly begins **No. 546, Juan
  Alberto**. No. 546 runs off the supplied image, so keeping it with
  `partial: true` is correct; the five-entry reference simply omits it.
- **65858**: the extra record is `Aos vinte e hum dias` (21 May) — a *different*
  date from the preceding `Aos vinte dias` (20 May). It is a **genuine new
  record** that runs off the last provided page (`partial:True`). Gold omits it;
  the segmenter is arguably *more* complete here.

## Why these are not dropped in the segmenter

Both extras contain anchored new-record evidence and distinct principals. A
rule that forces prediction count to match an incomplete reference would delete
real archival records.

The segmenter therefore keeps both trailing records and flags them `partial`—
the safe behavior under the confirmed "never drop partials" convention. A
regression test now runs the real 544367 pages through segmentation and
principal-aware QA and verifies that all six distinct records survive with no
duplicate flag.

## Open convention question for Daniel

The current behavior follows the 2026-07-16 "never drop partials" decision.
If supplied reference sheets intentionally exclude trailing partials, scoring
should mark that convention explicitly rather than calling correct segments
false positives.
