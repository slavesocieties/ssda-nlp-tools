# Daniel's grandparent-side ruling is implemented and inert — 2026-08-13

Found while reading the same-name pairs from
[latent_coparticipant_20260813.md](latent_coparticipant_20260813.md), not by
looking for it.

## The ruling

HANDOVER §5, Daniel 2026-08-10, answer 2 of four:

> *"Maternal/paternal grandparents should be labeled differently when
> extracted"* → done upstream. **97.8%** of entries with a grandparent edge
> already state the side in the transcription; only the schema discarded it.

The scorer implements it. `MAX_HOLDERS` carries the finer capacities:

```python
"grandparent": 4, "maternal grandparent": 2, "paternal grandparent": 2
```

## The data does not

Counting `relationship_type` across the delivered corpora:

| corpus | grandparent edges | sided |
|---|---:|---:|
| `production/luna_v3/assembled` | 2,268 | **0 (0.0%)** |
| `production/luna_v3/assembled_deduped` — **the default** | 2,235 | **0 (0.0%)** |
| `production/sided_7vol` | 2,268 | **2,080 (91.7%)** |

`sided_7vol` holds 1,696 maternal + 384 paternal + 188 still unsided, over the
full 6,794 entries. So the backfill ran, it worked, and its output was written —
and then nothing was pointed at it.

`backfill_grandparent_sides.py` writes only with `--out` and deliberately has no
in-place mode, because the assembled volumes are delivered artifacts. That is the
right design. The consequence is that promoting its output is a separate act that
nobody performed.

## What that costs

`MAX_HOLDERS["maternal grandparent"]` and `["paternal grandparent"]` **can never
fire on the delivered corpus**, because no edge in it ever carries a side. Every
grandparent falls under the pooled capacity of 4.

A role contradicts only when distinct holders exceed what one person can have, so
the ruling's whole effect is to tighten that test from *4 grandparents pooled* to
*2 maternal and 2 paternal*. Two mentions naming three distinct maternal
grandparents are a contradiction under the ruling and are silently fine without
it.

This is the "flag no code consumes" failure from §9 rule 1, one level out: not a
flag nothing reads, but a **capacity no data can trigger**. It produces no error,
no warning, and a merge result that looks entirely normal — and every accuracy
figure quoted since the ruling was implemented was measured without it.

The case that exposed it, from entry `176899-0217-A-02`:

```
"Abuela paterna Isabel ... materna Isabel"
```

Two grandmothers, both named Isabel, distinguished in the transcription by side
alone. In the delivered corpus both are plain `grandparent`, and the distinction
Daniel asked for is gone.

## Caveat on the 97.8%

§5's figure is the share of *entries* stating a side; 91.7% here is the share of
*edges* that got one. Different units, not a contradiction — but they are not the
same number and should not be quoted as one.

## What to do

Run the corpus A/B — `sided_7vol` against `assembled`, same everything else — and
see what the finer capacities actually change. That is the number that says
whether promoting the sided corpus matters or is bookkeeping. Until then nobody
should claim the ruling is in effect.

---

## The A/B: promoting the sided corpus is worth 1 identity

`sided_7vol` against `assembled`, same scorer, same blocking, same everything
else, both 6,794 entries / 39,697 mentions.

| | unsided (delivered) | sided |
|---|---:|---:|
| identities | 33,253 | **33,254** (+1) |
| auto-merges | 6,444 | **6,443** (−1) |
| review pairs | 641,669 | 641,666 (−3) |
| merged identities | 1,187 | 1,186 |
| every veto count | identical | identical |

**The direction is exactly right and the magnitude is nearly nothing.** A tighter
capacity means fewer merges and therefore more identities, and that is precisely
what moved: one merge prevented, one identity gained. The mechanism Daniel asked
for works; it just almost never has occasion to fire.

Why: the capacity contradicts only when two mentions name **more than two
distinct maternal** (or paternal) grandparents between them. Most entries name at
most two grandparents in total, so the tighter bound is reachable in a handful of
cases at this scale.

### What follows

- **The finding stands**: the ruling is implemented and not in force. That is
  worth fixing because a rule believed to be active and silently absent is a
  liability independent of its size.
- **The cost of the inertness is ~1 identity**, so nothing measured to date is
  materially wrong because of it. Nobody needs to re-run anything.
- **Promote it anyway, deliberately.** It is cheap, it is correct, and the case
  it guards against — a record naming grandparents on both sides — gets more
  common as volumes with fuller genealogies arrive. But it changes the delivered
  corpus, so it belongs with the dedupe and blocking decisions, not folded into
  something else.

This is §8's "uncomfortable summary" again, from a third direction: the
refinement is correct, it is worth keeping, and it moves under 1% of the corpus
while 99.82% of the archive remains unprocessed.
