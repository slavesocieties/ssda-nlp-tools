# Which weights actually fire — audit, 2026-08-13

`audit_inert_weights.py`, 250,000 of 14,472,013 candidate pairs (1.7%) drawn
from the real blocking keys, `assembled_deduped`.

Built to generalise the grandparent finding: a term can be inert because the data
never presents its condition, and the model still returns a confident number.

## Result: nothing else is inert

| term | fires | % of scored |
|---|---:|---:|
| name (rarity × similarity) | 246,530 | **100.00%** |
| year gap | 196,886 | 79.86% |
| **volumes-never-coexist** | 128,329 | **52.05%** |
| place:institution | 95,743 | 38.84% |
| place:none | 83,355 | 33.81% |
| attrs-agree ×1 | 31,972 | 12.97% |
| place:country | 30,921 | 12.54% |
| both-clergy | 28,540 | 11.58% |
| place:city | 18,596 | 7.54% |
| place:state | 17,915 | 7.27% |
| conflict:spouse | 4,969 | 2.02% |
| conflict:parent | 2,705 | 1.10% |
| conflict:godparent | 1,034 | 0.42% |
| disjoint-networks | 964 | 0.39% |
| conflict:enslaver | 261 | 0.11% |
| conflict:grandparent | 1 | 0.00% |

Every weight in the vocabulary fires. The maternal/paternal grandparent
capacities remain the only inert thing found, and they are consistent with this
table: `conflict:grandparent` fires **once in 250,000 pairs**, so the pooled
capacity barely engages either.

Two things worth acting on:

**`W_VOLUMES_NEVER_COEXIST = −1.5` fires on 52% of scored pairs.** It is the most
broadly applied non-name weight in the model, and §4 lists it as *"still a prior;
not separately measurable here"*. A chosen number pushing half the candidate pool
apart deserves more scrutiny than terms that fire on 2%. If anything in the
scorer is worth calibrating next, it is this rather than the conflict weights.

**The two caps behave completely differently.**

| cap | binds on |
|---|---:|
| `MAX_NAME_LLR = 5.5` | **95.9%** of name terms |
| `MAX_LLR_PER_ASSOCIATE = 7.0` | **0.6%** of shared associates |

§4 calls the name cap "the biggest single distortion" and this confirms it —
95.9% by term, and measured directly against the corpus, 100% of multi-token
names and 98.8% of mentions. But the associate cap is **not** a distortion: it
binds on 2 of 311. They are described together in §4 and behave nothing alike, so
raising one is not the same kind of decision as raising the other.

## The tool was wrong four times first

Each failure is the shape the project keeps hitting, and each was caught only by
an external contradiction rather than by the tool noticing:

1. **False zero.** Cap detection parsed `rarity ([\d.]+)` from a label that
   reads `name~0.55 rarity` — the number is in the tuple, not the string. It
   reported the cap binding on **0.0% of 246,517 terms**. Caught only because
   §4 claims 100% and the two could not both be true.
2. **Buried tally.** The shared-associate label carries the associate's *name*,
   so an unnormalised count produced one row per person and hid the finding
   under thousands of singletons.
3. **False inert.** Prefix matching declared `conflict` dead while
   `network:conflict:enslaver` was firing 261 times in the same output — the
   labels are namespaced.
4. **Invented vocabulary.** It reported `enslaver-gap` NEVER FIRED. There is no
   such label: `ENSLAVER_TAU_YEARS` decays the penalty *inside*
   `conflict:enslaver`. A hand-written list of expected labels turns a name I
   made up into a finding about the model.

(1) and (4) are the dangerous ones, because both produce a *confident negative* —
"this weight does nothing" — which is exactly the claim an audit exists to make
and the hardest to disbelieve. The vocabulary is now derived from the labels the
code constructs, and the cap is measured against the model rather than its own
display string.
