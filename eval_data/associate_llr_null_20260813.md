# What a shared associate is actually worth — measured, 2026-08-13

`measure_associate_null.py`, corpus `production/luna_v3/assembled_deduped`
(6,735 entries, 39,453 mentions). Nothing here changes a weight.

## The claim under test

HANDOVER §14 argues the shared-associate term is over-valued. It weights a shared
associate at `−ln(p)`, which is the log-likelihood ratio only under a null of
*two unrelated strangers*. §14's argument: in a parish register the realistic
alternative is *two different but related people*, relatives share associates
constantly, so the evidence is worth far less than `−ln(p)`. Its recommendation
was to "fix the associate LLR's null hypothesis, not move the bar."

**Measured, that recommendation is wrong as stated** — or rather, right about a
configuration and wrong about the corpus.

## The measurement

Two guaranteed-negative classes, because one would have answered the wrong
question. Within-entry pairs are co-participants *by construction* — a husband
and wife at one baptism — so they share associates at a rate no ordinary
candidate pair reaches. Cross-entry both-sacrament-principals (you are baptized,
born and buried once, so two principals of the same sacrament type in different
entries cannot be one person) are drawn from the population the scorer meets.

| class | pairs | P(share ≥1 associate) |
|---|---:|---:|
| NEG within-entry (co-participant) | 69,811 | **0.6214** |
| NEG cross-entry both-principals | 400,000 (capped) | **0.0011** |
| POS same record transcribed twice | 80 | 1.0000 |

Implied weight for one shared associate:

| null | empirical LLR | model assigns |
|---|---:|---:|
| co-participant | **+0.48 nats** | +5.49 mean |
| cross-entry | **+6.78 nats** | +5.49 mean |

## The control, which is what makes this trustworthy

The obvious objection is that cross-entry principals barely share because they
barely *have* associates. The opposite is true:

| class | mean \|associates\| | share with ≥1 |
|---|---:|---:|
| all mentions | 1.36 | 72.3% |
| within-entry | 1.36 | 72.6% |
| cross-entry both-principals | **2.81** | **92.2%** |

They are **twice as densely embedded as the corpus average and still almost
never share**. So 0.0011 is not a sparsity artifact, and if anything it is a
generous (high) estimate of the sharing rate among different people.

## What this actually says

1. **In the population the scorer meets, the weight is right — slightly
   conservative.** A shared associate is worth about **+6.8 nats**; the model
   gives ~5.5 and caps at 7.0. §14's "worth far less than −ln(p)" does not hold
   here.
2. **The over-valuation is real but confined to co-participants**, where the
   evidence is worth ~+0.5 nats and the model pays ~+5.5 — a ~5-nat error.
3. **So a global de-weighting would be a mistake.** It is the natural reading of
   §14's recommendation and it would degrade the 99.9% of the population where
   the term is correctly valued, to fix a configuration the same-entry veto
   already catches. §14 was right that the defect is latent; it named the wrong
   repair.

## Why the conclusion survives the weakest input

`P(share | same) = 1.0` comes from 80 byte-identical duplicate records, which
agree on everything by construction — the positives are far too easy, and that
inflates both LLRs. But it inflates them **equally**, because it is the same
numerator in both. The finding is the *gap* between the two nulls:

    LLR_cross − LLR_within = ln(0.6214 / 0.0011) = 6.30 nats

which involves no positive class at all. If real cross-entry same-person pairs
share at only half the rate of duplicates, both LLRs drop by 0.69 nats and the
6.30-nat gap is unchanged.

**A second construction artifact, stated because it nearly became a finding.**
Duplicate positives share in the *same role* 100% of the time — necessarily, they
are copies. It is tempting to read the within-entry class (only 13.2% same-role,
so ~79% of its sharing is cross-role) as showing that cross-role sharing is
evidence *against* identity, and therefore that the current ×0.6 cross-role
discount has the sign wrong. **That inference is not supported.** A real person
appearing in two entries changes role constantly — child at baptism, spouse at
marriage — and this positive class structurally cannot exhibit that. Measuring
it needs cross-entry same-person labels, which is exactly what the blocked-pair
sample is for and why recall is still unmeasured.

## What to do instead

The latent risk §14 identifies is real: when two registers cover one parish,
co-participants start appearing in *different* entries and the same-entry veto
stops firing. But the discriminating feature is not how much they share — it is
that they **participate in one event**. The repair is a same-event detector
(same register, same date, same event id, different entry), not a change to the
associate weight.

That is a different piece of work from re-weighting, and it is cheap: the
signal is already on the mention (`_entry`, `_year`, `_sacraments`).

## Reproduce

```bash
python measure_associate_null.py --out <somewhere outside production/>
```

Writes nothing under a locked label path, and refuses if asked to.
