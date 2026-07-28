# Identity resolution is over-merging at corpus scale

Found 2026-07-27 while regenerating the cross-volume graph. **Pre-existing, not
caused by the context-blocking change** — blocking slightly reduces it
(16,230 → 16,800 identities). It is now visible because the corpus is complete.

## The evidence

Six identities hold **3,828 of 27,631 mentions (14% of the corpus)**:

| identity | mentions | distinct surnames | canonical name |
|---|---:|---:|---|
| CORPUS-13736 | 1,007 | 24 | Miguel Alejo de P. |
| CORPUS-11541 | 964 | **98** | Antonio Fernández Sacendia |
| CORPUS-15515 | 934 | **209** | Francisco Fernández María de Regla Quintana |
| CORPUS-3724 | 446 | **1** | José Ramírez y Moreno |
| CORPUS-15762 | 274 | 69 | María Josefa de la Luz Alvarado |
| CORPUS-8120 | 203 | 50 | José Rafael de Villavicencio |

A person does not have 209 surnames. CORPUS-3724 is the control: 446 mentions
across **one** surname is exactly what a recurring officiating priest should
look like, and it is almost certainly correct. The others are not.

Inside CORPUS-13736, the name distribution shows the mechanism:

```
193  Miguel Llopiz      <- same person, scribal variants
130  Miguel Llopis
 27  Miguel Llepiz
 11  Miguel López       <- different people, chained in
  9  Miguel Alepuz
  7  Miguel Yépez
  3  Miguel Valdés
```

## Mechanism

Union-find merges transitively. Each individual link is defensible
(`Llopiz`~`Llopis` is strong, `Llopis`~`López` is plausible), but the transitive
closure joins `Llopiz` to `Valdés`, who share nothing but a given name. In a
corpus where a handful of given names cover most of the population, one weak
link anywhere in a chain collapses hundreds of distinct people into a single
node.

## Why the existing guard did not catch it

`disambiguate.py` records each cluster's weakest internal edge and flags
clusters whose cohesion falls below `auto_threshold`. It reported
**flagged_clusters: 0**. The check inspects edges that were actually scored, and
in a chain every scored edge is strong — the unrelated endpoints are never
compared, so no weak edge exists to find. The guard is structurally blind to the
failure mode it was written for.

## Impact

- The social graph's largest hubs are artefacts. Degree, centrality, and
  "people linked across registers" are all inflated for these nodes.
- The 778 cross-register links reported earlier include some of these merges.
- Per-person review inherits it: a screen for a 934-mention cluster is not a
  reviewable question.
- It does **not** affect the extracted records themselves — faithful text,
  normalized text, people and events per entry are unaffected. This is purely
  the identity layer on top.

## Candidate fixes, in the order I would try them

1. **Require surname compatibility for transitive merges.** Two mentions may
   merge on a given name alone only with corroborating context; joining two
   clusters should additionally require their surnames to be compatible.
2. **Cohesion over the cluster, not the chain.** After clustering, score a
   sample of *non-adjacent* pairs within each cluster and split where the
   average falls below threshold. This is what the current guard was trying to
   do and does not.
3. **Cap cluster size and route overflow to review.** Crude, but it makes the
   failure loud instead of silent.

## (1) implemented and measured

A cluster-level surname guard now refuses an auto-merge when both clusters carry
surnames and none of them are compatible. It compares the surnames ALREADY
ACCUMULATED IN BOTH CLUSTERS, not just the pair in hand, which is what catches
the transitive case. Refused merges become review items rather than vanishing.

| cluster | before | after |
|---|---|---|
| Francisco Fernández… | 934 mentions / **209 surnames** | 254 / **5** |
| Antonio Fernández Sacendia | 964 / 98 | 340 / 20 |
| Miguel (Llopiz family) | 1,007 / 24 | 923 / **11** |
| **José Ramírez y Moreno** (control) | **446 / 1** | **446 / 1** — unchanged |

The control matters most: the legitimate recurring priest, 446 mentions under a
single surname, is preserved exactly. The guard removes chained noise without
touching real recurrence.

Corpus effects: identities 16,800 → 18,143 (+1,343 split apart), 88,743 merges
refused, review queue 612,495 → 701,238 (+14%, because a refused merge becomes a
review item rather than disappearing).

**Not fully fixed.** The Miguel cluster is still 923 mentions across 11
surname variants (Llopiz/Llopis/Llepiz/Llepico…). Those are mutually compatible
under the name-similarity rule, so the guard permits them by design. Whether
that is one much-recorded priest or several similarly-named men is a
palaeographic judgement, not something the heuristic can settle — it is exactly
the kind of case for the trained model Daniel described.

Fix (2), cohesion over non-adjacent pairs within a cluster, remains
unimplemented and would be the next lever.

## Recommended disclosure

Daniel should be told before he looks at the graph. The extraction and
segmentation numbers reported to him are unaffected; the identity and network
figures (16,230/16,800 identities, 778 cross-register links, hub rankings) should
be treated as provisional until this is fixed.
