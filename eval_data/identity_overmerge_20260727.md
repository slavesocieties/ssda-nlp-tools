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

(1) and (2) are complementary. Neither is implemented yet: this is an
algorithmic change to the identity layer and should be a deliberate decision,
not a quiet patch — particularly since Daniel has said he ultimately wants a
trained probabilistic model here, which would supersede the heuristic entirely.

## Recommended disclosure

Daniel should be told before he looks at the graph. The extraction and
segmentation numbers reported to him are unaffected; the identity and network
figures (16,230/16,800 identities, 778 cross-register links, hub rankings) should
be treated as provisional until this is fixed.
