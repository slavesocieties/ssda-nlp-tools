"""Merge guards that need no human labels.

The rules here are the ones that are true about people rather than about names,
so they can be enforced before Daniel's 1,000 pair labels arrive and they do not
compete with whatever those labels teach.
"""
from ssda_nlp_tools.disambiguate import (_UnionFind,
                                         _clusters_attributes_compatible,
                                         attributes_contradict,
                                         _would_close_ancestry_cycle,
                                         surname_tier_allows)




# --- mutually exclusive attributes (found 2026-08-01 by validating the graph) - #

def test_infant_and_adult_are_not_the_same_person():
    a = {"name": "Maria de la Cruz", "age": "infant"}
    b = {"name": "Maria de la Cruz", "age": "adult"}
    assert attributes_contradict(a, b) == "age"


def test_free_and_enslaved_are_not_the_same_person():
    a = {"name": "Francisco Carabali", "free": True}
    b = {"name": "Francisco Carabali", "free": False}
    assert attributes_contradict(a, b) == "free"


def test_phenotype_variation_is_NOT_a_contradiction():
    """A scribe may write morena once and parda another time for the same woman.
    Only oppositions no reading of the register can reconcile are hard blocks;
    everything softer stays in the existing conflict COUNT."""
    a = {"name": "Maria", "phenotype": "morena"}
    b = {"name": "Maria", "phenotype": "parda"}
    assert attributes_contradict(a, b) is None


def test_a_missing_value_never_contradicts():
    assert attributes_contradict({"name": "Juan"}, {"name": "Juan", "age": "adult"}) is None
    assert attributes_contradict({"name": "Juan", "free": None},
                                 {"name": "Juan", "free": True}) is None


def test_the_contradiction_block_precedes_the_clergy_shortcut():
    """The clergy shortcut merges a priest across consecutive entries on his name
    and role. It must not rescue a pair that cannot be one person."""
    a = {"name": "Jose Domingo Sanchez", "occupation": "cleric", "age": "infant",
         "_entry": "29597-0001-A-01", "_ctx": set()}
    b = {"name": "Jose Domingo Sanchez", "occupation": "cleric", "age": "adult",
         "_entry": "29597-0001-A-02", "_ctx": set()}
    allowed, tier = surname_tier_allows(a, b)
    assert not allowed and tier == "blocked-contradiction-age"


def test_cluster_guard_catches_what_the_pairwise_one_cannot():
    """The contradiction is TRANSITIVE. An infant merges with a plausible adult,
    that adult merges with another adult, and the infant is never compared to the
    last one. Measured on the corpus: the pairwise check blocked 15 merges and
    left 21 impossible identities; the cluster check left 0."""
    uf = _UnionFind(3)
    sides = {0: {"age": {0}}, 1: {}, 2: {"age": {1}}}
    assert _clusters_attributes_compatible(uf, 0, 1, sides)      # infant + unknown
    uf.union(0, 1)
    root = uf.find(0)
    sides[root] = {"age": {0}}
    assert not _clusters_attributes_compatible(uf, 0, 2, sides)  # now infant vs adult


def test_compatible_clusters_still_merge():
    uf = _UnionFind(2)
    sides = {0: {"age": {0}}, 1: {"age": {0}}}
    assert _clusters_attributes_compatible(uf, 0, 1, sides)


# --- ancestry cycles (the class the attribute guard cannot see) -------------- #

def test_merging_along_a_descent_edge_is_refused():
    """Nobody is their own parent. If a descent path already runs between two
    clusters, merging the endpoints closes it into a loop."""
    uf = _UnionFind(2)
    parents = {0: {1}}                      # mention 0 is parent of mention 1
    assert _would_close_ancestry_cycle(uf, 0, 1, parents)
    assert _would_close_ancestry_cycle(uf, 1, 0, parents), "both directions"


def test_unrelated_clusters_still_merge():
    uf = _UnionFind(3)
    assert not _would_close_ancestry_cycle(uf, 0, 2, {0: {1}})


def test_a_grandparent_chain_is_caught():
    """Two generations deep: 0 -> 1 -> 2. Merging 0 with 2 makes someone their
    own grandmother."""
    uf = _UnionFind(3)
    assert _would_close_ancestry_cycle(uf, 0, 2, {0: {1}, 1: {2}})


def test_the_search_is_depth_bounded():
    """Descent chains in a register are shallow, and an unbounded search over a
    22,000-cluster graph would dominate the merge loop."""
    uf = _UnionFind(8)
    chain = {i: {i + 1} for i in range(7)}
    assert _would_close_ancestry_cycle(uf, 0, 3, chain, depth=4)
    assert not _would_close_ancestry_cycle(uf, 0, 7, chain, depth=2)


def test_already_merged_clusters_are_not_reported_as_cycles():
    uf = _UnionFind(2)
    uf.union(0, 1)
    assert not _would_close_ancestry_cycle(uf, 0, 1, {0: {1}})


def test_the_bernal_case():
    """Ramona Bernal is recorded as the PARENT of Rosalia Bernal on folio 0017
    and as her CHILD on folio 0195. Both are parda, both from Trinidad, no
    attribute conflict anywhere -- so attributes_contradict sees nothing and
    only the descent direction reveals it."""
    ramona_17, rosalia_17, ramona_195, rosalia_195 = 0, 1, 2, 3
    a = {"name": "Ramona Bernal", "phenotype": "parda", "origin": "Trinidad"}
    b = {"name": "Rosalia Bernal", "origin": "Trinidad"}
    assert attributes_contradict(a, b) is None, "attributes cannot see this"

    uf = _UnionFind(4)
    parents = {ramona_17: {rosalia_17}, rosalia_195: {ramona_195}}
    uf.union(ramona_17, ramona_195)                 # the two Ramonas merge
    root = uf.find(ramona_17)
    parents[root] = {rosalia_17}
    parents.setdefault(rosalia_195, set()).add(root)
    # now merging the two Rosalias would close the loop
    assert _would_close_ancestry_cycle(uf, rosalia_17, rosalia_195, parents)
