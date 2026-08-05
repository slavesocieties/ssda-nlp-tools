"""Merge guards that need no human labels.

The rules here are the ones that are true about people rather than about names,
so they can be enforced before Daniel's 1,000 pair labels arrive and they do not
compete with whatever those labels teach.
"""
from ssda_nlp_tools.disambiguate import (_UnionFind, _third_party_same,
                                         birth_window, corroborating_signals,
                                         lifespan_conflict, name_variants,
                                         _clusters_attributes_compatible,
                                         attributes_contradict,
                                         _clusters_share_an_entry,
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


# --- same entry, two people (the third guard defeated by transitivity) ------- #

def test_two_mentions_from_one_entry_never_share_an_identity():
    """The module has always refused this PAIRWISE -- "the extractor already
    separated them" -- but union-find routes around it: A from entry E merges
    with X from entry F, then B from entry E merges with X too, and A and B end
    up together without ever being compared. 35 delivered identities were in
    that state."""
    uf = _UnionFind(3)
    entries = {0: {"E"}, 1: {"F"}, 2: {"E"}}
    assert not _clusters_share_an_entry(uf, 0, 1, entries)   # E + F is fine
    uf.union(0, 1)
    root = uf.find(0)
    entries[root] = {"E", "F"}
    assert _clusters_share_an_entry(uf, root, 2, entries), "E is already in there"


def test_unrelated_entries_still_merge():
    uf = _UnionFind(2)
    assert not _clusters_share_an_entry(uf, 0, 1, {0: {"E"}, 1: {"F"}})


def test_it_does_not_fragment_a_legitimate_recurring_person():
    """A priest signs one entry after another, so his mentions come from
    DISTINCT entries and none of them collide. Measured: Miguel Llopiz stays at
    887 mentions, Jose Ramirez Moreno at 444, unchanged."""
    uf = _UnionFind(4)
    entries = {i: {f"E{i}"} for i in range(4)}
    for i in range(1, 4):
        assert not _clusters_share_an_entry(uf, 0, i, entries)
        uf.union(0, i)
        r = uf.find(0)
        entries[r] = set().union(*(entries.get(k, set()) for k in range(i + 1)))
    assert len(entries[uf.find(0)]) == 4


# --- chronology: Daniel, 2026-08-03 ----------------------------------------- #

def test_an_adult_in_1840_is_not_an_infant_in_1878():
    """Daniel: "nonsensical pairings like children born after a same-name adult
    died". The cause was that dates were used as PROXIMITY -- 38 years is inside
    the 40-year window, so it counted as corroboration rather than as a
    contradiction."""
    parent_1840 = {"name": "Juana", "_year": 1840,
                   "_ctx": {("child", "asuncion")}}
    infant_1878 = {"name": "Juana", "_year": 1878, "age": "infant", "_ctx": set()}
    assert lifespan_conflict(parent_1840, infant_1878)
    assert "date-overlap" not in corroborating_signals(parent_1840, infant_1878)


def test_an_infant_grows_into_an_adult():
    """The same two labels 25 years apart are one ordinary life."""
    infant = {"name": "Maria", "_year": 1850, "age": "infant", "_ctx": set()}
    adult = {"name": "Maria", "_year": 1875, "age": "adult", "_ctx": set()}
    assert lifespan_conflict(infant, adult) is None
    assert attributes_contradict(infant, adult) is None, "growing up is not a contradiction"


def test_infant_and_adult_in_the_same_year_still_contradict():
    infant = {"name": "Maria", "_year": 1850, "age": "infant"}
    adult = {"name": "Maria", "_year": 1851, "age": "adult"}
    assert attributes_contradict(infant, adult) == "age"


def test_events_more_than_a_lifetime_apart():
    a = {"name": "Juan", "_year": 1780, "_ctx": set()}
    b = {"name": "Juan", "_year": 1900, "_ctx": set()}
    assert lifespan_conflict(a, b)


def test_being_a_parent_bounds_the_birth_year_without_any_age_field():
    m = {"name": "Ana", "_year": 1840, "_ctx": {("child", "pedro")}}
    lo, hi = birth_window(m)
    assert hi <= 1840 - 14


def test_undated_mentions_are_never_refused_on_chronology():
    """Absence of a date is not evidence of separation."""
    a = {"name": "Juan", "_year": None, "age": "infant", "_ctx": set()}
    b = {"name": "Juan", "_year": 1850, "age": "adult", "_ctx": set()}
    assert lifespan_conflict(a, b) is None


# --- third-party names ------------------------------------------------------ #

def test_a_shared_given_name_does_not_make_one_enslaver():
    """Given names come from a tiny pool. "francisco pulgason" and "francisco
    challi" were treated as one man, handing a false merge two signals."""
    assert not _third_party_same("francisco pulgason", "francisco challi")


def test_the_estate_surname_rule_still_holds():
    assert not _third_party_same("hanna macqueen", "rachael macqueen")


def test_short_forms_still_match():
    assert _third_party_same("rachael", "rachael macqueen")
    assert _third_party_same("francisco", "francisco pulgason")


# --- name variants on merge (Daniel, 2026-08-03) ---------------------------- #

def test_a_clergy_cluster_collapses_to_the_modal_spelling():
    """"This is going to happen a lot with clergy, who will appear many times,
    in which case we can just go with the name that appears most frequently"."""
    members = ([{"name": "Jose Domingo Sanchez y Fleites"}] * 40
               + [{"name": "Jose Domingo Sanchez y Fleitas"}] * 2)
    r = name_variants(members, "Jose Domingo Sanchez y Fleites")
    assert r["policy"] == "modal"
    assert r["retained"] == ["Jose Domingo Sanchez y Fleites"]
    rare = [v for v in r["variants"] if v["count"] == 2][0]
    assert rare["role"] == "likely_mistranscription"


def test_a_weak_merge_retains_every_variant():
    """"in cases where we have a match based on only two or three very closely
    aligned data points... retain all name variants unless one is obviously
    mistranscribed"."""
    members = [{"name": "Maria Llopiz"}, {"name": "Maria Llopis"},
               {"name": "Maria Josefa Llopiz"}]
    r = name_variants(members, "Maria Llopiz")
    assert r["policy"] == "retain_all"
    assert len(r["retained"]) == 3


def test_a_fuller_name_is_a_variant_not_an_error():
    """"Maria Josefa Llopiz" is not a misreading of "Maria Llopiz"; it is more
    of the same person's name."""
    members = [{"name": "Maria Llopiz"}] * 3 + [{"name": "Maria Josefa Llopiz"}]
    r = name_variants(members, "Maria Llopiz")
    fuller = [v for v in r["variants"] if "Josefa" in v["name"]][0]
    assert fuller["role"] == "distinct_variant"


def test_a_spelling_in_real_circulation_is_not_called_a_mistranscription():
    """Near-equal counts mean both spellings were in use, not that one is a slip."""
    members = [{"name": "Maria Llopiz"}] * 10 + [{"name": "Maria Llopis"}] * 9
    r = name_variants(members, "Maria Llopiz")
    other = [v for v in r["variants"] if v["name"] == "Maria Llopis"][0]
    assert other["role"] == "distinct_variant"


def test_nothing_is_ever_discarded():
    """Whatever the policy, every spelling stays in the record with its count."""
    members = [{"name": "Juan"}] * 30 + [{"name": "Juan N."}] + [{"name": "Juan Perez"}]
    r = name_variants(members, "Juan")
    assert {v["name"] for v in r["variants"]} == {"Juan", "Juan N.", "Juan Perez"}
    assert sum(v["count"] for v in r["variants"]) == 32


def test_the_lifespan_ab_switch_actually_reaches_the_guard():
    """A control run that silently keeps the guard on reports "no difference",
    which is the most misleading result available. The first version of
    --no-lifespan set an argparse flag nothing consumed, and both runs came back
    byte-identical with blocked-lifespan=1,416 in each."""
    import ssda_nlp_tools.disambiguate as D
    a = {"name": "Juana", "_year": 1840, "_ctx": {("child", "asuncion")}}
    b = {"name": "Juana", "_year": 1878, "age": "infant", "_ctx": set()}
    assert surname_tier_allows(a, b) == (False, "blocked-lifespan")
    D.LIFESPAN_GUARD_ENABLED = False
    try:
        assert surname_tier_allows(a, b)[1] != "blocked-lifespan"
    finally:
        D.LIFESPAN_GUARD_ENABLED = True
    assert surname_tier_allows(a, b) == (False, "blocked-lifespan")


def test_capitalisation_alone_is_not_an_attribute_conflict():
    """'Cleric' vs 'cleric' was reported as a conflict on 7 of the 12 largest
    clusters, all priests. Decisions were never affected (attributes_contradict
    normalises via _val); only the identity report diverged from the scorer."""
    from ssda_nlp_tools.disambiguate import _merge_attributes
    merged, conflicts = _merge_attributes([{"occupation": "Cleric"},
                                           {"occupation": "cleric"}])
    assert conflicts == {}
    assert merged["occupation"] == "Cleric"      # first surface form kept


def test_a_genuine_attribute_conflict_is_still_reported():
    from ssda_nlp_tools.disambiguate import _merge_attributes
    _, conflicts = _merge_attributes([{"occupation": "Cleric"},
                                      {"occupation": "Carpenter"}])
    assert set(v.lower() for v in conflicts["occupation"]) == {"cleric", "carpenter"}


def test_whitespace_is_normalised_too():
    from ssda_nlp_tools.disambiguate import _merge_attributes
    _, conflicts = _merge_attributes([{"occupation": " Cleric "},
                                      {"occupation": "CLERIC"}])
    assert conflicts == {}


# --- Daniel 2026-08-05: only paternal surnames are definitive ----------------

def test_a_dropped_maternal_surname_no_longer_reads_as_a_different_family():
    """Spanish names run paternal then maternal, so the LAST token is the
    maternal surname when both are present. Comparing it made "Juan Gonzalez"
    and "Juan Gonzalez Torre" different families and refused 534 name pairs."""
    from ssda_nlp_tools.disambiguate import surname_affinity
    assert surname_affinity("Juan González", "Juan González Torre") == 1.0
    assert surname_affinity("José Domingo Sánchez",
                            "José Domingo Sánchez y Fleites") == 1.0


def test_genuinely_different_families_still_disagree():
    from ssda_nlp_tools.disambiguate import surname_affinity
    assert surname_affinity("Juan González", "Juan Corrales") < 0.7


def test_an_ethnonym_is_neither_auto_merged_nor_auto_refused():
    """Daniel: ethnonyms behave like surnames in many contexts, but these
    'should not be merged or discarded automatically. Further context is
    required.' So the surname bar must stop REFUSING them, while the
    corroboration requirement still decides."""
    from ssda_nlp_tools.disambiguate import surname_affinity, surname_tier_allows
    assert surname_affinity("María de la Concepción",
                            "María de la Concepción Lucumí") == 1.0
    a = {"name": "María de la Concepción", "_entry": "E1", "_year": 1850}
    b = {"name": "María de la Concepción Lucumí", "_entry": "E2", "_year": 1852}
    allowed, tier = surname_tier_allows(a, b)
    assert not allowed, "no context, so it must not merge on the name alone"


def test_prefix_rule_does_not_fire_on_equal_length_names():
    from ssda_nlp_tools.disambiguate import surname_affinity
    assert surname_affinity("Ana Pérez Soto", "Ana Pérez Vega") < 1.0
