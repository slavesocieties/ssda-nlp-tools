

# --- contradictory-roles reporting -----------------------------------------

def _edges(rows):
    return [{"source": s, "target": t, "type": ty} for s, t, ty in rows]


def test_a_contradiction_is_reported_once_not_once_per_direction():
    """by_pair is keyed on the ORDERED pair, so A-B and B-A both matched and the
    count read 8 where there were 5 distinct pairs."""
    from validate_graph import check
    nodes = [{"id": "A"}, {"id": "B"}]
    res = check(nodes, _edges([("A", "B", "parent"), ("A", "B", "spouse"),
                                  ("B", "A", "child"), ("B", "A", "spouse")]))
    assert len(res["contradictory_roles"]) == 1


def test_an_ordinary_parent_child_pair_is_NOT_a_contradiction():
    """The guard against the tidier-looking fix: unioning both directions makes
    A->B 'parent' plus B->A 'child' -- every real parent -- self-contradictory.
    That took the corpus count from 8 to 10,050."""
    from validate_graph import check
    nodes = [{"id": "A"}, {"id": "B"}]
    res = check(nodes, _edges([("A", "B", "parent"), ("B", "A", "child")]))
    assert res["contradictory_roles"] == []


def test_two_separate_contradicting_pairs_are_both_reported():
    from validate_graph import check
    nodes = [{"id": x} for x in "ABCD"]
    res = check(nodes, _edges([("A", "B", "parent"), ("A", "B", "spouse"),
                                  ("C", "D", "godparent"), ("C", "D", "parent")]))
    assert len(res["contradictory_roles"]) == 2
