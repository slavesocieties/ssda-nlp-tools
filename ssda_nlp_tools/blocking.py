"""Candidate generation that does not grow quadratically with the collection.

THE PROBLEM
-----------
Blocking on the phonetic name alone puts every "Maria" in one block, and the
pairs inside a block grow as the square of its size. Measured on 7 volumes:
39,697 mentions produce 24,035,951 in-block pairs. Projected to the 3,900
volumes in the collection that is ~7.46e12 pairs -- roughly 0.7 years on 24
cores, with one block holding ~2.28M mentions. More cores do not fix a quadratic.

THE OBSERVATION THAT MAKES IT TRACTABLE
---------------------------------------
`_shares_context` already discards 35.6% of those pairs AFTER enumerating them,
keeping a pair only when the records share a register, or name somebody in
common, or sit within 60 years. Those are exactly the conditions that make good
blocking keys, so the pairs can simply never be generated:

    (name, register)     same register
    (name, associate)    a person named in both entries
    (name, time bucket)  close in time

A pair is a candidate if it shares ANY key -- a disjunction of conjunctions.

WHY THE TIME KEY IS RESTRICTED, AND WHY THAT IS DANIEL'S RULE
-------------------------------------------------------------
The first two keys are naturally bounded: a register holds a bounded number of
people, and a specific associate name is rare. The time key is not -- "Maria,
1840s" is still enormous, because the corpus spans 271 years and bucketing by
time divides a name block by less than ten.

So the time key is emitted only for names rare enough that the resulting block
stays under `max_block`. For a common name, a pair from a different register
sharing nobody is then never a candidate. That is not a heuristic compromise; it
is Daniel's ruling, 2026-08-05:

    "In the absence of literally any other context, records should never be
     merged, however this should be so rare as to essentially not be worth
     discussing further."

Such a pair could not have merged anyway -- a bare common name is worth at most
MAX_NAME_LLR, which by construction cannot reach the auto-merge threshold. What
is dropped is work, not merges. `verify_blocking.py` measures that claim rather
than asserting it.

UNDATED MENTIONS
----------------
`_shares_context` keeps any pair where either side lacks a year, because an
undated entry must not be silently excluded from every comparison. They are 2.4%
of mentions here. They get the register and associate keys like everyone else,
plus a per-name key so an undated mention still meets dated ones of the same
name -- capped the same way, for the same reason.
"""
from __future__ import annotations

import collections
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

from .textmatch import phonetic_key

# Two mentions within this many years must share a time key. Matches the window
# `_shares_context` uses, so the candidate set is comparable.
YEAR_WINDOW = 60

# A block bigger than this is not worth its pairs: n(n-1)/2 comparisons to find
# merges that a bare common name cannot support anyway. 4,000 keeps the largest
# real block on this corpus intact while bounding the projection.
DEFAULT_MAX_BLOCK = 4000


def _time_keys(year: Optional[int]) -> Tuple[int, ...]:
    """Buckets such that any two years within YEAR_WINDOW share one.

    Emitting both floor(y/W) and floor(y/W)-1 bridges adjacent buckets, so a
    pair straddling a boundary is not lost. It over-generates out to 2W, which
    is safe -- blocking may propose too much, never too little.
    """
    if year is None:
        return ()
    b = int(year) // YEAR_WINDOW
    return (b, b - 1)


def keys_for(m: dict, allow_time: bool = True) -> Iterator[tuple]:
    """Every blocking key a mention belongs to."""
    p = phonetic_key(m.get("name"))
    if not p:
        return
    reg = m.get("_register")
    if reg:
        yield ("R", p, reg)
    for _role, other in (m.get("_ctx") or ()):
        if other:
            yield ("A", p, other)
    if allow_time:
        for b in _time_keys(m.get("_year")):
            yield ("Y", p, b)


def _name_is_rare_enough(mentions: List[dict], max_block: int) -> Dict[str, bool]:
    """Which phonetic names may use the unbounded time/undated keys."""
    counts = collections.Counter(phonetic_key(m.get("name")) for m in mentions)
    return {k: (n <= max_block) for k, n in counts.items()}


def candidate_pairs(mentions: List[dict], max_block: int = DEFAULT_MAX_BLOCK,
                    stats=None) -> Iterator[Tuple[int, int]]:
    """Yield each candidate pair of indices exactly once.

    De-duplicated across keys: a pair sharing both a register and an associate
    key is produced once, so the caller scores it once.
    """
    rare = _name_is_rare_enough(mentions, max_block)
    buckets: Dict[tuple, List[int]] = collections.defaultdict(list)
    for i, m in enumerate(mentions):
        p = phonetic_key(m.get("name"))
        # set(): one mention can name the SAME person under two roles (parent
        # and godparent, say), which yielded the same ("A", name, person) key
        # twice and put the index in that bucket twice -- emitting its pairs
        # twice and, worse, pairing the mention with itself.
        for k in set(keys_for(m, allow_time=rare.get(p, False))):
            buckets[k].append(i)

    # UNDATED MENTIONS NEED A CROSS PRODUCT, NOT A KEY.
    #
    # `_shares_context` keeps a pair whenever EITHER side lacks a year, so an
    # undated mention is a candidate against every namesake. A shared "undated"
    # key cannot express that: it would only pair undated mentions with each
    # other. Giving every mention that key instead would recreate the whole
    # phonetic block and with it the quadratic.
    #
    # The first version of this file made exactly that mistake and silently lost
    # 4,034,031 candidate pairs -- 947 undated mentions times their phonetic
    # blocks. So they are enumerated directly against their block, which is
    # O(undated x block) and bounded because undated mentions are 2.4% of the
    # corpus.
    by_name: Dict[str, List[int]] = collections.defaultdict(list)
    for i, m in enumerate(mentions):
        p = phonetic_key(m.get("name"))
        if p:
            by_name[p].append(i)

    # DE-DUPLICATE BY PRIORITY, NOT BY A SET.
    #
    # A pair usually matches several keys, so the first version kept a `seen`
    # set of every pair emitted. That is correct and ruinous: 14.7M tuples in a
    # Python set cost more than the pairs it saves, and the end-to-end merge went
    # from 1,319s to 33,901s despite scoring FEWER pairs. The asymptotics
    # improved and the constant destroyed the gain.
    #
    # Instead each pair is owned by exactly one key type, in priority order
    # R > A > Y, and a key only emits a pair it owns. Ownership is a direct test
    # on the two mentions -- same register? share an associate? -- so it is O(1)
    # and needs no memory. This mirrors the structure of `_shares_context`
    # itself, which is the point: the keys exist to reproduce it.
    def _same_register(x, y):
        r = mentions[x].get("_register")
        return bool(r) and r == mentions[y].get("_register")

    def _assoc(x):
        return {n for _, n in (mentions[x].get("_ctx") or ())}

    def _shares_assoc(x, y):
        ax = _assoc(x)
        return bool(ax) and bool(ax & _assoc(y))

    for i, m in enumerate(mentions):
        if m.get("_year") is not None:
            continue
        p = phonetic_key(m.get("name"))
        block = by_name.get(p) or ()
        if len(block) > max_block:
            if stats is not None:
                stats["undated_in_oversized_block"] += 1
            continue
        for j in block:
            if i == j:
                continue
            # when BOTH are undated this loop runs twice for the pair; the
            # lower index owns it
            if mentions[j].get("_year") is None and j < i:
                continue
            # This pass owns EVERY pair with an undated side, unconditionally.
            # An earlier version skipped ones that also shared a register, while
            # the key loops skipped anything undated -- so those pairs were
            # emitted by nobody and 300,000 candidates vanished. Ownership has
            # to be total as well as exclusive.
            yield (i, j) if i < j else (j, i)

    for key, idxs in buckets.items():
        if len(idxs) < 2:
            continue
        # ONLY THE TIME KEY MAY BE DROPPED FOR BEING TOO BIG.
        #
        # A register is one physical book, so ("R", name, register) is bounded by
        # how many same-named people that book holds -- it does not grow as the
        # collection grows, and dropping it would discard same-register merges,
        # which is where most real merges are. An associate key is bounded by how
        # often one specific person is named. The time key is the only one that
        # grows without limit, and it is the only one gated here.
        #
        # An earlier version capped every key, and the test for it caught that
        # same-register pairs disappeared once a register got big enough.
        if key[0] == "Y" and len(idxs) > max_block:
            if stats is not None:
                stats["skipped_oversized_keys"] += 1
                stats["skipped_pairs"] += len(idxs) * (len(idxs) - 1) // 2
            continue
        kind = key[0]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                if i > j:
                    i, j = j, i
                if (mentions[i].get("_year") is None
                        or mentions[j].get("_year") is None):
                    continue          # the undated pass above owns these
                # Priority R > A > Y: a key emits only pairs that no
                # higher-priority key owns, which is what removes the need to
                # remember every pair already emitted.
                if kind != "R" and _same_register(i, j):
                    continue
                if kind == "A":
                    # A pair sharing several associates sits in several A keys.
                    # The alphabetically first shared name owns it, so the other
                    # keys stay silent. Without this the pair is emitted once per
                    # shared associate -- 2.27M duplicates on this corpus.
                    shared = _assoc(i) & _assoc(j)
                    if not shared or key[2] != min(shared):
                        continue
                elif kind == "Y":
                    if _shares_assoc(i, j):
                        continue
                    # Likewise two mentions can share BOTH time buckets; the
                    # lower one owns the pair.
                    tb = set(_time_keys(mentions[i]["_year"])) & \
                        set(_time_keys(mentions[j]["_year"]))
                    if not tb or key[2] != min(tb):
                        continue
                yield i, j


def block_size_report(mentions: List[dict],
                      max_block: int = DEFAULT_MAX_BLOCK) -> dict:
    """Sizes without enumerating pairs, so the projection is cheap to check."""
    rare = _name_is_rare_enough(mentions, max_block)
    buckets: Dict[tuple, int] = collections.Counter()
    for m in mentions:
        p = phonetic_key(m.get("name"))
        for k in keys_for(m, allow_time=rare.get(p, False)):
            buckets[k] += 1
    sizes = [n for n in buckets.values() if n >= 2]
    kept = [n for n in sizes if n <= max_block]
    return {
        "keys": len(buckets),
        "keys_with_pairs": len(sizes),
        "largest": max(sizes) if sizes else 0,
        "oversized_keys": sum(1 for n in sizes if n > max_block),
        # upper bound: before cross-key de-duplication
        "pairs_upper_bound": sum(n * (n - 1) // 2 for n in kept),
        "names_denied_time_key": sum(1 for v in rare.values() if not v),
    }
