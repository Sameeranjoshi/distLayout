"""Random layout generator and independent oracles, shared by the combination tests.

Everything here re-derives answers from first principles (enumerating
elements, PEs and pieces) instead of calling the code under test, so a test
that compares the two is a real cross-check.
"""

from __future__ import annotations

import random
from itertools import product
from typing import List, Optional, Tuple

from distlayout import (Block, Box, Decomposition, Grid, Map, OwnerMap, Replicate, TableMap, UnevenBlock, bind,
                        nest, refine)
from distlayout.expr import Var


class Buf:
    """A stand-in for an existing buffer: anything with a shape."""

    def __init__(self, *shape):
        self.shape = shape

    def __getitem__(self, key):
        return ("region", key)


# -- the generator ----------------------------------------------------------------

def _cut(rng: random.Random, axis: int, n: int, parts: Optional[int] = None):
    """A random rule for an axis of length n (or None = whole)."""
    kinds = ["block", "uneven", "uneven_tail", "sizes"]
    kind = rng.choice(kinds)
    if parts is None:
        parts = rng.randint(1, min(n, 6))
    parts = min(parts, n)
    if kind == "block":
        divisors = [d for d in range(1, n + 1) if n % d == 0 and d <= 8]
        if parts not in divisors:
            parts = rng.choice(divisors)
        return Block(axis, parts)
    if kind == "uneven":
        return UnevenBlock(axis, parts=parts)
    if kind == "uneven_tail":
        b = -(-n // parts)
        if b * (parts - 1) >= n:            # tail policy would leave an empty part
            return UnevenBlock(axis, parts=parts)
        return UnevenBlock(axis, parts=parts, policy="tail")
    cuts = sorted(rng.sample(range(1, n), parts - 1)) if parts > 1 else []
    edges = [0] + cuts + [n]
    return UnevenBlock(axis, sizes=[b - a for a, b in zip(edges, edges[1:])])


def _num(rule, n):
    return rule.num_parts(n)


def random_view(rng: random.Random, allow_refine: bool = True, kinds: Optional[List[str]] = None):
    """A random valid DistributedView. Retries until construction succeeds."""
    kinds = kinds or ["placement", "placement", "owner", "owner", "table", "replicate"]
    for _ in range(200):
        try:
            v = _try_view(rng, rng.choice(kinds))
            if allow_refine and rng.random() < 0.35 and v.mapping.__class__ is not TableMap:
                axis = rng.randrange(len(v.shape))
                k = rng.randint(1, 3)
                if v.shape[axis] >= k * 2:
                    v = refine(v, axis=axis, k=k)
                    if rng.random() < 0.3 and v.shape[axis] >= 8:
                        v = refine(v, axis=axis, k=2)
            if allow_refine and rng.random() < 0.25:
                v = nest(v, *random_inner(rng, v))
            return v
        except (ValueError, NotImplementedError):
            continue
    raise RuntimeError("generator could not build a view")


def random_inner(rng: random.Random, view) -> Tuple[Decomposition, object]:
    """A random layout that fits inside every piece of ``view``."""
    sizes = [view.partition_region(p).size for p in view.partitions()]
    rules = []
    for a in range(len(view.shape)):
        extents = [s[a] for s in sizes]
        if rng.random() < 0.6 and min(extents) >= 2:
            parts = rng.randint(1, min(min(extents), 4))
            even = all(e % parts == 0 for e in extents)
            rules.append(Block(a, parts) if even and rng.random() < 0.7 else UnevenBlock(a, parts=parts))
    num = tuple(r.num_parts(sizes[0][r.axis]) for r in rules)
    rep = rng.choice([0, 0, 2])
    if rules and rng.random() < 0.5:                        # owner map: deal the pieces round-robin
        P = rng.randint(1, 3)
        k = rng.randrange(len(num))
        return Decomposition(*rules), OwnerMap(Grid((P, rep) if rep else P), {0: Var(k) % P}, replicate=[1] if rep else [])
    gshape = num + ((rep,) if rep else ()) or (1,)          # one PE per piece
    place = {k: k for k in range(len(num))}
    return Decomposition(*rules), Map(Grid(gshape), place, replicate=[g for g in range(len(gshape)) if g >= len(num)])


def _try_view(rng: random.Random, kind: str):
    rank = rng.choice([1, 1, 2])
    shape = tuple(rng.randint(4, 30 if rank == 1 else 12) for _ in range(rank))
    if kind == "replicate":
        g = tuple(rng.randint(1, 3) for _ in range(rng.choice([1, 2])))
        return bind(Buf(*shape), Decomposition(), Replicate(Grid(g)))
    cut_axes = [a for a in range(rank) if rng.random() < 0.8] or [0]
    rules = [_cut(rng, a, shape[a]) for a in cut_axes]
    num = [_num(r, shape[r.axis]) for r in rules]
    if kind == "placement":
        rep_extent = rng.choice([0, 0, 2, 3])
        gshape = num + ([rep_extent] if rep_extent else [])
        perm = list(range(len(num)))
        rng.shuffle(perm)                                   # piece component k -> grid axis perm[k]
        gs = [0] * len(num)
        for k, g in enumerate(perm):
            gs[g] = num[k]
        gshape = gs + ([rep_extent] if rep_extent else [])
        mapping = Map(Grid(tuple(gshape)), {k: perm[k] for k in range(len(num))},
                      replicate=[len(num)] if rep_extent else [])
        return bind(Buf(*shape), Decomposition(*rules), mapping)
    if kind == "table":
        r = _cut(rng, 0, shape[0])
        n = _num(r, shape[0])
        P = rng.randint(1, 5)
        rep = rng.choice([0, 0, 2])
        grid = Grid((P, rep)) if rep else Grid(P)
        table = [(rng.randrange(P), 0) if rep else rng.randrange(P) for _ in range(n)]
        return bind(Buf(*shape), Decomposition(r), TableMap(grid, table, replicate=[1] if rep else []))
    # owner map: each grid axis a random quasi-affine expression that stays in range
    G = rng.choice([1, 1, 2])
    gshape, exprs = [], {}
    for g in range(G):
        P = rng.randint(2, 5)
        form = rng.choice(["mod", "mod", "skew", "fdiv"])
        k = rng.randrange(len(num))
        if form == "mod":
            a = rng.choice([1, 1, 2, 3, -1])
            e = (Var(k) * a + rng.randint(0, 3)) % P
        elif form == "skew" and len(num) > 1:
            e = (Var(0) * rng.choice([1, 2]) + Var(1) * rng.choice([1, -1, 3])) % P
        else:
            q = max(1, -(-num[k] // P))
            e = Var(k) // q
            P = -(-num[k] // q)
        gshape.append(P)
        exprs[g] = e
    rep = rng.random() < 0.25
    if rep:
        gshape.append(rng.randint(2, 3))
    mapping = OwnerMap(Grid(tuple(gshape)), exprs, replicate=[G] if rep else [])
    return bind(Buf(*shape), Decomposition(*rules), mapping)


def random_request(rng: random.Random, shape) -> Box:
    lo = [rng.randrange(n) for n in shape]
    hi = [rng.randint(l + 1, n) for l, n in zip(lo, shape)]
    return Box.from_bounds(lo, hi)


# -- oracles -----------------------------------------------------------------------

def brute_owners(view):
    """{index: (primary, holders)} by evaluating the mapping on every piece and enumerating its box."""
    out = {}
    for pid in view.partitions():
        primary, holders = view.mapping.pes_of(pid)
        for i in view.partition_region(pid).indices():
            out[i] = (primary, set(holders))
    return out


def all_indices(shape):
    return product(*(range(n) for n in shape))


def pieces_by_pe(view):
    at = {pe: [] for pe in view.grid.coords()}
    for pid in view.partitions():
        for h in view.mapping.pes_of(pid)[1]:
            at[h].append(view.partition_region(pid))
    return at
