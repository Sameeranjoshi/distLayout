"""Category-2 layouts: a structured cut plus an affine owner map.

Each constructor returns ``(Decomposition, OwnerMap)``, so ``bind(src, *cyclic(100, 4))``
reads as "cut, then map". The cut is always rectangular tiles; the owner map
decides which PE each tile lands on, and several tiles may share a PE.

=============  ==================================  ============================
layout         cut                                 owner map
=============  ==================================  ============================
cyclic         ``Block(axis, n)``: tiles of 1       ``g0 = t0 mod P``
block-cyclic   ``Block(axis, n / b)``: tiles of b   ``g0 = t0 mod P``
skew           1 x 1 tiles                          ``g0 = (t0 + t1) mod P``
refine         one more cut inside each piece       ``(..., t floordiv k, t mod k)``
nest           any layout inside each piece         outer in ``t floordiv k``, inner in ``t mod k``
=============  ==================================  ============================

:func:`nest` lays a whole layout out inside every piece of an existing view, on
a new grid: ranks then cores, or rows of a model then data-parallel replicas.
:func:`refine` is its simplest case, rank splitting: cut each piece ``k`` ways
and send the slices to a new grid axis, ``owner(tx, ty)``.
"""

from __future__ import annotations

from typing import Any, Sequence, Tuple

from .decomposition import Block, Decomposition, UnevenBlock
from .expr import FloorDiv, Mod, Var, substitute_all
from .mapping import Grid, Map, Mapping, OwnerMap, TableMap
from .region import as_index
from .view import DistributedView


def cyclic(n: int, parts: int, axis: int = 0) -> Tuple[Decomposition, OwnerMap]:
    """Element ``i`` of an ``n``-long axis goes to PE ``i mod parts``."""
    return Decomposition(Block(axis, n)), OwnerMap(Grid(parts), {0: Var(0) % parts})


def block_cyclic(n: int, block: int, parts: int, axis: int = 0) -> Tuple[Decomposition, OwnerMap]:
    """Blocks of ``block`` elements dealt round-robin: block ``b`` goes to PE ``b mod parts``."""
    if n % block:
        raise ValueError(f"block_cyclic: {n} elements are not a whole number of blocks of {block}")
    return Decomposition(Block(axis, n // block)), OwnerMap(Grid(parts), {0: Var(0) % parts})


def skew(shape: Sequence[int], parts: int) -> Tuple[Decomposition, OwnerMap]:
    """Element ``(i, j)`` goes to PE ``(i + j) mod parts``: each row shifted by one."""
    r, c = as_index(shape)
    return Decomposition(Block(0, r), Block(1, c)), OwnerMap(Grid(parts), {0: (Var(0) + Var(1)) % parts})


def nest(view: DistributedView, decomposition: Decomposition, mapping: Any) -> DistributedView:
    """Lay ``(decomposition, mapping)`` out inside every piece of ``view``, on a new grid.

    The inner layout is written for one piece, in the piece's own coordinates,
    and every piece gets a copy of it. The new grid is the old one followed by
    the inner one: a PE is ``(outer PE, inner PE)``, and replication on either
    side stays replication. The source is unchanged.

    The result is an ordinary view, so everything downstream works on it. Each
    axis is cut by both layouts' cuts together, and if the inner cut makes ``k``
    parts, a piece coordinate ``t`` reads as outer ``t floordiv k``, inner
    ``t mod k``. Owner tables stay tables, which is possible only when the
    result has one piece coordinate.

    shortcut: each piece is split on its own, so a PE's several pieces are never
    treated as one stack. To cut across the stack, write the owner map directly:
    ``{0: "t0 mod P", 1: "(t0 floordiv P) floordiv b"}``.
    """
    sizes = {view.partition_region(p).size for p in view.partitions()}
    for size in sizes:
        decomposition.check(size)
    if len({decomposition.num_parts(size) for size in sizes}) > 1:
        raise NotImplementedError(f"nest: {decomposition!r} needs the same number of parts in every piece, or a "
                                  f"piece coordinate cannot be read as outer and inner parts")
    piece = next(iter(sizes))
    mapping.check(decomposition.num_parts(piece))

    rules = list(view.decomposition.rules)
    outer_t, inner_t = {}, {}                   # piece coordinates, rewritten in the combined cut
    for j, r in enumerate(decomposition.rules):
        pos = next((i for i, o in enumerate(rules) if o.axis == r.axis), None)
        if pos is None:
            inner_t[j] = Var(len(rules))
            rules.append(r)
            continue
        cuts = [s for size in rules[pos].sizes(view.shape[r.axis]) for s in r.sizes(size)]
        rules[pos] = Block(r.axis, len(cuts)) if len(set(cuts)) == 1 else UnevenBlock(r.axis, sizes=cuts)
        k = r.num_parts(piece[r.axis])
        outer_t[pos], inner_t[j] = FloorDiv(Var(pos), k), Mod(Var(pos), k)

    old, G = view.mapping, view.grid.rank
    names = view.grid.names + mapping.grid.names if view.grid.names and mapping.grid.names else None
    grid = Grid(view.grid.shape + mapping.grid.shape, names)
    replicate = tuple(old.replicate) + tuple(G + g for g in mapping.replicate)
    if isinstance(old, TableMap) or isinstance(mapping, TableMap):
        if len(rules) != 1:
            raise NotImplementedError(f"nest: an owner table needs one coordinate per piece, this result has "
                                      f"{len(rules)}; it would need a TableMap over several cuts")
        table = [old.pes_of(p)[0] + mapping.pes_of(q)[0] for p in view.partitions()
                 for q in decomposition.partitions(piece)]
        return DistributedView(view.source, Decomposition(*rules), TableMap(grid, table, replicate))
    exprs = {g: substitute_all(e, outer_t) for g, e in old.exprs().items()}
    exprs.update({G + g: substitute_all(e, inner_t) for g, e in mapping.exprs().items()})
    return DistributedView(view.source, Decomposition(*rules), OwnerMap(grid, exprs, replicate))


def refine(view: DistributedView, axis: int, k: int) -> DistributedView:
    """Split every piece of ``view`` ``k`` ways along tensor ``axis``; slices go to a new grid axis.

    :func:`nest` with one cut and a one-axis grid of ``k`` PEs. Ownership stays
    the same along the old axes: the PE that held a piece becomes the row of
    ``k`` PEs holding its slices. Cuts of different sizes are refined piece by
    piece (balanced); a whole axis gets a ``Block(axis, k)``.
    """
    if k < 1:
        raise ValueError(f"refine needs k >= 1, got {k}")
    if not 0 <= axis < len(view.shape):
        raise ValueError(f"tensor has no axis {axis}")
    rule = next((r for r in view.decomposition.rules if r.axis == axis), None)
    cut: Any = Block(axis, k)
    if isinstance(rule, UnevenBlock):
        smallest = min(rule.sizes(view.shape[axis]))
        if smallest < k:
            raise ValueError(f"refine: a piece of {smallest} elements cannot be cut {k} ways")
        cut = UnevenBlock(axis, parts=k)
    return nest(view, Decomposition(cut), Map(Grid(k), {0: 0}))
