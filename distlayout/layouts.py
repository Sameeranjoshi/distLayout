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
=============  ==================================  ============================

:func:`refine` is rank splitting: every piece of an existing view is cut ``k``
ways and the slices go to a new grid axis, ``owner(tx, ty)``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

from .decomposition import Block, Decomposition, UnevenBlock
from .expr import FloorDiv, Mod, Var, substitute
from .mapping import Grid, Mapping, OwnerMap
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


def refine(view: DistributedView, axis: int, k: int) -> DistributedView:
    """Split every piece of ``view`` ``k`` ways along tensor ``axis``; slices go to a new grid axis.

    The new grid is the old one with an extra trailing axis of extent ``k``.
    Ownership stays the same along the old axes: the PE that held a piece
    becomes the row of ``k`` PEs holding its slices. The source is unchanged.

    * if ``axis`` is cut by ``Block(axis, P)``, the new cut is ``Block(axis, P*k)``
      and the old piece coordinate ``t`` becomes ``t floordiv k`` (old axes) and
      ``t mod k`` (new axis);
    * an ``UnevenBlock`` is refined piece by piece (balanced);
    * if ``axis`` is whole, a ``Block(axis, k)`` is added on the new axis.
    """
    if k < 1:
        raise ValueError(f"refine needs k >= 1, got {k}")
    rules = list(view.decomposition.rules)
    old = view.mapping
    exprs = dict(old.exprs())
    G = view.grid.rank
    pos = next((i for i, r in enumerate(rules) if r.axis == axis), None)
    extent = view.shape[axis] if 0 <= axis < len(view.shape) else None
    if extent is None:
        raise ValueError(f"tensor has no axis {axis}")
    if pos is None:
        rules.append(Block(axis, k))
        exprs[G] = Var(len(rules) - 1)
    else:
        r = rules[pos]
        if isinstance(r, Block):
            rules[pos] = Block(axis, r.parts * k)
        elif isinstance(r, UnevenBlock):
            sizes = []
            for s in r.sizes(extent):
                q, rem = divmod(s, k)
                sizes += [q + 1 if j < rem else q for j in range(k)]
            if 0 in sizes:
                raise ValueError(f"refine: a piece of {min(r.sizes(extent))} elements cannot be cut {k} ways")
            rules[pos] = UnevenBlock(axis, sizes=sizes)
        else:
            raise NotImplementedError(f"refine cannot cut {r!r}")
        exprs = {g: substitute(e, pos, FloorDiv(Var(pos), k)) for g, e in exprs.items()}
        exprs[G] = Mod(Var(pos), k)
    grid = Grid(tuple(view.grid.shape) + (k,))
    return DistributedView(view.source, Decomposition(*rules), OwnerMap(grid, exprs, replicate=old.replicate))
