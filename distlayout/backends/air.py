"""MLIR-AIR backend over the ``air.api`` Python DSL.

Turns a :class:`~distlayout.view.DistributedView` into ordinary ``air.api``
calls inside a herd body. The herd coordinates (``tx``, ``ty``) are the view's
grid coordinates, in grid-axis order. Nothing here creates a new kind of IR:
the output is the same ``air.alloc`` / ``air.ops.load`` / ``air.ops.store`` a
hand-written kernel would contain, with the offsets derived from the layout
instead of written by hand::

    view = bind(l2_a, Decomposition(Block(0, n // tile_n)), Map(Grid(n // tile_n), {0: 0}))
    ...
    def _(tx):
        a = air_backend.alloc_local(view, dt, scope=h.private())
        air_backend.load_local(view, a, (tx,))        # == air.ops.load(a, l2_a[tx*tile_n : tx*tile_n+tile_n])

Offsets are built with ordinary ``*`` and ``+`` on the coordinates, so with
``air.api``'s ``IndexExpr`` they stay a single affine form and reach the IR as
one ``affine.apply`` -- the form AIR's DMA passes read. A coordinate on a
replicated grid axis does not appear in the offset at all, so every PE along it
names the same region: a broadcast, which AIR can lower to a multicast.

v0 limits:

* uniform layouts only. ``air.api`` needs static transfer sizes, and an uneven
  layout gives different PEs different sizes. :meth:`DistributedView.plan`
  still describes uneven layouts exactly; emitting them needs per-PE branches
  (``air.ops.branch``) or a padded L2 source, which is the next step.
* the herd must cover the grid one-to-one (its iteration space is the grid).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from ..decomposition import Block, UnevenBlock
from ..view import DistributedView


def _air():
    try:
        from air import api  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "distlayout.backends.air needs MLIR-AIR's Python package (`from air import api`). "
            "The layout core (distlayout) works without it."
        ) from e
    return api


def _placement(view: DistributedView):
    from ..mapping import as_placement
    return as_placement(view.mapping, view.decomposition.num_parts(view.shape))


def _check_emittable(view: DistributedView) -> None:
    from ..mapping import as_placement
    pm = as_placement(view.mapping, view.decomposition.num_parts(view.shape))
    if pm is None:
        raise NotImplementedError(
            f"owner maps ({view.mapping!r}) are not emitted to AIR yet; the AIR backend handles "
            f"placement maps (Map / Replicate) only")
    if not view.is_uniform():
        sizes = sorted({view.local_shape(pe) for pe in view.grid.coords()})
        raise NotImplementedError(
            f"uneven layout {view.decomposition!r}: local shapes {sizes} differ between PEs, "
            f"and air.api transfers need one static size. Use view.plan(pe) to inspect it; "
            f"AIR emission of uneven layouts is not implemented yet."
        )
    # Every PE's region must lie inside the source window (matters for partial sources).
    for pe in view.grid.coords():
        view.source_region(view.held_region(pe))


def offsets(view: DistributedView, coords: Sequence[Any]) -> Tuple[Any, ...]:
    """Per tensor axis, this PE's start offset in the *source handle's* coordinates.

    ``coords`` are the herd coordinates (``IndexExpr`` in ``air.api``, or plain
    ints). The result is ``coord * block - origin`` on a cut axis and the
    constant ``-origin`` on an uncut one.
    """
    _check_emittable(view)
    if len(coords) != view.grid.rank:
        raise ValueError(f"got {len(coords)} herd coordinates for a rank-{view.grid.rank} grid")
    rule_on_axis = {r.axis: (k, r) for k, r in enumerate(view.decomposition.rules)}
    placed = _placement(view).place_dict
    out: List[Any] = []
    for a, extent in enumerate(view.shape):
        base = -view.source.origin[a]
        if a not in rule_on_axis:
            out.append(base)
            continue
        k, rule = rule_on_axis[a]
        size = view.local_shape(next(view.grid.coords()))[a]
        if not isinstance(rule, (Block, UnevenBlock)):
            raise NotImplementedError(f"no AIR emission for {rule!r}")
        c = coords[placed[k]]
        out.append(c * size + base if base else c * size)
    return tuple(out)


def slice_key(view: DistributedView, coords: Sequence[Any]) -> Tuple[slice, ...]:
    """The subscript naming this PE's region of the source: ``src[slice_key(...)]``."""
    size = view.local_shape(next(view.grid.coords()))
    return tuple(slice(o, o + s) for o, s in zip(offsets(view, coords), size))


def local_region(view: DistributedView, coords: Sequence[Any], src: Any = None) -> Any:
    """``src[...]`` for this PE's region. ``src`` defaults to the bound handle."""
    src = view.source.handle if src is None else src
    return src[slice_key(view, coords)]


def alloc_local(view: DistributedView, dtype: Any, scope: Any = None, air: Any = None) -> Any:
    """``air.alloc`` of the per-PE local shape (the max over PEs, for future uneven support)."""
    air = air or _air()
    _check_emittable(view)
    kwargs = {} if scope is None else {"scope": scope}
    return air.alloc(list(view.max_local_shape()), dtype, **kwargs)


def load_local(view: DistributedView, dst: Any, coords: Sequence[Any], src: Any = None, air: Any = None) -> Any:
    """``air.ops.load(dst, src[this PE's region])``."""
    air = air or _air()
    return air.ops.load(dst, local_region(view, coords, src))


def store_local(view: DistributedView, src: Any, coords: Sequence[Any], dst: Any = None, air: Any = None) -> Any:
    """``air.ops.store(src, dst[this PE's region])``; ``dst`` defaults to the bound handle."""
    air = air or _air()
    return air.ops.store(src, local_region(view, coords, dst))


def herd_space(view: DistributedView) -> List[range]:
    """An ``air.herd`` iteration space with one point per grid coordinate.

    A placed axis gives ``range(0, extent, block)``, the idiom upstream examples
    use (``air.herd([range(0, n, tile_n)], ...)``: the coordinate counts tiles).
    A replicated axis gives ``range(0, k)``.
    """
    _check_emittable(view)
    placed = {g: k for k, g in _placement(view).place}
    size = view.local_shape(next(view.grid.coords()))
    out = []
    for g, extent in enumerate(view.grid.shape):
        if g in placed:
            a = view.decomposition.rules[placed[g]].axis
            out.append(range(0, view.shape[a], size[a]))
        else:
            out.append(range(0, extent))
    return out
