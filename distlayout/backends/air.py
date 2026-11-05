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

from typing import Any, Callable, List, Optional, Sequence, Tuple

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
    if not view.is_uniform():
        sizes = sorted({view.local_shape(pe) for pe in view.grid.coords()})
        raise NotImplementedError(
            f"uneven layout {view.decomposition!r}: local shapes {sizes} differ between PEs, "
            f"and air.api transfers need one static size. Use view.plan(pe) to inspect it; "
            f"AIR emission of uneven layouts is not implemented yet."
        )
    # Every PE's region must lie inside the source window (matters for partial sources).
    for pe in view.grid.coords():
        for r in view.held_regions(pe):
            view.source_region(r)


def _check_placement(view: DistributedView) -> None:
    _check_emittable(view)
    if _placement(view) is None:
        raise NotImplementedError(
            f"{view.mapping!r} is an owner map: a PE holds several pieces, so there is no single "
            f"region per PE. Use for_each_piece / load_local, which loop over the PE's pieces.")


def offsets(view: DistributedView, coords: Sequence[Any]) -> Tuple[Any, ...]:
    """Per tensor axis, this PE's start offset in the *source handle's* coordinates.

    ``coords`` are the herd coordinates (``IndexExpr`` in ``air.api``, or plain
    ints). The result is ``coord * block - origin`` on a cut axis and the
    constant ``-origin`` on an uncut one. Placement maps only.
    """
    _check_placement(view)
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


def schedule(view: DistributedView):
    """The loop nest giving each PE its pieces (see :mod:`distlayout.backends.schedule`)."""
    from .schedule import solve
    _check_emittable(view)
    return solve(view)


def _tile(view: DistributedView) -> Tuple[int, ...]:
    return view.partition_region(next(view.partitions())).size


def _piece_key(view: DistributedView, pid: Sequence[Any]) -> Tuple[slice, ...]:
    """Source subscript of piece ``pid`` (coordinates may be symbolic)."""
    tile = _tile(view)
    rule_on_axis = {r.axis: k for k, r in enumerate(view.decomposition.rules)}
    out = []
    for a, b in enumerate(tile):
        base = -view.source.origin[a]
        if a in rule_on_axis:
            o = pid[rule_on_axis[a]] * b
            o = o + base if base else o
        else:
            o = base
        out.append(slice(o, o + b))
    return tuple(out)


def _loop_nest(air: Any, extents: Sequence[int], body: Callable[[Tuple[Any, ...]], Any], idx: Tuple = ()) -> None:
    if not extents:
        body(idx)
        return
    n = extents[0]
    if n == 1:                                  # no loop for a single trip
        _loop_nest(air, extents[1:], body, idx + (0,))
        return
    for s in (air.sequential(0, n) if air is not None else range(n)):
        _loop_nest(air, extents[1:], body, idx + (s,))


def for_each_piece(view: DistributedView, coords: Sequence[Any], body: Callable[[Any, Tuple[slice, ...]], Any],
                   air: Any = None) -> None:
    """Call ``body(slot, src_key)`` once per piece this PE holds, inside ``air.sequential`` loops.

    ``src_key`` subscripts the bound source (``l2[src_key]`` is the piece);
    ``slot`` is its index in the PE's tile stack. A placement map has one piece
    and no loop. Inside ``body``, load, compute and store as usual::

        a = air.alloc(list(tile), f32, scope=h.private())       # one tile, reused
        def per_piece(slot, key):
            air.ops.load(a, l2_in[key]); ...; air.ops.store(a, l2_out[key])
        air_backend.for_each_piece(view, (tx,), per_piece)
    """
    air = air or _air()
    if len(coords) != view.grid.rank:
        raise ValueError(f"got {len(coords)} herd coordinates for a rank-{view.grid.rank} grid")
    if _placement(view) is not None:
        _check_emittable(view)
        body(0, slice_key(view, coords))
        return
    sched = schedule(view)
    _loop_nest(air, sched.loops, lambda s: body(sched.slot(s), _piece_key(view, sched.pieces(coords, s))))


def local_shape(view: DistributedView) -> Tuple[int, ...]:
    """The L1 buffer ``load_local`` fills: the piece (placement) or ``(slots, *tile)`` (owner map)."""
    if _placement(view) is not None:
        _check_emittable(view)
        return tuple(view.max_local_shape())
    return (schedule(view).slots,) + _tile(view)


def alloc_local(view: DistributedView, dtype: Any, scope: Any = None, air: Any = None) -> Any:
    """``air.alloc`` of :func:`local_shape`."""
    air = air or _air()
    kwargs = {} if scope is None else {"scope": scope}
    return air.alloc(list(local_shape(view)), dtype, **kwargs)


def _stack_key(view: DistributedView, slot: Any) -> Tuple[Any, ...]:
    return (slot,) + tuple(slice(0, b) for b in _tile(view))


def load_local(view: DistributedView, dst: Any, coords: Sequence[Any], src: Any = None, air: Any = None) -> Any:
    """Fill ``dst`` (from :func:`alloc_local`) with this PE's pieces of ``src`` (default: the bound handle)."""
    air = air or _air()
    src = view.source.handle if src is None else src
    if _placement(view) is not None:
        return air.ops.load(dst, local_region(view, coords, src))
    for_each_piece(view, coords, lambda slot, key: air.ops.load(dst[_stack_key(view, slot)], src[key]), air=air)


def store_local(view: DistributedView, src: Any, coords: Sequence[Any], dst: Any = None, air: Any = None) -> Any:
    """Drain ``src`` (from :func:`alloc_local`) into this PE's pieces of ``dst`` (default: the bound handle)."""
    air = air or _air()
    dst = view.source.handle if dst is None else dst
    if _placement(view) is not None:
        return air.ops.store(src, local_region(view, coords, dst))
    for_each_piece(view, coords, lambda slot, key: air.ops.store(src[_stack_key(view, slot)], dst[key]), air=air)


def herd_space(view: DistributedView) -> List[range]:
    """An ``air.herd`` iteration space with one point per grid coordinate.

    For a placement, a placed axis gives ``range(0, extent, block)``, the idiom
    upstream examples use (the coordinate counts tiles). Otherwise, and for
    replicated axes, ``range(0, k)``: the coordinate is the PE coordinate.
    """
    _check_emittable(view)
    pm = _placement(view)
    if pm is None:
        return [range(0, n) for n in view.grid.shape]
    placed = {g: k for k, g in pm.place}
    size = view.local_shape(next(view.grid.coords()))
    out = []
    for g, extent in enumerate(view.grid.shape):
        if g in placed:
            a = view.decomposition.rules[placed[g]].axis
            out.append(range(0, view.shape[a], size[a]))
        else:
            out.append(range(0, extent))
    return out


def copy_kernel(view: DistributedView, dtype: Any = None, cores: Optional[Sequence[int]] = None,
                name: str = "dist_copy", air: Any = None) -> Any:
    """A complete ``air.api`` launch that scatters through the layout and gathers back.

    L3 ``A`` -> L2 -> each PE loads its pieces (one reused L1 tile per core) ->
    stores them to a second L2 buffer -> L3 ``C``. Running it must give ``C ==
    A``, which checks every DMA window of the layout end to end. ``cores`` is
    the herd's physical shape (default: one core per grid point).
    """
    air = air or _air()
    if dtype is None:
        from air.api.types import f32 as dtype  # type: ignore[import-not-found]
    shape = list(view.shape)
    tile = list(_tile(view)) if _placement(view) is None else list(view.max_local_shape())
    A = air.tensor(shape, dtype)
    C = air.tensor(shape, dtype)
    with air.launch(name=name) as launch:
        @launch.body
        def _():
            with air.segment(name=name + "_seg") as seg:
                @seg.body
                def _():
                    l2_in = air.alloc(shape, dtype, scope=seg.private())
                    l2_out = air.alloc(shape, dtype, scope=seg.private())
                    air.ops.load(l2_in, A)
                    kw = {"shape": tuple(cores)} if cores else {}
                    with air.herd(herd_space(view), name=name + "_herd", **kw) as h:
                        def body(*coords):
                            a = air.alloc(tile, dtype, scope=h.private())

                            def piece(slot, key):
                                air.ops.load(a, l2_in[key])
                                air.ops.store(a, l2_out[key])
                            for_each_piece(view, coords, piece, air=air)
                        rank = view.grid.rank
                        h.body([lambda tx: body(tx), lambda tx, ty: body(tx, ty),
                                lambda tx, ty, tz: body(tx, ty, tz)][rank - 1])
                    air.ops.store(l2_out, C)
    return launch
