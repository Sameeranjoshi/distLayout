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
    # Every PE's region must lie inside the source window (matters for partial sources).
    for pe in view.grid.coords():
        for r in view.held_regions(pe):
            view.source_region(r)


def _check_uniform(view: DistributedView, what: str) -> None:
    if not view.is_uniform():
        sizes = sorted({view.local_shape(pe) for pe in view.grid.coords()})
        hint = ("load_local / store_local / for_each_region handle it with one branch per size class"
                if _placement(view) is not None else
                "owner maps with tiles of different sizes are not emitted yet (it needs size-class branches "
                "inside the piece loop)")
        raise NotImplementedError(
            f"{what}: uneven layout {view.decomposition!r} gives pieces different sizes {sizes}, so there is "
            f"no single region formula; {hint}.")


def _check_placement(view: DistributedView) -> None:
    _check_emittable(view)
    if _placement(view) is None:
        raise NotImplementedError(
            f"{view.mapping!r} is an owner map: a PE holds several pieces, so there is no single "
            f"region per PE. Use for_each_piece / load_local, which loop over the PE's pieces.")
    _check_uniform(view, "single-region offsets")


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
    _check_uniform(view, "owner-map emission")
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

    def trip(s):
        from .schedule import _eval
        pid = sched.pieces(coords, s)
        conds = [(pid[k], "lt", n) for k, n in sched.guards] + \
                [(_eval(e, pid), "eq", coords[g]) for g, e in sched.checks]

        def guarded(i):
            if i == len(conds):
                body(sched.slot(s), _piece_key(view, pid))
                return
            a, op, b = conds[i]
            if isinstance(a, int) and isinstance(b, int):   # concrete: decide in Python
                if (a < b) if op == "lt" else (a == b):
                    guarded(i + 1)
                return
            with air.ops.branch(a < b if op == "lt" else a == b):   # only on PEs that hold this trip's piece
                guarded(i + 1)
        guarded(0)
    _loop_nest(air, sched.loops, trip)


# -- size classes: uneven layouts and halos (placement maps) ---------------------

def _axis_runs(view: DistributedView, a: int, radius: int, boundary: str):
    """``(grid axis or None, runs)`` for tensor axis ``a``.

    A run is a range of PE coordinates ``c0..c1`` along the grid axis over which
    everything static agrees -- request size, pads, owned size and position in
    the tile -- and the request start is affine, ``lo0 + (c - c0) * d``.
    """
    n = view.shape[a]
    padded = boundary in ("zero", "pad")
    pb_whole = radius if padded else 0
    rule_k = next((k for k, r in enumerate(view.decomposition.rules) if r.axis == a), None)
    if rule_k is None:
        return None, [dict(c0=0, c1=0, lo0=0, d=0, size=n, pb=pb_whole, pa=pb_whole, own_off=pb_whole, own=n)]
    g = _placement(view).place_dict[rule_k]
    rule = view.decomposition.rules[rule_k]
    sizes = rule.sizes(n)
    starts = [sum(sizes[:i]) for i in range(len(sizes))]
    runs: List[dict] = []
    for c, (st, sz) in enumerate(zip(starts, sizes)):
        lo, hi = max(0, st - radius), min(n, st + sz + radius)
        pb = (lo - (st - radius)) if padded else 0
        pa = ((st + sz + radius) - hi) if padded else 0
        key = (hi - lo, pb, pa, st - lo + pb, sz)
        r = runs[-1] if runs else None
        if r is not None and r["key"] == key and (r["d"] is None or lo - r["last"] == r["d"]):
            r["d"] = lo - r["last"] if r["d"] is None else r["d"]
            r["c1"], r["last"] = c, lo
            continue
        runs.append(dict(c0=c, c1=c, lo0=lo, d=None, last=lo, key=key, size=hi - lo, pb=pb, pa=pa,
                         own_off=st - lo + pb, own=sz))
    for r in runs:
        r["d"] = r["d"] or 0
    return g, runs


def region_shape(view: DistributedView, halo: int = 0, boundary: str = "zero") -> Tuple[int, ...]:
    """L1 tile that holds any PE's region (+ halo): the per-axis maximum over size classes."""
    return tuple(max(r["size"] + r["pb"] + r["pa"] for r in _axis_runs(view, a, halo, boundary)[1])
                 for a in range(len(view.shape)))


def size_classes(view: DistributedView, halo: int = 0, boundary: str = "zero") -> List[List[dict]]:
    """Per tensor axis, its runs (for printing and tests)."""
    return [_axis_runs(view, a, halo, boundary)[1] for a in range(len(view.shape))]


def for_each_region(view: DistributedView, coords: Sequence[Any], body: Callable[..., Any], halo: int = 0,
                    boundary: str = "zero", air: Any = None) -> None:
    """Call ``body(keys)`` for this PE's region, inside one ``air.ops.branch`` per size class.

    For placement maps whose PEs differ in size (uneven) or in their halo (edge
    PEs). ``keys`` holds static-size subscripts for that class:

    * ``src``: the request window of the source (own region grown by ``halo``,
      clipped to the tensor);
    * ``tile``: the part of the L1 tile (from :func:`region_shape`) this PE uses;
    * ``window``: where the source window lands inside it;
    * ``pad_before`` / ``pad_after``: cells per axis outside the tensor, which
      ``boundary="zero"`` / ``"pad"`` turn into zeros so every PE sees a full
      ``b + 2*halo`` tile (``None`` when there are none);
    * ``own_tile`` / ``own_src``: the PE's owned region inside the tile and in
      the source -- what a store writes back (never the halo).

    With concrete ``coords`` (ints) the class is chosen in Python, without branches.
    """
    air = air or _air()
    _check_emittable(view)
    if _placement(view) is None:
        raise NotImplementedError("for_each_region is for placement maps; owner maps use for_each_piece")
    if boundary not in ("zero", "pad", "clip"):
        raise ValueError(f"boundary must be 'zero', 'pad' or 'clip', got {boundary!r}")
    axes = [_axis_runs(view, a, halo, boundary) for a in range(len(view.shape))]
    origin = view.source.origin

    def emit(i: int, chosen: List[Tuple[Any, dict]]) -> None:
        if i == len(axes):
            src, tile, win, own_tile, own_src, pb, pa = [], [], [], [], [], [], []
            for a, (x, r) in enumerate(chosen):
                lo = r["lo0"] - r["c0"] * r["d"] - origin[a]
                if x is not None and r["d"]:
                    lo = x * r["d"] + lo if lo else x * r["d"]
                src.append(slice(lo, lo + r["size"]))
                tile.append(slice(0, r["size"] + r["pb"] + r["pa"]))
                win.append(slice(r["pb"], r["pb"] + r["size"]))
                own_tile.append(slice(r["own_off"], r["own_off"] + r["own"]))
                st = lo + (r["own_off"] - r["pb"])
                own_src.append(slice(st, st + r["own"]))
                pb.append(r["pb"]); pa.append(r["pa"])
            body(dict(src=tuple(src), tile=tuple(tile), window=tuple(win), own_tile=tuple(own_tile), own_src=tuple(own_src),
                      pad_before=pb if any(pb) else None, pad_after=pa if any(pa) else None))
            return
        g, runs = axes[i]
        x = coords[g] if g is not None else None
        if len(runs) == 1:
            emit(i + 1, chosen + [(x, runs[0])])
        elif isinstance(x, int):
            emit(i + 1, chosen + [(x, next(r for r in runs if r["c0"] <= x <= r["c1"]))])
        else:
            def chain(j: int) -> None:
                if j == len(runs) - 1:
                    emit(i + 1, chosen + [(x, runs[j])])
                    return
                with air.ops.branch(x <= runs[j]["c1"]) as b:
                    emit(i + 1, chosen + [(x, runs[j])])
                with b.otherwise():
                    chain(j + 1)
            chain(0)
    emit(0, [])


def _sub(buf: Any, key: Tuple[slice, ...], full: Sequence[int]) -> Any:
    """``buf[key]``, or ``buf`` itself when the key covers all of it."""
    return buf if all(k.start == 0 and k.stop == n for k, n in zip(key, full)) else buf[key]


def halo_load(view: DistributedView, dst: Any, coords: Sequence[Any], halo: int, src: Any = None,
              boundary: str = "zero", air: Any = None) -> None:
    """Fill ``dst`` (shape :func:`region_shape`) with this PE's region plus ``halo`` cells per side.

    The neighbors' cells are read from the shared source buffer (L2), not sent
    PE to PE: the L2 buffer is the segment-wide view of the tensor. Cells
    outside the tensor, at its edges:

    * ``boundary="zero"`` (default): the edge PEs fill their tile with 0 on the
      core, then load the window at its offset;
    * ``"pad"``: the memtile DMA zero-fills (``pad_before``/``pad_after``; needs
      an ``air.api`` that supports DMA padding);
    * ``"clip"``: no ghost cells outside the tensor; edge tiles are smaller.
    """
    air = air or _air()
    src = view.source.handle if src is None else src
    full = region_shape(view, halo, boundary)

    def body(k):
        if boundary == "pad" or not (k["pad_before"] or k["pad_after"]):
            air.ops.load(_sub(dst, k["tile"], full), src[k["src"]],
                         **({"pad_before": k["pad_before"], "pad_after": k["pad_after"]} if boundary == "pad" else {}))
        else:
            air.ops.fill(dst, 0.0)
            air.ops.load(_sub(dst, k["window"], full), src[k["src"]])
    for_each_region(view, coords, body, halo=halo, boundary=boundary, air=air)


def store_owned(view: DistributedView, tile: Any, coords: Sequence[Any], halo: int = 0, dst: Any = None,
                boundary: str = "zero", air: Any = None) -> None:
    """Write the owned part of ``tile`` (not its halo) back to this PE's region of ``dst``."""
    air = air or _air()
    dst = view.source.handle if dst is None else dst
    full = region_shape(view, halo, boundary)
    for_each_region(view, coords, lambda k: air.ops.store(_sub(tile, k["own_tile"], full), dst[k["own_src"]]),
                    halo=halo, boundary=boundary, air=air)


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
        if not view.is_uniform():
            return halo_load(view, dst, coords, 0, src=src, air=air)
        return air.ops.load(dst, local_region(view, coords, src))
    for_each_piece(view, coords, lambda slot, key: air.ops.load(dst[_stack_key(view, slot)], src[key]), air=air)


def store_local(view: DistributedView, src: Any, coords: Sequence[Any], dst: Any = None, air: Any = None) -> Any:
    """Drain ``src`` (from :func:`alloc_local`) into this PE's pieces of ``dst`` (default: the bound handle)."""
    air = air or _air()
    dst = view.source.handle if dst is None else dst
    if _placement(view) is not None:
        if not view.is_uniform():
            return store_owned(view, src, coords, 0, dst=dst, air=air)
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
    if pm is None or not view.is_uniform():
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
                name: str = "dist_copy", halo: int = 0, boundary: str = "zero", air: Any = None) -> Any:
    """A complete ``air.api`` launch that scatters through the layout and gathers back.

    L3 ``A`` -> L2 -> each PE loads its pieces (one reused L1 tile per core;
    with ``halo``, its region plus halo) -> stores what it owns to a second L2
    buffer -> L3 ``C``. Running it must give ``C ==
    A``, which checks every DMA window of the layout end to end. ``cores`` is
    the herd's physical shape (default: one core per grid point).
    """
    air = air or _air()
    if dtype is None:
        from air.api.types import f32 as dtype  # type: ignore[import-not-found]
    shape = list(view.shape)
    regions = _placement(view) is not None and (halo or not view.is_uniform())
    tile = (list(region_shape(view, halo, boundary)) if regions else
            list(_tile(view)) if _placement(view) is None else list(view.max_local_shape()))
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

                            if regions:
                                halo_load(view, a, coords, halo, src=l2_in, boundary=boundary, air=air)
                                store_owned(view, a, coords, halo, dst=l2_out, boundary=boundary, air=air)
                                return

                            def piece(slot, key):
                                air.ops.load(a, l2_in[key])
                                air.ops.store(a, l2_out[key])
                            for_each_piece(view, coords, piece, air=air)
                        rank = view.grid.rank
                        h.body([lambda tx: body(tx), lambda tx, ty: body(tx, ty),
                                lambda tx, ty, tz: body(tx, ty, tz)][rank - 1])
                    air.ops.store(l2_out, C)
    return launch


# -- irregular: ghost gathers from the inspector's schedule --------------------------

def gather_runs(plan) -> List[Tuple[int, int, int]]:
    """``[(dst_offset, src_start, length), ...]`` filling a PE's local x in local order (owned, then ghosts)."""
    out, off = [], 0
    for ids in (plan.owned, plan.ghosts):
        for a, b in ids.runs:
            out.append((off, a, b - a))
            off += b - a
    return out


def for_each_pe(grid_shape: Sequence[int], coords: Sequence[Any], body: Callable[[Tuple[int, ...]], Any],
                air: Any = None) -> None:
    """Call ``body(pe)`` with the concrete PE coordinate, inside one ``air.ops.branch`` chain per grid axis.

    For code that differs per PE (irregular schedules). With concrete ``coords``
    the PE is picked in Python.
    """
    air = air or _air()

    def go(g: int, prefix: Tuple[int, ...]) -> None:
        if g == len(grid_shape):
            body(prefix)
            return
        x, n = coords[g], grid_shape[g]
        if isinstance(x, int) or n == 1:
            go(g + 1, prefix + ((x if isinstance(x, int) else 0),))
            return

        def chain(v: int) -> None:
            if v == n - 1:
                go(g + 1, prefix + (v,))
                return
            with air.ops.branch(x <= v) as b:
                go(g + 1, prefix + (v,))
            with b.otherwise():
                chain(v + 1)
        chain(0)
    go(0, ())


def ghost_footprint(sched) -> int:
    """L1 elements for the largest local x (owned + ghosts) over all PEs."""
    return max(p.x_footprint for p in sched.values())


def ghost_load(sched, dst: Any, coords: Sequence[Any], src: Any, air: Any = None) -> None:
    """Fill ``dst`` with this PE's local x -- owned ids, then ghosts -- straight from the L2 copy ``src``.

    One static DMA per contiguous run of ids, one branch per PE. The order is
    the local numbering :meth:`GhostPlan.local_csr` uses, so a local SpMV can
    run on ``dst`` as is.
    """
    air = air or _air()
    grid_shape = tuple(max(pe[g] for pe in sched) + 1 for g in range(len(next(iter(sched)))))
    F = ghost_footprint(sched)

    def body(pe):
        for off, start, n in gather_runs(sched[pe]):
            d = dst if (off == 0 and n == F) else dst[off:off + n]
            air.ops.load(d, src[start:start + n])
    for_each_pe(grid_shape, coords, body, air=air)


def ghost_gather_kernel(sched, n: int, dtype: Any = None, name: str = "ghost_gather", air: Any = None) -> Any:
    """``X`` (L3, n) -> L2 -> each PE gathers owned + ghost x into L1 -> ``Y[p, :footprint_p]`` (L3, P x F).

    Row ``p`` of ``Y`` must equal ``x[local_ids(p)]``: this checks every gather
    DMA end to end. Grids are flattened to a linear PE index for ``Y``.
    """
    air = air or _air()
    if dtype is None:
        from air.api.types import f32 as dtype  # type: ignore[import-not-found]
    pes = sorted(sched)
    grid_shape = tuple(max(pe[g] for pe in pes) + 1 for g in range(len(pes[0])))
    F, P = ghost_footprint(sched), len(pes)
    X = air.tensor([n], dtype)
    Y = air.tensor([P, F], dtype)
    with air.launch(name=name) as launch:
        @launch.body
        def _():
            with air.segment(name=name + "_seg") as seg:
                @seg.body
                def _():
                    l2_x = air.alloc([n], dtype, scope=seg.private())
                    l2_y = air.alloc([P, F], dtype, scope=seg.private())
                    air.ops.load(l2_x, X)
                    with air.herd([range(g) for g in grid_shape], name=name + "_herd") as h:
                        def body(*coords):
                            a = air.alloc([F], dtype, scope=h.private())
                            ghost_load(sched, a, coords, l2_x, air=air)

                            def store(pe):
                                p = pes.index(pe)
                                fp = sched[pe].x_footprint
                                air.ops.store(a if fp == F else a[0:fp], l2_y[p:p + 1, 0:fp])
                            for_each_pe(grid_shape, coords, store, air=air)
                        h.body([lambda tx: body(tx), lambda tx, ty: body(tx, ty),
                                lambda tx, ty, tz: body(tx, ty, tz)][len(grid_shape) - 1])
                    air.ops.store(l2_y, Y)
    return launch
