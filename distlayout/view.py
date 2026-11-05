"""Binding a layout to existing data: :class:`DataSource` + :func:`bind`.

``bind`` never allocates or copies. It pairs a handle to storage that already
exists (an ``air.tensor``, an ``air.alloc`` L2 buffer, an MLIR value, a numpy
array -- anything) with a decomposition and a mapping, and answers ownership
questions about it.

Two entry points, distinguished by :class:`DataSource`:

* **global source** -- the handle holds the whole logical tensor
  (``DataSource(l2, global_shape=(100,))``).
* **partial source** -- the handle holds one window of a larger logical tensor
  (``DataSource(l2_piece, global_shape=(100,), origin=(50,), extent=(25,))``),
  as when an MPI rank's buffer is already one block of the global array.
  Region queries are still in *global* coordinates; :meth:`DistributedView.source_region`
  translates them into the handle's own coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .decomposition import Decomposition, PartitionId
from .mapping import Coord, Mapping, as_placement
from .region import Box, Index, as_index


@dataclass(frozen=True)
class DataSource:
    handle: Any
    global_shape: Index
    origin: Index
    extent: Index

    def __init__(self, handle: Any, global_shape: Sequence[int], origin: Optional[Sequence[int]] = None,
                 extent: Optional[Sequence[int]] = None) -> None:
        gs = as_index(global_shape)
        org = as_index(origin) if origin is not None else (0,) * len(gs)
        ext = as_index(extent) if extent is not None else tuple(g - o for g, o in zip(gs, org))
        window = Box(org, ext)
        if window.rank != len(gs) or not Box.full(gs).contains_box(window):
            raise ValueError(f"source window {window} does not fit in global shape {gs}")
        hshape = _shape_of(handle)
        if hshape is not None and tuple(hshape) != ext:
            raise ValueError(f"handle has shape {tuple(hshape)} but the source window is {ext}")
        object.__setattr__(self, "handle", handle)
        object.__setattr__(self, "global_shape", gs)
        object.__setattr__(self, "origin", org)
        object.__setattr__(self, "extent", ext)

    @property
    def window(self) -> Box:
        return Box(self.origin, self.extent)

    @property
    def is_global(self) -> bool:
        return self.extent == self.global_shape


@dataclass(frozen=True)
class Transfer:
    """One piece of a :class:`MaterializationPlan`.

    ``region`` is in global coordinates; ``dst_offset`` is where it lands in the
    requesting PE's local buffer. ``src_pe`` is the primary owner of the piece,
    or ``None`` when the piece is already held by the requester (``local``).
    """

    region: Box
    dst_offset: Index
    src_pe: Optional[Coord]
    local: bool


@dataclass(frozen=True)
class MaterializationPlan:
    pe: Coord
    request: Box
    transfers: Tuple[Transfer, ...]

    @property
    def local(self) -> Tuple[Transfer, ...]:
        return tuple(t for t in self.transfers if t.local)

    @property
    def remote(self) -> Tuple[Transfer, ...]:
        return tuple(t for t in self.transfers if not t.local)

    @property
    def is_local(self) -> bool:
        return all(t.local for t in self.transfers)


class DistributedView:
    """A non-owning ``DataSource + Decomposition + Mapping`` (placement ``Map`` or ``OwnerMap``).

    A PE holds zero or more pieces. With one piece per PE (any placement
    ``Map``), the local buffer is the piece itself. When some PE holds several
    (owner maps such as cyclic), every PE's local buffer is a stack of tiles,
    ``(slot, *tile_shape)``, slot ``s`` being the PE's s-th piece in piece
    order; with tiles of different shapes it is flat, ``(n_elements,)``.
    """

    def __init__(self, source: DataSource, decomposition: Decomposition, mapping: Any) -> None:
        decomposition.check(source.global_shape)
        mapping.check(decomposition.num_parts(source.global_shape))
        self.source = source
        self.decomposition = decomposition
        self.mapping = mapping
        at: Dict[Coord, List[PartitionId]] = {pe: [] for pe in mapping.grid.coords()}
        for pid in decomposition.partitions(source.global_shape):
            for h in mapping.pes_of(pid)[1]:
                at[h].append(pid)
        self._at = {pe: tuple(v) for pe, v in at.items()}
        self.max_pieces = max((len(v) for v in self._at.values()), default=0)
        sizes = {self.partition_region(pid).size for pid in self.partitions()}
        self._tile = next(iter(sizes)) if len(sizes) == 1 else None

    # -- basic facts --------------------------------------------------------

    @property
    def shape(self) -> Index:
        return self.source.global_shape

    @property
    def grid(self):
        return self.mapping.grid

    @property
    def stacked(self) -> bool:
        """True when local buffers are stacks of tiles (some PE holds several pieces)."""
        return self.max_pieces > 1

    def partitions(self) -> Iterator[PartitionId]:
        return self.decomposition.partitions(self.shape)

    def partition_region(self, pid: PartitionId) -> Box:
        return self.decomposition.region(pid, self.shape)

    def pieces_at(self, pe: Sequence[int]) -> Tuple[PartitionId, ...]:
        """The pieces this PE holds (primary or replica), in piece order."""
        return self._at[self.grid.check(pe)]

    def partition_at(self, pe: Sequence[int]) -> PartitionId:
        """The single piece at ``pe``; raises when it holds none or several (see :meth:`pieces_at`)."""
        ps = self.pieces_at(pe)
        if len(ps) != 1:
            raise ValueError(f"PE {tuple(as_index(pe))} holds {len(ps)} pieces; use pieces_at()")
        return ps[0]

    def partition_of(self, index: int | Sequence[int]) -> PartitionId:
        return self.decomposition.partition_of(as_index(index), self.shape)

    # -- ownership ----------------------------------------------------------

    def primary_owner(self, index: int | Sequence[int]) -> Coord:
        return self.mapping.pes_of(self.partition_of(index))[0]

    def holders(self, index: int | Sequence[int]) -> Tuple[Coord, ...]:
        """Every PE holding a valid copy: the primary owner first, then replicas."""
        primary, holders = self.mapping.pes_of(self.partition_of(index))
        return (primary,) + tuple(h for h in holders if h != primary)

    def held_regions(self, pe: Sequence[int]) -> Tuple[Box, ...]:
        return tuple(self.partition_region(pid) for pid in self.pieces_at(pe))

    def held_region(self, pe: Sequence[int]) -> Box:
        """What this PE stores, when it is one piece (see :meth:`held_regions`)."""
        return self.partition_region(self.partition_at(pe))

    def owned_regions(self, pe: Sequence[int]) -> Tuple[Box, ...]:
        """What this PE is the primary owner of; empty on a pure replica."""
        return self.held_regions(pe) if self.is_primary(pe) else ()

    def owned_region(self, pe: Sequence[int]) -> Optional[Box]:
        """Single-piece form of :meth:`owned_regions`; ``None`` on a pure replica."""
        return self.held_region(pe) if self.is_primary(pe) else None

    def is_primary(self, pe: Sequence[int]) -> bool:
        return self.mapping.is_primary(self.grid.check(pe))

    def local_shape(self, pe: Sequence[int]) -> Index:
        """Valid local extent at ``pe`` (exact; no padding)."""
        regs = self.held_regions(pe)
        if not self.stacked:
            return regs[0].size if regs else (0,) * len(self.shape)
        if self._tile is not None:
            return (len(regs),) + self._tile
        return (sum(r.volume for r in regs),)

    def max_local_shape(self) -> Index:
        """Per-axis maximum of :meth:`local_shape` -- an allocation size that fits every PE."""
        shapes = [self.local_shape(pe) for pe in self.grid.coords()]
        return tuple(max(s[d] for s in shapes) for d in range(len(shapes[0])))

    def is_uniform(self) -> bool:
        return self.decomposition.is_uniform(self.shape)

    def is_local(self, index: int | Sequence[int], pe: Sequence[int]) -> bool:
        index = as_index(index)
        return any(r.contains(index) for r in self.held_regions(pe))

    def local_index(self, index: int | Sequence[int], pe: Sequence[int]) -> Index:
        index, pe = as_index(index), as_index(pe)
        regs = self.held_regions(pe)
        base = 0
        for slot, r in enumerate(regs):
            if r.contains(index):
                off = tuple(i - o for i, o in zip(index, r.offset))
                if not self.stacked:
                    return off
                if self._tile is not None:
                    return (slot,) + off
                return (base + _rowmajor(off, r.size),)
            base += r.volume
        raise KeyError(f"PE {tuple(pe)} does not hold element {index} (it holds {', '.join(map(str, regs)) or 'nothing'})")

    def global_index(self, local: int | Sequence[int], pe: Sequence[int]) -> Index:
        local, pe = as_index(local), as_index(pe)
        regs = self.held_regions(pe)
        if not Box.full(self.local_shape(pe)).contains(local):
            raise IndexError(f"local index {local} outside PE {tuple(pe)}'s extent {self.local_shape(pe)}")
        if not self.stacked:
            r, off = regs[0], local
        elif self._tile is not None:
            r, off = regs[local[0]], local[1:]
        else:
            k = local[0]
            for r in regs:
                if k < r.volume:
                    break
                k -= r.volume
            off = _unrowmajor(k, r.size)
        return tuple(l + o for l, o in zip(off, r.offset))

    # -- source coordinates -------------------------------------------------

    def source_region(self, region: Box) -> Box:
        """Translate a global region into the source handle's coordinates."""
        if not self.source.window.contains_box(region):
            raise ValueError(f"region {region} is not inside the source window {self.source.window}")
        return region.shift(tuple(-o for o in self.source.origin))

    # -- materialization ----------------------------------------------------

    def plan(self, pe: Sequence[int], request: Optional[Box] = None) -> MaterializationPlan:
        """How ``request`` (default: what ``pe`` holds, if one piece) becomes available at ``pe``.

        Splits the request at piece boundaries. Pieces ``pe`` already holds are
        ``local``; the rest come from their primary owner. The request is
        explicit: inferring it from a computation's accesses is a later pass.
        """
        pe = self.grid.check(pe)
        request = self.held_region(pe) if request is None else request
        if not Box.full(self.shape).contains_box(request):
            raise ValueError(f"request {request} outside tensor of shape {self.shape}")
        mine = self.held_regions(pe)
        out: List[Transfer] = []
        for pid in self.partitions():
            piece = self.partition_region(pid).intersect(request)
            if piece is None:
                continue
            dst = tuple(p - r for p, r in zip(piece.offset, request.offset))
            local = any(m.contains_box(piece) for m in mine)
            out.append(Transfer(piece, dst, None if local else self.mapping.pes_of(pid)[0], local))
        return MaterializationPlan(pe, request, tuple(out))

    # -- checking and printing ---------------------------------------------

    def validate(self) -> None:
        """Check the invariants by enumeration. For tests and small shapes.

        * every element has exactly one piece (cover, no overlap);
        * the primary owner holds the element;
        * replicas never change the primary owner;
        * local_index / global_index round-trip on every PE, and local indices
          are distinct within a PE.
        """
        seen: Dict[Index, PartitionId] = {}
        for pid in self.partitions():
            for i in self.partition_region(pid).indices():
                if i in seen:
                    raise AssertionError(f"element {i} in pieces {seen[i]} and {pid}")
                seen[i] = pid
        full = set(Box.full(self.shape).indices())
        if set(seen) != full:
            missing = sorted(full - set(seen))[:5]
            raise AssertionError(f"elements not in any piece, e.g. {missing}")
        for i in seen:
            p = self.primary_owner(i)
            if not self.is_primary(p) or not self.is_local(i, p):
                raise AssertionError(f"primary owner {p} of {i} is not a primary holder")
            for h in self.holders(i):
                if not self.is_local(i, h):
                    raise AssertionError(f"holder {h} of {i} does not hold it")
        for pe in self.grid.coords():
            used = set()
            for r in self.held_regions(pe):
                for i in r.indices():
                    li = self.local_index(i, pe)
                    if li in used or self.global_index(li, pe) != i:
                        raise AssertionError(f"index round trip failed at PE {pe}, element {i}")
                    used.add(li)

    def table(self) -> str:
        """One line per PE: pieces held, local shape, primary or replica."""
        lines = [f"tensor {self.shape} on grid {self.grid.shape}: {self.decomposition!r}, {self.mapping!r}"]
        for pe in self.grid.coords():
            role = "primary" if self.is_primary(pe) else "replica"
            regs = ", ".join(map(str, self.held_regions(pe))) or "nothing"
            lines.append(f"  PE {pe}: {regs}  local {self.local_shape(pe)}  {role}")
        return "\n".join(lines)

    def to_attr(self) -> str:
        """Textual form of the layout, a sketch of the future ``#dist.layout`` attribute."""
        rules = ", ".join(_rule_attr(r) for r in self.decomposition.rules)
        rep = ", ".join(f"g{g}" for g in self.mapping.replicate)
        head = f"#dist.layout<shape = {list(self.shape)}, grid = {list(self.grid.shape)}, decomp = [{rules}], "
        pm = as_placement(self.mapping, self.decomposition.num_parts(self.shape))
        if pm is not None:
            place = ", ".join(f"{k} -> g{g}" for k, g in pm.place)
            return head + f"place = [{place}], replicate = [{rep}]>"
        owner = ", ".join(f"g{g} = {e}" for g, e in sorted(self.mapping.exprs().items()))
        return head + f"owner = [{owner}], replicate = [{rep}]>"

    def __repr__(self) -> str:
        return f"DistributedView({self.to_attr()})"


def _rowmajor(off: Index, size: Index) -> int:
    k = 0
    for o, s in zip(off, size):
        k = k * s + o
    return k


def _unrowmajor(k: int, size: Index) -> Index:
    out = []
    for s in reversed(size):
        out.append(k % s)
        k //= s
    return tuple(reversed(out))


def bind(source: Any, decomposition: Decomposition, mapping: Any,
         global_shape: Optional[Sequence[int]] = None) -> DistributedView:
    """Attach a layout to existing data. Metadata only: nothing is copied.

    ``source`` is a :class:`DataSource`, or any handle with a ``.shape`` (taken
    as a global source of that shape), or any handle plus ``global_shape``.
    """
    if not isinstance(source, DataSource):
        shape = global_shape if global_shape is not None else _shape_of(source)
        if shape is None:
            raise TypeError("cannot infer the shape of the source; pass global_shape= or a DataSource")
        source = DataSource(source, shape)
    elif global_shape is not None and as_index(global_shape) != source.global_shape:
        raise ValueError("global_shape disagrees with the DataSource")
    return DistributedView(source, decomposition, mapping)


def _shape_of(handle: Any) -> Optional[Tuple[int, ...]]:
    shape = getattr(handle, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(s) for s in shape)
    except (TypeError, ValueError):
        return None


def _rule_attr(rule) -> str:
    from .decomposition import Block, UnevenBlock
    if isinstance(rule, Block):
        return f"block<axis = {rule.axis}, parts = {rule.parts}>"
    if isinstance(rule, UnevenBlock):
        if rule.sizes_ is not None:
            return f"uneven<axis = {rule.axis}, sizes = {list(rule.sizes_)}>"
        return f"uneven<axis = {rule.axis}, parts = {rule.parts}, policy = {rule.policy}>"
    return repr(rule)
