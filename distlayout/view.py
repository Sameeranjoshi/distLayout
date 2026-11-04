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
from .mapping import Coord, Mapping
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
    """A non-owning ``DataSource + Decomposition + Mapping``."""

    def __init__(self, source: DataSource, decomposition: Decomposition, mapping: Mapping) -> None:
        decomposition.check(source.global_shape)
        mapping.check(decomposition.num_parts(source.global_shape))
        self.source = source
        self.decomposition = decomposition
        self.mapping = mapping

    # -- basic facts --------------------------------------------------------

    @property
    def shape(self) -> Index:
        return self.source.global_shape

    @property
    def grid(self):
        return self.mapping.grid

    def partitions(self) -> Iterator[PartitionId]:
        return self.decomposition.partitions(self.shape)

    def partition_region(self, pid: PartitionId) -> Box:
        return self.decomposition.region(pid, self.shape)

    def partition_at(self, pe: Sequence[int]) -> PartitionId:
        pe = self.grid.check(pe)
        return self.mapping.partition_at(pe, len(self.decomposition.rules))

    def partition_of(self, index: int | Sequence[int]) -> PartitionId:
        return self.decomposition.partition_of(as_index(index), self.shape)

    # -- ownership ----------------------------------------------------------

    def primary_owner(self, index: int | Sequence[int]) -> Coord:
        return self.mapping.pes_of(self.partition_of(index))[0]

    def holders(self, index: int | Sequence[int]) -> Tuple[Coord, ...]:
        """Every PE holding a valid copy: the primary owner first, then replicas."""
        primary, holders = self.mapping.pes_of(self.partition_of(index))
        return (primary,) + tuple(h for h in holders if h != primary)

    def held_region(self, pe: Sequence[int]) -> Box:
        """What this PE stores, primary or replica."""
        return self.partition_region(self.partition_at(pe))

    def owned_region(self, pe: Sequence[int]) -> Optional[Box]:
        """What this PE is the primary owner of; ``None`` on a pure replica."""
        pe = self.grid.check(pe)
        return self.held_region(pe) if self.mapping.is_primary(pe) else None

    def is_primary(self, pe: Sequence[int]) -> bool:
        return self.mapping.is_primary(self.grid.check(pe))

    def local_shape(self, pe: Sequence[int]) -> Index:
        """Valid local extent at ``pe`` (exact; no padding)."""
        return self.held_region(pe).size

    def max_local_shape(self) -> Index:
        """Per-axis maximum of :meth:`local_shape` -- an allocation size that fits every PE."""
        shapes = [self.local_shape(pe) for pe in self.grid.coords()]
        return tuple(max(s[d] for s in shapes) for d in range(len(self.shape)))

    def is_uniform(self) -> bool:
        return self.decomposition.is_uniform(self.shape)

    def is_local(self, index: int | Sequence[int], pe: Sequence[int]) -> bool:
        return self.held_region(pe).contains(as_index(index))

    def local_index(self, index: int | Sequence[int], pe: Sequence[int]) -> Index:
        index, pe = as_index(index), as_index(pe)
        region = self.held_region(pe)
        if not region.contains(index):
            raise KeyError(f"PE {tuple(pe)} does not hold element {index} (it holds {region})")
        return tuple(i - o for i, o in zip(index, region.offset))

    def global_index(self, local: int | Sequence[int], pe: Sequence[int]) -> Index:
        local, pe = as_index(local), as_index(pe)
        region = self.held_region(pe)
        if not Box.full(region.size).contains(local):
            raise IndexError(f"local index {local} outside PE {tuple(pe)}'s extent {region.size}")
        return tuple(l + o for l, o in zip(local, region.offset))

    # -- source coordinates -------------------------------------------------

    def source_region(self, region: Box) -> Box:
        """Translate a global region into the source handle's coordinates."""
        if not self.source.window.contains_box(region):
            raise ValueError(f"region {region} is not inside the source window {self.source.window}")
        return region.shift(tuple(-o for o in self.source.origin))

    # -- materialization ----------------------------------------------------

    def plan(self, pe: Sequence[int], request: Optional[Box] = None) -> MaterializationPlan:
        """How ``request`` (default: what ``pe`` holds) becomes available at ``pe``.

        Splits the request at partition boundaries. Pieces ``pe`` already holds
        are ``local``; the rest come from their primary owner. The request is
        given explicitly: inferring it from a computation's accesses is a later
        compiler pass.
        """
        pe = self.grid.check(pe)
        request = self.held_region(pe) if request is None else request
        if not Box.full(self.shape).contains_box(request):
            raise ValueError(f"request {request} outside tensor of shape {self.shape}")
        mine = self.held_region(pe)
        out: List[Transfer] = []
        for pid in self.partitions():
            piece = self.partition_region(pid).intersect(request)
            if piece is None:
                continue
            dst = tuple(p - r for p, r in zip(piece.offset, request.offset))
            local = mine.contains_box(piece)
            out.append(Transfer(piece, dst, None if local else self.mapping.pes_of(pid)[0], local))
        return MaterializationPlan(pe, request, tuple(out))

    # -- checking and printing ---------------------------------------------

    def validate(self) -> None:
        """Check the v0 invariants by enumeration. For tests and small shapes.

        * every element has exactly one primary partition (cover, no overlap);
        * the primary owner holds the element;
        * replicas never change the primary owner;
        * local_index / global_index round-trip on every PE.
        """
        seen: Dict[Index, PartitionId] = {}
        for pid in self.partitions():
            for i in self.partition_region(pid).indices():
                if i in seen:
                    raise AssertionError(f"element {i} in partitions {seen[i]} and {pid}")
                seen[i] = pid
        full = set(Box.full(self.shape).indices())
        if set(seen) != full:
            missing = sorted(full - set(seen))[:5]
            raise AssertionError(f"elements not in any partition, e.g. {missing}")
        for i, pid in seen.items():
            p = self.primary_owner(i)
            if not self.is_primary(p) or not self.is_local(i, p):
                raise AssertionError(f"primary owner {p} of {i} is not a primary holder")
            for h in self.holders(i):
                if not self.is_local(i, h):
                    raise AssertionError(f"holder {h} of {i} does not hold it")
        for pe in self.grid.coords():
            for i in self.held_region(pe).indices():
                if self.global_index(self.local_index(i, pe), pe) != i:
                    raise AssertionError(f"index round trip failed at PE {pe}, element {i}")

    def table(self) -> str:
        """One line per PE: held region, local shape, primary or replica."""
        lines = [f"tensor {self.shape} on grid {self.grid.shape}: {self.decomposition!r}"]
        for pe in self.grid.coords():
            role = "primary" if self.is_primary(pe) else "replica"
            lines.append(f"  PE {pe}: {self.held_region(pe)}  local {self.local_shape(pe)}  {role}")
        return "\n".join(lines)

    def to_attr(self) -> str:
        """Textual form of the layout, a sketch of the future ``#dist.layout`` attribute."""
        rules = ", ".join(_rule_attr(r) for r in self.decomposition.rules)
        place = ", ".join(f"{k} -> g{g}" for k, g in self.mapping.place)
        rep = ", ".join(f"g{g}" for g in self.mapping.replicate)
        return (f"#dist.layout<shape = {list(self.shape)}, grid = {list(self.grid.shape)}, "
                f"decomp = [{rules}], place = [{place}], replicate = [{rep}]>")

    def __repr__(self) -> str:
        return f"DistributedView({self.to_attr()})"


def bind(source: Any, decomposition: Decomposition, mapping: Mapping,
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
