"""Layout trees: a layout inside each piece of another, with the levels kept.

:func:`~distlayout.layouts.nest` flattens two layouts into one view, so every
piece must get the same inner layout and the levels are lost. A tree keeps them:
each node is an ordinary :class:`DistributedView` over its parent's piece, in
that piece's own coordinates, so any cut or mapping works at any node and each
piece can have its own inner layout.

A PE is named by its path, one coordinate per level from the root:
``((0, 1), (3,))`` is PE 3 of the piece at outer PE (0, 1).

    Layout(cut, map, level, inner)                  a value, tied to no data
    bind_tree(data, layout)  -->  TreeView          the layout on real shapes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from .decomposition import Decomposition, PartitionId
from .layouts import nest
from .mapping import Coord
from .region import Box, Index, as_index
from .view import DataSource, DistributedView, bind

Path = Tuple[Coord, ...]


@dataclass(frozen=True)
class Layout:
    """How to cut one region and place the pieces, and what goes inside each piece.

    ``inner`` goes inside every piece, and ``overrides`` ``{piece: Layout}`` replaces
    it in the pieces it names; without ``inner`` it must name them all. With neither,
    this node is a leaf. ``level`` names the memory that holds this node's pieces
    (``"L2"``, ``"L1"``); only :meth:`TreeView.fit` reads it.
    """

    cut: Decomposition
    map: Any
    level: Optional[str] = None
    inner: Optional["Layout"] = None
    overrides: Optional[Dict[Any, "Layout"]] = None


@dataclass(frozen=True)
class Fit:
    """One memory level: the fullest memory on it, and what it would take to fit."""

    level: str
    needed: int             # bytes in the fullest memory
    available: int
    where: Path

    @property
    def fits(self) -> bool:
        return self.needed <= self.available

    @property
    def split(self) -> int:
        """The fewest equal parts that would fit, ignoring how the data can be cut."""
        return -(-self.needed // self.available)


class TreeView:
    """A :class:`Layout` bound to shapes. ``view`` is this node, ``children`` its pieces' trees."""

    def __init__(self, view: DistributedView, layout: Layout) -> None:
        self.view, self.layout = view, layout
        self.children: Dict[PartitionId, TreeView] = {}
        if layout.inner is None and layout.overrides is None:
            return
        pieces = list(view.partitions())
        inner = {} if layout.inner is None else dict.fromkeys(pieces, layout.inner)
        overrides = {as_index(k): v for k, v in (layout.overrides or {}).items()}
        if not set(overrides) <= set(pieces):
            raise ValueError(f"overrides name {sorted(overrides)}, but the pieces are {pieces}")
        inner.update(overrides)
        if len(inner) != len(pieces):
            raise ValueError(f"with no inner layout, overrides must name all the pieces {pieces}")
        # shortcut: a PE holds one piece when it has inner layouts, else two pieces share its leaf
        # coordinates; to support it, add the slot to the path.
        if view.max_pieces > 1:
            raise NotImplementedError("a PE with several pieces cannot have inner layouts yet")
        for pid, child in inner.items():
            self.children[pid] = bind_tree(DataSource(None, view.partition_region(pid).size), child)

    # -- ownership ----------------------------------------------------------

    def _child_of(self, index: Index) -> Tuple["TreeView", Index]:
        pid = self.view.partition_of(index)
        offset = self.view.partition_region(pid).offset
        return self.children[pid], tuple(i - o for i, o in zip(index, offset))

    def primary_owner(self, index: int | Sequence[int]) -> Path:
        index = as_index(index)
        pe = self.view.primary_owner(index)
        if not self.children:
            return (pe,)
        child, sub = self._child_of(index)
        return (pe,) + child.primary_owner(sub)

    def holders(self, index: int | Sequence[int]) -> Tuple[Path, ...]:
        """Every leaf PE with a copy, the primary owner first."""
        index = as_index(index)
        here = self.view.holders(index)
        if not self.children:
            return tuple((pe,) for pe in here)
        child, sub = self._child_of(index)
        return tuple((pe,) + rest for pe in here for rest in child.holders(sub))

    def paths(self) -> Iterator[Path]:
        """Every leaf PE, in order. An inner PE holding no piece has no leaves."""
        for pe in self.view.grid.coords():
            if not self.children:
                yield (pe,)
            for pid in self.view.pieces_at(pe) if self.children else ():
                for rest in self.children[pid].paths():
                    yield (pe,) + rest

    # -- one leaf PE --------------------------------------------------------

    def _leaf(self, path: Path) -> Tuple["TreeView", Coord, Index]:
        """The leaf node on ``path``, its PE, and where the leaf's region starts globally."""
        node, offset = self, (0,) * len(self.view.shape)
        for pe in path[:-1]:
            if not node.children:
                raise ValueError(f"path {path} goes below a leaf")
            pid = node.view.partition_at(pe)
            shift = node.view.partition_region(pid).offset
            node, offset = node.children[pid], tuple(o + s for o, s in zip(offset, shift))
        if node.children:
            raise ValueError(f"path {path} stops above a leaf")
        return node, path[-1], offset

    def held_regions(self, path: Path) -> Tuple[Box, ...]:
        """What this leaf PE stores, in global coordinates."""
        node, pe, offset = self._leaf(path)
        return tuple(Box(tuple(a + b for a, b in zip(r.offset, offset)), r.size) for r in node.view.held_regions(pe))

    def local_shape(self, path: Path) -> Index:
        node, pe, _ = self._leaf(path)
        return node.view.local_shape(pe)

    def local_index(self, index: int | Sequence[int], path: Path) -> Index:
        node, pe, offset = self._leaf(path)
        return node.view.local_index(tuple(i - o for i, o in zip(as_index(index), offset)), pe)

    def global_index(self, local: int | Sequence[int], path: Path) -> Index:
        node, pe, offset = self._leaf(path)
        return tuple(i + o for i, o in zip(node.view.global_index(local, pe), offset))

    # -- the tree as a whole ------------------------------------------------

    def flatten(self) -> DistributedView:
        """This tree as one view, for code that takes views. Needs one inner layout per level."""
        view, layout = self.view, self.layout
        while True:
            if layout.overrides:
                raise ValueError("a tree with different inner layouts per piece has no single view")
            if layout.inner is None:
                return view
            layout = layout.inner
            view = nest(view, layout.cut, layout.map)

    def validate(self) -> None:
        """Check every node, by enumeration. For tests and small shapes."""
        self.view.validate()
        for child in self.children.values():
            child.validate()

    def classes(self) -> List[Tuple[Layout, Index, List[PartitionId]]]:
        """Pieces grouped by ``(inner layout, piece shape)``, as ``(layout, shape, pieces)``.

        Backends pay per class, not per piece: on AIR each one is a segment or a branch.
        """
        out: List[Tuple[Layout, Index, List[PartitionId]]] = []
        for pid, child in self.children.items():
            for layout, shape, pieces in out:
                if layout == child.layout and shape == child.view.shape:
                    pieces.append(pid)
                    break
            else:
                out.append((child.layout, child.view.shape, [pid]))
        return out

    def fit(self, capacity: Dict[str, int], element_bytes: int, depth: int = 1) -> Tuple[Fit, ...]:
        """For each named level, does its fullest memory fit in ``capacity[level]`` bytes?

        ``capacity`` is what is usable, so take off stacks and anything else stored
        there first. ``depth`` is the buffers per piece (2 for ping-pong). Replicas
        count, since each holds a copy.

        shortcut: counts this tensor without halos, per PE; pass a smaller capacity for
        the rest until the tree knows its other tensors.
        """
        if element_bytes < 1 or depth < 1:
            raise ValueError(f"need element_bytes >= 1 and depth >= 1, got {element_bytes}, {depth}")
        fullest: Dict[str, Tuple[int, Path]] = {}
        self._measure((), fullest)
        fits = []
        for level, (elements, where) in fullest.items():
            if capacity.get(level, 0) < 1:
                raise ValueError(f"no capacity given for level {level!r}")
            fits.append(Fit(level, elements * element_bytes * depth, capacity[level], where))
        return tuple(fits)

    def _measure(self, prefix: Path, fullest: Dict[str, Tuple[int, Path]]) -> None:
        for pe in self.view.grid.coords():
            level = self.layout.level
            n = sum(r.volume for r in self.view.held_regions(pe))
            if level is not None and n > fullest.get(level, (-1,))[0]:
                fullest[level] = (n, prefix + (pe,))
            for pid in self.view.pieces_at(pe) if self.children else ():
                self.children[pid]._measure(prefix + (pe,), fullest)


def bind_tree(source: Any, layout: Layout, global_shape: Optional[Sequence[int]] = None) -> TreeView:
    """Attach a layout tree to existing data, as :func:`bind` does. Nothing is copied."""
    return TreeView(bind(source, layout.cut, layout.map, global_shape), layout)
