"""Rectangular regions of a logical index space.

A :class:`Box` is a half-open hyper-rectangle ``[offset, offset + size)``. It is
the only region kind in v0: every block, uneven-block and replicated layout
produces boxes. Strided regions (cyclic) and index sets (irregular) come later
and will implement the same small interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, List, Optional, Sequence, Tuple

Index = Tuple[int, ...]


def as_index(index: int | Sequence[int]) -> Index:
    """Normalise ``5`` and ``(5,)`` and ``[5]`` to ``(5,)``."""
    if isinstance(index, int):
        return (index,)
    return tuple(int(i) for i in index)


@dataclass(frozen=True)
class Box:
    offset: Index
    size: Index

    def __post_init__(self) -> None:
        object.__setattr__(self, "offset", as_index(self.offset))
        object.__setattr__(self, "size", as_index(self.size))
        if len(self.offset) != len(self.size):
            raise ValueError(f"rank mismatch: offset {self.offset} size {self.size}")
        if any(s < 0 for s in self.size):
            raise ValueError(f"negative size {self.size}")

    @staticmethod
    def full(shape: Sequence[int]) -> "Box":
        shape = as_index(shape)
        return Box((0,) * len(shape), shape)

    @staticmethod
    def from_bounds(start: Sequence[int], stop: Sequence[int]) -> "Box":
        start, stop = as_index(start), as_index(stop)
        return Box(start, tuple(b - a for a, b in zip(start, stop)))

    @property
    def rank(self) -> int:
        return len(self.offset)

    @property
    def stop(self) -> Index:
        return tuple(o + s for o, s in zip(self.offset, self.size))

    @property
    def volume(self) -> int:
        v = 1
        for s in self.size:
            v *= s
        return v

    @property
    def empty(self) -> bool:
        return self.volume == 0

    def contains(self, index: int | Sequence[int]) -> bool:
        index = as_index(index)
        if len(index) != self.rank:
            raise ValueError(f"index {index} has rank {len(index)}, box has {self.rank}")
        return all(o <= i < o + s for i, o, s in zip(index, self.offset, self.size))

    def contains_box(self, other: "Box") -> bool:
        return other.empty or all(
            a0 <= b0 and b1 <= a1
            for a0, a1, b0, b1 in zip(self.offset, self.stop, other.offset, other.stop)
        )

    def intersect(self, other: "Box") -> Optional["Box"]:
        """The overlap of two boxes, or ``None`` if they are disjoint."""
        lo = tuple(max(a, b) for a, b in zip(self.offset, other.offset))
        hi = tuple(min(a, b) for a, b in zip(self.stop, other.stop))
        if any(h <= l for l, h in zip(lo, hi)):
            return None
        return Box.from_bounds(lo, hi)

    def shift(self, delta: Sequence[int]) -> "Box":
        """Translate by ``delta`` (e.g. global -> local coordinates with ``-origin``)."""
        return Box(tuple(o + d for o, d in zip(self.offset, as_index(delta))), self.size)

    def indices(self) -> Iterator[Index]:
        """Every index in the box, row-major. Intended for tests and small shapes."""
        return product(*(range(o, o + s) for o, s in zip(self.offset, self.size)))

    def __str__(self) -> str:
        return "[" + ", ".join(f"{a}:{b}" for a, b in zip(self.offset, self.stop)) + ")"


class IndexSet:
    """A set of ids on one axis, kept as sorted, merged half-open runs ``[(start, stop), ...]``.

    The region kind of irregular layouts: owned ids, ghost ids, import lists.
    Mostly-contiguous sets (a renumbered partition) stay a few runs long.
    """

    __slots__ = ("runs",)

    def __init__(self, runs: Sequence[Tuple[int, int]] = ()) -> None:
        merged: list = []
        for a, b in sorted((int(a), int(b)) for a, b in runs if b > a):
            if merged and a <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b))
            else:
                merged.append((a, b))
        self.runs = tuple(merged)

    @staticmethod
    def from_ids(ids) -> "IndexSet":
        ids = sorted(set(int(i) for i in ids))
        runs, start, prev = [], None, None
        for i in ids:
            if start is None:
                start = prev = i
            elif i == prev + 1:
                prev = i
            else:
                runs.append((start, prev + 1))
                start = prev = i
        if start is not None:
            runs.append((start, prev + 1))
        return IndexSet(runs)

    @staticmethod
    def from_boxes(boxes, axis: int = 0) -> "IndexSet":
        return IndexSet([(b.offset[axis], b.stop[axis]) for b in boxes])

    def ids(self) -> List[int]:
        return [i for a, b in self.runs for i in range(a, b)]

    def __iter__(self):
        return iter(self.ids())

    def __len__(self) -> int:
        return sum(b - a for a, b in self.runs)

    def __bool__(self) -> bool:
        return bool(self.runs)

    def __contains__(self, i: int) -> bool:
        return any(a <= i < b for a, b in self.runs)

    def union(self, o: "IndexSet") -> "IndexSet":
        return IndexSet(self.runs + o.runs)

    def intersect(self, o: "IndexSet") -> "IndexSet":
        out, i, j = [], 0, 0
        while i < len(self.runs) and j < len(o.runs):
            (a, b), (c, d) = self.runs[i], o.runs[j]
            lo, hi = max(a, c), min(b, d)
            if lo < hi:
                out.append((lo, hi))
            if b < d:
                i += 1
            else:
                j += 1
        return IndexSet(out)

    def difference(self, o: "IndexSet") -> "IndexSet":
        out = []
        for a, b in self.runs:
            cur = a
            for c, d in o.runs:
                if d <= cur or c >= b:
                    continue
                if c > cur:
                    out.append((cur, c))
                cur = max(cur, d)
            if cur < b:
                out.append((cur, b))
        return IndexSet(out)

    __or__, __and__, __sub__ = union, intersect, difference

    def boxes(self) -> List[Box]:
        return [Box((a,), (b - a,)) for a, b in self.runs]

    def __eq__(self, o: object) -> bool:
        return isinstance(o, IndexSet) and self.runs == o.runs

    def __hash__(self) -> int:
        return hash(self.runs)

    def __repr__(self) -> str:
        return "IndexSet{" + ", ".join(f"{a}" if b == a + 1 else f"{a}:{b}" for a, b in self.runs) + "}"
