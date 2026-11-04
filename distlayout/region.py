"""Rectangular regions of a logical index space.

A :class:`Box` is a half-open hyper-rectangle ``[offset, offset + size)``. It is
the only region kind in v0: every block, uneven-block and replicated layout
produces boxes. Strided regions (cyclic) and index sets (irregular) come later
and will implement the same small interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterator, Optional, Sequence, Tuple

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
