"""Data decomposition: how a logical index space is cut into partitions.

A decomposition knows nothing about processors. It answers *what the pieces
are*; :mod:`distlayout.mapping` answers *where they live*.

A :class:`Decomposition` is a product of per-axis rules. Each rule cuts one
tensor axis into contiguous parts; axes without a rule stay whole. A partition
is identified by a tuple with one part index per rule, in rule order, so a
``Decomposition(Block(0, 2), Block(1, 2))`` of a 16x32 tensor has partitions
``(0, 0) .. (1, 1)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product
from typing import Iterator, Optional, Sequence, Tuple

from .region import Box, Index, as_index

PartitionId = Tuple[int, ...]


class AxisRule(ABC):
    """Cuts one tensor axis into ``num_parts`` contiguous, ordered parts."""

    axis: int

    @abstractmethod
    def num_parts(self, extent: int) -> int: ...

    @abstractmethod
    def bounds(self, part: int, extent: int) -> Tuple[int, int]:
        """``[start, stop)`` of ``part`` on an axis of length ``extent``."""

    def part_of(self, i: int, extent: int) -> int:
        """The part holding element ``i``. Overridden with closed forms."""
        for p in range(self.num_parts(extent)):
            start, stop = self.bounds(p, extent)
            if start <= i < stop:
                return p
        raise IndexError(f"element {i} outside axis of extent {extent}")

    def sizes(self, extent: int) -> Tuple[int, ...]:
        return tuple(b - a for a, b in (self.bounds(p, extent) for p in range(self.num_parts(extent))))

    def is_uniform(self, extent: int) -> bool:
        return len(set(self.sizes(extent))) == 1

    def check(self, extent: int) -> None:
        """Raise if this rule cannot cut an axis of this length."""
        self.num_parts(extent)
        for p in range(self.num_parts(extent)):
            self.bounds(p, extent)


@dataclass(frozen=True)
class Block(AxisRule):
    """Equal contiguous parts. ``extent`` must be divisible by ``parts``.

    An indivisible extent is an error rather than silently rounded: use
    :class:`UnevenBlock` to say which PEs get the remainder.
    """

    axis: int
    parts: int

    def __post_init__(self) -> None:
        if self.parts < 1:
            raise ValueError(f"Block needs parts >= 1, got {self.parts}")

    def num_parts(self, extent: int) -> int:
        return self.parts

    def block_size(self, extent: int) -> int:
        if extent % self.parts:
            raise ValueError(
                f"Block(axis={self.axis}, parts={self.parts}) cannot cut extent {extent} "
                f"evenly ({extent} % {self.parts} = {extent % self.parts}); "
                f"use UnevenBlock to place the remainder"
            )
        return extent // self.parts

    def bounds(self, part: int, extent: int) -> Tuple[int, int]:
        b = self.block_size(extent)
        _check_part(part, self.parts)
        return part * b, (part + 1) * b

    def part_of(self, i: int, extent: int) -> int:
        _check_elem(i, extent)
        return i // self.block_size(extent)


@dataclass(frozen=True)
class UnevenBlock(AxisRule):
    """Contiguous parts of different lengths.

    Give exactly one of:

    * ``sizes``: explicit part lengths, e.g. ``(25, 25, 25, 24)``; they must sum
      to the axis extent.
    * ``parts`` (+ ``policy``): computed from the extent at query time.
      ``"balanced"`` gives the first ``extent % parts`` parts one extra element
      (99 over 4 -> 25, 25, 25, 24); ``"tail"`` makes every part
      ``ceil(extent / parts)`` except a shorter last one (99 over 4 -> 25, 25, 25, 24,
      but 10 over 4 -> 3, 3, 3, 1).

    Padding is *not* part of the layout: the valid extents are exact, and a
    backend may still allocate ``max(sizes)`` slots per PE.
    """

    axis: int
    parts: Optional[int] = None
    sizes_: Optional[Tuple[int, ...]] = field(default=None)
    policy: str = "balanced"

    def __init__(self, axis: int, parts: Optional[int] = None, sizes: Optional[Sequence[int]] = None,
                 policy: str = "balanced") -> None:
        if (parts is None) == (sizes is None):
            raise ValueError("UnevenBlock takes exactly one of parts= or sizes=")
        if policy not in ("balanced", "tail"):
            raise ValueError(f"unknown policy {policy!r}; use 'balanced' or 'tail'")
        if sizes is not None:
            sizes = tuple(int(s) for s in sizes)
            if not sizes or any(s < 0 for s in sizes):
                raise ValueError(f"bad sizes {sizes}")
        if parts is not None and parts < 1:
            raise ValueError(f"UnevenBlock needs parts >= 1, got {parts}")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "sizes_", sizes)
        object.__setattr__(self, "policy", policy)

    def num_parts(self, extent: int) -> int:
        return len(self.sizes_) if self.sizes_ is not None else self.parts  # type: ignore[return-value]

    def sizes(self, extent: int) -> Tuple[int, ...]:
        if self.sizes_ is not None:
            if sum(self.sizes_) != extent:
                raise ValueError(f"UnevenBlock sizes {self.sizes_} sum to {sum(self.sizes_)}, axis extent is {extent}")
            return self.sizes_
        p = self.parts
        assert p is not None
        if self.policy == "balanced":
            q, r = divmod(extent, p)
            return tuple(q + 1 if k < r else q for k in range(p))
        b = -(-extent // p)  # ceil
        out = []
        left = extent
        for _ in range(p):
            out.append(min(b, left))
            left -= out[-1]
        return tuple(out)

    def bounds(self, part: int, extent: int) -> Tuple[int, int]:
        s = self.sizes(extent)
        _check_part(part, len(s))
        start = sum(s[:part])
        return start, start + s[part]

    def __repr__(self) -> str:
        if self.sizes_ is not None:
            return f"UnevenBlock(axis={self.axis}, sizes={self.sizes_})"
        return f"UnevenBlock(axis={self.axis}, parts={self.parts}, policy={self.policy!r})"


@dataclass(frozen=True)
class Decomposition:
    """A product of per-axis rules. ``Decomposition()`` is one whole partition."""

    rules: Tuple[AxisRule, ...]

    def __init__(self, *rules: AxisRule) -> None:
        axes = [r.axis for r in rules]
        if len(set(axes)) != len(axes):
            raise ValueError(f"two rules cut the same tensor axis: {axes}")
        object.__setattr__(self, "rules", tuple(rules))

    def check(self, shape: Sequence[int]) -> None:
        shape = as_index(shape)
        for r in self.rules:
            if not 0 <= r.axis < len(shape):
                raise ValueError(f"{r!r} cuts axis {r.axis}, tensor has rank {len(shape)}")
            r.check(shape[r.axis])

    def num_parts(self, shape: Sequence[int]) -> Tuple[int, ...]:
        shape = as_index(shape)
        return tuple(r.num_parts(shape[r.axis]) for r in self.rules)

    def partitions(self, shape: Sequence[int]) -> Iterator[PartitionId]:
        return product(*(range(n) for n in self.num_parts(shape)))

    def region(self, pid: PartitionId, shape: Sequence[int]) -> Box:
        shape = as_index(shape)
        pid = as_index(pid)
        if len(pid) != len(self.rules):
            raise ValueError(f"partition id {pid} needs {len(self.rules)} components")
        start, stop = [0] * len(shape), list(shape)
        for r, p in zip(self.rules, pid):
            start[r.axis], stop[r.axis] = r.bounds(p, shape[r.axis])
        return Box.from_bounds(start, stop)

    def partition_of(self, index: Index, shape: Sequence[int]) -> PartitionId:
        shape, index = as_index(shape), as_index(index)
        if not Box.full(shape).contains(index):
            raise IndexError(f"index {index} outside tensor of shape {shape}")
        return tuple(r.part_of(index[r.axis], shape[r.axis]) for r in self.rules)

    def is_uniform(self, shape: Sequence[int]) -> bool:
        shape = as_index(shape)
        return all(r.is_uniform(shape[r.axis]) for r in self.rules)

    def __repr__(self) -> str:
        return "Decomposition(" + ", ".join(map(repr, self.rules)) + ")"


def _check_part(part: int, n: int) -> None:
    if not 0 <= part < n:
        raise IndexError(f"part {part} out of range [0, {n})")


def _check_elem(i: int, extent: int) -> None:
    if not 0 <= i < extent:
        raise IndexError(f"element {i} outside axis of extent {extent}")
