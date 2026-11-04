"""Processor mapping: where partitions live on a logical PE grid.

The grid is a logical coordinate space -- for AIR, the herd's *iteration*
space, not its physical core count (``air.herd([range(0, n, tile)], shape=(2,))``
has n/tile logical points strip-mined onto 2 cores).

v0 rule set, kept deliberately strict so every PE's role is explicit:

* ``place={k: g}`` puts partition component ``k`` (the k-th decomposition rule)
  on grid axis ``g``. The rule's part count must equal the grid extent there:
  one partition per PE along that axis.
* ``replicate=(g, ...)`` makes every coordinate along grid axis ``g`` hold the
  same data. The primary owner is coordinate 0 on that axis; the others hold
  read-only replicas.
* A grid axis that is neither placed nor replicated is an error.

Several partitions per PE (cyclic, overdecomposition) is a later extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterator, Mapping as TMapping, Optional, Sequence, Tuple

from .region import Index, as_index

Coord = Tuple[int, ...]


@dataclass(frozen=True)
class Grid:
    shape: Tuple[int, ...]
    names: Optional[Tuple[str, ...]] = None

    def __init__(self, shape: int | Sequence[int], names: Optional[Sequence[str]] = None) -> None:
        shape = as_index(shape)
        if not shape or any(s < 1 for s in shape):
            raise ValueError(f"bad grid shape {shape}")
        if names is not None and len(names) != len(shape):
            raise ValueError(f"{len(names)} names for a rank-{len(shape)} grid")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "names", tuple(names) if names is not None else None)

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        n = 1
        for s in self.shape:
            n *= s
        return n

    def coords(self) -> Iterator[Coord]:
        return product(*(range(s) for s in self.shape))

    def check(self, pe: Sequence[int]) -> Coord:
        pe = as_index(pe)
        if len(pe) != self.rank or not all(0 <= c < s for c, s in zip(pe, self.shape)):
            raise IndexError(f"PE {pe} outside grid {self.shape}")
        return pe

    def axis_name(self, g: int) -> str:
        return self.names[g] if self.names else f"g{g}"


@dataclass(frozen=True)
class Mapping:
    grid: Grid
    place: Tuple[Tuple[int, int], ...]   # (partition component, grid axis)
    replicate: Tuple[int, ...]

    def __init__(self, grid: Grid, place: Optional[TMapping[int, int]] = None,
                 replicate: Sequence[int] = ()) -> None:
        place = dict(place or {})
        replicate = tuple(sorted(set(replicate)))
        used = list(place.values())
        if len(set(used)) != len(used):
            raise ValueError(f"two partition components placed on one grid axis: {place}")
        for g in used + list(replicate):
            if not 0 <= g < grid.rank:
                raise ValueError(f"grid axis {g} outside rank-{grid.rank} grid")
        both = set(used) & set(replicate)
        if both:
            raise ValueError(f"grid axes {sorted(both)} are both placed and replicated")
        missing = set(range(grid.rank)) - set(used) - set(replicate)
        if missing:
            raise ValueError(
                f"grid axes {sorted(missing)} are neither placed nor replicated; "
                f"say which with place= or replicate="
            )
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "place", tuple(sorted(place.items())))
        object.__setattr__(self, "replicate", replicate)

    @property
    def place_dict(self) -> Dict[int, int]:
        return dict(self.place)

    def check(self, num_parts: Tuple[int, ...]) -> None:
        """Every partition component placed, and part counts match the grid."""
        placed = self.place_dict
        if set(placed) != set(range(len(num_parts))):
            raise ValueError(
                f"decomposition has {len(num_parts)} components, mapping places {sorted(placed)}"
            )
        for k, g in placed.items():
            if num_parts[k] != self.grid.shape[g]:
                raise ValueError(
                    f"component {k} has {num_parts[k]} parts but grid axis {g} has "
                    f"{self.grid.shape[g]} PEs (v0 needs one partition per PE)"
                )

    def partition_at(self, pe: Coord, ncomp: int) -> Tuple[int, ...]:
        placed = self.place_dict
        return tuple(pe[placed[k]] for k in range(ncomp))

    def pes_of(self, pid: Tuple[int, ...]) -> Tuple[Coord, Coord]:
        """``(primary, [all holders])`` coordinates of a partition."""
        base = [0] * self.grid.rank
        for k, g in self.place:
            base[g] = pid[k]
        primary = tuple(base)
        holders = []
        for rep in product(*(range(self.grid.shape[g]) for g in self.replicate)):
            c = list(base)
            for g, r in zip(self.replicate, rep):
                c[g] = r
            holders.append(tuple(c))
        return primary, tuple(holders)  # type: ignore[return-value]

    def is_primary(self, pe: Coord) -> bool:
        return all(pe[g] == 0 for g in self.replicate)


def Map(grid: Grid, place: Optional[TMapping[int, int]] = None, replicate: Sequence[int] = ()) -> Mapping:
    """Shorthand: ``Map(grid, {0: 0}, replicate=[1])``."""
    return Mapping(grid, place, replicate)


def Replicate(grid: Grid) -> Mapping:
    """Every PE holds the whole tensor; the primary owner is the origin PE."""
    return Mapping(grid, {}, tuple(range(grid.rank)))
