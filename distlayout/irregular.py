"""Category 3: irregular partitions -- owner arrays, sparse access patterns, ghost schedules.

The standard decomposition, as in PETSc, Tpetra, Zoltan and the PARTI/CHAOS
inspector-executor line, keeps three things apart:

* **the partition** -- an owner array ``owner[i]`` over an index space (rows,
  vertices, mesh cells), from any partitioner;
* **the data's sparsity** -- a :class:`CSRPattern` (a graph is its adjacency
  matrix), which only says *which ids each row touches*;
* **the communication** -- derived from both by :func:`inspect`: per PE, its
  owned ids, ghost ids, imports grouped by owner, exports, and a local CSR
  renumbered owned-first-then-ghosts.

::

    pattern = grid_graph(8, 8)                       # 5-point stencil on an 8x8 mesh
    owner = bfs_partition(pattern, 4)                # or METIS, or your own
    rows = bind(y, *owner_array(owner))              # pieces = runs of equal owner
    sched = inspect(rows, pattern)                   # x laid out like the rows
    sched[2].ghosts, sched[2].imports                # IndexSet, {src_pe: IndexSet}

``owner_array`` turns the owner array into ordinary pieces (an ``UnevenBlock``
of runs) plus a :class:`~distlayout.mapping.TableMap`, so views, ``plan``,
local numbering and ``validate`` all work unchanged. :func:`renumber` permutes
ids so each PE's ids are contiguous: the table becomes a placement, and the
owned part becomes an ordinary uneven block.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .decomposition import Decomposition, UnevenBlock
from .mapping import Grid, TableMap
from .region import IndexSet
from .view import DistributedView


# -- partitions -----------------------------------------------------------------

def owner_array(owner: Sequence[int], axis: int = 0, grid: Optional[Grid] = None) -> Tuple[Decomposition, TableMap]:
    """``(Decomposition, TableMap)`` for ``owner[i]`` = linear PE index of id ``i`` along ``axis``.

    Consecutive ids with the same owner form one piece, so a contiguous
    partition has one piece per PE and a scattered one has many.
    """
    owner = [int(o) for o in owner]
    if not owner:
        raise ValueError("empty owner array")
    if min(owner) < 0:
        raise ValueError("owner ids must be >= 0")
    grid = grid or Grid(max(owner) + 1)
    sizes, table = [], []
    for o in owner:
        if table and table[-1] == o:
            sizes[-1] += 1
        else:
            sizes.append(1)
            table.append(o)
    return Decomposition(UnevenBlock(axis, sizes=sizes)), TableMap(grid, table)


def block_partition(n: int, parts: int) -> List[int]:
    """Contiguous, balanced: the first ``n % parts`` PEs get one extra id."""
    q, r = divmod(n, parts)
    out: List[int] = []
    for p in range(parts):
        out += [p] * (q + (1 if p < r else 0))
    return out


def random_partition(n: int, parts: int, seed: int = 0) -> List[int]:
    """Balanced but scattered: what a partition that ignores the graph looks like."""
    owner = block_partition(n, parts)
    random.Random(seed).shuffle(owner)
    return owner


def bfs_partition(pattern: "CSRPattern", parts: int) -> List[int]:
    """Greedy graph growing: fill each part breadth-first from an unassigned vertex.

    A small stand-in for METIS / Scotch: balanced to within one, and parts are
    connected regions, so few edges cross parts.
    """
    n = pattern.nrows
    owner = [-1] * n
    targets = [len(r) for r in _counts(n, parts)]
    nxt = 0
    for p in range(parts):
        while nxt < n and owner[nxt] != -1:
            nxt += 1
        if nxt == n:
            break
        q, taken = deque([nxt]), 0
        while taken < targets[p]:
            if not q:                                 # region exhausted: restart from the next free vertex
                rest = [i for i in range(n) if owner[i] == -1]
                if not rest:
                    break
                q.append(rest[0])
            v = q.popleft()
            if owner[v] != -1:
                continue
            owner[v] = p
            taken += 1
            q.extend(u for u in pattern.row(v) if u < n and owner[u] == -1)
    return [o if o >= 0 else parts - 1 for o in owner]


def _counts(n: int, parts: int) -> List[range]:
    q, r = divmod(n, parts)
    return [range(q + (1 if p < r else 0)) for p in range(parts)]


def renumber(owner: Sequence[int]) -> Tuple[List[int], List[int]]:
    """``(perm, new_owner)``: ``perm[new] = old``, stable by owner, so each PE's ids become contiguous.

    Apply ``perm`` to the data and :meth:`CSRPattern.permute` to the pattern.
    """
    perm = sorted(range(len(owner)), key=lambda i: (owner[i], i))
    return perm, [owner[i] for i in perm]


# -- sparse access pattern -----------------------------------------------------

@dataclass(frozen=True)
class CSRPattern:
    """Rows of a sparse matrix (or adjacency lists of a graph): row ``i`` touches ``colidx[rowptr[i]:rowptr[i+1]]``."""

    rowptr: Tuple[int, ...]
    colidx: Tuple[int, ...]
    ncols: int

    def __init__(self, rowptr: Sequence[int], colidx: Sequence[int], ncols: Optional[int] = None) -> None:
        rowptr, colidx = tuple(int(v) for v in rowptr), tuple(int(v) for v in colidx)
        if not rowptr or rowptr[0] != 0 or rowptr[-1] != len(colidx) or any(b < a for a, b in zip(rowptr, rowptr[1:])):
            raise ValueError("rowptr must start at 0, be non-decreasing and end at len(colidx)")
        ncols = ncols if ncols is not None else (max(colidx) + 1 if colidx else 0)
        if colidx and (min(colidx) < 0 or max(colidx) >= ncols):
            raise ValueError("column index outside [0, ncols)")
        object.__setattr__(self, "rowptr", rowptr)
        object.__setattr__(self, "colidx", colidx)
        object.__setattr__(self, "ncols", ncols)

    @staticmethod
    def from_rows(rows: Sequence[Sequence[int]], ncols: Optional[int] = None) -> "CSRPattern":
        rowptr, colidx = [0], []
        for r in rows:
            colidx += sorted(set(int(c) for c in r))
            rowptr.append(len(colidx))
        return CSRPattern(rowptr, colidx, ncols)

    @property
    def nrows(self) -> int:
        return len(self.rowptr) - 1

    @property
    def nnz(self) -> int:
        return len(self.colidx)

    def row(self, i: int) -> Tuple[int, ...]:
        return self.colidx[self.rowptr[i]:self.rowptr[i + 1]]

    def needed(self, rows: IndexSet) -> IndexSet:
        """Columns touched by ``rows``: what a PE owning those rows must read."""
        return IndexSet.from_ids(c for i in rows for c in self.row(i))

    def permute(self, perm: Sequence[int]) -> "CSRPattern":
        """Symmetric renumbering: new row/col ``k`` is old ``perm[k]``."""
        inv = [0] * len(perm)
        for new, old in enumerate(perm):
            inv[old] = new
        return CSRPattern.from_rows([[inv[c] for c in self.row(old)] for old in perm], self.ncols)

    def cut_edges(self, owner: Sequence[int]) -> int:
        """Entries ``(i, j)`` with ``owner[i] != owner[j]`` (each undirected edge counted from both ends)."""
        return sum(1 for i in range(self.nrows) for j in self.row(i) if owner[i] != owner[j])


def grid_graph(nx: int, ny: int, perm: Optional[Sequence[int]] = None) -> CSRPattern:
    """5-point stencil on an ``nx x ny`` mesh, node ``(i, j)`` = id ``i*ny + j`` (diagonal included).

    With ``perm`` the ids are scrambled (``perm[new] = old``): the same mesh, but
    a numbering in which contiguous blocks of ids are scattered over the mesh.
    """
    rows = []
    for i in range(nx):
        for j in range(ny):
            r = [i * ny + j]
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny:
                    r.append(a * ny + b)
            rows.append(r)
    p = CSRPattern.from_rows(rows)
    return p.permute(perm) if perm is not None else p


def grid_positions(nx: int, ny: int, perm: Optional[Sequence[int]] = None) -> List[Tuple[int, int]]:
    """Mesh coordinates of each id (matching :func:`grid_graph` with the same ``perm``)."""
    pos = [(i, j) for i in range(nx) for j in range(ny)]
    return [pos[old] for old in perm] if perm is not None else pos


# -- the inspector --------------------------------------------------------------

@dataclass
class GhostPlan:
    """What one PE needs for ``y[rows] = A[rows, :] @ x`` under the given layouts."""

    pe: Tuple[int, ...]
    rows: IndexSet                                  # rows this PE computes
    owned: IndexSet                                 # x ids this PE holds
    needed: IndexSet                                # x ids its rows touch
    ghosts: IndexSet                                # needed - owned
    imports: Dict[Tuple[int, ...], IndexSet]        # src PE -> ids to receive
    exports: Dict[Tuple[int, ...], IndexSet] = field(default_factory=dict)   # dst PE -> ids to send

    def local_ids(self) -> List[int]:
        """Local x numbering: owned ids, then ghost ids (each ascending)."""
        return self.owned.ids() + self.ghosts.ids()

    def local_csr(self, pattern: CSRPattern) -> CSRPattern:
        """This PE's rows with column ids renumbered into :meth:`local_ids` positions."""
        pos = {g: k for k, g in enumerate(self.local_ids())}
        return CSRPattern.from_rows([[pos[c] for c in pattern.row(i)] for i in self.rows], ncols=len(pos))

    @property
    def x_footprint(self) -> int:
        """x entries this PE stores: owned + ghosts (vs ``ncols`` if x were replicated)."""
        return len(self.owned) + len(self.ghosts)


class GhostSchedule(dict):
    """``{pe: GhostPlan}`` plus totals."""

    @property
    def messages(self) -> int:
        return sum(len(p.imports) for p in self.values())

    @property
    def volume(self) -> int:
        return sum(len(p.ghosts) for p in self.values())

    def summary(self) -> str:
        lines = [f"{'PE':>10} {'rows':>5} {'owned x':>8} {'ghosts':>7} {'from PEs':>10}"]
        for pe, p in self.items():
            src = ",".join(str(s[0] if len(s) == 1 else s) for s in p.imports)
            lines.append(f"{str(pe):>10} {len(p.rows):>5} {len(p.owned):>8} {len(p.ghosts):>7} {src:>10}")
        lines.append(f"total ghost volume {self.volume}, messages {self.messages}")
        return "\n".join(lines)


def _ids_at(view: DistributedView, pe) -> IndexSet:
    if len(view.shape) != 1:
        raise NotImplementedError("the inspector works on 1-D index spaces (vectors / matrix rows)")
    return IndexSet.from_boxes(view.held_regions(pe))


def inspect(rows: DistributedView, pattern: CSRPattern, x: Optional[DistributedView] = None) -> GhostSchedule:
    """Ghost sets and import/export lists for ``y = A @ x`` with ``A``'s rows laid out by ``rows``.

    ``x`` defaults to ``rows`` (a square system, x laid out like y). Both views
    must be on the same grid. Ghost owners are found by run, not per id.
    """
    x = x or rows
    if rows.grid.shape != x.grid.shape:
        raise ValueError("rows and x must be laid out on the same grid")
    if rows.shape[0] != pattern.nrows or x.shape[0] != pattern.ncols:
        raise ValueError(f"pattern is {pattern.nrows} x {pattern.ncols}, layouts are {rows.shape[0]} rows, "
                         f"{x.shape[0]} x entries")
    pieces = [(x.partition_region(pid), x.mapping.pes_of(pid)[0]) for pid in x.partitions()]
    sched = GhostSchedule()
    for pe in rows.grid.coords():
        r = _ids_at(rows, pe)
        owned = _ids_at(x, pe)
        needed = pattern.needed(r)
        ghosts = needed - owned
        imports: Dict[Tuple[int, ...], IndexSet] = {}
        for box, src in pieces:
            got = ghosts & IndexSet([(box.offset[0], box.stop[0])])
            if got:
                imports[src] = imports.get(src, IndexSet()) | got
        sched[pe] = GhostPlan(pe, r, owned, needed, ghosts, dict(sorted(imports.items())))
    for pe, p in sched.items():
        for src, ids in p.imports.items():
            sched[src].exports[pe] = ids
    return sched
