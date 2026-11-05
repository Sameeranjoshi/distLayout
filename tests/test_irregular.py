"""Category 3: owner arrays, CSR access patterns, the ghost inspector."""

import random

import pytest

from distlayout import (Block, Decomposition, Grid, IndexSet, Map, TableMap, bfs_partition, bind, block_partition,
                        grid_graph, inspect, owner_array, random_partition, renumber)
from distlayout.irregular import CSRPattern, grid_positions
from distlayout.mapping import as_placement


class Buf:
    def __init__(self, *shape):
        self.shape = shape


# -- IndexSet ------------------------------------------------------------------

def test_indexset_ops():
    a = IndexSet.from_ids([0, 1, 2, 5, 7, 8])
    b = IndexSet([(2, 6)])
    assert a.runs == ((0, 3), (5, 6), (7, 9)) and len(a) == 6
    assert (a | b).runs == ((0, 6), (7, 9))
    assert (a & b).ids() == [2, 5]
    assert (a - b).ids() == [0, 1, 7, 8]
    assert 8 in a and 4 not in a and repr(a) == "IndexSet{0:3, 5, 7:9}"
    assert IndexSet.from_ids([]) == IndexSet() and not IndexSet()


# -- owner arrays ---------------------------------------------------------------

def test_owner_array_runs_and_table():
    owner = [0, 0, 1, 1, 1, 0, 2, 2, 3]
    d, m = owner_array(owner)
    assert d.rules[0].sizes(9) == (2, 3, 1, 2, 1) and m.linear() == [0, 1, 0, 2, 3]
    v = bind(Buf(9), d, m)
    v.validate()
    assert [v.primary_owner(i)[0] for i in range(9)] == owner
    assert v.held_regions(0)[1].offset == (5,) and v.local_index(5, 0) == (2,)      # owned ids, renumbered densely
    assert v.to_attr().endswith("owner = table<[0, 1, 0, 2, 3]>, replicate = []>")


def test_contiguous_owner_array_is_a_placement():
    d, m = owner_array(block_partition(10, 3))
    assert d.rules[0].sizes(10) == (4, 3, 3)
    assert as_placement(m, d.num_parts((10,))) is not None


def test_table_map_with_replication():
    v = bind(Buf(6), Decomposition(Block(0, 3)), TableMap(Grid((2, 2)), [(1, 0), (0, 0), (1, 0)], replicate=[1]))
    v.validate()
    assert v.holders(0) == ((1, 0), (1, 1)) and v.pieces_at((1, 1)) == ((0,), (2,))
    with pytest.raises(ValueError):
        TableMap(Grid((2, 2)), [(1, 1)], replicate=[1])         # table must name the primary


# -- the inspector, against brute force ----------------------------------------

def brute(owner, pattern, pe):
    rows = [i for i, o in enumerate(owner) if o == pe]
    needed = {c for i in rows for c in pattern.row(i)}
    owned = set(rows)
    ghosts = needed - owned
    imports = {}
    for g in ghosts:
        imports.setdefault(owner[g], set()).add(g)
    return set(rows), owned, ghosts, imports


@pytest.mark.parametrize("which", ["block", "random", "bfs"])
def test_inspector_matches_brute_force(which):
    nx = ny = 6
    perm = list(range(nx * ny))
    random.Random(3).shuffle(perm)
    pat = grid_graph(nx, ny, perm)
    owner = {"block": block_partition(36, 4), "random": random_partition(36, 4, seed=5),
             "bfs": bfs_partition(pat, 4)}[which]
    v = bind(Buf(36), *owner_array(owner))
    sched = inspect(v, pat)
    for pe in v.grid.coords():
        rows, owned, ghosts, imports = brute(owner, pat, pe[0])
        p = sched[pe]
        assert set(p.rows) == rows and set(p.owned) == owned and set(p.ghosts) == ghosts
        assert {s[0]: set(ids) for s, ids in p.imports.items()} == imports
        for dst, ids in p.exports.items():                           # exports mirror imports
            assert sched[dst].imports[pe] == ids
        local = p.local_csr(pat)                                     # local CSR maps back to the same columns
        ids = p.local_ids()
        assert [sorted(ids[c] for c in local.row(k)) for k in range(local.nrows)] == [sorted(pat.row(i)) for i in p.rows]
    assert sched.volume == sum(len(brute(owner, pat, q)[2]) for q in range(4))


def test_bfs_beats_id_blocks_on_a_scrambled_mesh_and_renumbering_makes_it_contiguous():
    nx = ny = 8
    perm = list(range(64))
    random.Random(1).shuffle(perm)
    pat = grid_graph(nx, ny, perm)
    good, bad = bfs_partition(pat, 4), block_partition(64, 4)
    assert sorted(good.count(p) for p in range(4)) == [16] * 4
    assert pat.cut_edges(good) < pat.cut_edges(bad) / 2
    assert inspect(bind(Buf(64), *owner_array(good)), pat).volume < inspect(bind(Buf(64), *owner_array(bad)), pat).volume / 2
    p2, new = renumber(good)
    v = bind(Buf(64), *owner_array(new))
    assert len(list(v.partitions())) == 4 and as_placement(v.mapping, v.decomposition.num_parts(v.shape)) is not None
    assert inspect(v, pat.permute(p2)).volume == inspect(bind(Buf(64), *owner_array(good)), pat).volume


def test_de_replication_footprint():
    pat = grid_graph(8, 8)
    s = inspect(bind(Buf(64), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0})), pat)
    assert [p.x_footprint for p in s.values()] == [24, 32, 32, 24]      # owned 16 + one or two mesh rows of ghosts


def test_rectangular_system_with_separate_x_layout():
    pat = CSRPattern.from_rows([[0, 5], [1], [2, 3], [7]], ncols=8)       # 4 x 8
    rows = bind(Buf(4), Decomposition(Block(0, 2)), Map(Grid(2), {0: 0}))
    x = bind(Buf(8), Decomposition(Block(0, 2)), Map(Grid(2), {0: 0}))
    s = inspect(rows, pat, x)
    assert s[(0,)].ghosts.ids() == [5] and s[(0,)].imports == {(1,): IndexSet.from_ids([5])}
    assert s[(1,)].ghosts.ids() == [2, 3] and s[(1,)].exports == {(0,): IndexSet.from_ids([5])}


def test_mesh_helpers_and_errors():
    assert grid_positions(2, 3)[4] == (1, 1) and grid_graph(2, 3).row(4) == (1, 3, 4, 5)
    with pytest.raises(ValueError):
        CSRPattern([0, 2], [0])                          # rowptr does not end at len(colidx)
    with pytest.raises(ValueError):
        inspect(bind(Buf(5), *owner_array([0, 0, 1, 1, 1])), grid_graph(2, 3))
