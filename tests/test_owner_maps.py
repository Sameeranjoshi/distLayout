"""Category 2: structured affine owner maps (cut into tiles + owner(t) expression)."""

import pytest

from distlayout import (Block, Box, Decomposition, Grid, Map, OwnerMap, UnevenBlock, bind, block_cyclic,
                        cyclic, parse, refine, skew, t)
from distlayout.backends import air as air_backend
from distlayout.expr import substitute


class Buf:
    def __init__(self, *shape):
        self.shape = shape


# -- expressions -------------------------------------------------------------

@pytest.mark.parametrize("text,printed,at,value", [
    ("t0 mod 4", "t0 mod 4", (6,), 2),
    ("(t0 // 2) % 4", "(t0 floordiv 2) mod 4", (11,), 1),
    ("(t0 + t1) mod 4", "(t0 + t1) mod 4", (3, 2), 1),
    ("(t0 + 2*t1) mod 4", "(t0 + t1 * 2) mod 4", (1, 1), 3),
    ("t0 - 1", "t0 - 1", (5,), 4),
])
def test_parse_print_eval(text, printed, at, value):
    e = parse(text)
    assert str(e) == printed and e.eval(at) == value and parse(str(e)) == e


def test_operators_build_the_same_expression():
    assert (t(0) + 2 * t(1)) % 4 == parse("(t0 + t1 * 2) mod 4")
    assert substitute(t(0) % 4, 0, t(0) // 2) == parse("(t0 floordiv 2) mod 4")


@pytest.mark.parametrize("bad", ["t0 * t1", "t0 mod t1", "t0 +", "(t0", "x0"])
def test_parse_rejects_non_affine(bad):
    with pytest.raises((ValueError, TypeError)):
        parse(bad)


# -- the slide-5/6 layouts -----------------------------------------------------

def test_cyclic():
    v = bind(Buf(16), *cyclic(16, 4))
    v.validate()
    assert [v.primary_owner(i)[0] for i in range(8)] == [0, 1, 2, 3, 0, 1, 2, 3]
    assert v.held_regions(1) == tuple(Box((i,), (1,)) for i in (1, 5, 9, 13))
    assert v.local_shape(1) == (4, 1) and v.local_index(9, 1) == (2, 0)


def test_block_cyclic():
    v = bind(Buf(16), *block_cyclic(16, 2, 4))
    v.validate()
    assert [v.primary_owner(i)[0] for i in range(16)] == [0, 0, 1, 1, 2, 2, 3, 3] * 2
    assert v.held_regions(1) == (Box((2,), (2,)), Box((10,), (2,)))
    assert v.local_index(11, 1) == (1, 1) and v.global_index((1, 1), 1) == (11,)


def test_skew():
    v = bind(Buf(4, 4), *skew((4, 4), 4))
    v.validate()
    owners = [[v.primary_owner((i, j))[0] for j in range(4)] for i in range(4)]
    assert owners == [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]


def test_tile_splitting_owner_map():
    # 4x4 logical tiles of a 8x8 tensor; tile (tx, ty) goes to PE (tx + 2*ty) mod 4
    v = bind(Buf(8, 8), Decomposition(Block(0, 4), Block(1, 4)), OwnerMap(Grid(4), {0: "(t0 + 2*t1) mod 4"}))
    v.validate()
    assert v.primary_owner((2, 4)) == ((1 + 2 * 2) % 4,)
    assert len(v.pieces_at(0)) == 4 and v.local_shape(0) == (4, 2, 2)


def test_uneven_tiles_get_a_flat_local_buffer():
    v = bind(Buf(10), Decomposition(UnevenBlock(0, sizes=[3, 3, 2, 2])), OwnerMap(Grid(2), {0: "t0 mod 2"}))
    v.validate()
    assert v.local_shape(0) == (5,) and v.local_index(6, 0) == (3,)    # piece [6:8) after [0:3)


def test_replicated_owner_map_and_empty_pes():
    v = bind(Buf(8), Decomposition(Block(0, 2)), OwnerMap(Grid((4, 2)), {0: "t0"}, replicate=[1]))
    v.validate()
    assert v.holders(5) == ((1, 0), (1, 1))
    assert v.pieces_at((3, 0)) == () and v.local_shape((3, 0)) == (0,)
    assert v.owned_regions((1, 1)) == ()


def test_halo_plan_over_cyclic():
    v = bind(Buf(16), *block_cyclic(16, 4, 2))       # PE0: [0:4) [8:12), PE1: [4:8) [12:16)
    p = v.plan(0, Box((3,), (6,)))                    # [3, 9)
    assert [(t.region, t.local, t.src_pe) for t in p.transfers] == [
        (Box((3,), (1,)), True, None), (Box((4,), (4,)), False, (1,)), (Box((8,), (1,)), True, None)]


@pytest.mark.parametrize("make", [
    lambda: OwnerMap(Grid((4, 4)), {0: "t0 mod 4"}),                                     # g1 unspecified
    lambda: bind(Buf(16), Decomposition(Block(0, 16)), OwnerMap(Grid(4), {0: "t0"})),     # lands outside grid
    lambda: bind(Buf(16), Decomposition(Block(0, 4)), OwnerMap(Grid(4), {0: "t1"})),      # no t1
])
def test_owner_map_rejects(make):
    with pytest.raises(ValueError):
        make()


def test_to_attr_and_repr():
    v = bind(Buf(16), *block_cyclic(16, 2, 4))
    assert v.to_attr().endswith("owner = [g0 = t0 mod 4], replicate = []>")
    assert repr(v.mapping) == "OwnerMap(Grid(4), {0: 't0 mod 4'})"


# -- rank splitting = refine --------------------------------------------------

def test_refine_block_is_owner_tx_ty():
    rank = bind(Buf(64), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))   # 4 ranks of 16
    split = refine(rank, axis=0, k=2)
    split.validate()
    assert split.grid.shape == (4, 2)
    assert split.to_attr().endswith("owner = [g0 = t0 floordiv 2, g1 = t0 mod 2], replicate = []>")
    for r in range(4):                                    # rank r's 16 elements now on PEs (r, 0), (r, 1)
        assert split.held_region((r, 0)) == Box((16 * r,), (8,))
        assert split.held_region((r, 1)) == Box((16 * r + 8,), (8,))
    assert not split.stacked


def test_refine_whole_axis_adds_a_cut():
    split = refine(bind(Buf(8, 4096), Decomposition(Block(0, 8)), Map(Grid(8), {0: 0})), axis=1, k=4)
    assert split.held_region((3, 2)) == Box((3, 2048), (1, 1024)) and split.local_shape((3, 2)) == (1, 1024)


def test_refine_uneven_and_keeps_replication():
    v = bind(Buf(10), Decomposition(UnevenBlock(0, sizes=[6, 4])), Map(Grid((2, 2)), {0: 0}, replicate=[1]))
    split = refine(v, axis=0, k=2)
    split.validate()
    assert [split.held_region((p, 0, s)) for p in range(2) for s in range(2)] == [
        Box((0,), (3,)), Box((3,), (3,)), Box((6,), (2,)), Box((8,), (2,))]
    assert split.holders(7) == ((1, 0, 0), (1, 1, 0)) and split.holders(8) == ((1, 0, 1), (1, 1, 1))


def test_air_backend_refuses_owner_maps_for_now():
    with pytest.raises(NotImplementedError, match="owner maps"):
        air_backend.slice_key(bind(Buf(16), *cyclic(16, 4)), (0,))
