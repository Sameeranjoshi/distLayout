import pytest

from distlayout import (Block, Box, DataSource, Decomposition, Grid, Map, Replicate,
                        UnevenBlock, bind)


class Handle:
    """Stand-in for an existing buffer: anything with a shape."""

    def __init__(self, *shape):
        self.shape = shape


# -- the five design examples ----------------------------------------------

def test_block_100_over_4x1():
    v = bind(Handle(100), Decomposition(Block(0, 4)), Map(Grid((4, 1)), {0: 0}, replicate=[1]))
    v.validate()
    assert [v.held_region((p, 0)) for p in range(4)] == [Box((25 * p,), (25,)) for p in range(4)]
    assert v.local_shape((3, 0)) == (25,)
    assert v.primary_owner(55) == (2, 0)
    assert v.local_index(55, (2, 0)) == (5,)
    assert v.global_index(5, (2, 0)) == (55,)


def test_block_plus_replication_100_on_4x4():
    v = bind(Handle(100), Decomposition(Block(0, 4)), Map(Grid((4, 4)), {0: 0}, replicate=[1]))
    v.validate()
    assert v.held_region((2, 1)) == Box((50,), (25,))
    assert v.owned_region((2, 1)) is None            # a replica, not the owner
    assert v.owned_region((2, 0)) == Box((50,), (25,))
    assert v.primary_owner(55) == (2, 0)
    assert v.holders(55) == ((2, 0), (2, 1), (2, 2), (2, 3))


def test_uneven_99_over_4x1():
    v = bind(Handle(99), Decomposition(UnevenBlock(0, parts=4)), Map(Grid((4, 1)), {0: 0}, replicate=[1]))
    v.validate()
    assert [v.local_shape((p, 0)) for p in range(4)] == [(25,), (25,), (25,), (24,)]
    assert v.held_region((3, 0)) == Box((75,), (24,))
    assert v.max_local_shape() == (25,)              # what a padded allocation would use
    assert v.shape == (99,)                          # padding never changes the tensor
    assert not v.is_uniform()


def test_full_replication_100_on_4x4():
    v = bind(Handle(100), Decomposition(), Replicate(Grid((4, 4))))
    v.validate()
    assert all(v.held_region(pe) == Box((0,), (100,)) for pe in v.grid.coords())
    assert v.primary_owner(42) == (0, 0)
    assert len(v.holders(42)) == 16
    assert sum(v.owned_region(pe) is not None for pe in v.grid.coords()) == 1


def test_2d_block_16x32_on_2x2():
    v = bind(Handle(16, 32), Decomposition(Block(0, 2), Block(1, 2)), Map(Grid((2, 2)), {0: 0, 1: 1}))
    v.validate()
    assert v.held_region((1, 0)) == Box((8, 0), (8, 16))
    assert v.held_region((0, 1)) == Box((0, 16), (8, 16))
    assert v.primary_owner((9, 20)) == (1, 1)
    assert v.local_index((9, 20), (1, 1)) == (1, 4)


# -- more decomposition behaviour ------------------------------------------

def test_uneven_policies_and_explicit_sizes():
    assert UnevenBlock(0, parts=4, policy="balanced").sizes(10) == (3, 3, 2, 2)
    assert UnevenBlock(0, parts=4, policy="tail").sizes(10) == (3, 3, 3, 1)
    v = bind(Handle(100), Decomposition(UnevenBlock(0, sizes=[20, 25, 25, 30])), Map(Grid(4), {0: 0}))
    v.validate()
    assert v.held_region(3) == Box((70,), (30,))
    assert v.partition_of(69) == (2,)


def test_placement_can_transpose_axes():
    # tensor axis 0 on grid axis 1 and vice versa
    v = bind(Handle(4, 6), Decomposition(Block(0, 2), Block(1, 3)), Map(Grid((3, 2)), {0: 1, 1: 0}))
    v.validate()
    assert v.held_region((2, 1)) == Box((2, 4), (2, 2))


def test_bind_does_not_copy():
    h = Handle(100)
    v = bind(h, Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    assert v.source.handle is h


def test_partial_source_window():
    # the handle holds global [50, 100) of a 100-element tensor
    piece = Handle(50)
    src = DataSource(piece, global_shape=(100,), origin=(50,))
    v = bind(src, Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    assert v.source_region(v.held_region(2)) == Box((0,), (25,))
    assert v.source_region(v.held_region(3)) == Box((25,), (25,))
    with pytest.raises(ValueError):
        v.source_region(v.held_region(0))       # not inside this handle


# -- materialization plans ---------------------------------------------------

def test_plan_own_region_is_local():
    v = bind(Handle(100), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    p = v.plan(1)
    assert p.is_local and p.transfers[0].region == Box((25,), (25,))


def test_plan_halo_request_splits_at_owner_boundaries():
    v = bind(Handle(100), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    p = v.plan(1, Box((24,), (27,)))            # [24, 51): one halo element each side
    assert [(t.region, t.dst_offset, t.src_pe, t.local) for t in p.transfers] == [
        (Box((24,), (1,)), (0,), (0,), False),
        (Box((25,), (25,)), (1,), None, True),
        (Box((50,), (1,)), (26,), (2,), False),
    ]


def test_plan_on_replica_is_local():
    v = bind(Handle(100), Decomposition(Block(0, 4)), Map(Grid((4, 4)), {0: 0}, replicate=[1]))
    assert v.plan((3, 2)).is_local


# -- errors ----------------------------------------------------------------

@pytest.mark.parametrize("make", [
    lambda: bind(Handle(99), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0})),          # indivisible
    lambda: Map(Grid((4, 4)), {0: 0}),                                                   # axis 1 unspecified
    lambda: bind(Handle(100), Decomposition(Block(0, 4)), Map(Grid(8), {0: 0})),         # 4 parts on 8 PEs
    lambda: bind(Handle(100), Decomposition(Block(1, 4)), Map(Grid(4), {0: 0})),         # no axis 1
    lambda: Decomposition(Block(0, 2), Block(0, 2)),                                     # axis cut twice
    lambda: bind(Handle(100), Decomposition(UnevenBlock(0, sizes=[50, 49])), Map(Grid(2), {0: 0})),
    lambda: DataSource(Handle(30), global_shape=(100,), origin=(50,)),                   # handle != window
])
def test_rejects(make):
    with pytest.raises(ValueError):
        make()


def test_local_index_of_foreign_element():
    v = bind(Handle(100), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    with pytest.raises(KeyError):
        v.local_index(10, 3)


def test_to_attr():
    v = bind(Handle(100), Decomposition(Block(0, 4)), Map(Grid((4, 4)), {0: 0}, replicate=[1]))
    assert v.to_attr() == ("#dist.layout<shape = [100], grid = [4, 4], decomp = [block<axis = 0, parts = 4>], "
                           "place = [0 -> g0], replicate = [g1]>")
