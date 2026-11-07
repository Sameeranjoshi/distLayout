"""Layout trees: a layout inside every piece of another, kept as a tree, levels intact."""

from __future__ import annotations

import random

import pytest

from combos import Buf, all_indices, random_inner, random_view
from distlayout import (Block, Box, Decomposition, Grid, Layout, Map, Replicate, UnevenBlock, bind_tree, cyclic, nest)

G4, G22 = Grid(4), Grid((2, 2))
QUARTERS = Decomposition(Block(0, 2), Block(1, 2))
ON_G22 = Map(G22, {0: 0, 1: 1})

TILES = Layout(QUARTERS, ON_G22)
ROWS = Layout(Decomposition(Block(0, 4)), Map(G4, {0: 0}))
COLS = Layout(Decomposition(Block(1, 4)), Map(G4, {0: 0}))
CYCLIC = Layout(*cyclic(8, 4))
MIXED = {(0, 0): TILES, (0, 1): ROWS, (1, 0): COLS, (1, 1): CYCLIC}


def mixed():
    return bind_tree(Buf(16, 16), Layout(QUARTERS, ON_G22, level="L2", overrides=MIXED))


def check_every_element(tree):
    """Each element has one primary path, and that path's PE stores it once and gets it back."""
    seen = set()
    for path in tree.paths():
        for box in tree.held_regions(path):
            for i in box.indices():
                local = tree.local_index(i, path)
                assert tree.global_index(local, path) == i
                assert (path, local) not in seen
                seen.add((path, local))
    for i in all_indices(tree.view.shape):
        holders = tree.holders(i)
        assert holders[0] == tree.primary_owner(i)
        assert len(set(holders)) == len(holders)
        assert all(any(b.contains(i) for b in tree.held_regions(p)) for p in holders)


@pytest.mark.parametrize("index, owner", [
    ((0, 0), ((0, 0), (0, 0))),
    ((3, 7), ((0, 0), (0, 1))),
    ((4, 0), ((0, 0), (1, 0))),
    ((3, 12), ((0, 1), (1,))),      # row blocks: local row 3
    ((10, 3), ((1, 0), (1,))),      # column blocks: local column 3
    ((12, 8), ((1, 1), (0,))),      # cyclic rows: local row 4
    ((13, 8), ((1, 1), (1,))),
])
def test_each_quarter_uses_its_own_inner_layout(index, owner):
    assert mixed().primary_owner(index) == owner


def test_mixed_tree_stores_every_element_once():
    check_every_element(mixed())


def test_local_shape_comes_from_the_leaf_layout():
    tree = mixed()
    assert tree.local_shape(((0, 0), (0, 0))) == (4, 4)
    assert tree.local_shape(((0, 1), (2,))) == (2, 8)
    assert tree.local_shape(((1, 0), (2,))) == (8, 2)


def test_held_regions_are_global():
    assert mixed().held_regions(((1, 1), (1,))) == (Box((9, 8), (1, 8)), Box((13, 8), (1, 8)))


def test_outer_pieces_may_differ_in_size():
    cut = Decomposition(UnevenBlock(0, sizes=[6, 10]))
    tree = bind_tree(Buf(16, 4), Layout(cut, Map(Grid(2), {0: 0}), overrides={
        0: Layout(Decomposition(Block(0, 2)), Map(Grid(2), {0: 0})),
        1: Layout(Decomposition(Block(0, 5)), Map(Grid(5), {0: 0})),
    }))
    check_every_element(tree)
    assert tree.primary_owner((5, 0)) == ((0,), (1,))
    assert tree.primary_owner((15, 0)) == ((1,), (4,))


def test_three_levels():
    middle = Layout(Decomposition(Block(0, 2)), Map(Grid(2), {0: 0}), inner=ROWS)
    tree = bind_tree(Buf(32, 16), Layout(Decomposition(Block(0, 2)), Map(Grid(2), {0: 0}), inner=middle))
    check_every_element(tree)
    assert tree.primary_owner((31, 0)) == ((1,), (1,), (3,))


def test_replicated_outer_piece_has_a_path_per_copy():
    tree = bind_tree(Buf(8), Layout(Decomposition(), Replicate(Grid(2)), inner=Layout(*cyclic(8, 2))))
    assert tree.holders(3) == (((0,), (1,)), ((1,), (1,)))
    check_every_element(tree)


def test_a_uniform_tree_owns_what_nest_owns():
    built = 0
    for seed in range(200):
        rng = random.Random(seed)
        outer = random_view(rng)
        if outer.stacked:
            continue
        d, m = random_inner(rng, outer)
        try:
            flat = nest(outer, d, m)
        except NotImplementedError:
            continue
        tree = bind_tree(outer.source, Layout(outer.decomposition, outer.mapping, inner=Layout(d, m)))
        for i in all_indices(outer.shape):
            assert sum(tree.primary_owner(i), ()) == flat.primary_owner(i)
            assert {sum(p, ()) for p in tree.holders(i)} == set(flat.holders(i))
        assert tree.flatten().grid == flat.grid
        built += 1
    assert built > 20


def test_flatten_refuses_a_per_piece_tree():
    with pytest.raises(ValueError, match="different inner layouts"):
        mixed().flatten()


def test_overrides_replace_the_default_inner_layout_in_named_pieces():
    tree = bind_tree(Buf(16, 16), Layout(QUARTERS, ON_G22, inner=TILES, overrides={(1, 1): ROWS}))
    assert tree.primary_owner((3, 3)) == ((0, 0), (0, 0))
    assert tree.primary_owner((12, 12)) == ((1, 1), (2,))
    assert [(c[0], c[2]) for c in tree.classes()] == [(TILES, [(0, 0), (0, 1), (1, 0)]), (ROWS, [(1, 1)])]
    check_every_element(tree)


def test_overrides_without_a_default_must_name_every_piece():
    with pytest.raises(ValueError, match="pieces"):
        bind_tree(Buf(16, 16), Layout(QUARTERS, ON_G22, overrides={(0, 0): TILES}))


def test_overrides_cannot_name_a_piece_that_does_not_exist():
    with pytest.raises(ValueError, match="pieces"):
        bind_tree(Buf(16, 16), Layout(QUARTERS, ON_G22, inner=TILES, overrides={(5, 5): ROWS}))


def test_validate_checks_every_node():
    mixed().validate()


def test_inner_layout_that_cannot_cut_a_piece_is_refused():
    with pytest.raises(ValueError):
        bind_tree(Buf(4), Layout(Decomposition(Block(0, 2)), Map(Grid(2), {0: 0}), inner=Layout(*cyclic(3, 3))))


def test_several_pieces_on_one_pe_with_inner_layouts_are_refused():
    with pytest.raises(NotImplementedError, match="several pieces"):
        bind_tree(Buf(8), Layout(*cyclic(8, 2), inner=Layout(*cyclic(1, 1))))


def test_path_must_reach_a_leaf():
    with pytest.raises(ValueError, match="leaf"):
        mixed().held_regions(((0, 0),))
    with pytest.raises(ValueError, match="leaf"):
        mixed().held_regions(((0, 0), (0, 0), (0,)))


def test_element_outside_the_path_is_a_key_error():
    with pytest.raises(KeyError):
        mixed().local_index((15, 15), ((0, 0), (0, 0)))


def test_classes_group_pieces_by_inner_layout_and_shape():
    assert [c[2] for c in mixed().classes()] == [[(0, 0)], [(0, 1)], [(1, 0)], [(1, 1)]]
    same = bind_tree(Buf(16, 16), Layout(QUARTERS, ON_G22, inner=TILES, overrides={(1, 1): ROWS}))
    assert [(c[0], c[1], c[2]) for c in same.classes()] == [(TILES, (8, 8), [(0, 0), (0, 1), (1, 0)]), (ROWS, (8, 8), [(1, 1)])]


def test_same_inner_layout_on_different_shapes_is_two_classes():
    cut = Decomposition(UnevenBlock(0, sizes=[6, 10]))
    two = Layout(Decomposition(UnevenBlock(0, parts=2)), Map(Grid(2), {0: 0}))
    tree = bind_tree(Buf(16), Layout(cut, Map(Grid(2), {0: 0}), inner=two))
    assert [c[1] for c in tree.classes()] == [(6,), (10,)]


NPU = {"L2": 512 * 1024, "L1": 64 * 1024}


def levels(outer_cut, outer_map, inner_cut, inner_map):
    return Layout(outer_cut, outer_map, level="L2", inner=Layout(inner_cut, inner_map, level="L1"))


def test_fit_reports_the_fullest_memory_of_each_level():
    tree = bind_tree(Buf(16, 16), levels(QUARTERS, ON_G22, QUARTERS, ON_G22))
    l2, l1 = tree.fit(NPU, element_bytes=4)
    assert (l2.level, l2.needed, l2.available, l2.fits, l2.split) == ("L2", 256, 512 * 1024, True, 1)
    assert (l1.level, l1.needed, l1.fits) == ("L1", 64, True)
    assert l1.where == ((0, 0), (0, 0))


def test_fit_says_how_many_ways_to_split_a_piece_that_does_not_fit():
    whole = Layout(Decomposition(), Replicate(Grid(1)), level="L2")
    l2, = bind_tree(Buf(1024, 1024), whole).fit(NPU, element_bytes=4)
    assert (l2.needed, l2.fits, l2.split) == (4 * 1024 * 1024, False, 8)


def test_fit_counts_buffer_depth():
    whole = Layout(Decomposition(), Replicate(Grid(1)), level="L1")
    small = bind_tree(Buf(100, 100), whole)
    assert small.fit(NPU, 4, depth=1)[0].fits
    assert not small.fit(NPU, 4, depth=2)[0].fits


def test_fit_counts_every_piece_on_a_pe():
    l1, = bind_tree(Buf(8), Layout(*cyclic(8, 2), level="L1")).fit(NPU, 4)
    assert l1.needed == 16


def test_fit_skips_nodes_without_a_level_and_refuses_unknown_levels():
    assert bind_tree(Buf(8), Layout(*cyclic(8, 2))).fit({}, 4) == ()
    with pytest.raises(ValueError, match="L9"):
        bind_tree(Buf(8), Layout(*cyclic(8, 2), level="L9")).fit(NPU, 4)
