"""Uneven layouts and halos for AIR: one static-size branch per size class (no MLIR-AIR needed)."""

from contextlib import contextmanager

import pytest

from distlayout import Block, Box, Decomposition, Grid, Map, UnevenBlock, bind
from distlayout.backends import air as air_backend
from distlayout.backends.schedule import Text


class Buf:
    def __init__(self, *shape):
        self.shape = shape


class Recorder:
    """Records the branch tree and the keys handed to the body."""

    def __init__(self):
        self.log = []
        rec = self

        class ops:
            @staticmethod
            def branch(cond):
                return rec._branch(cond)

        self.ops = ops

    def _branch(self, cond):
        rec = self

        class B:
            def __enter__(self):
                rec.log.append(f"if {cond}:")
                return self

            def __exit__(self, *a):
                return False

            @contextmanager
            def otherwise(self):
                rec.log.append("else:")
                yield

        return B()


def s(key):
    return [f"{k.start}:{k.stop}" for k in key]


def concrete(view, pe, halo=0, boundary="zero"):
    out = []
    air_backend.for_each_region(view, pe, out.append, halo=halo, boundary=boundary, air=Recorder())
    assert len(out) == 1
    return out[0]


def check_against_layout(view, halo, boundary="zero"):
    """For every PE: source window == clipped halo request; owned window == held region; tile lines up."""
    tile_shape = air_backend.region_shape(view, halo, boundary)
    for pe in view.grid.coords():
        k = concrete(view, pe, halo, boundary)
        held = view.held_region(pe)
        lo = [max(0, o - halo) for o in held.offset]
        hi = [min(n, e + halo) for e, n in zip(held.stop, view.shape)]
        assert [(x.start, x.stop) for x in k["src"]] == list(zip(lo, hi))
        assert [(x.start, x.stop) for x in k["own_src"]] == list(zip(held.offset, held.stop))
        pb = k["pad_before"] or [0] * len(view.shape)
        pa = k["pad_after"] or [0] * len(view.shape)
        for a, (src, tile, own_t, own_s) in enumerate(zip(k["src"], k["tile"], k["own_tile"], k["own_src"])):
            assert tile.stop - tile.start == (src.stop - src.start) + pb[a] + pa[a] <= tile_shape[a]
            assert own_t.start - pb[a] + src.start == own_s.start          # tile index -> global index
            assert own_t.stop - own_t.start == own_s.stop - own_s.start
        for a, (src, win) in enumerate(zip(k["src"], k["window"])):
            assert win.start == pb[a] and win.stop - win.start == src.stop - src.start
        if boundary in ("zero", "pad"):                                     # every PE sees a full tile
            assert [t.stop for t in k["tile"]] == [h.size[a] + 2 * halo for a, h in enumerate([held] * len(held.size))]


def test_uneven_99_over_4_has_two_classes():
    v = bind(Buf(99), Decomposition(UnevenBlock(0, parts=4)), Map(Grid(4), {0: 0}))
    check_against_layout(v, 0)
    rec, out = Recorder(), []
    air_backend.for_each_region(v, [Text.var("tx")], out.append, air=rec)
    assert rec.log == ["if tx <= 2:", "else:"]
    assert [s(k["src"]) for k in out] == [["tx*25:tx*25 + 25"], ["75:99"]]
    assert air_backend.region_shape(v) == (25,)
    assert [s(k["tile"]) for k in out] == [["0:25"], ["0:24"]]          # last PE fills 24 of its 25 slots


def test_explicit_uneven_sizes_runs():
    v = bind(Buf(100), Decomposition(UnevenBlock(0, sizes=[20, 25, 25, 30])), Map(Grid(4), {0: 0}))
    check_against_layout(v, 0)
    assert [(r["c0"], r["c1"], r["size"]) for r in air_backend.size_classes(v)[0]] == [(0, 0, 20), (1, 2, 25), (3, 3, 30)]


def test_halo_1d_zero_boundary():
    v = bind(Buf(48), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    check_against_layout(v, 1)
    rec, out = Recorder(), []
    air_backend.for_each_region(v, [Text.var("tx")], out.append, halo=1, air=rec)
    assert rec.log == ["if tx <= 0:", "else:", "if tx <= 2:", "else:"]       # first, interior, last
    assert [s(k["src"]) for k in out] == [["0:13"], ["tx*12 - 1:tx*12 + 13"], ["35:48"]]
    assert [(k["pad_before"], k["pad_after"]) for k in out] == [([1], None), (None, None), (None, [1])]
    assert [s(k["window"]) for k in out] == [["1:14"], ["0:14"], ["0:13"]]   # where the window sits in the 14-tile
    assert air_backend.region_shape(v, 1) == (14,)


def test_halo_clip_boundary_gives_smaller_edge_tiles():
    v = bind(Buf(48), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    check_against_layout(v, 1, boundary="clip")
    assert s(concrete(v, (0,), 1, "clip")["tile"]) == ["0:13"] and s(concrete(v, (1,), 1, "clip")["tile"]) == ["0:14"]


def test_halo_2d_nine_classes_reduce_to_corners_edges_interior():
    v = bind(Buf(16, 32), Decomposition(Block(0, 4), Block(1, 4)), Map(Grid((4, 4)), {0: 0, 1: 1}))
    check_against_layout(v, 1)
    rec = Recorder()
    out = []
    air_backend.for_each_region(v, [Text.var("tx"), Text.var("ty")], out.append, halo=1, air=rec)
    assert len(out) == 9 and air_backend.region_shape(v, 1) == (6, 10)


def test_halo_with_whole_axis_pads_it_too():
    v = bind(Buf(8, 16), Decomposition(Block(0, 2)), Map(Grid(2), {0: 0}))
    check_against_layout(v, 2)
    k = concrete(v, (0,), 2)
    assert s(k["src"]) == ["0:6", "0:16"] and k["pad_before"] == [2, 2] and k["pad_after"] == [0, 2]


def test_uneven_plus_halo():
    v = bind(Buf(10), Decomposition(UnevenBlock(0, sizes=[3, 3, 4])), Map(Grid(3), {0: 0}))
    check_against_layout(v, 1)


def test_load_local_uses_classes_for_uneven():
    calls = []

    class Air(Recorder):
        def __init__(self):
            super().__init__()
            base = self.ops

            class ops(base):
                @staticmethod
                def load(dst, src, pad_before=None, pad_after=None):
                    calls.append((dst, src, pad_before, pad_after))
            self.ops = ops

    class B(Buf):
        def __getitem__(self, key):
            return (self.shape, tuple(s(key)))

    v = bind(B(99), Decomposition(UnevenBlock(0, parts=4)), Map(Grid(4), {0: 0}))
    a = B(25)
    air_backend.load_local(v, a, (3,), air=Air())
    assert calls == [(((25,), ("0:24",)), ((99,), ("75:99",)), None, None)]


def test_single_region_api_still_refuses_uneven():
    v = bind(Buf(99), Decomposition(UnevenBlock(0, parts=4)), Map(Grid(4), {0: 0}))
    with pytest.raises(NotImplementedError, match="size class"):
        air_backend.slice_key(v, (0,))
