"""Owner-map inversion for AIR emission (no MLIR-AIR needed).

``schedule.solve`` turns "piece t -> PE owner(t)" into "PE x, slot s -> piece
t(x, s)". Every schedule is verified by enumeration inside ``solve``; these
tests pin the closed forms and the emitted subscripts.
"""

import pytest

from distlayout import Block, Decomposition, Grid, Map, OwnerMap, bind, block_cyclic, cyclic, refine, skew
from distlayout.backends import air as air_backend
from distlayout.backends.schedule import Text, solve


class Buf:
    def __init__(self, *shape):
        self.shape = shape

    def __getitem__(self, key):
        return ("region", key)


class Recorder:
    """Records air.ops calls; air.sequential yields printable loop variables."""

    def __init__(self):
        self.calls, self.loops = [], []
        rec = self

        class ops:
            @staticmethod
            def load(dst, src):
                rec.calls.append(("load", dst, src))

            @staticmethod
            def store(src, dst):
                rec.calls.append(("store", src, dst))

        self.ops = ops

    def sequential(self, lo, hi):
        name = f"s{len(self.loops)}"
        self.loops.append((name, hi))
        yield Text.var(name)

    def alloc(self, shape, dtype, scope=None):
        return Buf(*shape)


def subs(key):
    return [f"{k.start} : {k.stop}" if isinstance(k, slice) else str(k) for k in key]


def emitted(view):
    rec = Recorder()
    x = [Text.var(n) for n in ["tx", "ty", "tz"][: view.grid.rank]]
    air_backend.for_each_piece(view, x, lambda slot, key: rec.calls.append((str(slot), subs(key))), air=rec)
    return rec.loops, rec.calls


def test_cyclic_closed_form():
    v = bind(Buf(16), *cyclic(16, 4))
    sch = solve(v)
    assert sch.loops == [4] and sch.concrete((1,)) == [(0, (1,)), (1, (5,)), (2, (9,)), (3, (13,))]
    loops, calls = emitted(v)
    assert loops == [("s0", 4)] and calls == [("s0", ["tx % 4 + s0*4 : tx % 4 + s0*4 + 1"])]


def test_block_cyclic_closed_form():
    v = bind(Buf(16), *block_cyclic(16, 2, 4))
    loops, calls = emitted(v)
    assert loops == [("s0", 2)] and calls == [("s0", ["(tx % 4)*2 + s0*8 : (tx % 4)*2 + s0*8 + 2"])]
    assert [p for _, p in solve(v).concrete((3,))] == [(3,), (7,)]


def test_skew_closed_form_keeps_piece_order():
    v = bind(Buf(8, 8), *skew((8, 8), 4))
    sch = solve(v)
    assert sch.loops == [8, 2]
    for pe in v.grid.coords():                       # slot order == the view's piece order
        assert [p for _, p in sch.concrete(pe)] == list(v.pieces_at(pe))
    loops, calls = emitted(v)
    assert calls == [("s0*2 + s1", ["s0 : s0 + 1", "(tx - s0) % 4 + s1*4 : (tx - s0) % 4 + s1*4 + 1"])]


def test_tile_split_uses_the_unit_coefficient():
    v = bind(Buf(8, 8), Decomposition(Block(0, 4), Block(1, 4)), OwnerMap(Grid(4), {0: "(t0 + 2*t1) mod 4"}))
    sch = solve(v)
    for pe in v.grid.coords():
        assert sorted(p for _, p in sch.concrete(pe)) == sorted(v.pieces_at(pe))
    _, calls = emitted(v)
    # t0 has a single trip (no loop), so the one loop variable is t1's
    assert calls[0][1] == ["((tx - s0*2) % 4)*2 : ((tx - s0*2) % 4)*2 + 2", "s0*2 : s0*2 + 2"]


def test_refine_is_loop_free():
    v = refine(bind(Buf(64), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0})), axis=0, k=2)
    sch = solve(v)
    assert sch.loops == [] and sch.concrete((2, 1)) == [(0, (5,))]
    loops, calls = emitted(v)
    assert loops == [] and calls == [("0", ["tx*16 + ty*8 : tx*16 + ty*8 + 8"])]


def test_replicated_owner_map():
    v = bind(Buf(16), Decomposition(Block(0, 8)), OwnerMap(Grid((4, 2)), {0: "t0 mod 4"}, replicate=[1]))
    sch = solve(v)
    assert sch.concrete((2, 0)) == sch.concrete((2, 1)) == [(0, (2,)), (1, (6,))]


def test_local_shape_is_a_tile_stack():
    assert air_backend.local_shape(bind(Buf(16), *block_cyclic(16, 2, 4))) == (2, 2)
    assert air_backend.local_shape(bind(Buf(8, 8), *skew((8, 8), 4))) == (16, 1, 1)
    assert air_backend.herd_space(bind(Buf(16), *cyclic(16, 4))) == [range(0, 4)]


def test_load_local_fills_the_stack():
    rec = Recorder()
    v = bind(Buf(16), *block_cyclic(16, 2, 4))
    a = Buf(2, 2)
    air_backend.load_local(v, a, (Text.var("tx"),), air=rec)
    kind, dst, src = rec.calls[0]
    assert kind == "load" and subs(dst[1]) == ["s0", "0 : 2"] and subs(src[1]) == ["(tx % 4)*2 + s0*8 : (tx % 4)*2 + s0*8 + 2"]


@pytest.mark.parametrize("make,msg", [
    (lambda: bind(Buf(10), *cyclic(10, 4)), "not a multiple"),                                    # 10 pieces on 4 PEs
    (lambda: bind(Buf(8), Decomposition(Block(0, 8)), OwnerMap(Grid(4), {0: "(t0 * 2) mod 4"})), "coefficient"),
    (lambda: bind(Buf(8, 8), Decomposition(Block(0, 2), Block(1, 2)),
                  OwnerMap(Grid((2, 2)), {0: "t0", 1: "t0"})), "two grid axes"),
])
def test_refuses_what_it_cannot_invert(make, msg):
    with pytest.raises(NotImplementedError, match=msg):
        solve(make())
