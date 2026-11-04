"""The AIR adapter, without compiled MLIR-AIR.

``air.api``'s ``IndexExpr`` (``python/air/api/_index.py``) is pure Python, so the
herd coordinates here are the *real* upstream index type when an MLIR-AIR
checkout is available (``MLIR_AIR_SRC``, default ``~/xilinx/mlir-air``). Buffers
and ``air.ops`` are recording fakes: the test checks the exact subscripts
handed to ``air.ops.load``/``store``, which is everything the adapter decides.
"""

import importlib.util
import os
from pathlib import Path

import pytest

from distlayout import Block, DataSource, Decomposition, Grid, Map, Replicate, UnevenBlock, bind
from distlayout.backends import air as air_backend


def _load_index_module():
    candidates = [os.environ.get("MLIR_AIR_SRC"), Path.home() / "xilinx" / "mlir-air", "/home/user/xilinx/mlir-air"]
    for root in filter(None, candidates):
        path = Path(root) / "python" / "air" / "api" / "_index.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("air_api_index", path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


INDEX = _load_index_module()


class FakeBuffer:
    def __init__(self, name, *shape):
        self.name, self.shape = name, shape

    def __getitem__(self, key):
        return ("region", self.name, key)


class FakeAir:
    def __init__(self):
        self.calls = []
        outer = self

        class ops:
            @staticmethod
            def load(dst, src):
                outer.calls.append(("load", dst, src))

            @staticmethod
            def store(src, dst):
                outer.calls.append(("store", src, dst))

        self.ops = ops

    def alloc(self, shape, dtype, scope=None):
        self.calls.append(("alloc", shape, dtype, scope))
        return FakeBuffer("l1", *shape)


def coord(name):
    if INDEX is not None:
        return INDEX.IndexExpr.leaf(value=None, name=name, spatial=True)
    pytest.skip("no MLIR-AIR checkout; set MLIR_AIR_SRC")


def as_form(x):
    """An IndexExpr or int as ({leaf name: coeff}, const) for comparison."""
    if isinstance(x, int):
        return {}, x
    return {leaf.name: c for leaf, c in x.terms.items()}, x.const


def key_forms(key):
    return [(as_form(s.start), as_form(s.stop)) for s in key]


# -- the upstream eltwise_add_with_l2 pattern -------------------------------

def test_block_matches_handwritten_eltwise_add():
    n, tile_n = 16384, 1024
    l2 = FakeBuffer("l2_a", n)
    v = bind(l2, Decomposition(Block(0, n // tile_n)), Map(Grid(n // tile_n), {0: 0}))
    tx = coord("tx")
    key = air_backend.slice_key(v, (tx,))
    hand = slice(tx * tile_n, tx * tile_n + tile_n)    # what the example writes
    assert key_forms(key) == key_forms((hand,)) == [(({"tx": 1024}, 0), ({"tx": 1024}, 1024))]
    # static size recovered the way air.api does it
    assert (key[0].stop - key[0].start).as_const() == tile_n
    assert air_backend.herd_space(v) == [range(0, n, tile_n)]


def test_load_store_and_alloc_calls():
    fake = FakeAir()
    l2 = FakeBuffer("l2_a", 100)
    v = bind(l2, Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    tx = coord("tx")
    a = air_backend.alloc_local(v, "f32", scope="h.private", air=fake)
    air_backend.load_local(v, a, (tx,), air=fake)
    air_backend.store_local(v, a, (tx,), dst=FakeBuffer("l2_c", 100), air=fake)
    assert fake.calls[0] == ("alloc", [25], "f32", "h.private")
    kind, dst, (_, name, key) = fake.calls[1]
    assert (kind, dst, name) == ("load", a, "l2_a")
    assert key_forms(key) == [(({"tx": 25}, 0), ({"tx": 25}, 25))]
    kind, src, (_, name, key) = fake.calls[2]
    assert (kind, name) == ("store", "l2_c")


def test_replicated_axis_does_not_enter_offset():
    # 4 blocks on a 4x4 herd, replicated along ty: every ty names the same region
    v = bind(FakeBuffer("l2", 100), Decomposition(Block(0, 4)), Map(Grid((4, 4)), {0: 0}, replicate=[1]))
    tx, ty = coord("tx"), coord("ty")
    key = air_backend.slice_key(v, (tx, ty))
    assert key_forms(key) == [(({"tx": 25}, 0), ({"tx": 25}, 25))]


def test_full_replication_is_constant_region():
    v = bind(FakeBuffer("l2", 100), Decomposition(), Replicate(Grid((4, 4))))
    key = air_backend.slice_key(v, (coord("tx"), coord("ty")))
    assert key_forms(key) == [(({}, 0), ({}, 100))]


def test_2d_block():
    v = bind(FakeBuffer("l2", 16, 32), Decomposition(Block(0, 2), Block(1, 2)), Map(Grid((2, 2)), {0: 0, 1: 1}))
    key = air_backend.slice_key(v, (coord("tx"), coord("ty")))
    assert key_forms(key) == [(({"tx": 8}, 0), ({"tx": 8}, 8)), (({"ty": 16}, 0), ({"ty": 16}, 16))]


def test_partial_source_subtracts_origin():
    # the handle holds global [50, 100); only PEs 2 and 3 of the view live in it
    src = DataSource(FakeBuffer("piece", 50), global_shape=(100,), origin=(50,))
    v = bind(src, Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    with pytest.raises(ValueError, match="source window"):
        air_backend.slice_key(v, (coord("tx"),))      # PEs 0, 1 fall outside the handle
    # the offset formula itself carries -origin
    full = DataSource(FakeBuffer("whole", 100), global_shape=(100,))
    assert air_backend.offsets(bind(full, Decomposition(Block(0, 4)), Map(Grid(4), {0: 0})), (3,)) == (75,)
    assert v.source_region(v.held_region(3)).offset == (25,)


def test_int_coords_give_concrete_slices():
    v = bind(FakeBuffer("l2", 100), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    assert air_backend.slice_key(v, (2,)) == (slice(50, 75),)


def test_uneven_is_refused_with_a_reason():
    v = bind(FakeBuffer("l2", 99), Decomposition(UnevenBlock(0, parts=4)), Map(Grid(4), {0: 0}))
    with pytest.raises(NotImplementedError, match="uneven"):
        air_backend.slice_key(v, (0,))
