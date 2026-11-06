"""Against real MLIR-AIR (``from air import api``). Skipped when it is not installed.

Install the prebuilt wheel, e.g.::

    pip install --no-deps mlir_air-<ver>-cp313-cp313-linux_x86_64.whl numpy ml_dtypes filelock==3.13.1

The equivalence test needs upstream's ``programming_examples/eltwise_add_with_l2``
from a checkout at ``$MLIR_AIR_SRC`` (default ``~/xilinx/mlir-air``) whose
``build_module`` matches the installed wheel.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

air = pytest.importorskip("air.api", reason="MLIR-AIR Python package not installed")
np = pytest.importorskip("numpy")

from air.api.types import f32  # noqa: E402

from distlayout import Block, Decomposition, Grid, Map, Replicate, UnevenBlock, bind  # noqa: E402
from distlayout.backends import air as air_backend  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _upstream_example():
    roots = [os.environ.get("MLIR_AIR_SRC"), Path.home() / "xilinx" / "mlir-air", "/home/user/xilinx/mlir-air"]
    rel = Path("programming_examples") / "eltwise_add_with_l2" / "eltwise_add.py"
    for root in filter(None, roots):
        if (Path(root) / rel).exists():
            return _load(Path(root) / rel, "upstream_eltwise_add_with_l2")
    pytest.skip(f"no upstream {rel}; set MLIR_AIR_SRC")


@pytest.mark.parametrize("n,tile", [(16384, 1024), (4096, 512), (2048, 1024)])
def test_eltwise_add_matches_upstream_module(n, tile):
    up = _upstream_example()
    mine = _load(ROOT / "examples" / "eltwise_add_with_l2_dist.py", "distlayout_eltwise_add")
    a = str(up.build_module(n, tile, np.float32).build(target="npu2"))
    b = str(mine.build_module(n, tile, np.float32).build(target="npu2"))
    assert a == b


def _build_copy(view_of, herd_shape, n_local):
    """L3 -> L2 -> (layout) -> L1 -> L2' -> L3, returning the module text."""
    A = air.tensor(list(view_of.shape), f32)
    C = air.tensor(list(view_of.shape), f32)
    with air.launch(name="k") as launch:
        @launch.body
        def _():
            with air.segment(name="s") as seg:
                @seg.body
                def _():
                    l2 = air.alloc(list(view_of.shape), f32, scope=seg.private())
                    out = air.alloc(list(view_of.shape), f32, scope=seg.private())
                    v = bind(l2, view_of.decomposition, view_of.mapping)
                    air.ops.load(l2, A)
                    with air.herd(air_backend.herd_space(v), name="h", shape=herd_shape) as h:
                        def body(*coords):
                            a = air_backend.alloc_local(v, f32, scope=h.private())
                            air_backend.load_local(v, a, coords)
                            if n_local is not None:
                                air_backend.store_local(v, a, coords, dst=out)
                        # air.api counts the body's named parameters
                        if len(herd_shape) == 1:
                            h.body(lambda tx: body(tx))
                        else:
                            h.body(lambda tx, ty: body(tx, ty))
                    air.ops.store(out, C)
    return str(launch.build(target="npu2"))


class _Shape:
    def __init__(self, *shape):
        self.shape = shape


def test_replication_builds_and_every_core_reads_the_whole_buffer():
    v = bind(_Shape(256), Decomposition(), Replicate(Grid(4)))
    text = _build_copy(v, (4,), None)
    assert "memref<256xf32, 2" in text          # each core's L1 holds all 256


def test_2d_block_builds():
    v = bind(_Shape(64, 128), Decomposition(Block(0, 2), Block(1, 2)), Map(Grid((2, 2)), {0: 0, 1: 1}))
    text = _build_copy(v, (2, 2), 1)
    assert "memref<32x64xf32, 2" in text        # one quarter per core


# -- category 2: owner maps through the schedule ---------------------------------

from distlayout import OwnerMap, block_cyclic, cyclic, nest, refine, skew  # noqa: E402

OWNER_MAP_CASES = {
    "cyclic": (lambda: bind(_Shape(64), *cyclic(64, 4)), "(s0 mod 4 + s1 * 4)", 1),
    "block_cyclic": (lambda: bind(_Shape(256), *block_cyclic(256, 16, 4)), "((s0 mod 4) * 16 + s1 * 64)", 1),
    "skew": (lambda: bind(_Shape(16, 16), *skew((16, 16), 4)), "((s0 - s1) mod 4 + s2 * 4)", 2),
    "tile_split": (lambda: bind(_Shape(16, 16), Decomposition(Block(0, 4), Block(1, 4)),
                                OwnerMap(Grid(4), {0: "(t0 + 2*t1) mod 4"})), "mod 4", 1),
    "refine": (lambda: refine(bind(_Shape(1024), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0})), axis=0, k=2),
               "* 256", None),
    "nest": (lambda: nest(bind(_Shape(64), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0})), *cyclic(16, 2)),
             "(s0 * 32 + s1 * 16 + s2)", 2),     # rank, then element; a guard picks this core's half
    "inverse": (lambda: bind(_Shape(64), Decomposition(Block(0, 64)), OwnerMap(Grid(4), {0: "(t0 * 3) mod 4"})),
                "((s0 * 3) mod 4 + s1 * 4)", 1),
    "cyclic_tail": (lambda: bind(_Shape(10), *cyclic(10, 4)), "scf.if", 1),     # 3 trips, 3rd guarded
}


@pytest.mark.parametrize("name", sorted(OWNER_MAP_CASES))
def test_owner_map_copy_kernel_builds(name):
    make, offset_text, loops = OWNER_MAP_CASES[name]
    text = str(air_backend.copy_kernel(make(), name=name).build(target="npu2"))
    assert offset_text in text                      # the inverted owner map reaches the DMA offset
    if loops is not None:
        assert text.count("scf.for") >= loops       # one sequential loop per piece coordinate solved by a loop


# -- category 1 completion: uneven and halo through size classes -----------------

REGION_CASES = {
    "uneven": (lambda: bind(_Shape(99), Decomposition(UnevenBlock(0, parts=4)), Map(Grid(4), {0: 0})), 0,
               ["[75] [24] [1]"]),
    "halo1d": (lambda: bind(_Shape(48), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0})), 1,
               ["%alloc_3[1] [13] [1], %arg14[0] [13] [1]", "[35] [13] [1]", "[14] [1]"]),
    "halo2d": (lambda: bind(_Shape(16, 32), Decomposition(Block(0, 2), Block(1, 2)), Map(Grid((2, 2)), {0: 0, 1: 1})), 1,
               ["[9, 17]", "[8, 16]"]),
    "uneven_halo": (lambda: bind(_Shape(10), Decomposition(UnevenBlock(0, sizes=[3, 3, 4])), Map(Grid(3), {0: 0})), 1,
                    ["[5] [5] [1]", "[6] [4] [1]"]),
}


@pytest.mark.parametrize("name", sorted(REGION_CASES))
def test_region_copy_kernel_builds(name):
    make, halo, fragments = REGION_CASES[name]
    text = str(air_backend.copy_kernel(make(), name=name, halo=halo).build(target="npu2"))
    assert "scf.if" in text                         # one branch per size class
    for f in fragments:
        assert f in text, f
    if halo:
        assert "memref.store %cst" in text          # edge PEs zero their ghost cells


# -- category 3: ghost gathers ---------------------------------------------------------

def test_ghost_gather_kernel_builds_one_dma_per_run():
    import random as _random
    from distlayout import bfs_partition, grid_graph, inspect, owner_array, renumber
    perm = list(range(64))
    _random.Random(1).shuffle(perm)
    pat = grid_graph(8, 8, perm)
    owner = bfs_partition(pat, 4)
    p2, new = renumber(owner)
    for o, pt in ((owner, pat), (new, pat.permute(p2))):
        sched = inspect(bind(_Shape(64), *owner_array(o)), pt)
        text = str(air_backend.ghost_gather_kernel(sched, 64).build(target="npu2"))
        loads = [l for l in text.splitlines() if "dma_memcpy_nd" in l and "memref<64xf32, 1" in l and "2 : i32>, memref" in l]
        assert len(loads) == sum(len(air_backend.gather_runs(p)) for p in sched.values())
        assert text.count("scf.if") >= 3                       # one branch per PE
