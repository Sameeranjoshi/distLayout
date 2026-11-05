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

from distlayout import Block, Decomposition, Grid, Map, Replicate, bind  # noqa: E402
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
