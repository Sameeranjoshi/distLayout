"""Upstream ``programming_examples/eltwise_add_with_l2`` with a distlayout layout.

The only change from upstream is where the per-core window comes from. Upstream:

    i0 = tx * tile_n
    air.ops.load(a, l2_a[i0 : i0 + tile_n])

Here the L2 buffers are bound to a block layout once, and every load/store
derives its region from it:

    va = bind(l2_a, decomp, mapping)
    air_backend.load_local(va, a, (tx,))

The emitted module should be identical. Needs MLIR-AIR's Python package; run
with ``-p`` to print the module, as upstream.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from air import api as air  # noqa: E402
from air.api.types import dtype_of  # noqa: E402

from distlayout import Block, Decomposition, Grid, Map, bind  # noqa: E402
from distlayout.backends import air as air_backend  # noqa: E402

NUM_TILES = 2


def build_module(n, tile_n, np_dtype_in):
    assert n % (tile_n * NUM_TILES) == 0
    dt = dtype_of(np_dtype_in)

    A = air.tensor([n], dt)
    B = air.tensor([n], dt)
    C = air.tensor([n], dt)

    # One layout for all three vectors: n // tile_n blocks, one per herd point.
    tiles = n // tile_n
    decomp = Decomposition(Block(axis=0, parts=tiles))
    mapping = Map(Grid(tiles), {0: 0})

    with air.launch(name="eltwise_add") as launch:

        @launch.body
        def _():
            with air.segment(name="segment_0") as seg:

                @seg.body
                def _():
                    l2_a = air.alloc([n], dt, scope=seg.private())
                    l2_b = air.alloc([n], dt, scope=seg.private())
                    l2_c = air.alloc([n], dt, scope=seg.private())

                    # Metadata only: no copy, no new allocation.
                    va = bind(l2_a, decomp, mapping, global_shape=[n])
                    vb = bind(l2_b, decomp, mapping, global_shape=[n])
                    vc = bind(l2_c, decomp, mapping, global_shape=[n])

                    air.ops.load(l2_a, A)
                    air.ops.load(l2_b, B)

                    with air.herd(air_backend.herd_space(va), name="herd_0", shape=(NUM_TILES,)) as h:

                        @h.body
                        def _(tx):
                            a = air_backend.alloc_local(va, dt, scope=h.private())
                            b = air_backend.alloc_local(vb, dt, scope=h.private())
                            c = air_backend.alloc_local(vc, dt, scope=h.private())

                            air_backend.load_local(va, a, (tx,))
                            air_backend.load_local(vb, b, (tx,))

                            c[:] = a[:] + b[:]

                            air_backend.store_local(vc, c, (tx,))

                    air.ops.store(l2_c, C)

    return launch


if __name__ == "__main__":
    import numpy as np

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16384)
    p.add_argument("--tile-n", type=int, default=1024)
    p.add_argument("-p", "--print-module-only", action="store_true")
    p.add_argument("--target", default="auto")
    args = p.parse_args()
    module = build_module(args.n, args.tile_n, np.float32).build(target=args.target)
    if args.print_module_only:
        print(module)
