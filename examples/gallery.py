"""Render the layout explorer: ``python3 examples/gallery.py [out.html]`` (default ``docs/explorer.html``).

One page, one preset per case. If MLIR-AIR is installed, the eltwise-add
preset also carries the real MLIR that ``air.api`` emits for it.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distlayout import Block, Decomposition, Grid, Map, Replicate, UnevenBlock, bind  # noqa: E402
from distlayout.viz import render_explorer  # noqa: E402


class Buf:
    """Stand-in for an existing L2 buffer."""

    def __init__(self, *shape):
        self.shape = shape


def eltwise_mlir(n, tile):
    try:
        import numpy as np
        spec = importlib.util.spec_from_file_location("ex", ROOT / "examples" / "eltwise_add_with_l2_dist.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return str(mod.build_module(n, tile, np.float32).build(target="npu2"))
    except Exception as e:  # MLIR-AIR not installed
        print(f"  (eltwise preset without MLIR: {type(e).__name__}: {e})")
        return None


def presets():
    g41 = Grid((4, 1))
    v2 = bind(Buf(16, 32), Decomposition(Block(0, 2), Block(1, 2)), Map(Grid((2, 2)), {0: 0, 1: 1}))
    n, tile = 2048, 256
    return [
        ("Block", bind(Buf(100), Decomposition(Block(0, 4)), Map(g41, {0: 0}, replicate=[1])),
         {"note": "100 elements in 4 equal blocks, one per PE."}),
        ("Block + replicate", bind(Buf(100), Decomposition(Block(0, 4)), Map(Grid((4, 4)), {0: 0}, replicate=[1])),
         {"note": "4 blocks on a 4 × 4 grid. Grid axis g1 holds nothing new, so each block is replicated along it "
                  "(lighter cards are replicas)."}),
        ("Uneven", bind(Buf(99), Decomposition(UnevenBlock(0, parts=4)), Map(g41, {0: 0}, replicate=[1])),
         {"note": "99 elements over 4 PEs: 25, 25, 25, 24. Try 'block' on axis 0 to see why uneven is needed."}),
        ("Replicate", bind(Buf(100), Decomposition(), Replicate(Grid((4, 4)))),
         {"note": "Nothing is split, so every PE holds the whole tensor; PE 0 is the owner."}),
        ("2-D block", v2, {"note": "16 × 32 in 2 × 2 tiles: axis 0 over g0, axis 1 over g1."}),
        ("2-D halo", v2, {"halo": 1, "note": "Each PE also needs a 1-cell halo around its tile. Red arrows are "
                                             "the imports; memory bars include the halo."}),
        ("Rank split", bind(Buf(8, 4096), Decomposition(Block(0, 8), Block(1, 1)), Map(Grid((8, 1)), {0: 0, 1: 1})),
         {"elem_bytes": 16, "note": "8 MPI ranks, each with a 4096-element row (16 B per element: the program's 4 "
                               "arrays). At k = 1 a rank is one PE and needs 64 KiB > 48 KiB. Pin it, then change "
                               "the PE grid to 8 × 2 or 8 × 4: each rank is split k ways and fits again."}),
        ("eltwise_add", bind(Buf(n), Decomposition(Block(0, n // tile)), Map(Grid(n // tile), {0: 0})),
         {"src": "l2_a", "mlir": eltwise_mlir(n, tile),
          "note": f"Upstream eltwise_add_with_l2, n = {n}, tile = {tile}: one block per herd point. The MLIR tab "
                  f"is the real module from air.api."}),
    ]


def main(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, view, kw in presets():
        view.validate()
        entries.append((name, view, {k: v for k, v in kw.items() if v is not None}))
    out.write_text(render_explorer(entries))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "explorer.html")
