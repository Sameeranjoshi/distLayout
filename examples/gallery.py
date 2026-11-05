"""Render the layout gallery: ``python3 examples/gallery.py [out_dir]`` (default ``docs/gallery``).

One HTML page per case plus ``index.html``. If MLIR-AIR is installed, the
eltwise-add page also embeds the real MLIR that ``air.api`` emits.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from distlayout import Block, Box, Decomposition, Grid, Map, Replicate, UnevenBlock, bind  # noqa: E402
from distlayout.viz import render_html, render_index  # noqa: E402


class Buf:
    """Stand-in for an existing L2 buffer."""

    def __init__(self, *shape):
        self.shape = shape


def halo_requests(view, radius=1):
    """Each PE asks for its own block grown by ``radius``, clipped to the tensor."""
    out = {}
    for pe in view.grid.coords():
        r = view.held_region(pe)
        lo = [max(0, o - radius) for o in r.offset]
        hi = [min(n, s + radius) for s, n in zip(r.stop, view.shape)]
        out[pe] = Box.from_bounds(lo, hi)
    return out


def eltwise_mlir(n, tile):
    try:
        import numpy as np
        import importlib.util
        spec = importlib.util.spec_from_file_location("ex", ROOT / "examples" / "eltwise_add_with_l2_dist.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return str(mod.build_module(n, tile, np.float32).build(target="npu2"))
    except Exception as e:  # MLIR-AIR not installed
        print(f"  (no MLIR for eltwise page: {type(e).__name__}: {e})")
        return None


def cases():
    g41 = Grid((4, 1))
    yield ("block", "Block", "100 elements in 4 equal blocks",
           bind(Buf(100), Decomposition(Block(0, 4)), Map(g41, {0: 0}, replicate=[1])), {}, None)
    yield ("block_replicate", "Block + replication", "4 blocks on a 4×4 grid, each replicated along ty",
           bind(Buf(100), Decomposition(Block(0, 4)), Map(Grid((4, 4)), {0: 0}, replicate=[1])), {}, None)
    yield ("uneven", "Uneven block", "99 elements over 4 PEs: 25, 25, 25, 24",
           bind(Buf(99), Decomposition(UnevenBlock(0, parts=4)), Map(g41, {0: 0}, replicate=[1])), {}, None)
    yield ("replicate", "Full replication", "every PE holds all 100 elements",
           bind(Buf(100), Decomposition(), Replicate(Grid((4, 4)))), {}, None)
    v2 = bind(Buf(16, 32), Decomposition(Block(0, 2), Block(1, 2)), Map(Grid((2, 2)), {0: 0, 1: 1}))
    yield ("block2d", "2-D block", "16×32 tensor in 2×2 tiles", v2, {}, None)
    yield ("halo2d", "2-D block with halo", "each PE requests its tile plus a 1-cell halo: owned vs ghost cells",
           v2, {"requests": halo_requests(v2)}, None)
    v1 = bind(Buf(48), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    yield ("halo1d", "1-D halo plan", "48 elements, 4 PEs, radius-1 halo: which neighbor sends what",
           v1, {"requests": halo_requests(v1)}, None)
    n, tile = 2048, 256
    ve = bind(Buf(n), Decomposition(Block(0, n // tile)), Map(Grid(n // tile), {0: 0}))
    yield ("eltwise_add", "eltwise_add_with_l2", f"upstream example, n={n}, tile={tile}: layout → air.api → MLIR",
           ve, {"src": "l2_a"}, (n, tile))


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pages = []
    for slug, name, desc, view, kw, mlir_args in cases():
        view.validate()
        mlir = eltwise_mlir(*mlir_args) if mlir_args else None
        (out / f"{slug}.html").write_text(render_html(view, title=name, subtitle=desc, mlir=mlir, **kw))
        pages.append((f"{slug}.html", name, desc))
        print(f"  wrote {out / (slug + '.html')}")
    (out / "index.html").write_text(render_index(pages))
    print(f"  wrote {out / 'index.html'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "gallery")
