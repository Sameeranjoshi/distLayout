"""Render the layout explorer: ``python3 examples/gallery.py [out.html]`` (default ``docs/explorer.html``).

One page, one preset per case. If MLIR-AIR is installed, the eltwise-add
preset also carries the real MLIR that ``air.api`` emits for it.
"""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import random  # noqa: E402

from distlayout import (Block, Decomposition, Grid, Map, OwnerMap, Replicate, UnevenBlock, bind,  # noqa: E402
                        bfs_partition, block_cyclic, block_partition, cyclic, grid_graph, owner_array, refine,
                        renumber, skew)
from distlayout.irregular import CSRPattern  # noqa: E402
from distlayout.irregular import grid_positions  # noqa: E402
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


def _rect_rows():
    """A fixed sparse 10 x 7 pattern (seeded), with the diagonal present."""
    rng = random.Random(4)
    return [sorted({c for c in range(7) if rng.random() < 0.35} | ({i} if i < 7 else set())) for i in range(10)]


def _rect_summary(view, pattern):
    from distlayout import rectangle_fetch
    fx = rectangle_fetch(view, pattern)
    parts = []
    for pe, d in fx.items():
        src = sorted({s for s, _, _ in d["fetch"]})
        parts.append(f"PE {pe[0] * 2 + pe[1]}: {d['local_nnz']} local, "
                     + (f"fetches {len(d['fetch'])} slice(s) from PE {', '.join(str(s[0] * 2 + s[1]) for s in src)}"
                        if d["fetch"] else "no fetch"))
    return "; ".join(parts) + "."


def presets():
    g41 = Grid((4, 1))
    rank_view = bind(Buf(8, 4096), Decomposition(Block(0, 8)), Map(Grid(8), {0: 0}))
    rect_view = bind(Buf(10, 7), Decomposition(UnevenBlock(0, sizes=[3, 7]), UnevenBlock(1, sizes=[2, 5])),
                     Map(Grid((2, 2)), {0: 0, 1: 1}))
    rect_pattern = CSRPattern.from_rows(_rect_rows(), ncols=7)
    rect_note = _rect_summary(rect_view, rect_pattern)
    scramble = list(range(64))
    random.Random(1).shuffle(scramble)
    mesh, mesh_pos = grid_graph(8, 8, scramble), grid_positions(8, 8, scramble)
    bfs_owner = bfs_partition(mesh, 4)
    perm, renum_owner = renumber(bfs_owner)
    renum_mesh, renum_pos = mesh.permute(perm), [mesh_pos[old] for old in perm]
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
        ("Transposed mapping", bind(Buf(16, 32), Decomposition(Block(0, 2), Block(1, 2)), Map(Grid((2, 2)), {0: 1, 1: 0})),
         {"note": "The same 2 × 2 cuts as 2-D block, mapped differently: tensor axis 0 goes over grid axis g1. "
                  "Piece P(1,0) now lands on PE 1, not PE 2. Cutting and mapping are separate choices."}),
        ("2-D halo", v2, {"halo": 1, "note": "Each PE also needs a 1-cell halo around its tile. Red arrows are "
                                             "the imports; memory bars include the halo."}),
        ("Cyclic", bind(Buf(16), *cyclic(16, 4)),
         {"note": "Cut into 16 one-element pieces, then deal them round-robin: g0 = t0 mod 4. Every PE holds 4 pieces."}),
        ("Block-cyclic", bind(Buf(16), *block_cyclic(16, 2, 4)),
         {"note": "Cut into 2-element blocks, then deal the blocks round-robin: g0 = t0 mod 4."}),
        ("Skew", bind(Buf(8, 8), *skew((8, 8), 4)),
         {"note": "1 × 1 pieces, g0 = (t0 + t1) mod 4: each row is the one above shifted by one PE."}),
        ("Tile split", bind(Buf(8, 8), Decomposition(Block(0, 4), Block(1, 4)), OwnerMap(Grid(4), {0: "(t0 + 2*t1) mod 4"})),
         {"note": "A structured 4 × 4 cut into 2 × 2 tiles, but tiles go to PEs by (t0 + 2·t1) mod 4: the cut "
                  "being regular does not make the mapping regular."}),
        ("Rank split k=1", rank_view,
         {"elem_bytes": 16, "note": "8 MPI ranks, each with a 4096-element row (16 B per element: the program's 4 "
                                    "arrays). One rank per PE needs 64 KiB > 48 KiB. Pin it, then open 'Rank split k=4'."}),
        ("Rank split k=4", refine(rank_view, axis=1, k=4),
         {"elem_bytes": 16, "note": "refine(view, axis=1, k=4): each rank's row is cut 4 ways and the slices go to a "
                                    "new grid axis, owner(tx, ty). 16 KiB per PE: fits."}),
        ("Refine 1-D", refine(bind(Buf(64), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0})), axis=0, k=2),
         {"note": "4 ranks of 16 elements, each refined 2 ways: g0 = t0 floordiv 2 (the rank), g1 = t0 mod 2 (the "
                  "slice). This is rank splitting written as an affine owner map."}),
        ("Irregular: block of ids", bind(Buf(64), *owner_array(block_partition(64, 4))),
         {"pattern": mesh, "positions": mesh_pos,
          "note": "An 8 × 8 mesh whose ids are scrambled. Cutting the ids into 4 blocks looks tidy in the tensor "
                  "but scatters every PE over the mesh (④): many cut edges, many ghosts."}),
        ("Irregular: BFS partition", bind(Buf(64), *owner_array(bfs_owner)),
         {"pattern": mesh, "positions": mesh_pos,
          "note": "owner[i] from a graph partitioner (bfs_partition; METIS works the same way). The ids each PE owns "
                  "are scattered (many pieces in ②), but on the mesh they are compact regions: far fewer ghosts."}),
        ("Irregular: renumbered", bind(Buf(64), *owner_array(renum_owner)),
         {"pattern": renum_mesh, "positions": renum_pos,
          "note": "renumber(owner) permutes ids so each PE's are contiguous: the owner table becomes a placement "
                  "(an uneven block, emittable to AIR) with the same ghosts as the BFS partition."}),
        ("PartTensor-style rectangles", rect_view,
         {"note": "A sparse 10 × 7 matrix cut into uneven disjoint rectangles (rows 3 + 7, columns 2 + 5), one per PE. "
                  "PE 0 computes the rows of its rectangles and needs the nonzeros of those rows: " + rect_note + " "
                  "Each foreign slice is one fetch, like a PartTensor get_slice."}),
        ("eltwise_add", bind(Buf(n), Decomposition(Block(0, n // tile)), Map(Grid(n // tile), {0: 0})),
         {"src": "l2_a", "mlir": eltwise_mlir(n, tile),
          "note": f"Upstream eltwise_add_with_l2, n = {n}, tile = {tile}: one block per herd point. The MLIR tab "
                  f"is the real module from air.api."}),
    ]


def copy_mlir(view, halo=0):
    """Real MLIR of the layout's scatter/gather kernel, when MLIR-AIR is installed and the layout emits."""
    try:
        from distlayout.backends import air as air_backend
        return str(air_backend.copy_kernel(view, halo=halo).build(target="npu2"))
    except ImportError:
        return None
    except NotImplementedError as e:
        return f"// not emitted: {e}"


def ghost_mlir(view, pattern):
    """Real MLIR of the ghost-gather kernel for an irregular preset."""
    try:
        from distlayout import inspect
        from distlayout.backends import air as air_backend
        return str(air_backend.ghost_gather_kernel(inspect(view, pattern), view.shape[0]).build(target="npu2"))
    except ImportError:
        return None


STRUCTURED = "1. Structured tensor sharding"
AFFINE = "2. Structured affine owner maps"
IRREGULAR = "3. Irregular partitions"
GROUPS = {
    "Block": STRUCTURED, "Block + replicate": STRUCTURED, "Uneven": STRUCTURED, "Replicate": STRUCTURED,
    "2-D block": STRUCTURED, "Transposed mapping": STRUCTURED, "2-D halo": STRUCTURED, "eltwise_add": STRUCTURED,
    "Cyclic": AFFINE, "Block-cyclic": AFFINE, "Skew": AFFINE, "Tile split": AFFINE,
    "Rank split k=1": AFFINE, "Rank split k=4": AFFINE, "Refine 1-D": AFFINE,
    "Irregular: block of ids": IRREGULAR, "Irregular: BFS partition": IRREGULAR,
    "Irregular: renumbered": IRREGULAR, "PartTensor-style rectangles": IRREGULAR,
}


def main(out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, view, kw in presets():
        view.validate()
        if "mlir" not in kw:
            kw["mlir"] = ghost_mlir(view, kw["pattern"]) if "pattern" in kw else copy_mlir(view, kw.get("halo", 0))
        kw["group"] = GROUPS[name]
        entries.append((name, view, {k: v for k, v in kw.items() if v is not None}))
    out.write_text(render_explorer(entries))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "docs" / "explorer.html")
