# distlayout

Distributed data layouts over **existing** tensors and buffers.

```
Data source + Decomposition + Mapping  --bind-->  DistributedView     (who owns what; no copy)
DistributedView + requested region     --plan-->  MaterializationPlan (what moves where)
DistributedView + herd coordinates     --air -->  air.alloc / air.ops.load / store
```

- **Decomposition**: what the pieces are. It knows nothing about processors.
- **Mapping**: which PE holds each piece, and which grid axes replicate.
- **Bind**: attaches the two to storage that already exists, such as an `air.alloc` L2 buffer. Bind is metadata only.

## v0 vocabulary

| Module | Contents |
|---|---|
| `region` | `Box`: half-open hyper-rectangle with `intersect`, `contains`, `shift`, `indices` |
| `decomposition` | `Block(axis, parts)`, `UnevenBlock(axis, parts= \| sizes=, policy='balanced'\|'tail')`, `Decomposition(*rules)` |
| `mapping` | `Grid(shape)`, `Map(grid, {component: grid_axis}, replicate=[axes])`, `Replicate(grid)` |
| `view` | `DataSource(handle, global_shape, origin=, extent=)`, `bind(...)` → `DistributedView` |
| `DistributedView` | `primary_owner`, `holders`, `held_region`, `owned_region`, `local_shape`, `max_local_shape`, `local_index`, `global_index`, `is_local`, `source_region`, `plan(pe, request)`, `validate()`, `table()`, `to_attr()` |
| `backends.air` | `offsets`, `slice_key`, `local_region`, `alloc_local`, `load_local`, `store_local`, `herd_space` |

Semantics that v0 enforces:
- **One primary owner per element.** Replicas hold copies, and `owned_region` is `None` on a replica.
- **Every grid axis is explicitly placed or replicated.**
- **Valid extents are exact.** Padding is a backend decision (`max_local_shape`); it never changes the logical tensor.
- **An indivisible `Block` is an error.** Use `UnevenBlock` to say where the remainder goes.

## Use

```python
from distlayout import Block, Decomposition, Grid, Map, bind
from distlayout.backends import air as air_backend

view = bind(l2_a, Decomposition(Block(0, n // tile_n)), Map(Grid(n // tile_n), {0: 0}))

with air.herd(air_backend.herd_space(view), shape=(2,)) as h:
    @h.body
    def _(tx):
        a = air_backend.alloc_local(view, dt, scope=h.private())
        air_backend.load_local(view, a, (tx,))   # == air.ops.load(a, l2_a[tx*tile_n : tx*tile_n + tile_n])
```

- `examples/demo.py` prints the five design cases. It needs no AIR.
- `examples/eltwise_add_with_l2_dist.py` is upstream's `eltwise_add_with_l2` with the layout supplying every L2↔L1 window.

## Coverage

| Category | Feature | Core (Python) | Explorer | AIR backend |
|---|---|---|---|---|
| 1. Structured sharding | Block | ✅ | ✅ | ✅ (byte-identical to upstream) |
| | Uneven block | ✅ | ✅ | ❌ (on hold) |
| | Replication | ✅ | ✅ | ✅ |
| | Rectangular halo (`plan`) | ✅ | ✅ | ❌ (on hold) |
| | All-reduce, all-to-all | ⏸ on hold (compute-side collectives) | | |
| 2. Affine owner maps | `OwnerMap` + quasi-affine `expr` (`+ - *const floordiv mod`) | ✅ | ✅ | ❌ (on hold; placement-shaped maps emit) |
| | Cyclic, block-cyclic, skew, tile split | ✅ | ✅ | ❌ (on hold) |
| | Several pieces per PE (local `(slot, *tile)`) | ✅ | ✅ | ❌ |
| | Rank splitting `refine(view, axis, k)` → `owner(tx, ty)` | ✅ | ✅ | ✅ when the result is a placement |
| 3. Irregular | Owner arrays, graph partitions, ghost sets, import/export, inspector–executor | ❌ | ❌ | ❌ |
| Transitions | `redistribute`, `plan_redistribution` | ❌ | ❌ | ❌ |
| Native MLIR | `#dist.layout` attribute, `dist.bind` op | text only (`to_attr`) | | |

## Owner maps (category 2)

A layout is a **structured cut** plus an **owner map** from piece coordinates `t0, t1, …` to grid coordinates:

```python
from distlayout import Block, Decomposition, Grid, OwnerMap, bind, cyclic, block_cyclic, skew, refine

bind(x, *cyclic(16, 4))                    # Block(0, 16) + g0 = t0 mod 4
bind(x, *block_cyclic(16, 2, 4))           # Block(0, 8)  + g0 = t0 mod 4
bind(a, *skew((8, 8), 4))                  # 1x1 pieces   + g0 = (t0 + t1) mod 4
bind(a, Decomposition(Block(0, 4), Block(1, 4)), OwnerMap(Grid(4), {0: "(t0 + 2*t1) mod 4"}))
refine(rank_view, axis=0, k=2)             # owner = [g0 = t0 floordiv 2, g1 = t0 mod 2]
```

- Expressions use MLIR's quasi-affine language and print in its syntax (`(t0 floordiv 2) mod 4`).
- Grid axes without an expression are replicated, and must be listed in `replicate=`.
- When a PE holds several pieces, every local buffer is a stack of tiles, `(slot, *tile)`. With tiles of different sizes it is flat.

## Visualize

```bash
python3 examples/gallery.py          # writes docs/explorer.html, open it in a browser
```

`docs/explorer.html` is one self-contained page with no network or dependencies:

- **Presets:** Block, Block + replicate, Uneven, Replicate, 2-D block, Transposed mapping, 2-D halo, Cyclic, Block-cyclic, Skew, Tile split, Rank split k=1 / k=4, Refine 1-D, eltwise_add.
- **Builder,** in the library's two steps: *① Cut*, per tensor axis whole / block / uneven and a piece count; *③ Map*, per grid axis an owner expression such as `t0`, `t0 mod 4` or `(t0 + t1) mod 4` (blank = replicated). Halo radius, bytes per element and PE memory are also editable, and errors are explained inline.
- **Picture, in three steps:** ① *Decompose*: a neutral cutter, the tensor's outline with dashed cut lines and piece names (`P2`, `P(1,0)`), no PEs yet. ② *Cut the tensor*: the same dashed cuts over the elements, each piece in its own color; hover an element for its formula (`p0 = i0 // 8 … → piece P(1,0) → PE 1`). ③ *Map pieces to PEs*: each card says which piece it got (`PE 1 ← P(1,0)`), with a memory-fit bar against the PE budget and red arrows for halo imports. The *Transposed mapping* preset uses the same cuts as *2-D block* but a different mapping.
- **Code tabs:** Python spec · `#dist.layout` · `air.api` herd body · MLIR (the real module, when rendered with MLIR-AIR installed).
- **Pin as A** to compare bytes/PE with the current layout. In *Rank split*, pin at 8 × 1, then set the grid to 8 × 4.
- **Export SVG** of the picture.

For your own views, use `distlayout.viz.render_html(view, halo=1, mlir=text)` or `render_explorer([(name, view, opts), ...])`. The page runs a JavaScript port of the v0 layout rules so the builder updates live. `tests/test_viz.py` checks in a headless browser that it agrees with the Python library on every preset.

## Tests

```bash
python3 -m pytest -q
```

- **Core tests** need no dependencies.
- **AIR-adapter tests** build herd coordinates with upstream's real `IndexExpr` (`python/air/api/_index.py`, plain Python). They look for a checkout in `$MLIR_AIR_SRC` or `~/xilinx/mlir-air`.
- **`tests/test_air_real.py`** runs against real MLIR-AIR and is skipped when it isn't installed. It checks:
  - `examples/eltwise_add_with_l2_dist.py` builds an MLIR module **byte-identical** to upstream's handwritten `eltwise_add_with_l2`, at three sizes.
  - Replication and a 2-D block layout build valid modules.

The prebuilt wheel works with no compiler build:

```bash
pip install --no-deps mlir_air-0.0.1.2026100604+94f274f-cp313-cp313-linux_x86_64.whl numpy ml_dtypes filelock==3.13.1
```

(from `https://github.com/Xilinx/mlir-air/releases/download/latest-air-wheels/`; the `[aie]` extra, i.e. mlir_aie and llvm-aie, is only needed to compile for the NPU.)

## Status and limits

- **Checked against real `air.api` up to the emitted MLIR, not yet compiled or run on an NPU.** That needs the `[aie]` extra and hardware/XRT.
- **The AIR backend emits uniform layouts only.** `air.api` needs one static transfer size, and uneven layouts give PEs different sizes. `plan()` describes them exactly. Emitting them needs `air.ops.branch` or a padded L2 source.
- **One partition per PE along each placed axis.** Cyclic layouts and overdecomposition come later.
- **The partial-source binding works only within its window.** A `DataSource` with `origin` gives correct offsets for PEs whose region lies in the window. Decomposing *the window itself* (an MPI rank's piece split k ways) is the next feature, needed for rank splitting.

## Roadmap

1. **V1:** uneven emission, `Cyclic`/`BlockCyclic`, halo requests to AIR transfers.
2. **V2:** `redistribute(view, layout)` with a "no movement needed" check.
3. **V3:** a `#dist.layout` attribute and `dist.bind` op in MLIR (`to_attr()` is the textual sketch), consumed by the MPI→AIR rank-split pass.
4. **Later:** irregular `IndexMap` ownership, and LEGO for layout inside a PE.
