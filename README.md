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

## Visualize

```bash
python3 examples/gallery.py          # writes docs/gallery/*.html, open docs/gallery/index.html
```

`distlayout.viz.render_html(view, requests={pe: Box}, mlir=text)` turns any view into one self-contained HTML page. It needs no network or dependencies, and the panels are linked: hover an element to light up every PE holding it, or a PE to light up its region.

| Panel | Shows |
|---|---|
| 01 Input | the logical tensor, each element colored by its primary owner |
| 02 Mapping | the PE grid: each PE's region, with primary vs replica (striped) |
| 03 Local views | each PE's local buffer; with `requests=`, owned cells vs `g` ghost cells and where each ghost comes from |
| 04 Layout | the Python spec (`spec_code`) and the `#dist.layout` text |
| 05 Generated code | the `air.api` herd body with the derived affine window, plus a per-PE table of the region each reads |
| 06 MLIR | the real module from `air.api`, when MLIR-AIR is installed |

Gallery cases: block, block + replication, uneven, full replication, 2-D block, 2-D halo, 1-D halo plan, and upstream `eltwise_add_with_l2`.

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
