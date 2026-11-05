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
| 1. Structured sharding | Block | ✅ | ✅ | ✅ byte-identical to upstream eltwise_add |
| | Uneven block | ✅ | ✅ | ✅ one `air.ops.branch` per size class, L1 tile at the largest size |
| | Replication | ✅ | ✅ | ✅ |
| | Rectangular halo (`plan`) | ✅ | ✅ | ✅ `halo_load` / `store_owned`: read from L2, zeros outside the tensor (core fill, DMA pad or clip) |
| | All-reduce, all-to-all | ⏸ on hold (compute-side collectives) | | |
| 2. Affine owner maps | `OwnerMap` + quasi-affine `expr` (`+ - *const floordiv mod`) | ✅ | ✅ | ✅ via owner-map inversion (`backends/schedule.py`) |
| | Cyclic, block-cyclic, skew, tile split | ✅ | ✅ | ✅ `air.sequential` loops over each PE's pieces; built with real `air.api` |
| | Several pieces per PE (local `(slot, *tile)`) | ✅ | ✅ | ✅ `for_each_piece`, `load_local` / `store_local` |
| | Rank splitting `refine(view, axis, k)` → `owner(tx, ty)` | ✅ | ✅ | ✅ |
| | Coefficients invertible mod P (modular inverse); pieces not dividing evenly over PEs (uniform trips + per-trip guard) | ✅ | ✅ | ✅ |
| | Same coordinate on several grid axes; coefficients with gcd(a, P) > 1; non-linear owner expressions | ✅ | ✅ | ✅ per-trip consistency guard (`expr == x`), guarded scan of the free coordinates |
| | Owner maps with halo or uneven tiles; 3-D PE grids | ✅ | ✅ | ❌ refused with a reason (size classes inside the piece loop; `air.api` herds are 1-D / 2-D) |
| 3. Irregular | Owner arrays (`owner_array` → runs + `TableMap`) | ✅ | ✅ | ✅ owned + ghost x via `ghost_load`, any table; renumbered tables also as an uneven block |
| | Sparse access pattern (`CSRPattern`; graphs as adjacency) | ✅ | ✅ graph view, cut edges | — |
| | Inspector: owned / ghost sets, import & export lists, local CSR (`inspect`) | ✅ | ✅ ghost counts, import arcs | ✅ `ghost_load`: one branch per PE, one static DMA per id run from L2 |
| | Partitioners: block, random, BFS graph growing; `renumber` | ✅ | ✅ | — |
| | Executor: `scatter` / `exchange` (along export lists) / `spmv` (local CSR) / `gather` | ✅ checked against dense `A @ x` | — | data movement only (no sparse compute in `air.api`) |
| | Patterns that change at run time (run-time inspector) | ❌ | ❌ | ❌ |
| Transitions | `redistribute`, `plan_redistribution` | ❌ | ❌ | ❌ |
| Native MLIR | `#dist.layout` attribute, `dist.bind` op | text only (`to_attr`) | | |

AIR emission is checked by building kernels with the real `air.api` (`tests/test_air_real.py`). Nothing has been compiled for or run on an NPU yet.

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

### Uneven layouts and halos to AIR

`air.api` transfers need static sizes, so PEs whose regions differ in size get one branch each. PEs whose regions have the same size form a **size class**, and within a class the offset is affine. Examples:
- uneven 99 over 4: PEs 0–2 vs PE 3;
- halo: first / interior / last PE.

```python
with air.herd(air_backend.herd_space(view)) as h:
    @h.body
    def _(tx):
        a = air.alloc(list(air_backend.region_shape(view, halo=1)), f32, scope=h.private())
        air_backend.halo_load(view, a, (tx,), halo=1, src=l2_in)      # own block + 1 cell per side
        ...                                                          # stencil on a
        air_backend.store_owned(view, a, (tx,), halo=1, dst=l2_out)   # writes the owned cells only
```

For block 48 over 4 with halo 1, this emits three `scf.if` branches:
- **first PE:** zero the tile, load `l2[0:13]` into `a[1:14]`;
- **interior PEs:** load `l2[tx*12 - 1 : tx*12 + 13]`;
- **last PE:** zero the tile, load `l2[35:48]` into `a[0:13]`.

Halo cells are read from the shared L2 buffer, the segment-wide view of the tensor, not exchanged PE to PE. Boundary options:
- `boundary="zero"` (default): the core zero-fills edge tiles.
- `"pad"`: the memtile DMA does the zero-fill; needs an `air.api` with DMA padding, which the Oct-2026 wheel lacks.
- `"clip"`: edge tiles are smaller, with no ghost cells outside the tensor.

`load_local` / `store_local` use the same classes for uneven layouts automatically.

### Owner maps to AIR

The herd body runs *on* a PE, so the backend inverts the owner map: "I am PE `tx`; my s-th piece is `t(tx, s)`". `distlayout.backends.schedule.solve` does this in closed form for:
- bare coordinates;
- `floordiv`/`mod` pairs (refine);
- `(linear) mod P` where some coefficient is invertible mod P (modular inverse).

When pieces don't divide evenly over PEs, every PE runs the same number of trips, which AIR requires, and the surplus trip is skipped by `air.ops.branch(t < n)`.

Each schedule is checked by enumeration against the layout before anything is emitted.

```python
with air.herd(air_backend.herd_space(view)) as h:
    @h.body
    def _(tx):
        a = air.alloc(tile, f32, scope=h.private())           # one tile, reused
        def per_piece(slot, key):                             # key: this piece's window of l2
            air.ops.load(a, l2_in[key]); ...; air.ops.store(a, l2_out[key])
        air_backend.for_each_piece(view, (tx,), per_piece)    # air.sequential loops over my pieces
```

For block-cyclic 256/16 over 4 PEs this emits `scf.for` with `affine_map<()[s0, s1] -> ((s0 mod 4) * 16 + s1 * 64)>`. `air_backend.copy_kernel(view)` builds a complete L3 → L2 → layout → L2 → L3 kernel for any emittable layout; running it must return its input.

## Irregular partitions (category 3)

The usual decomposition, as in PETSc, Tpetra, Zoltan and PARTI/CHAOS, keeps three things apart:
- the **partition**: an owner array over an index space;
- the **sparsity**: a CSR pattern, where a graph is its adjacency matrix;
- the **communication**: derived from the two by an inspector.

```python
from distlayout import bind, owner_array, grid_graph, bfs_partition, inspect, renumber

pattern = grid_graph(8, 8, perm)            # 5-point stencil on a mesh (perm: a scrambled numbering)
owner = bfs_partition(pattern, 4)           # or METIS / Scotch / your own owner[i]
rows = bind(y, *owner_array(owner))         # pieces = runs of equal owner, TableMap: run -> PE
sched = inspect(rows, pattern)              # x laid out like the rows (or pass x=)
sched[(2,)].ghosts, sched[(2,)].imports     # IndexSet{...}, {src PE: IndexSet{...}}
sched[(2,)].local_csr(pattern)              # columns renumbered: owned first, then ghosts
perm2, owner2 = renumber(owner)             # contiguous ids per PE: the table becomes a placement
```

Behaviour:
- `owner_array` reuses the existing pieces: an `UnevenBlock` of runs plus a `TableMap`. So views, `plan`, local numbering and `validate` all work unchanged.
- Ghost owners are found per run, not per id.

On a scrambled 8×8 mesh over 4 PEs:

| Partition | Cut edges | Ghost volume |
|---|---|---|
| Blocks of ids | 166 | 123 |
| BFS graph growing | 56 | 41 |

With rows and x partitioned alike, each PE stores 16 owned + 8 to 12 ghost x entries instead of all 64. That's de-replication.

**Executor.** `y = gather(rows, spmv(sched, pattern, values, scatter(x_view, x)))` computes `A @ x` the distributed way:
1. ghosts travel only along the export lists;
2. each PE multiplies with its local CSR.

The tests check the result against a dense product for block, random, BFS, renumbered and replicated layouts.

**AIR.** `ghost_load(sched, x_local, coords, l2_x)` fills each PE's local x (owned, then ghosts, in local-CSR order) from the L2 copy:
- one `air.ops.branch` per PE, since every PE's list differs;
- one static DMA per contiguous run of ids.

`ghost_gather_kernel(sched, n)` is a complete, checkable kernel: `Y[p] == x[local_ids(p)]`. On the scrambled mesh, renumbering cuts the gather from 84 DMAs to 39, with each PE's owned part becoming one 16-element DMA. The rest are single-element ghost runs, the real cost of irregular data on a DMA machine.

### PartTensor-style rectangle partitions

A sparse matrix cut into disjoint, non-uniform rectangles, one per PE, is `Decomposition(UnevenBlock(0, sizes=...), UnevenBlock(1, sizes=...))` with a `Map`.

`rectangle_fetch(view, pattern)` gives, for each PE:
- how many nonzeros of its row band are local;
- which foreign rectangle slices it must fetch, one `(owner, Box, nnz)` per owner.

That's the decision a PartTensor `get_slice` makes. It's checked against brute force on random rectangle layouts. The explorer preset *PartTensor-style rectangles* shows a 10 × 7 example. AIR emission of these slices is not implemented: the layout is uneven in 2-D.

## Visualize

```bash
python3 examples/gallery.py          # writes docs/explorer.html, open it in a browser
```

`docs/explorer.html` is one self-contained page with no network or dependencies:

- **Presets:** Block, Block + replicate, Uneven, Replicate, 2-D block, Transposed mapping, 2-D halo, Cyclic, Block-cyclic, Skew, Tile split, Rank split k=1 / k=4, Refine 1-D, Irregular: block of ids / BFS partition / renumbered (with a ④ graph view of the mesh), PartTensor-style rectangles, eltwise_add.
- **Builder,** in the library's two steps: *① Cut*, per tensor axis whole / block / uneven and a piece count; *③ Map*, per grid axis an owner expression such as `t0`, `t0 mod 4` or `(t0 + t1) mod 4` (blank = replicated). Halo radius, bytes per element and PE memory are also editable, and errors are explained inline.
- **Picture, in three steps:** ① *Decompose*: a neutral cutter, the tensor's outline with dashed cut lines and piece names (`P2`, `P(1,0)`), no PEs yet. ② *Cut the tensor*: the same dashed cuts over the elements, each piece in its own color; hover an element for its formula (`p0 = i0 // 8 … → piece P(1,0) → PE 1`). ③ *Map pieces to PEs*: each card says which piece it got (`PE 1 ← P(1,0)`), with a memory-fit bar against the PE budget and red arrows for halo imports. The *Transposed mapping* preset uses the same cuts as *2-D block* but a different mapping.
- **Code tabs:** Python spec · `#dist.layout` · `air.api` herd body · MLIR (the real module, when rendered with MLIR-AIR installed).
- **Pin as A** to compare bytes/PE with the current layout. In *Rank split*, pin at 8 × 1, then set the grid to 8 × 4.
- **Export SVG** of the picture.

For your own views, use `distlayout.viz.render_html(view, halo=1, mlir=text)` or `render_explorer([(name, view, opts), ...])`. The page runs a JavaScript port of the v0 layout rules so the builder updates live. `tests/test_viz.py` checks in a headless browser that it agrees with the Python library on every preset.

## Tests

**Combination suite** (`tests/test_combinations.py`, with the generator and oracles in `tests/combos.py`). Seeded random layouts are checked against brute-force enumeration through every layer. The layouts vary:
- 1-D / 2-D tensors;
- whole, block and uneven cuts (balanced, tail, explicit sizes);
- placement, owner-map, table and replicate mappings;
- replication, and `refine` once or twice.

The checks:
- core invariants;
- plans for random requests;
- the generated-Python round trip;
- owner-map inversion;
- piece loops and size-class windows with random halos and every boundary mode;
- the explorer's JavaScript port;
- real `air.api` builds.

Metamorphic tests check that two spellings of one layout own the same elements: refine twice vs once, any 1-D layout as an owner array, the cyclic family, transposed placements. Irregular tests cross 6 row layouts × 6 x layouts on random sparse patterns, checking the inspector against brute force, SpMV against the dense product, and the AIR gather by replay. The suite runs about 1,600 cases in about 15 s.

### Unit tests

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
