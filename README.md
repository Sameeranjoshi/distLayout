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

## How it compares

This comparison is based on our reading of these projects, not on running them, so treat the entries for the other three as approximate.

| Capability | **distlayout** | **LEGO** | **PartTensor** | **Spensor** |
|---|---|---|---|---|
| Main purpose | ownership: who holds what, across PEs | layout inside one memory (index → address) | distributed sparse tensors in MLIR | spatial tensor ops on AMD NPU |
| Distribution across PEs | ✅ | ❌ | ✅ | ✅ |
| Layout *inside* a PE (tiling, swizzle) | ❌ | ✅ | ❌ | ⚠️ partial |
| Layout separate from the tensor | ✅ | ✅ | ⚠️ encoding on the tensor type | ❌ split is an op |
| Binds to an existing buffer (`memref.alloc`) | ✅ | ❌ | ❌ | ❌ |
| Block / uneven block | ✅ | ✅ (inside one memory) | ✅ rectangles | ⚠️ even splits only |
| Replication (primary vs replica) | ✅ | ❌ | ❌ | ⚠️ broadcast, no owner semantics |
| Cyclic / block-cyclic / skew (formula owners) | ✅ | ⚠️ as permutations, not ownership | ❌ | ❌ |
| Irregular owner arrays / graph partitions | ✅ | ❌ | ❌ only rectangles | ❌ |
| Sparse access → ghost / import lists (inspector) | ✅ | ❌ | ⚠️ slice movement | ❌ |
| Halo / ghost regions | ✅ | ❌ | ❌ | ❌ |
| Splitting below an MPI rank (`refine`) | ✅ | ❌ | ❌ | ❌ |
| Generates AIR code | ✅ (`air.api`) | ❌ | ❌ (lowers to MPI) | ✅ |
| Native MLIR dialect | ❌ text only | ✅ | ✅ | ✅ (xDSL) |
| Runs on hardware today | ❌ not yet | ✅ | ✅ | ✅ |
| Interactive visualizer | ✅ | ✅ | ❌ | ❌ |

✅ = yes, ❌ = no, ⚠️ = partly.

- **LEGO** decides *where an element sits inside one memory*; distlayout decides *which PE owns it*. They are complementary: LEGO could later supply the layout inside a PE.
- **PartTensor** covers distributed sparse tensors split into rectangles, lowered to MPI. distlayout covers the rectangle case (`rectangle_fetch`) and adds formula owners, owner arrays and ghost schedules.
- **Spensor** is the closest relative: spatial splits on AIR. It ties the split to its own tensor types and memory levels; distlayout keeps the layout separate, binds to existing buffers, and covers more layout kinds.

Where distlayout is behind: no native MLIR dialect yet, and nothing has run on hardware.

## Coverage

The overall feature list, with what's done and what isn't, is in [`docs/FEATURES.md`](docs/FEATURES.md).

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
- **Builder,** in the library's two steps. *① Decomposition layout*: per tensor axis whole / block / uneven, with a piece count; uneven axes also take explicit **sizes** (`3, 7`) or a **balanced / tail** policy. *③ Mapping layout*: per grid axis an owner expression such as `t0`, `t0 mod 4` or `(t0 + t1) mod 4` (blank = replicated). Also halo radius, bytes per element and PE memory, with errors explained inline (e.g. "sizes 3, 6 sum to 9, axis has 10").
- **Picture, in three steps:** *① Decomposition layout*: an outline of the tensor drawn on the tensor's own grid, so its cuts line up with the tensor and can be laid on it mentally. *② Tensor*: the same cuts over the elements, colored by owner PE; hover an element for its formula. *③ Mapping layout*: each PE card names the piece(s) it got, with a memory-fit bar and red arrows for halo or ghost imports. Code tabs (Python, `#dist.layout`, `air.api`, MLIR) sit in a full-width panel below.
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

Checked against real `air.api` up to the emitted MLIR, but not yet compiled or run on an NPU. That needs the `[aie]` extra and hardware/XRT. See [`docs/FEATURES.md`](docs/FEATURES.md) for what comes next.
