# Feature list

Where distlayout stands overall. This is the one list to keep current: update it in the same commit as any feature.

✅ done · ⚠️ partly · ❌ not yet · ⏸ on hold · — doesn't apply

"AIR ✅" means the real `air.api` builds the module and it's checked structurally. Nothing has been compiled for or run on an NPU yet.

## 1. Structured sharding

| Feature | Core | Explorer | AIR |
|---|---|---|---|
| Block | ✅ | ✅ | ✅ byte-identical to upstream `eltwise_add` |
| Uneven block (`parts=` / `sizes=`, balanced / tail) | ✅ | ✅ | ✅ one branch per size class |
| Replication (primary + replicas) | ✅ | ✅ | ✅ |
| Rectangular halo (`plan`, any boundary mode) | ✅ | ✅ | ✅ read from L2 |
| Partial source (a buffer holding one window of the tensor) | ✅ | — | ⚠️ inside the window only |
| Cuts that fall only on a granularity (`granularity=128`) | ❌ | ❌ | ❌ |
| 3-D PE grids | ✅ | ✅ | ❌ `air.api` herds are 1-D / 2-D |

## 2. Affine owner maps

| Feature | Core | Explorer | AIR |
|---|---|---|---|
| Quasi-affine owner expressions (`OwnerMap`) | ✅ | ✅ | ✅ owner-map inversion |
| Cyclic, block-cyclic, skew, tile split | ✅ | ✅ | ✅ |
| Several pieces per PE | ✅ | ✅ | ✅ |
| Rank split (`refine`) | ✅ | ✅ | ✅ |
| Non-invertible maps (gcd > 1, non-linear, repeated coordinate) | ✅ | ✅ | ✅ guards and guarded scans |
| Owner maps with uneven tiles | ✅ | ✅ | ❌ refused with a reason |
| Owner maps with halo | ✅ | ✅ | ❌ refused with a reason |

## 3. Irregular partitions

| Feature | Core | Explorer | AIR |
|---|---|---|---|
| Owner arrays (`owner_array`, `TableMap`) | ✅ | ✅ | ✅ `ghost_load` |
| Sparse access pattern (`CSRPattern`) | ✅ | ✅ | — |
| Inspector: owned, ghosts, import / export lists, local CSR | ✅ | ✅ | ✅ ghost gather |
| Count-balanced partitioners (block, random, BFS), `renumber` | ✅ | ✅ | — |
| Cost-aware partitioner (balance FLOPs or bytes, not counts) | ❌ | ❌ | — |
| Rectangle partitions (`rectangle_fetch`) | ✅ | ✅ | ❌ |
| Executor: `scatter` / `exchange` / `spmv` / `gather` | ✅ | — | ⚠️ data movement only |
| Owner tables over several cuts (`TableMap` on 2-D pieces) | ❌ | ❌ | ❌ |
| Patterns that change at run time | ❌ | ❌ | ❌ |

## 4. Composition

| Feature | Core | Explorer | AIR |
|---|---|---|---|
| Any layout inside any layout | ❌ | ❌ | ❌ |
| Several tensors assigned together (whole tensors as pieces) | ❌ | ❌ | ❌ |
| Layout inside one PE (tiling, swizzle) | ❌ | ❌ | ❌ |

## 5. Transitions

| Feature | Core | Explorer | AIR |
|---|---|---|---|
| Storage layout vs compute layout | ❌ | ❌ | ❌ |
| `plan_redistribution` between two views | ❌ | ❌ | ❌ |
| "No movement needed" detection | ❌ | ❌ | ❌ |
| `redistribute` executor | ❌ | ❌ | ❌ |

## 6. Compute side

| Feature | Core | Explorer | AIR |
|---|---|---|---|
| Requests given explicitly (`plan(pe, request)`) | ✅ | ✅ | ✅ |
| Requests inferred from a computation's accesses | ❌ | ❌ | ❌ |
| Pending-reduction state ("holds a partial sum") | ❌ | ❌ | ❌ |
| All-reduce, all-to-all | ⏸ | ⏸ | ⏸ |

## 7. Compiler integration

| Feature | Status |
|---|---|
| Python spec and `#dist.layout` text (`to_attr`) | ✅ |
| Parser for the `#dist.layout` text | ❌ |
| Native MLIR attribute and `dist.bind` op | ❌ |
| MPI → AIR rank-split pass consuming layouts | ❌ |

## 8. Verification and tooling

| Feature | Status |
|---|---|
| Unit tests and seeded combination suite with independent oracles | ✅ |
| Runs against real `air.api` | ✅ |
| Interactive explorer, kept equal to Python by a browser test | ✅ |
| Soak mode (many more seeds on demand) | ❌ |
| Compiled through to an NPU binary | ❌ |
| Run on NPU hardware | ❌ |
