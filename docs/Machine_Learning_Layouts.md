# Data partitioning in ML frameworks, and how it relates to distlayout

A study of how ML training frameworks partition data, centred on Meta's **FlexShard**, and mapped onto distlayout's three categories.

**Sources.** FlexShard's README and source were read: [`meta-pytorch/flex_shard`](https://github.com/meta-pytorch/flex_shard), in particular:
- `flex_shard/placement_contract.py`;
- `custom_placements/owned.py` and `custom_placements/block_shard.py`;
- `dist_muon/optimizer_reshard.py` and `dist_muon/optimizer.py`.

Not read:
- the torchtitan copy (`torchtitan/distributed/flex_shard`, which appears to be the same library vendored into torchtitan);
- the accompanying video and LinkedIn posts.

The "Wider context" section is from general knowledge, not studied here.

## FlexShard in one paragraph

FlexShard is a data-parallel sharding library for PyTorch, like FSDP2. It stores each parameter sharded between uses, all-gathers it for forward and backward, and reduce-scatters gradients. The difference is that you choose **how each group of parameters (a "bucket") is stored**, through a pluggable `Placement`.

The motivation is that some optimizers and kernels need data **whole**:
- **Muon** orthogonalizes each weight matrix, so the whole matrix must sit on one GPU. FSDP2's equal `Shard(0)` slices every matrix across every GPU, so each optimizer step needs a gather and scatter.
- **Blockwise FP8** gives each 128 × 128 tile its own scale, so a tile must not be split.

Reported result: on DeepSeek V3 671B with 256 H100s, the slowest GPU's optimizer step went from 527 ms to 301 ms, because no exchange is needed.

## How FlexShard partitions

| Concept | What it does | Where in the code |
|---|---|---|
| **`Shard(i)`** | equal slice of each parameter along dim *i* (as in FSDP2) | `placement_contract.py` |
| **`BucketedOwned`** (the owner placement rule) | each **whole** parameter (or whole dim-0 slabs) is owned by **one** rank | `custom_placements/owned.py` |
| **Owner assignment** | greedy **LPT** (largest first): each matrix goes to the least-loaded host, then the least-loaded rank in it. Load is numel for storage, Newton–Schulz cost for Muon (`2·s²·l + s³` per step). `carry_loads` balances across buckets; fixed owners can be pinned. | `_assign_params_to_ranks`, `_balance_loads_across_partitions` |
| **`BlockShard(dim, blocks_per_rank)`** | cuts only **between** blocks (attention heads, 128-row FP8 tiles), with a possibly different number of blocks per rank | `custom_placements/block_shard.py` |
| **Storage vs compute layout** | parameters are stored in one layout and the optimizer computes in another (`Owned`, `BlockShard`, `Replicate`); the library plans the redistribution between them | `dist_muon/optimizer.py` (`_resolve_storage_to_compute_redistribution_requirement`) |
| **`GlobalLayout`** | each rank's local tensor declares which chunks of the full tensor it holds (offsets, sizes); `compose_global_layouts` stacks an inner sharding inside an outer one (TP / EP) | `placement_contract.py` |
| **Collectives per placement** | `Shard` / `BlockShard`: all-gather + reduce-scatter; `Owned` is lowered to a `BlockShard` with one-hot blocks per rank | `owned.py` → `_lower_to_block_shards` |

## How it maps to distlayout

| FlexShard | distlayout equivalent | Match |
|---|---|---|
| `Shard(0)` | `Decomposition(Block(0, P))` + `Map` | ✅ same |
| `BucketedOwned`: whole parameter → owner rank | one piece per parameter + **`TableMap`** (piece → PE lookup), i.e. the irregular owner array | ✅ same idea. Their piece is a whole parameter in a bucket; ours is a region of one tensor. |
| LPT owner balancing by compute cost | `bfs_partition` / `block_partition` balance **counts**, not cost | ⚠️ no cost-aware assignment |
| `BlockShard`: cut only between blocks, uneven blocks per rank | `UnevenBlock(sizes=...)`, without enforcing block boundaries | ⚠️ expressible, but no granularity rule |
| Storage layout ≠ compute layout + planned redistribution | both views exist, but **`redistribute` is not built** | ❌ our largest gap, and their core feature |
| `GlobalLayout` + `compose_global_layouts` (TP / EP inside FSDP) | `held_regions(pe)` (boxes); `refine` (split inside a rank) | ✅ same concept; their composition is general, `refine` covers splitting |
| Collective chosen per placement | `plan` / `inspect` give point-to-point transfers; no collectives | ❌ collectives are on hold |
| `Partial` gradients (`set_partial_grad_group`) | none | ❌ no "pending reduction" state |

**The main point.** FlexShard's owner placement rule is distlayout's **category 3 idea applied at whole-parameter granularity**: an owner table chosen to balance cost. Its `BlockShard` is our **uneven block with a granularity rule**. Its key insight, **separating the storage layout from the compute layout and planning the redistribution between them**, is what distlayout's planned `redistribute` would provide.

## Wider context: other ML frameworks

From general knowledge, not studied in this document.

| Framework | Partitioning model | Relation to distlayout |
|---|---|---|
| PyTorch **DTensor** / FSDP2 | per mesh dimension: `Shard(d)`, `Replicate`, `Partial` | category 1; distlayout lacks `Partial` |
| **JAX / GSPMD / Shardy** (`PartitionSpec`, MLIR `sdy` / `shard` dialects) | tensor axes → mesh axes; the compiler inserts collectives | category 1 |
| **Megatron-LM** | hand-written tensor / sequence / pipeline-parallel splits | fixed block patterns (category 1) |
| **Megatron distributed optimizer, veScale-FSDP, FSDP1** | one flat buffer per bucket, cut into equal byte ranges | like FlexShard's `BucketedBlockShard`; distlayout has no flat-buffer mode |
| **Alpa** and other auto-sharding tools | search over shardings with a cost model | distlayout has no search; FlexShard's LPT is a simple cost model |

None of these mainstream ML systems express formula owners (cyclic, skew) or graph partitions with ghost schedules. Those come from HPC (HPF, PETSc, Tpetra) and remain distlayout's distinctive part.

## Ideas worth borrowing

1. **Cost-aware owner assignment:** an LPT partitioner that balances a cost function (FLOPs, bytes) instead of counts. Small to add next to `bfs_partition`.
2. **A granularity constraint on cuts:** e.g. `UnevenBlock(..., granularity=128)`, so cuts fall only on tile boundaries (FP8 tiles, attention heads). It fits the size-class machinery directly.
3. **Storage layout vs compute layout + `redistribute`:** already the next task in `HANDOFF.md`; FlexShard shows it is the feature that pays off in practice.
4. **A `Partial` state:** "this PE holds a partial sum still to be reduced", the missing piece for the collectives currently on hold.
5. **General layout composition**, like `compose_global_layouts`, generalizing `refine`.
