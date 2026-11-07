# distlayout: handoff

Read this first. Then `README.md` (user-facing docs and the coverage table, kept current) and this file's "Next tasks".

## What the project is

`distlayout` describes how a logical tensor (or a sparse graph) is distributed over a grid of processing elements (PEs), then derives local storage, transfers and communication. The first backend is MLIR-AIR's Python DSL (`air.api`). The core has no dependencies.

The model, in three steps:

1. **Cut** the index space into pieces: `Decomposition(Block | UnevenBlock, ...)`, or an owner array for irregular data (`owner_array`).
2. **Map** pieces to PEs: `Map` (placement), `OwnerMap` (quasi-affine expressions over piece coordinates `t0, t1, ...`), `TableMap` (lookup), or `Replicate`. Grid axes with no expression are replicated and must be listed in `replicate=`.
3. **Bind** to existing storage: `bind(handle, decomposition, mapping)` is metadata only; nothing is copied.

From a `DistributedView` you can ask ownership questions, plan transfers for a request, and generate code.

## Repository map

| Path | Role |
|---|---|
| `distlayout/region.py` | `Box` (half-open hyper-rectangle), `IndexSet` (sorted runs; the irregular region kind) |
| `distlayout/decomposition.py` | `Block`, `UnevenBlock` (`parts=` with `balanced`/`tail`, or `sizes=`), `Decomposition` (product of cuts) |
| `distlayout/mapping.py` | `Grid`, `Mapping` / `Map` / `Replicate`, `OwnerMap`, `TableMap`, `as_placement` (canonicalises an owner map that is really a placement) |
| `distlayout/expr.py` | quasi-affine expressions: `+ - *const floordiv mod`, `parse`, MLIR-syntax printing, `substitute` |
| `distlayout/view.py` | `DataSource`, `DistributedView` (ownership queries, `plan`, `validate`, `to_attr`), `bind` |
| `distlayout/layouts.py` | `cyclic`, `block_cyclic`, `skew`, `nest` (any layout inside each piece), `refine` (`nest` with one cut: owner(tx, ty)) |
| `distlayout/tree.py` | `Layout` (a tree of layouts), `bind_tree`, `TreeView` (paths, `classes`, `fit`, `flatten`). Each node is a `DistributedView` over its parent's piece. |
| `distlayout/irregular.py` | `CSRPattern`, `grid_graph`, `owner_array`, partitioners (`block_partition`, `random_partition`, `bfs_partition`), `renumber`, `inspect` (inspector), `rectangle_fetch`, `scatter` / `exchange` / `spmv` / `gather` (reference executor) |
| `distlayout/backends/schedule.py` | owner-map inversion: `solve(view)` gives loop nest + recipes; `Text` for printing; per-trip guards and guarded scans |
| `distlayout/backends/air.py` | AIR emission: `copy_kernel`, `for_each_piece`, `for_each_region` / `halo_load` / `store_owned` (size classes), `ghost_load` / `ghost_gather_kernel`, `herd_space`, `region_shape` |
| `distlayout/viz.py` | self-contained HTML explorer (`render_explorer`, `render_html`, `to_spec`); a JavaScript port of the algebra |
| `examples/gallery.py` | builds `docs/explorer.html` from presets; embeds real MLIR when `air.api` is importable |
| `examples/eltwise_add_with_l2_dist.py` | upstream's `eltwise_add_with_l2` with the layout supplying every window (byte-identical MLIR) |
| `tests/` | see Testing |
| `docs/FEATURES.md` | the overall feature list |
| `docs/ML_PATTERNS.md` | study of ML data partitioning. Learn from it, but keep our own design and names. |

## Key invariants (keep these true)

- **One primary owner per element.** Replicas hold copies; `owned_region` is `None` on a replica.
- **Every grid axis is placed or replicated.** Nothing is implicit.
- **Valid extents are exact.** Padding is a backend decision (`max_local_shape`, `region_shape`), never part of the logical tensor.
- **A PE with several pieces has a stacked local buffer**: `(slot, *tile)` when tiles are equal, flat `(n,)` otherwise. `local_index` / `global_index` must round-trip (tested).
- **AIR needs static sizes and equal trip counts on every core.** Uneven layouts are emitted as one `air.ops.branch` per size class; uneven piece counts use a uniform trip count with a per-trip guard `t < n`.
- **Ghost cells outside the tensor** are zero-filled on the core (`boundary="zero"`), via DMA padding (`"pad"`, not supported by the pinned `air.api`), or left out (`"clip"`).
- **Owner maps invert or are refused with a reason.** `solve` never emits an unverified schedule: `_verify` enumerates every PE.

## Environment and how to run things

- Python 3.13 with the stdlib plus `pytest` (installed with `pip install pytest`). The core needs nothing else.
- MLIR-AIR's Python package is optional, and tests that need it skip cleanly without it. Wheels come from `https://github.com/Xilinx/mlir-air/releases/download/latest-air-wheels/` (install with `--no-deps`, then `numpy ml_dtypes filelock==3.13.1`). The release keeps about a month of nightlies, so a pinned name goes stale; first verified with `94f274f`.
- On CHPC (Rocky 8, glibc 2.28) the wheels don't load, because they need glibc 2.32. Use the local build instead: `PYTHONPATH=~/other/spatialAIR/mlir-air/build/python MLIR_AIR_SRC=~/other/spatialAIR/mlir-air ~/work/venv_work/bin/python3 -m pytest -q`. Run on a compute node, not a login node.
- Node plus Playwright (Chromium is preinstalled in the session container) is needed for the explorer test. Use `NODE_PATH=$(npm root -g)` if `require('playwright')` does not resolve.
- Do not run `playwright install` in the session container; use the preinstalled browser. Elsewhere, install it in a temporary directory and point `NODE_PATH` at its `node_modules`.

Commands:

```bash
python3 -m pytest -q                      # core only
<venv>/bin/python -m pytest -q            # with MLIR-AIR
python3 examples/gallery.py               # regenerates docs/explorer.html (run with the venv to embed MLIR)
python3 examples/demo.py                  # prints the five design cases
```

Keep the explorer page committed: `docs/explorer.html` is generated, so regenerate it after any change to `viz.py`, `to_spec`, or presets, and check it in.

## Testing

| File | What it checks |
|---|---|
| `test_core.py`, `test_owner_maps.py`, `test_viz.py`, `test_irregular.py` | unit tests for each layer; hand-computed expectations |
| `test_air_backend.py`, `test_air_schedule.py`, `test_air_regions.py`, `test_air_ghosts.py` | AIR code generation without `air.api`: the exact subscripts, branch trees and replays of emitted DMAs on concrete data |
| `test_air_real.py` | builds real `air.api` modules (eltwise byte-identity, owner maps, regions, ghost gathers); skipped without the wheel |
| `combos.py` + `test_combinations.py` | seeded random layouts against brute-force oracles through every layer (see below) |

The combination suite, `combos.py` and `test_combinations.py`:
- `random_view(rng)` builds 1-D / 2-D layouts across all cut kinds and mapping kinds, optionally refined.
- Oracles re-derive answers by enumeration (`brute_owners`, `pieces_by_pe`, `brute_schedule`). They must not call the code under test.
- Metamorphic tests compare two spellings of one layout. These are the strongest checks; add more of them.
- The explorer test runs the page's JavaScript on random layouts and compares it with Python. It is skipped without Node/Playwright.

When a combination test fails, first decide which side is wrong. Two of the bugs found so far were in the oracle or the expected text, not in the code.

## Coverage

`docs/FEATURES.md` is the one feature list. Update it in the same commit as any feature, and show it to the user at each milestone.

Nothing has been compiled or run on an NPU. "AIR yes" means the module is generated by the real `air.api` and checked structurally.

## Conventions

- Tests are the spec. Add a test before or with a feature. Prefer one combination or metamorphic test over many single-case tests.
- Keep the core free of AIR imports. Backend code imports `air` lazily inside functions.
- Refuse unsupported inputs with `NotImplementedError` and a reason that says what would be needed. Silent degradation is a bug.
- Do not put secrets in the repo. Do not use the word "fake" in code or tests (use "virtual" or "stand-in"); this is the user's style rule.
- Commits: one logical change per commit, with a message that explains why. Push to `origin main` after the suite passes in both modes. Commit trailers come from the session harness; do not put model identifiers in commit messages or code.
- The user reads the explorer and the screenshots. Visual changes should be checked in a headless browser (Playwright screenshot), not only by tests.

## Next tasks

Each has a "done" line. Do them in this order unless the user says otherwise.

1. **Redistribution (`redistribute`, `plan_redistribution`).** Given `src` and `dst` views of the same tensor, produce the transfers that move each element from its source holder to its destination owners, reusing `plan` and the inspector's import/export lists.
   - *Done when:* for random pairs of views (block ↔ cyclic ↔ table ↔ owner map), executing the plan on scattered data reproduces `gather(dst, ...)`, and "no movement needed" is detected exactly when every element keeps the same owner set. Add the combination test to `test_combinations.py`, and a real `air.api` build of a copy kernel for one pair.
2. **Owner maps with uneven tiles, and halo on owner maps.** Combine the piece loop (`for_each_piece`) with size-class branches (`for_each_region`) inside the loop. Branch conditions are on `t(x, s)`.
   - *Done when:* the two refusals in `test_combinations.py` become positive tests, `test_air_real.py` builds them, and the "not emittable" count drops to zero for 2-D owner maps.
3. **Native MLIR attribute and op (`#dist.layout`, `dist.bind`).** Start with a textual parser for the `to_attr` form, round-trip tests, then decide on an MLIR dialect. Requires C++ work in the AIR fork; discuss with the user first.
4. **Soak mode.** An environment variable (e.g. `DISTLAYOUT_SEEDS=5000`) that multiplies the seed count in `test_combinations.py`, for occasional runs.

## Pitfalls already hit (so you do not repeat them)

- **Printing**: `Text` (in `schedule.py`) builds Python-source text for offsets; its `__eq__` returns a printed condition, so never compare two `Text` values with `==` in test logic. Use `str(...)`.
- **Constant expressions**: `-3` parses as `-(3)`. Treat any `Expr` with no variables as a constant (`o.vars()`), not only `Const`.
- **Operator precedence in printed expressions**: `a - (b + c)` must keep parentheses; `_atom` decides this.
- **`air.api` gaps** (checked on `94f274f`): `pad_before`/`pad_after` raise `NotImplementedError`; herds beyond 2-D are rejected; a loop bound that depends on a tile coordinate is refused.
- **`gather_runs`** returns runs in local order: owned ids, then ghosts. The local CSR (`GhostPlan.local_csr`) assumes the same order.
- **`ghost_load` replays the schedule**, it does not infer it, so it works for any owner table. Do not replace it with an owner-map formula.
- **The explorer's JavaScript** duplicates the algebra. Any change to `build`, `plan`, or the text functions in `viz.py` needs a matching change in Python, and the explorer test is the check.
- **`nest` splits pieces, not stacks**: a PE with several pieces has each one split the same way. The inner cut's part count must not depend on the piece size, or `t floordiv k` stops being the outer coordinate; every `AxisRule` today meets this.
- **Owner-map tile order**: when a PE holds several pieces, the stack order is the loop nest order (`Schedule.slot`), which equals the view's piece order only for some maps. Tests that compare slots must use the schedule, not `pieces_at` order.

## Where the open decisions are

These are the user's calls. Ask before implementing:
- whether redistribution should emit AIR transfers, or stop at planning for now;
- whether `#dist.layout` should be an MLIR attribute in the AIR fork or stay a text sketch until the Python side is finished (the user said the Python side comes first);
- whether collectives (all-reduce, all-to-all) belong to this library at all, or to the compute side.
