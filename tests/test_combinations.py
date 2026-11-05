"""Combination tests: random layouts through every layer, checked against independent oracles.

Each seeded case builds a random layout -- 1-D or 2-D, whole / block / uneven
cuts in every policy, placement / owner-map / table / replicate mappings,
replication, refine (sometimes twice) -- and checks it against brute-force
enumeration. Metamorphic tests check that two different spellings of the same
layout own the same elements.
"""

from __future__ import annotations

import random
from itertools import product

import pytest

from combos import Buf, all_indices, brute_owners, pieces_by_pe, random_request, random_view
from distlayout import (Block, Box, DataSource, Decomposition, Grid, IndexSet, Map, OwnerMap, TableMap, UnevenBlock,
                        bfs_partition, bind, block_cyclic, block_partition, cyclic, gather, grid_graph, inspect,
                        owner_array, random_partition, refine, renumber, scatter, skew, spmv)
from distlayout.backends import air as air_backend
from distlayout.backends.schedule import solve
from distlayout.expr import Add, Const, FloorDiv, Mod, Mul, Var, parse
from distlayout.irregular import CSRPattern
from distlayout.mapping import as_placement
from distlayout.viz import spec_code

SEEDS = range(160)


def placement_of(v):
    return as_placement(v.mapping, v.decomposition.num_parts(v.shape))


class Concrete:
    """An ``air`` stand-in for concrete coordinates: loops are ranges, branches never happen."""

    class ops:
        @staticmethod
        def branch(cond):  # pragma: no cover - must not be reached with int coordinates
            raise AssertionError(f"branch emitted for concrete coordinates: {cond}")

    @staticmethod
    def sequential(lo, hi):
        return range(lo, hi)


def key_box(key):
    return Box.from_bounds([k.start for k in key], [k.stop for k in key])


# -- 1. core invariants on random layouts -------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_core_invariants(seed):
    rng = random.Random(seed)
    v = random_view(rng)
    v.validate()
    oracle = brute_owners(v)
    assert set(oracle) == set(all_indices(v.shape))
    for i, (primary, holders) in oracle.items():
        assert v.primary_owner(i) == primary
        assert v.holders(i)[0] == primary and set(v.holders(i)) == holders
        assert all(v.is_local(i, h) for h in holders)
    at = pieces_by_pe(v)
    for pe in v.grid.coords():
        assert sorted(v.held_regions(pe), key=lambda b: b.offset) == sorted(at[pe], key=lambda b: b.offset)
        locals_ = [v.local_index(i, pe) for r in v.held_regions(pe) for i in r.indices()]
        assert len(set(locals_)) == len(locals_)                        # a bijection onto ...
        assert all(Box.full(v.local_shape(pe)).contains(l) for l in locals_)   # ... the local buffer
        if not v.stacked:
            assert len(locals_) == Box.full(v.local_shape(pe)).volume  # dense when one piece per PE


@pytest.mark.parametrize("seed", SEEDS)
def test_plans_cover_requests_exactly(seed):
    rng = random.Random(1000 + seed)
    v = random_view(rng)
    oracle = brute_owners(v)
    pes = list(v.grid.coords())
    for _ in range(3):
        pe, req = rng.choice(pes), random_request(rng, v.shape)
        if not v.held_regions(pe):
            continue
        plan = v.plan(pe, req)
        assert sum(t.region.volume for t in plan.transfers) == req.volume
        for a in range(len(plan.transfers)):
            t = plan.transfers[a]
            assert req.contains_box(t.region)
            assert t.dst_offset == tuple(o - r for o, r in zip(t.region.offset, req.offset))
            assert t.local == any(h.contains_box(t.region) for h in v.held_regions(pe))
            if not t.local:
                assert t.src_pe == oracle[t.region.offset][0]
            for b in plan.transfers[a + 1:]:
                assert t.region.intersect(b.region) is None


@pytest.mark.parametrize("seed", SEEDS)
def test_generated_python_rebuilds_the_layout(seed):
    v = random_view(random.Random(2000 + seed))
    env = {"buf": Buf(*v.shape)}
    exec("from distlayout import *\n" + spec_code(v), env)
    w = env["view"]
    assert w.to_attr() == v.to_attr()
    assert all(w.primary_owner(i) == v.primary_owner(i) for i in all_indices(v.shape))


# -- 2. AIR layer on random layouts -----------------------------------------------

@pytest.mark.parametrize("seed", SEEDS)
def test_owner_map_inversion_or_a_reasoned_refusal(seed):
    v = random_view(random.Random(3000 + seed), kinds=["owner", "owner", "placement"])
    try:
        sched = solve(v)
    except NotImplementedError as e:
        assert "uneven" in str(e), e                  # only uneven tiles may still be refused
        return
    for pe in v.grid.coords():
        got = sched.concrete(pe)
        assert sorted(p for _, p in got) == sorted(v.pieces_at(pe))
        assert len({s for s, _ in got}) == len(got) and all(0 <= s < max(sched.slots, 1) for s, _ in got)


def test_every_uniform_random_owner_map_inverts():
    scans = checks = 0
    for seed in range(300):
        v = random_view(random.Random(3500 + seed), kinds=["owner"], allow_refine=False)
        if not v.is_uniform():
            continue
        sched = solve(v)                                    # must not refuse
        scans += bool(sched.scanned)
        checks += bool(sched.checks)
        for pe in v.grid.coords():
            assert sorted(p for _, p in sched.concrete(pe)) == sorted(v.pieces_at(pe))
    assert checks and scans                                 # both mechanisms were exercised


@pytest.mark.parametrize("seed", SEEDS)
def test_piece_loops_emit_exactly_each_pes_pieces(seed):
    v = random_view(random.Random(4000 + seed))
    if not v.is_uniform() or isinstance(v.mapping, TableMap) and placement_of(v) is None:
        pytest.skip("not emittable through piece loops")
    for pe in v.grid.coords():
        keys = []
        try:
            air_backend.for_each_piece(v, pe, lambda slot, key: keys.append((slot, key_box(key))), air=Concrete)
        except NotImplementedError:
            pytest.skip("owner map not invertible")
        assert sorted(b.offset for _, b in keys) == sorted(r.offset for r in v.held_regions(pe))
        assert all(b in v.held_regions(pe) for _, b in keys)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("boundary", ["zero", "clip", "pad"])
def test_size_class_windows_for_random_halos(seed, boundary):
    rng = random.Random(5000 + seed)
    v = random_view(rng, kinds=["placement", "placement", "replicate"])
    if placement_of(v) is None:
        pytest.skip("not a placement")
    halo = rng.randint(0, 3)
    tile_shape = air_backend.region_shape(v, halo, boundary)
    for pe in v.grid.coords():
        got = []
        air_backend.for_each_region(v, pe, got.append, halo=halo, boundary=boundary, air=Concrete)
        assert len(got) == 1
        k, held = got[0], v.held_region(pe)
        lo = [max(0, o - halo) for o in held.offset]
        hi = [min(n, e + halo) for e, n in zip(held.stop, v.shape)]
        assert [(s.start, s.stop) for s in k["src"]] == list(zip(lo, hi))
        assert [(s.start, s.stop) for s in k["own_src"]] == list(zip(held.offset, held.stop))
        pb = k["pad_before"] or [0] * len(v.shape)
        pa = k["pad_after"] or [0] * len(v.shape)
        for a in range(len(v.shape)):
            src, tile, win, own_t, own_s = k["src"][a], k["tile"][a], k["window"][a], k["own_tile"][a], k["own_src"][a]
            assert tile.start == 0 and tile.stop <= tile_shape[a]
            assert win.start == pb[a] and win.stop - win.start == src.stop - src.start
            assert tile.stop == win.stop + pa[a]
            assert own_t.start - win.start + src.start == own_s.start        # tile index -> global index
            if boundary == "clip":
                assert pb[a] == pa[a] == 0
            else:                                                            # full b + 2*halo tile everywhere
                assert tile.stop == held.size[a] + 2 * halo


# -- 3. metamorphic: two spellings, one ownership ------------------------------

@pytest.mark.parametrize("seed", range(40))
def test_refine_twice_equals_refine_once(seed):
    rng = random.Random(6000 + seed)
    P, k1, k2 = rng.randint(1, 4), rng.randint(1, 3), rng.randint(1, 3)
    n = P * k1 * k2 * rng.randint(1, 3)
    base = bind(Buf(n), Decomposition(Block(0, P)), Map(Grid(P), {0: 0}))
    twice = refine(refine(base, 0, k1), 0, k2)
    once = refine(base, 0, k1 * k2)
    for i in range(n):
        r2, s1, s2 = twice.primary_owner(i)
        r1, s = once.primary_owner(i)
        assert (r2, s1 * k2 + s2) == (r1, s) and r1 == base.primary_owner(i)[0]


@pytest.mark.parametrize("seed", range(40))
def test_refine_by_one_only_adds_an_axis(seed):
    v = random_view(random.Random(6500 + seed), allow_refine=False, kinds=["placement", "owner"])
    w = refine(v, rng_axis(seed, v), 1)
    assert all(w.primary_owner(i) == v.primary_owner(i) + (0,) for i in all_indices(v.shape))


def rng_axis(seed, v):
    return seed % len(v.shape)


@pytest.mark.parametrize("seed", range(60))
def test_any_1d_layout_rewritten_as_an_owner_array(seed):
    v = random_view(random.Random(7000 + seed))
    if len(v.shape) != 1 or any(v.grid.shape[g] > 1 for g in v.mapping.replicate):
        pytest.skip("1-D, unreplicated layouts only")
    lin = lambda pe: sum(c * s for c, s in zip(pe, [prod_(v.grid.shape[g + 1:]) for g in range(len(pe))]))
    owner = [lin(v.primary_owner(i)) for i in range(v.shape[0])]
    w = bind(Buf(v.shape[0]), *owner_array(owner, grid=Grid(prod_(v.grid.shape))))   # keep empty PEs
    w.validate()
    for i in range(v.shape[0]):
        assert w.primary_owner(i)[0] == owner[i]
    for pe in v.grid.coords():
        mine = IndexSet.from_boxes(v.held_regions(pe))
        assert IndexSet.from_boxes(w.held_regions((lin(pe),))) == mine


def prod_(xs):
    out = 1
    for x in xs:
        out *= x
    return out


@pytest.mark.parametrize("n,P", [(12, 4), (20, 4), (9, 3), (24, 6), (10, 4)])
def test_cyclic_family_equivalences(n, P):
    a, b = bind(Buf(n), *cyclic(n, P)), bind(Buf(n), *block_cyclic(n, 1, P))
    assert [a.primary_owner(i) for i in range(n)] == [b.primary_owner(i) for i in range(n)]
    if n % P == 0:
        big = bind(Buf(n), *block_cyclic(n, n // P, P))
        blk = bind(Buf(n), Decomposition(Block(0, P)), Map(Grid(P), {0: 0}))
        assert [big.primary_owner(i) for i in range(n)] == [blk.primary_owner(i) for i in range(n)]


@pytest.mark.parametrize("seed", range(30))
def test_transposed_placement_is_a_transpose(seed):
    rng = random.Random(7500 + seed)
    r, c = rng.randint(2, 10), rng.randint(2, 10)
    p0 = rng.choice([d for d in range(1, r + 1) if r % d == 0])
    p1 = rng.choice([d for d in range(1, c + 1) if c % d == 0])
    d = Decomposition(Block(0, p0), Block(1, p1))
    straight = bind(Buf(r, c), d, Map(Grid((p0, p1)), {0: 0, 1: 1}))
    crossed = bind(Buf(r, c), d, Map(Grid((p1, p0)), {0: 1, 1: 0}))
    owner_map = bind(Buf(r, c), d, OwnerMap(Grid((p1, p0)), {0: "t1", 1: "t0"}))
    for i, j in product(range(r), range(c)):
        a, b = straight.primary_owner((i, j)), crossed.primary_owner((i, j))
        assert b == (a[1], a[0]) and owner_map.primary_owner((i, j)) == b
    assert placement_of(owner_map) is not None


def test_skew_is_a_latin_square():
    v = bind(Buf(8, 8), *skew((8, 8), 4))
    for i in range(8):
        assert sorted(v.primary_owner((i, j))[0] for j in range(4)) == [0, 1, 2, 3]
        assert sorted(v.primary_owner((j, i))[0] for j in range(4)) == [0, 1, 2, 3]


# -- 4. irregular: rows layout x  x layout  x  random sparse patterns -----------

def layouts_1d(n, P, pattern, rng):
    out = {
        "block": bind(Buf(n), Decomposition(UnevenBlock(0, parts=P)), Map(Grid(P), {0: 0})),
        "uneven_sizes": bind(Buf(n), *owner_array(sorted(rng.choice(range(P)) for _ in range(n)), grid=Grid(P))),
        "cyclic": bind(Buf(n), Decomposition(Block(0, n)), OwnerMap(Grid(P), {0: f"t0 mod {P}"})),
        "block_cyclic": bind(Buf(n), Decomposition(UnevenBlock(0, parts=-(-n // 2))), OwnerMap(Grid(P), {0: f"t0 mod {P}"})),
        "random_owner": bind(Buf(n), *owner_array(random_partition(n, P, seed=rng.randrange(99)), grid=Grid(P))),
    }
    if pattern.nrows == n:
        out["bfs"] = bind(Buf(n), *owner_array(bfs_partition(pattern, P), grid=Grid(P)))
    return out


def random_pattern(rng, nrows, ncols, density):
    rows = []
    for i in range(nrows):
        r = {c for c in range(ncols) if rng.random() < density}
        if i < ncols:
            r.add(i)
        rows.append(r)
    return CSRPattern.from_rows(rows, ncols=ncols)


def brute_schedule(rows, x, pattern):
    """Per PE: rows, owned x, ghosts, {owner: ids} -- by enumerating ids and holders."""
    out = {}
    xowner = {i: x.primary_owner(i) for i in range(x.shape[0])}
    for pe in rows.grid.coords():
        my_rows = {i for i in range(rows.shape[0]) if rows.is_local(i, pe)}
        owned = {i for i in range(x.shape[0]) if x.is_local(i, pe)}
        needed = {c for i in my_rows for c in pattern.row(i)}
        ghosts = needed - owned
        imports = {}
        for g in ghosts:
            imports.setdefault(xowner[g], set()).add(g)
        out[pe] = (my_rows, owned, ghosts, imports)
    return out


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("square", [True, False])
def test_irregular_rows_times_x_layouts(seed, square):
    rng = random.Random(8000 + seed)
    n = rng.randint(12, 30)
    m = n if square else rng.randint(8, 30)
    P = rng.choice([2, 3, 4])
    pattern = grid_graph(*divisors_pair(n), list(rng.sample(range(n), n))) if square and seed % 2 else \
        random_pattern(rng, n, m, rng.choice([0.05, 0.15, 0.3]))
    row_layouts = layouts_1d(n, P, pattern, rng)
    x_layouts = layouts_1d(m, P, pattern if square else CSRPattern.from_rows([[]] * m, ncols=1), rng)
    values = [rng.uniform(-1, 1) for _ in range(pattern.nnz)]
    xv = [rng.uniform(-1, 1) for _ in range(m)]
    dense = [sum(values[k] * xv[pattern.colidx[k]] for k in range(pattern.rowptr[i], pattern.rowptr[i + 1]))
             for i in range(n)]
    for (rn, rows), (xn, x) in product(row_layouts.items(), x_layouts.items()):
        sched = inspect(rows, pattern, x)
        want = brute_schedule(rows, x, pattern)
        for pe, (my_rows, owned, ghosts, imports) in want.items():
            p = sched[pe]
            assert (set(p.rows), set(p.owned), set(p.ghosts)) == (my_rows, owned, ghosts), (rn, xn, pe)
            assert {s: set(ids) for s, ids in p.imports.items()} == imports, (rn, xn, pe)
            for dst, ids in p.exports.items():
                assert sched[dst].imports[pe] == ids
        y = spmv(sched, pattern, values, scatter(x, xv))
        got = [None] * n
        for pe, plan in sched.items():
            for i, val in zip(plan.rows, y[pe]):
                got[i] = val
        assert got == pytest.approx(dense), (rn, xn)
        # AIR gather replay: every PE ends up with exactly x[local_ids]
        F = air_backend.ghost_footprint(sched)
        for pe, plan in sched.items():
            assert replay_gather(sched, pe, F, xv) == [xv[g] for g in plan.local_ids()], (rn, xn, pe)


def divisors_pair(n):
    for a in range(int(n ** 0.5), 0, -1):
        if n % a == 0:
            return a, n // a
    return 1, n


class _Win:
    def __init__(self, n):
        self.shape = (n,)

    def __getitem__(self, k):
        return (k.start, k.stop)


def replay_gather(sched, pe, F, x):
    dst, src, local = _Win(F), _Win(len(x)), [None] * F
    loads = []

    class A:
        class ops:
            @staticmethod
            def load(d, s):
                loads.append((d, s))

    air_backend.ghost_load(sched, dst, pe, src, air=A)
    for d, s in loads:
        lo, hi = (0, F) if d is dst else d
        local[lo:hi] = x[s[0]:s[1]]
    return local[:sched[pe].x_footprint]


@pytest.mark.parametrize("seed", range(10))
def test_renumbering_preserves_ghosts_and_spmv(seed):
    rng = random.Random(8500 + seed)
    nx, ny = rng.randint(3, 7), rng.randint(3, 7)
    n = nx * ny
    perm = list(range(n))
    rng.shuffle(perm)
    pat = grid_graph(nx, ny, perm)
    owner = bfs_partition(pat, rng.choice([2, 3, 4]))
    p2, new = renumber(owner)
    a = inspect(bind(Buf(n), *owner_array(owner)), pat)
    b = inspect(bind(Buf(n), *owner_array(new)), pat.permute(p2))
    assert a.volume == b.volume and a.messages == b.messages
    assert sorted(len(p.ghosts) for p in a.values()) == sorted(len(p.ghosts) for p in b.values())
    # renumbered: one owned DMA per PE
    assert all(len(p.owned.runs) == 1 for p in b.values())


# -- 5. expression fuzz ----------------------------------------------------------

def random_expr(rng, depth, nvars):
    if depth == 0 or rng.random() < 0.3:
        return Var(rng.randrange(nvars)) if rng.random() < 0.7 else Const(rng.randint(-5, 9))
    kind = rng.choice(["add", "sub", "mul", "fdiv", "mod", "neg"])
    a = random_expr(rng, depth - 1, nvars)
    if kind == "add":
        return a + random_expr(rng, depth - 1, nvars)
    if kind == "sub":
        return a - random_expr(rng, depth - 1, nvars)
    if kind == "mul":
        return a * rng.choice([-3, -1, 2, 3, 5])
    if kind == "fdiv":
        return a // rng.randint(1, 6)
    if kind == "neg":
        return -a
    return a % rng.randint(1, 7)


def reference_eval(e, t):
    if isinstance(e, Var):
        return t[e.k]
    if isinstance(e, Const):
        return e.k
    if isinstance(e, Add):
        return reference_eval(e.a, t) + reference_eval(e.b, t)
    if isinstance(e, Mul):
        return reference_eval(e.a, t) * e.k
    if isinstance(e, FloorDiv):
        return reference_eval(e.a, t) // e.k
    return reference_eval(e.a, t) % e.k


@pytest.mark.parametrize("seed", range(200))
def test_expression_print_parse_eval(seed):
    rng = random.Random(9000 + seed)
    e = random_expr(rng, 4, 3)
    back = parse(str(e))
    assert str(parse(str(back))) == str(back)          # printing is a fixpoint after one round
    for _ in range(10):
        t = [rng.randint(-20, 40) for _ in range(3)]
        assert back.eval(t) == e.eval(t) == reference_eval(e, t)


# -- 6. the explorer's JavaScript port on random layouts --------------------------

def _drawable(v):
    rules = v.decomposition.rules
    return (len(v.shape) <= 2 and len(v.grid.shape) <= 3 and v.source.is_global
            and [r.axis for r in rules] == sorted(r.axis for r in rules)
            and all(isinstance(r, (Block, UnevenBlock)) for r in rules)
            and prod_(v.shape) <= 900)


def test_explorer_port_on_random_layouts(tmp_path):
    import json
    import shutil
    import subprocess
    if not shutil.which("node") or subprocess.run(["node", "-e", "require('playwright')"], capture_output=True).returncode:
        pytest.skip("needs node + playwright")
    from distlayout.viz import render_explorer
    views = []
    seed = 0
    while len(views) < 60:
        v = random_view(random.Random(10000 + seed))
        seed += 1
        if _drawable(v):
            views.append(v)
    page = tmp_path / "p.html"
    page.write_text(render_explorer([(f"v{i}", v, {"src": "SRC"}) for i, v in enumerate(views)]))
    script = tmp_path / "c.js"
    script.write_text(r"""
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + process.argv[2]);
  const out = await p.evaluate(() => PRESETS.map(spec => {
    try {
      const D = window.distlayout, L = D.build(spec);
      const all = s => s.reduce((acc, n) => acc.flatMap(a => [...Array(n).keys()].map(x => a.concat([x]))), [[]]);
      return { owners: all(L.shape).map(i => L.primaryOf(L.partOf(i))),
               held: L.pes.map(pe => L.piecesAt(pe).map(q => { const r = L.region(q); return [r.lo, r.hi]; })),
               attr: D.attrText(L), py: D.pyText(L) };
    } catch (e) { return { error: e.message }; }
  }));
  console.log(JSON.stringify(out)); await b.close();
})();
""")
    res = subprocess.run(["node", str(script), str(page)], capture_output=True, text=True, timeout=180)
    assert res.returncode == 0, res.stderr
    got = json.loads(res.stdout)
    for v, g in zip(views, got):
        assert "error" not in g, (v.to_attr(), g)
        assert g["attr"] == v.to_attr()
        assert g["py"] == spec_code(v, src="SRC")
        assert [tuple(o) for o in g["owners"]] == [v.primary_owner(i) for i in all_indices(v.shape)]
        want = [[[list(r.offset), list(r.stop)] for r in v.held_regions(pe)] for pe in v.grid.coords()]
        assert g["held"] == want


# -- 7. real air.api builds of random combinations -----------------------------

@pytest.mark.parametrize("seed", range(30))
def test_random_layouts_build_with_real_air(seed):
    pytest.importorskip("air.api")
    rng = random.Random(11000 + seed)
    v = random_view(rng)
    halo = rng.randint(0, 2) if placement_of(v) is not None else 0
    try:
        launch = air_backend.copy_kernel(v, name=f"combo{seed}", halo=halo)
        text = str(launch.build(target="npu2"))
    except NotImplementedError as e:
        pytest.skip(f"not emittable: {e}")
    assert "air.herd" in text and "air.dma_memcpy_nd" in text


@pytest.mark.parametrize("seed", range(6))
def test_random_ghost_gathers_build_with_real_air(seed):
    pytest.importorskip("air.api")
    rng = random.Random(12000 + seed)
    n, P = rng.randint(10, 40), rng.choice([2, 3, 4])
    pattern = random_pattern(rng, n, n, 0.12)
    rows = bind(Buf(n), *owner_array(random_partition(n, P, seed=seed), grid=Grid(P)))
    sched = inspect(rows, pattern)
    text = str(air_backend.ghost_gather_kernel(sched, n, name=f"g{seed}").build(target="npu2"))
    loads = [l for l in text.splitlines() if "dma_memcpy_nd" in l and f"memref<{n}xf32, 1" in l and "2 : i32>, memref" in l]
    assert len(loads) == sum(len(air_backend.gather_runs(p)) for p in sched.values())
