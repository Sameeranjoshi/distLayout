"""AIR ghost gathers from the inspector's schedule (no MLIR-AIR needed)."""

import random

from distlayout import bfs_partition, bind, block_partition, grid_graph, inspect, owner_array, renumber
from distlayout.backends import air as air_backend
from distlayout.backends.schedule import Text


class Buf:
    def __init__(self, *shape):
        self.shape = shape


class Window:
    """A 1-D buffer recording slices as (start, stop)."""

    def __init__(self, name, n):
        self.name, self.shape = name, (n,)

    def __getitem__(self, k):
        return (self.name, k.start, k.stop)


class Recorder:
    def __init__(self):
        self.log = []
        rec = self

        class ops:
            @staticmethod
            def load(dst, src):
                rec.log.append(("load", dst, src))

            @staticmethod
            def branch(cond):
                rec.log.append(("if", cond))

                class B:
                    def __enter__(self):
                        return self

                    def __exit__(self, *a):
                        return False

                    def otherwise(self):
                        rec.log.append(("else",))
                        return self
                return B()

        self.ops = ops


def scrambled(n=8, seed=1):
    perm = list(range(n * n))
    random.Random(seed).shuffle(perm)
    return grid_graph(n, n, perm)


def replay(sched, pe, F, x):
    """Run a PE's emitted loads on concrete data: the local x the PE ends up with."""
    rec = Recorder()
    dst = Window("a", F)
    air_backend.ghost_load(sched, dst, pe, Window("x", len(x)), air=rec)
    local = [None] * F
    for kind, d, s in rec.log:
        lo, hi = (0, F) if d is dst else (d[1], d[2])
        local[lo:hi] = x[s[1]:s[2]]
    return local[:sched[pe].x_footprint]


def test_every_pe_gathers_exactly_its_local_x():
    pat = scrambled()
    for owner in (bfs_partition(pat, 4), block_partition(64, 4)):
        sched = inspect(bind(Buf(64), *owner_array(owner)), pat)
        F = air_backend.ghost_footprint(sched)
        x = [100 + i for i in range(64)]
        for pe, plan in sched.items():
            assert replay(sched, pe, F, x) == [x[g] for g in plan.local_ids()]


def test_renumbering_shrinks_the_dma_count():
    pat = scrambled()
    owner = bfs_partition(pat, 4)
    scattered = inspect(bind(Buf(64), *owner_array(owner)), pat)
    p2, new = renumber(owner)
    tidy = inspect(bind(Buf(64), *owner_array(new)), pat.permute(p2))
    count = lambda s: sum(len(air_backend.gather_runs(p)) for p in s.values())
    assert count(tidy) < count(scattered)
    assert all(len(p.owned.runs) == 1 for p in tidy.values())          # owned part: one DMA per PE


def test_symbolic_emission_is_one_branch_per_pe():
    sched = inspect(bind(Buf(16), *owner_array(block_partition(16, 4))), grid_graph(4, 4))
    rec = Recorder()
    air_backend.ghost_load(sched, Window("a", air_backend.ghost_footprint(sched)), (Text.var("tx"),),
                           Window("x", 16), air=rec)
    assert [e[1] for e in rec.log if e[0] == "if"] == ["tx <= 0", "tx <= 1", "tx <= 2"]
    first_pe = [e for e in rec.log[: rec.log.index(("else",))] if e[0] == "load"]
    assert [e[2][1:] for e in first_pe] == [(0, 4), (4, 8)]              # owned 0:4, then ghosts 4:8
