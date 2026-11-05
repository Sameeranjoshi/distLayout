"""The explorer page.

The page re-implements the v0 layout rules in JavaScript so its builder can
update live. ``test_page_agrees_with_python`` loads the page in a headless
browser (Playwright, if available) and checks, for every gallery preset, that
the page computes the same owners, PE regions, halo plans and layout text as
the Python library.
"""

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from distlayout import Block, Box, Decomposition, Grid, Map, Replicate, UnevenBlock, bind
from distlayout.viz import air_code, render_explorer, render_html, spec_code, to_spec

ROOT = Path(__file__).resolve().parents[1]


class Buf:
    def __init__(self, *shape):
        self.shape = shape


def presets():
    spec = importlib.util.spec_from_file_location("gallery", ROOT / "examples" / "gallery.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.eltwise_mlir = lambda n, tile: None          # no MLIR-AIR needed here
    return mod.presets()


def test_spec_code_round_trips():
    for v in [bind(Buf(16, 32), Decomposition(Block(0, 2), Block(1, 2)), Map(Grid((2, 2)), {0: 0, 1: 1})),
              bind(Buf(100), Decomposition(), Replicate(Grid((4, 4)))),
              bind(Buf(99), Decomposition(UnevenBlock(0, parts=4)), Map(Grid((4, 1)), {0: 0}, replicate=[1]))]:
        env = {"buf": Buf(*v.shape)}
        exec("from distlayout import *\n" + spec_code(v), env)
        assert env["view"].to_attr() == v.to_attr()


def test_air_code():
    v = bind(Buf(100), Decomposition(Block(0, 4)), Map(Grid((4, 4)), {0: 0}, replicate=[1]))
    code, table = air_code(v)
    assert "l2[tx*25 : tx*25 + 25]" in code and len(table) == 16
    uneven = bind(Buf(99), Decomposition(UnevenBlock(0, parts=4)), Map(Grid(4), {0: 0}))
    assert air_code(uneven)[1] == [] and "air.ops.branch(tx <= 2)" in air_code(uneven)[0]


def test_page_is_self_contained():
    v = bind(Buf(100), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    page = render_html(v, title="Block", mlir="module {}")
    assert page.startswith("<!doctype html>")
    assert not re.search(r"<(script|link)[^>]+(src|href)=", page)     # nothing loaded from outside
    presets_json = json.loads(re.search(r"const PRESETS=(\[.*?\]), COLORS", page, re.S).group(1))
    assert presets_json[0]["mlir"] == "module {}"
    assert presets_json[0]["cuts"] == [{"axis": 0, "kind": "block", "parts": 4}] and presets_json[0]["map"] == ["t0"]


def test_to_spec_renumbers_cuts_into_axis_order():
    from distlayout import OwnerMap
    v = bind(Buf(4, 6), Decomposition(Block(1, 3), Block(0, 2)), OwnerMap(Grid((2, 3)), {0: "t1", 1: "t0"}))
    spec = to_spec(v)
    assert [c["axis"] for c in spec["cuts"]] == [0, 1] and spec["map"] == ["t0", "t1"]


def test_to_spec_rejects_what_it_cannot_draw():
    with pytest.raises(NotImplementedError):
        to_spec(bind(Buf(2, 2, 2), Decomposition(Block(0, 2)), Map(Grid(2), {0: 0})))


# -- the JavaScript port against Python ----------------------------------------

def _python_facts(view, halo):
    owners = [_pe_ix(view, view.primary_owner(i)) for i in Box.full(view.shape).indices()]
    held = [[_box(r) for r in view.held_regions(pe)] for pe in view.grid.coords()]
    plans = []
    for pe in view.grid.coords():
        if len(view.pieces_at(pe)) != 1:
            plans.append(None)
            continue
        r = view.held_region(pe)
        req = Box.from_bounds([max(0, o - halo) for o in r.offset], [min(n, s + halo) for s, n in zip(r.stop, view.shape)])
        plans.append(sorted([_box(t.region), t.local, None if t.local else _pe_ix(view, t.src_pe)]
                            for t in view.plan(pe, req).transfers))
    return {"owners": owners, "held": held, "plans": plans, "attr": view.to_attr(),
            "py": spec_code(view, src="SRC")}


def _pe_ix(view, pe):
    ix = 0
    for c, s in zip(pe, view.grid.shape):
        ix = ix * s + c
    return ix


def _box(b):
    return [list(b.offset), list(b.stop)]


JS = r"""
const { chromium } = require('playwright');
(async () => {
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://' + process.argv[2]);
  const out = await p.evaluate(() => PRESETS.map(spec => {
    const D = window.distlayout, L = D.build(spec);
    const all = (shape) => shape.reduce((acc, n) => acc.flatMap(a => [...Array(n).keys()].map(x => a.concat([x]))), [[]]);
    return {
      owners: all(L.shape).map(i => L.peIx(L.primaryOf(L.partOf(i)))),
      held: L.pes.map(pe => L.piecesAt(pe).map(p => { const r = L.region(p); return [r.lo, r.hi]; })),
      plans: L.pes.map(pe => { const pl = D.plan(L, pe, +spec.halo || 0);
        return pl ? pl.pieces.map(x => [[x.r.lo, x.r.hi], x.local, x.local ? null : x.src]) : null; }),
      attr: D.attrText(L), py: D.pyText(Object.assign({}, L, { spec: Object.assign({}, spec, { src: 'SRC' }) })),
    };
  }));
  console.log(JSON.stringify(out)); await b.close();
})();
"""


def _have_playwright():
    if not shutil.which("node"):
        return False
    r = subprocess.run(["node", "-e", "require('playwright')"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _have_playwright(), reason="needs node + playwright")
def test_page_agrees_with_python(tmp_path):
    entries = presets()
    page = tmp_path / "explorer.html"
    page.write_text(render_explorer([(n, v, {k: x for k, x in kw.items() if x is not None}) for n, v, kw in entries]))
    script = tmp_path / "check.js"
    script.write_text(JS)
    res = subprocess.run(["node", str(script), str(page)], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr
    js = json.loads(res.stdout)
    assert len(js) == len(entries)
    for (name, view, kw), got in zip(entries, js):
        want = _python_facts(view, kw.get("halo", 0))
        got["plans"] = [None if pl is None else sorted(pl) for pl in got["plans"]]
        for key in ("owners", "held", "attr", "py"):
            assert got[key] == want[key], f"{name}: {key} differs"
        assert got["plans"] == want["plans"], f"{name}: halo plans differ"
