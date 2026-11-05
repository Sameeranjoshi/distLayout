"""Layout explorer: one interactive HTML page for picturing layouts.

    from distlayout.viz import render_html
    open("explorer.html", "w").write(render_html(view, title="My layout"))

or, for several views as presets::

    render_explorer([("Block", view1, {}), ("Halo", view2, {"halo": 1})])

The page is a small workbench:

* **presets** across the top;
* a **builder** on the left, in the same two steps as the library:
  *cut* (per tensor axis: whole, block or uneven, and how many pieces) and
  *map* (per grid axis, an owner expression over piece coordinates such as
  ``t0`` or ``(t0 + t1) mod 4``; blank = replicated), plus halo radius,
  element size and PE memory;
* **one picture** on the right: ① the decomposition as a neutral cutter,
  ② the tensor with the same cuts, colored by owner PE, ③ the PE grid --
  pieces held, memory-fit bar, red arrows for halo imports;
* **code tabs**: Python, ``#dist.layout``, ``air.api`` (placements only), MLIR;
* **pin** to compare bytes/PE, **export** the picture as SVG.

The layout rules run in the page (a port of the Python algebra) so the
builder updates live; ``tests/test_viz.py`` checks the port against Python in a
headless browser. No network, fonts or libraries are needed.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .decomposition import Block, UnevenBlock
from .expr import Var, substitute
from .mapping import as_placement
from .view import DistributedView

PE_COLORS = ["#3978CF", "#E59B38", "#249A82", "#A36BC6", "#D95455", "#52657B",
             "#2FA7C9", "#C58B1E", "#5E9E4A", "#7A5BC7", "#C2577F", "#3E8E8A",
             "#8C6A4A", "#4867B8", "#B8743A", "#6B7F99"]


# -- Python-side text, also used by tests -------------------------------------

class _Sym:
    """A tiny affine form (``tx*25 + 3``) for printing the backend's offsets."""

    def __init__(self, terms: Optional[Dict[str, int]] = None, const: int = 0) -> None:
        self.terms = {k: v for k, v in (terms or {}).items() if v}
        self.const = const

    @staticmethod
    def of(name: str) -> "_Sym":
        return _Sym({name: 1})

    def _lift(self, o: Any) -> "_Sym":
        return o if isinstance(o, _Sym) else _Sym({}, int(o))

    def __add__(self, o: Any) -> "_Sym":
        o = self._lift(o)
        t = dict(self.terms)
        for k, v in o.terms.items():
            t[k] = t.get(k, 0) + v
        return _Sym(t, self.const + o.const)

    __radd__ = __add__

    def __mul__(self, o: Any) -> "_Sym":
        o = self._lift(o)
        if self.terms and o.terms:
            raise TypeError("non-affine")
        a, k = (self, o.const) if not o.terms else (o, self.const)
        return _Sym({n: c * k for n, c in a.terms.items()}, a.const * k)

    __rmul__ = __mul__

    def __str__(self) -> str:
        parts = [n if c == 1 else f"{n}*{c}" for n, c in self.terms.items()]
        if self.const or not parts:
            parts.append(str(self.const))
        return " + ".join(parts).replace("+ -", "- ")


def _grid_repr(shape: Sequence[int]) -> str:
    return f"Grid({tuple(shape)})" if len(shape) > 1 else f"Grid({shape[0]})"


def spec_code(view: DistributedView, src: str = "buf") -> str:
    """Python that rebuilds ``view``'s layout over a handle named ``src``."""
    rules = ", ".join(repr(r) for r in view.decomposition.rules)
    m = as_placement(view.mapping, view.decomposition.num_parts(view.shape))
    grid = _grid_repr(view.grid.shape)
    if m is None:
        mapping = repr(view.mapping)
    elif not m.place and len(m.replicate) == m.grid.rank:
        mapping = f"Replicate({grid})"
    else:
        rep = f", replicate={list(m.replicate)}" if m.replicate else ""
        mapping = f"Map({grid}, {m.place_dict}{rep})"
    return f"view = bind({src}, Decomposition({rules}), {mapping})"


def air_code(view: DistributedView, src: str = "l2", dtype: str = "f32",
             halo: int = 0) -> Tuple[str, List[Tuple[Tuple[int, ...], Tuple[slice, ...]]]]:
    """(``air.api`` herd body, per-PE slices). The slices are given for uniform placements without halo."""
    from .backends import air as backend

    names = ["tx", "ty", "tz"][: view.grid.rank]
    if (halo or not view.is_uniform()
            or as_placement(view.mapping, view.decomposition.num_parts(view.shape)) is None):
        return _traced_code(view, names, src, dtype, halo), []
    key = backend.slice_key(view, [_Sym.of(n) for n in names])
    sub = ", ".join(f"{k.start} : {k.stop}" for k in key)
    herd = ", ".join(f"range({r.start}, {r.stop}, {r.step})" if r.step != 1 else f"range({r.stop})"
                     for r in backend.herd_space(view))
    code = (f"with air.herd([{herd}]) as h:\n"
            f"  @h.body\n"
            f"  def _({', '.join(names)}):\n"
            f"    a = air.alloc({list(view.max_local_shape())}, {dtype}, scope=h.private())\n"
            f"    air.ops.load(a, {src}[{sub}])")
    return code, [(pe, backend.slice_key(view, pe)) for pe in view.grid.coords()]


class _TextAir:
    """A stand-in for ``air.api`` that writes the calls it receives as Python source."""

    def __init__(self, indent: str = "    ") -> None:
        self.lines: List[str] = []
        self.ind = indent
        self.nb = 0
        outer = self

        class ops:
            @staticmethod
            def load(dst, src, pad_before=None, pad_after=None):
                pads = (f", pad_before={pad_before}" if pad_before else "") + (f", pad_after={pad_after}" if pad_after else "")
                outer.emit(f"air.ops.load({dst}, {src}{pads})")

            @staticmethod
            def store(src, dst):
                outer.emit(f"air.ops.store({src}, {dst})")

            @staticmethod
            def fill(buf, value):
                outer.emit(f"air.ops.fill({buf}, {value})               # ghost cells outside the tensor")

            @staticmethod
            def branch(cond):
                return outer._branch(cond)

        self.ops = ops

    def emit(self, line: str) -> None:
        self.lines.append(self.ind + line)

    def sequential(self, lo, hi):
        from .backends.schedule import Text
        name = f"s{sum(l.strip().startswith('for ') for l in self.lines)}"
        self.emit(f"for {name} in air.sequential({lo}, {hi}):")
        self.ind += "  "
        yield Text.var(name)
        self.ind = self.ind[:-2]

    def _branch(self, cond):
        outer = self
        name = f"b{self.nb}"
        self.nb += 1

        class B:
            def __enter__(self):
                outer.emit(f"with air.ops.branch({cond}) as {name}:")
                outer.ind += "  "
                return self

            def __exit__(self, *exc):
                outer.ind = outer.ind[:-2]
                return False

            def otherwise(self):
                class O:
                    def __enter__(self_inner):
                        outer.emit(f"with {name}.otherwise():")
                        outer.ind += "  "

                    def __exit__(self_inner, *exc):
                        outer.ind = outer.ind[:-2]
                        return False
                return O()

        return B()


class _TextBuf:
    def __init__(self, name: str, shape: Sequence[int]) -> None:
        self.name, self.shape = name, tuple(shape)

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        return _TextBuf(self.name + "[" + ", ".join(f"{k.start} : {k.stop}" if isinstance(k, slice) else str(k)
                                                    for k in key) + "]", ())

    def __str__(self) -> str:
        return self.name


def _ghost_code(sched: Any, src: str) -> str:
    """The ghost-gather herd body for an irregular schedule, traced into text."""
    from .backends import air as backend
    from .backends.schedule import Text
    w = _TextAir()
    F = backend.ghost_footprint(sched)
    pe0 = next(iter(sched))
    names = ["tx", "ty", "tz"][: len(pe0)]
    backend.ghost_load(sched, _TextBuf("x_local", [F]), [Text.var(n) for n in names], _TextBuf(src, [0]), air=w)
    runs = sum(len(backend.gather_runs(p)) for p in sched.values())
    grid = [max(pe[g] for pe in sched) + 1 for g in range(len(pe0))]
    head = [f"with air.herd([{', '.join(f'range({n})' for n in grid)}]) as h:", "  @h.body", f"  def _({', '.join(names)}):",
            f"    x_local = air.alloc([{F}], f32, scope=h.private())    # owned, then ghosts (local CSR order)"]
    return "\n".join(head + w.lines + ["    ...                                   # local SpMV on x_local", "",
                                        f"# one branch per PE, one static DMA per run of ids: {runs} DMAs in total",
                                        "# air_backend.ghost_load(sched, x_local, coords, l2_x)"])


def _traced_code(view: DistributedView, names: List[str], src: str, dtype: str, halo: int) -> str:
    """The herd body the AIR backend emits for ``view``, traced into text."""
    from .backends import air as backend
    from .backends.schedule import Text
    coords = [Text.var(n) for n in names]
    w = _TextAir()
    l2 = _TextBuf(src, view.shape)
    placement = as_placement(view.mapping, view.decomposition.num_parts(view.shape)) is not None
    try:
        if placement:
            tile = list(backend.region_shape(view, halo))
            a = _TextBuf("a", tile)
            backend.halo_load(view, a, coords, halo, src=l2, air=w)
            w.emit("...                                   # compute on a")
            backend.store_owned(view, a, coords, halo, dst=_TextBuf(src + "_out", view.shape), air=w)
            note = (f"# {len(backend.size_classes(view, halo)[0])} size class(es) per cut axis; "
                    f"each branch has static sizes" + (f"; halo {halo} read from {src}, zeros outside the tensor"
                                                       if halo else ""))
        else:
            tile = list(view.partition_region(next(view.partitions())).size)
            a = _TextBuf("a", tile)

            def piece(slot, key):
                w.ops.load(a, l2[key])
                w.emit("  ...                                 # compute on piece a"[2:])
                w.ops.store(a, _TextBuf(src + "_out", view.shape)[key])
            backend.for_each_piece(view, coords, piece, air=w)
            pieces = len(view.pieces_at(next(view.grid.coords())))
            note = (f"# {pieces} pieces per PE; owner map inverted by distlayout.backends.schedule: "
                    + ", ".join(f"g{g} = {e}" for g, e in sorted(view.mapping.exprs().items())))
    except NotImplementedError as e:
        return f"# not emitted: {e}"
    herd = ", ".join(f"range({r.start}, {r.stop}, {r.step})" if r.step != 1 else f"range({r.stop})"
                     for r in backend.herd_space(view))
    head = [f"with air.herd([{herd}]) as h:", "  @h.body", f"  def _({', '.join(names)}):",
            f"    a = air.alloc({tile}, {dtype}, scope=h.private())"]
    return "\n".join(head + w.lines + ["", note])


# -- view -> page spec -------------------------------------------------------------

def to_spec(view: DistributedView, title: str = "", note: str = "", halo: int = 0,
            mlir: Optional[str] = None, src: str = "l2", elem_bytes: int = 4,
            pe_kib: float = 48, pattern: Any = None, positions: Optional[Sequence[Tuple[int, int]]] = None) -> Dict[str, Any]:
    """The JSON the page understands. Raises for layouts the page cannot draw.

    Cuts are listed in tensor-axis order (as the builder shows them); owner
    expressions are renumbered to match.
    """
    if len(view.shape) > 2:
        raise NotImplementedError("the explorer draws rank-1 and rank-2 tensors")
    if not view.source.is_global:
        raise NotImplementedError("the explorer draws global sources")
    rules = list(view.decomposition.rules)
    order = sorted(range(len(rules)), key=lambda k: rules[k].axis)
    new_of = {old: new for new, old in enumerate(order)}
    cuts = []
    for k in order:
        r = rules[k]
        n = view.shape[r.axis]
        if isinstance(r, Block):
            cuts.append({"axis": r.axis, "kind": "block", "parts": r.parts})
        elif isinstance(r, UnevenBlock):
            c = {"axis": r.axis, "kind": "uneven", "parts": r.num_parts(n), "policy": r.policy}
            if r.sizes_ is not None:
                c["sizes"] = list(r.sizes_)
            cuts.append(c)
        else:
            raise NotImplementedError(f"the explorer cannot draw {r!r}")
    from .mapping import TableMap
    table = isinstance(view.mapping, TableMap)
    exprs = {} if table else view.mapping.exprs()
    mapping: List[Optional[str]] = [None] * view.grid.rank
    for g, e in exprs.items():
        for old in range(len(rules)):                      # two-phase renumbering avoids collisions
            e = substitute(e, old, Var(1000 + new_of[old]))
        for new in range(len(rules)):
            e = substitute(e, 1000 + new, Var(new))
        mapping[g] = str(e)
    spec = {"title": title, "note": note, "shape": list(view.shape), "grid": list(view.grid.shape),
            "cuts": cuts, "map": mapping, "halo": halo, "bytes": elem_bytes, "kib": pe_kib, "src": src}
    if table:
        spec["table"], spec["rep"] = view.mapping.linear(), list(view.mapping.replicate)
    if pattern is not None:                      # irregular access: ghosts and imports from the inspector
        from .irregular import inspect
        sched = inspect(view, pattern)
        ix = {pe: k for k, pe in enumerate(view.grid.coords())}
        spec["ghosts"] = [len(sched[pe].ghosts) for pe in view.grid.coords()]
        spec["imports"] = [[ix[pe], ix[src], len(ids)] for pe, pl in sched.items() for src, ids in pl.imports.items()]
        spec["air"] = _ghost_code(sched, src)
        if positions is not None:
            spec["graph"] = {"pos": [list(map(int, q)) for q in positions],
                             "edges": [[i, j] for i in range(pattern.nrows) for j in pattern.row(i) if i < j]}
    if mlir:
        spec["mlir"] = mlir
    if (halo or not view.is_uniform()
            or as_placement(view.mapping, view.decomposition.num_parts(view.shape)) is None):
        spec["air"] = air_code(view, src, halo=halo)[0]
    return spec


def render_explorer(entries: Sequence[Tuple[str, DistributedView, Dict[str, Any]]],
                    title: str = "distlayout explorer") -> str:
    """A page whose presets are ``[(name, view, to_spec kwargs), ...]``."""
    presets = [to_spec(v, title=name, **kw) for name, v, kw in entries]
    return (_PAGE.replace("%TITLE%", html.escape(title))
            .replace("%PRESETS%", json.dumps(presets).replace("</", "<\\/"))
            .replace("%COLORS%", json.dumps(PE_COLORS)))


def render_html(view: DistributedView, title: str = "Data layout", **kw: Any) -> str:
    """One view as an explorer page (it is the only preset; the builder still edits it)."""
    return render_explorer([(title, view, kw)], title=title)


# -- the page -------------------------------------------------------------------------

_PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>%TITLE%</title>
<style>
:root{--ink:#17253C;--muted:#52657B;--faint:#8896A8;--card:#F6F9FD;--line:#DDE7F2;--bg:#FFFFFF;
--ghost:#A8C9F1;--comm:#D95455;--code:#F4F7FB;--accent:#3978CF;--ok:#249A82}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ink:#E8EEF6;--muted:#A9B7C8;--faint:#7D8FA3;
--card:#18222F;--line:#2A3747;--bg:#0F1722;--ghost:#3C5A80;--code:#141D29}}
:root[data-theme="dark"]{--ink:#E8EEF6;--muted:#A9B7C8;--faint:#7D8FA3;--card:#18222F;--line:#2A3747;--bg:#0F1722;--ghost:#3C5A80;--code:#141D29}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Aptos,"Segoe UI",Inter,system-ui,-apple-system,sans-serif;font-size:14px}
.wrap{margin:0 auto;padding:20px 28px 32px}
h1{font-size:24px;margin:0 0 12px;font-weight:700}
.presets{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.presets button,.btn{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--ink);
border-radius:8px;padding:6px 12px;cursor:pointer}
.presets button.on{background:var(--accent);border-color:var(--accent);color:#fff}
.main{display:grid;grid-template-columns:360px 1fr;gap:18px;align-items:start}
@media(max-width:860px){.main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.card h2{font-size:15px;margin:0 0 4px}
h3{font-size:12.5px;margin:14px 0 2px;color:var(--ink)}
label{display:block;color:var(--muted);font-size:12px;margin:8px 0 4px}
input,select{font:inherit;font-size:13px;width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
input.mono{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px}
.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}.row3{display:grid;grid-template-columns:1.3fr 1fr;gap:8px}
.help{color:var(--faint);font-size:11.5px;margin:2px 0 0}
.err{color:var(--comm);font-size:12.5px;min-height:18px;margin-top:8px}
.tabs{display:flex;gap:4px;margin:0;border-bottom:1px solid var(--line)}
.tabs button{font:inherit;font-size:14px;border:0;background:none;color:var(--muted);padding:6px 10px;cursor:pointer;border-bottom:2px solid transparent}
.tabs button.on{color:var(--ink);border-bottom-color:var(--accent)}
pre{background:var(--code);border:1px solid var(--line);border-top:0;border-radius:0 0 8px 8px;margin:0;padding:10px 12px;
font:14px/1.55 ui-monospace,Menlo,Consolas,monospace;white-space:pre;max-height:560px;overflow:auto}
.codecard{margin-top:18px}
.note{color:var(--muted);font-size:13px;margin:0 0 10px}
.formula{font:12.5px ui-monospace,Menlo,Consolas,monospace;color:var(--muted);min-height:20px;margin:8px 0 0}
.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:12px}
.stat{font-size:12.5px;color:var(--muted)}.stat b{color:var(--ink)}
svg text{font-family:Aptos,"Segoe UI",Inter,system-ui,sans-serif}
.legend{display:flex;flex-wrap:wrap;gap:12px;color:var(--muted);font-size:12px;margin-top:8px}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;vertical-align:-1px;margin-right:5px}
</style></head><body><div class="wrap">
<h1>%TITLE%</h1>
<div class="presets" id="presets"></div>
<div class="main">
 <div class="card">
  <h2>Layout</h2>
  <div class="row"><div><label>Tensor shape</label><input id="shape"></div><div><label>PE grid</label><input id="grid"></div></div>
  <h3>① Decomposition layout</h3>
  <div id="cuts"></div>
  <h3>③ Mapping layout</h3>
  <div id="maps"></div>
  <div class="row"><div><label>Halo radius</label><input id="halo" type="number" min="0" value="0"></div>
   <div><label>Bytes / element · PE KiB</label><div class="row"><input id="bytes" type="number" min="1"><input id="kib" type="number" min="1"></div></div></div>
  <div class="err" id="err"></div>
 </div>
 <div class="card">
  <div id="pic"></div>
  <div class="formula" id="formula"></div>
  <div class="legend" id="legend"></div>
  <div class="bar"><button class="btn" id="pin">Pin as A</button><button class="btn" id="svg">Export SVG</button><span class="stat" id="stat"></span></div>
 </div>
</div>
<div class="card codecard"><div class="tabs" id="tabs"></div><pre id="code"></pre></div>
</div>
<script>
const PRESETS=%PRESETS%, COLORS=%COLORS%;
const $=id=>document.getElementById(id);
let cur=null, pinned=null, tab="python";

// ---------- owner-map expressions (same grammar and printing as distlayout.expr) ----------
const E={var:k=>({t:"var",k}),con:k=>({t:"con",k}),add:(a,b)=>({t:"add",a,b}),mul:(a,k)=>({t:"mul",a,k}),fdiv:(a,k)=>({t:"fdiv",a,k}),mod:(a,k)=>({t:"mod",a,k})};
function esub(a,b){return E.add(a,b.t==="con"?E.con(-b.k):E.mul(b,-1))}
function emul(a,b){if(b.t==="con")return E.mul(a,b.k);if(a.t==="con")return E.mul(b,a.k);throw new Error("non-affine: multiply by a constant only")}
function parseExpr(text){
  const toks=[],re=/\s*(?:(\d+)|(t\d+)|(floordiv|mod|\/\/|[%*+()-]))/y;let s=text.trim(),m;re.lastIndex=0;
  while(re.lastIndex<s.length){const p=re.lastIndex;m=re.exec(s);if(!m||re.lastIndex===p)throw new Error(`cannot parse "${text}" at "${s.slice(p)}"`);toks.push(m[1]||m[2]||m[3])}
  let i=0;const peek=()=>toks[i],take=()=>{if(i>=toks.length)throw new Error(`unexpected end of "${text}"`);return toks[i++]};
  function add(){let e=mul();while(peek()==="+"||peek()==="-"){const op=take(),r=mul();e=op==="+"?E.add(e,r):esub(e,r)}return e}
  function mul(){let e=un();while(["*","floordiv","//","mod","%"].includes(peek())){const op=take(),r=un();
    if(op==="*")e=emul(e,r);else{if(r.t!=="con"||r.k<=0)throw new Error(`${op} needs a positive constant in "${text}"`);e=(op==="floordiv"||op==="//")?E.fdiv(e,r.k):E.mod(e,r.k)}}return e}
  function un(){if(peek()==="-"){take();return E.mul(un(),-1)}return atom()}
  function atom(){const t=take();if(t==="("){const e=add();if(take()!==")")throw new Error(`missing ")" in "${text}"`);return e}
    if(/^\d+$/.test(t))return E.con(+t);if(/^t\d+$/.test(t))return E.var(+t.slice(1));throw new Error(`unexpected "${t}" in "${text}"`)}
  const e=add();if(i!==toks.length)throw new Error(`trailing input in "${text}"`);return e}
function evalE(e,t){switch(e.t){case"var":return t[e.k];case"con":return e.k;case"add":return evalE(e.a,t)+evalE(e.b,t);
  case"mul":return evalE(e.a,t)*e.k;case"fdiv":return Math.floor(evalE(e.a,t)/e.k);case"mod":{const v=evalE(e.a,t);return((v%e.k)+e.k)%e.k}}}
function varsE(e){return e.t==="var"?[e.k]:e.t==="con"?[]:e.t==="add"?varsE(e.a).concat(varsE(e.b)):varsE(e.a)}
function atomS(e){return e.t==="var"||e.t==="con"?strE(e):`(${strE(e)})`}
function strE(e){switch(e.t){case"var":return "t"+e.k;case"con":return String(e.k);
  case"add":{const b=e.b;if(b.t==="con"&&b.k<0)return `${strE(e.a)} - ${-b.k}`;if(b.t==="mul"&&b.k<0)return `${strE(e.a)} - ${strE(E.mul(b.a,-b.k))}`;return `${strE(e.a)} + ${strE(b)}`}
  case"mul":return e.k===1?strE(e.a):e.k===-1?`-${atomS(e.a)}`:`${atomS(e.a)} * ${e.k}`;
  case"fdiv":return `${atomS(e.a)} floordiv ${e.k}`;case"mod":return `${atomS(e.a)} mod ${e.k}`}}

// ---------- layout algebra (port of distlayout) ----------
function sizesOf(c,extent){const parts=c.parts;
  if(!(parts>=1))throw new Error(`axis ${c.axis}: needs at least 1 piece`);
  if(c.kind==="block"){if(extent%parts)throw new Error(`axis ${c.axis} of ${extent} is not divisible into ${parts} blocks (${extent} % ${parts} = ${extent%parts}); choose uneven`);return Array(parts).fill(extent/parts)}
  if(c.sizes){const s=c.sizes.reduce((a,b)=>a+b,0);if(c.sizes.some(v=>!(v>=1)||!Number.isInteger(v)))throw new Error(`axis ${c.axis}: sizes must be positive integers`);
    if(s!==extent)throw new Error(`axis ${c.axis}: sizes ${c.sizes.join(", ")} sum to ${s}, axis has ${extent}`);return c.sizes.slice()}
  if((c.policy||"balanced")==="balanced"){const q=Math.floor(extent/parts),r=extent%parts;return Array.from({length:parts},(_,k)=>k<r?q+1:q)}
  const b=Math.ceil(extent/parts);let left=extent;return Array.from({length:parts},()=>{const v=Math.min(b,left);left-=v;return v})}
function prod(lists){return lists.reduce((acc,l)=>acc.flatMap(a=>l.map(x=>a.concat([x]))),[[]])}
function build(spec){
  const shape=spec.shape, grid=spec.grid, rank=shape.length, G=grid.length;
  if(!rank||rank>2)throw new Error("tensor must be 1-D or 2-D");
  if(!G||G>3)throw new Error("grid must be 1-D to 3-D");
  if(shape.concat(grid).some(v=>!(v>=1)||!Number.isInteger(v)))throw new Error("shapes must be positive integers");
  const rules=spec.cuts.map(c=>{if(!(c.axis>=0&&c.axis<rank))throw new Error(`cut on axis ${c.axis}: tensor has rank ${rank}`);
    const sz=sizesOf(c,shape[c.axis]);if(sz.some(s=>s<1))throw new Error(`axis ${c.axis}: ${c.parts} pieces leave an empty piece`);
    const st=[0];sz.forEach(s=>st.push(st[st.length-1]+s));return {axis:c.axis,c,sizes:sz,starts:st}});
  const counts=rules.map(r=>r.sizes.length), ncomp=rules.length;
  const table=spec.table||null;
  const exprs=grid.map((n,g)=>{const s=table?"":(spec.map[g]||"").trim();if(!s)return null;
    let e;try{e=parseExpr(s)}catch(err){throw new Error(`g${g}: ${err.message}`)}
    const bad=varsE(e).filter(k=>k>=ncomp);if(bad.length)throw new Error(`g${g} = ${strE(e)} uses t${bad[0]}, but pieces have ${ncomp} coordinate${ncomp===1?"":"s"}`);return e});
  const replicate=table?(spec.rep||[]).slice():[...Array(G).keys()].filter(g=>!exprs[g]);
  const pids=prod(counts.map(n=>[...Array(n).keys()]));
  const npe=grid.reduce((a,b)=>a*b,1);
  if(table){if(ncomp!==1)throw new Error("an owner table needs a single cut (pieces = runs of ids)");
    if(table.length!==pids.length)throw new Error(`the owner table has ${table.length} entries for ${pids.length} pieces: it came from an owner array in Python (owner_array), so edit it there`);
    if(table.some(t=>!(t>=0&&t<npe)))throw new Error("owner table names a PE outside the grid")}
  pids.forEach(pid=>exprs.forEach((e,g)=>{if(!e)return;const v=evalE(e,pid);if(!(v>=0&&v<grid[g]))throw new Error(`piece (${pid}): g${g} = ${strE(e)} = ${v}, outside grid axis of ${grid[g]} PEs`)}));
  const region=pid=>{const lo=shape.map(()=>0),hi=shape.slice();rules.forEach((r,k)=>{lo[r.axis]=r.starts[pid[k]];hi[r.axis]=r.starts[pid[k]+1]});return{lo,hi}};
  const partOf=idx=>rules.map(r=>{let p=0;while(idx[r.axis]>=r.starts[p+1])p++;return p});
  const pidIx=pid=>pid.reduce((a,p,k)=>a*counts[k]+p,0);
  const peIx=c=>c.reduce((a,v,g)=>a*grid[g]+v,0);
  const pes=prod(grid.map(n=>[...Array(n).keys()]));
  const unl=ix=>{const c=[];for(let g=G-1;g>=0;g--){c.unshift(ix%grid[g]);ix=Math.floor(ix/grid[g])}return c};
  const primaryOf=pid=>table?unl(table[pidIx(pid)]):grid.map((n,g)=>exprs[g]?evalE(exprs[g],pid):0);
  const holdersOf=pid=>{const b=primaryOf(pid);return pes.filter(pe=>pe.every((v,g)=>replicate.includes(g)?true:v===b[g]))};
  const isPrimary=pe=>replicate.every(g=>pe[g]===0);
  const at={};pes.forEach(pe=>at[peIx(pe)]=[]);pids.forEach(pid=>holdersOf(pid).forEach(h=>at[peIx(h)].push(pid)));
  const piecesAt=pe=>at[peIx(pe)];
  const maxPieces=Math.max(0,...Object.values(at).map(v=>v.length));
  const uniform=rules.every(r=>new Set(r.sizes).size===1);
  // placement-shaped owner map? (each expr a distinct bare t_k, one piece per PE along it)
  let place=null;
  if(table){const free=[...Array(G).keys()].filter(g=>!replicate.includes(g));
    if(ncomp===1&&free.length===1&&grid[free[0]]===counts[0]&&table.every((t,k)=>unl(t)[free[0]]===k))place=[[0,free[0]]]}
  else{const bare=exprs.filter(Boolean);
    if(bare.every(e=>e.t==="var")&&bare.map(e=>e.k).sort().join()===[...Array(ncomp).keys()].join()&&exprs.every((e,g)=>!e||counts[e.k]===grid[g]))
      place=exprs.map((e,g)=>e?[e.k,g]:null).filter(Boolean).sort((a,b)=>a[0]-b[0])}
  const placement=!!place;
  // owner-PE colors: distinct primaries in PE order
  const prim=[...new Set(pids.map(pid=>peIx(primaryOf(pid))))].sort((a,b)=>a-b), primRank={};prim.forEach((p,i)=>primRank[p]=i);
  return {spec,shape,grid,rules,counts,exprs,table,replicate,pids,region,partOf,pidIx,peIx,pes,primaryOf,holdersOf,isPrimary,piecesAt,maxPieces,uniform,placement,place,prim,primRank};
}
function inter(a,b){const lo=a.lo.map((v,i)=>Math.max(v,b.lo[i])),hi=a.hi.map((v,i)=>Math.min(v,b.hi[i]));return hi.every((v,i)=>v>lo[i])?{lo,hi}:null}
function contains(a,b){return b.lo.every((v,i)=>v>=a.lo[i]&&b.hi[i]<=a.hi[i])}
function vol(r){return r.lo.reduce((a,v,i)=>a*(r.hi[i]-v),1)}
function rstr(r){return "["+r.lo.map((v,i)=>v+":"+r.hi[i]).join(", ")+")"}
function plan(L,pe,h){  // halo plan for a single-piece PE (own piece grown by h)
  const held=L.piecesAt(pe).map(L.region);if(held.length!==1)return null;
  const req={lo:held[0].lo.map(v=>Math.max(0,v-h)),hi:held[0].hi.map((v,i)=>Math.min(L.shape[i],v+h))};
  const pieces=[];L.pids.forEach(pid=>{const p=inter(L.region(pid),req);if(p)pieces.push({r:p,local:held.some(m=>contains(m,p)),src:L.peIx(L.primaryOf(pid))})});
  return {held:held[0],req,pieces};
}
function pname(pid){return pid.length===0?"P":pid.length===1?`P${pid[0]}`:`P(${pid.join(",")})`}
function mix(hex,t){const n=parseInt(hex.slice(1),16),c=[n>>16,(n>>8)&255,n&255].map(v=>Math.round(v+(255-v)*t));return "#"+c.map(v=>v.toString(16).padStart(2,"0")).join("")}
function peColor(L,ix){  // color of an owner PE
  const r=L.primRank[ix];if(r===undefined)return "#8896A8";
  if(L.prim.length<=COLORS.length)return COLORS[r];
  const c=[];let k=ix;for(let g=L.grid.length-1;g>=0;g--){c.unshift(k%L.grid[g]);k=Math.floor(k/L.grid[g])}
  const rest=c.slice(1),rc=L.grid.slice(1),ri=rest.reduce((a,v,g)=>a*rc[g]+v,0),rn=rc.reduce((a,b)=>a*b,1);
  return mix(COLORS[c[0]%COLORS.length],rn>1?ri/(rn-1)*0.5:0)}
function primaryIxOfPe(L,pe){return L.peIx(pe.map((v,g)=>L.replicate.includes(g)?0:v))}

// ---------- text ----------
function brief(a){return a.length>12?`${a.slice(0,8).join(", ")}, … (${a.length} sizes)`:a.join(", ")}
function pyTuple(a){return a.length===1?`(${a[0]},)`:`(${a.join(", ")})`}
function ruleRepr(r){const c=r.c;if(c.kind==="block")return `Block(axis=${r.axis}, parts=${r.sizes.length})`;
  if(c.sizes)return `UnevenBlock(axis=${r.axis}, sizes=${pyTuple(c.sizes)})`;
  return `UnevenBlock(axis=${r.axis}, parts=${r.sizes.length}, policy='${c.policy||"balanced"}')`}
function gridRepr(g){return g.length>1?`Grid(${pyTuple(g)})`:`Grid(${g[0]})`}
function mappingRepr(L){
  const rep=L.replicate.length?`, replicate=[${L.replicate.join(", ")}]`:"";
  if(L.placement){if(!L.rules.length&&L.replicate.length===L.grid.length)return `Replicate(${gridRepr(L.grid)})`;
    return `Map(${gridRepr(L.grid)}, {${L.place.map(([k,g])=>`${k}: ${g}`).join(", ")}}${rep})`}
  if(L.table)return `TableMap(${gridRepr(L.grid)}, [${L.table.join(", ")}]${rep})`;
  return `OwnerMap(${gridRepr(L.grid)}, {${L.exprs.map((e,g)=>e?`${g}: '${strE(e)}'`:null).filter(Boolean).join(", ")}}${rep})`}
function pyText(L){return `view = bind(${L.spec.src||"l2"}, Decomposition(${L.rules.map(ruleRepr).join(", ")}), ${mappingRepr(L)})`}
function attrText(L){
  const rr=L.rules.map(r=>r.c.kind==="block"?`block<axis = ${r.axis}, parts = ${r.sizes.length}>`:
    r.c.sizes?`uneven<axis = ${r.axis}, sizes = [${r.c.sizes.join(", ")}]>`:`uneven<axis = ${r.axis}, parts = ${r.sizes.length}, policy = ${r.c.policy||"balanced"}>`).join(", ");
  const head=`#dist.layout<shape = [${L.shape.join(", ")}], grid = [${L.grid.join(", ")}], decomp = [${rr}], `, rep=`replicate = [${L.replicate.map(g=>"g"+g).join(", ")}]>`;
  if(L.placement)return head+`place = [${L.place.map(([k,g])=>`${k} -> g${g}`).join(", ")}], `+rep;
  if(L.table)return head+`owner = table<[${L.table.join(", ")}]>, `+rep;
  return head+`owner = [${L.exprs.map((e,g)=>e?`g${g} = ${strE(e)}`:null).filter(Boolean).join(", ")}], `+rep}
function airText(L){
  if(L.spec.imports&&cur.air)return cur.air;          // irregular access: the ghost gather
  if(!L.placement||!L.uniform||(+L.spec.halo||0)>0)return cur.air||"# This layout is emitted with loops (owner maps) or one branch per size class\n# (uneven, halo), which the Python backend computes: air_backend.for_each_piece /\n# halo_load. python3 examples/gallery.py embeds the emitted code for each preset;\n# editing the layout here drops it.";
  const names=["tx","ty","tz"].slice(0,L.grid.length), src=L.spec.src||"l2", gOf=k=>(L.place.find(p=>p[0]===k)||[0,-1])[1];
  const sub=L.shape.map((n,a)=>{const k=L.rules.findIndex(r=>r.axis===a);if(k<0)return `0 : ${n}`;const b=L.rules[k].sizes[0],c=names[gOf(k)];return `${c}*${b} : ${c}*${b} + ${b}`}).join(", ");
  const herd=L.grid.map((n,g)=>{const p=L.place.find(p=>p[1]===g);if(!p)return `range(${n})`;const r=L.rules[p[0]];return `range(0, ${L.shape[r.axis]}, ${r.sizes[0]})`}).join(", ");
  const loc=L.shape.map((n,a)=>{const r=L.rules.find(r=>r.axis===a);return r?r.sizes[0]:n});
  const rep=L.replicate.filter(g=>L.grid[g]>1).map(g=>names[g]);
  return `with air.herd([${herd}]) as h:\n  @h.body\n  def _(${names.join(", ")}):\n    a = air.alloc([${loc.join(", ")}], f32, scope=h.private())\n    air.ops.load(a, ${src}[${sub}])`+
    (rep.length?`\n    # ${rep.join(", ")} not in the offset: every PE along it reads the same region (broadcast)`:"")+
    `\n\n# with distlayout:\n#   a = air_backend.alloc_local(view, f32, scope=h.private())\n#   air_backend.load_local(view, a, (${names.join(", ")}${names.length===1?",":""}))`}
function formula(L,idx){
  const pid=L.partOf(idx);
  const cut=L.rules.map((r,k)=>r.c.kind==="block"?`t${k} = i${r.axis} // ${r.sizes[0]} = ${pid[k]}`:`t${k} = piece of i${r.axis} in [${brief(r.sizes)}] = ${pid[k]}`);
  const map=L.table?[`table[${L.pidIx(pid)}] = PE ${L.table[L.pidIx(pid)]}`]:L.exprs.map((e,g)=>e?`g${g} = ${strE(e)} = ${evalE(e,pid)}`:null).filter(Boolean);
  const pe=L.primaryOf(pid), hs=L.holdersOf(pid);
  return `element (${idx.join(", ")}): ${cut.join("; ")||"one piece"} → ${pname(pid)} → ${map.join("; ")||"(replicated)"} → PE ${L.peIx(pe)}`+(hs.length>1?`, held by ${hs.length} PEs`:"");
}
function hint(L){return ""}

function fmtB(b){return b>=1024?(b/1024).toFixed(b%1024?1:0)+" KiB":b+" B"}

// ---------- drawing (SVG) ----------
function draw(L){
  const W=Math.min(1600,$("pic").clientWidth||1100), h=+L.spec.halo||0, bytes=+L.spec.bytes||4, budget=(+L.spec.kib||48)*1024;
  const rank=L.shape.length, ink="var(--ink)", muted="var(--muted)";
  const cap=(x,y,t)=>`<text x="${x}" y="${y}" font-size="12" font-weight="600" fill="${ink}">${t}</text>`;
  const sub=(x,y,t)=>`<text x="${x}" y="${y}" font-size="11.5" fill="${muted}">${t}</text>`;
  let s="";
  // the tensor's drawing grid (1-D tensors wrap into rows); the decomposition layout uses the same grid
  let rows,cols,at0;
  if(rank===1){const n=L.shape[0],b=L.rules.length?Math.max(...L.rules[0].sizes):n;const w=(b>=4&&b<=64)?b:b>64?64:Math.min(n,32);cols=Math.min(n,w);rows=Math.ceil(n/cols);at0=(r,c)=>r*cols+c<n?[r*cols+c]:null}
  else{rows=L.shape[0];cols=L.shape[1];at0=(r,c)=>[r,c]}
  const sr=Math.ceil(rows/48), sc=Math.ceil(cols/64), at=(r,c)=>at0(r*sr,c*sc);
  rows=Math.ceil(rows/sr);cols=Math.ceil(cols/sc);
  const pidAt=[], bc={};
  for(let r=0;r<rows;r++){pidAt.push([]);for(let c=0;c<cols;c++){const idx=at(r,c);if(!idx){pidAt[r].push(-1);continue}
    const pi=L.pidIx(L.partOf(idx));pidAt[r].push(pi);const q=bc[pi]=bc[pi]||[c,r,c+1,r+1];q[0]=Math.min(q[0],c);q[1]=Math.min(q[1],r);q[2]=Math.max(q[2],c+1);q[3]=Math.max(q[3],r+1)}}
  const cutPath=(x0,y0,sx,sy,gap)=>{let d="";for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const p0=pidAt[r][c];if(p0<0)continue;
    if(c+1<cols&&pidAt[r][c+1]>=0&&pidAt[r][c+1]!==p0){const x=x0+(c+1)*sx-gap/2;d+=`M${x},${y0+r*sy-gap/2}v${sy}`}
    if(r+1<rows&&pidAt[r+1][c]>=0&&pidAt[r+1][c]!==p0){const y=y0+(r+1)*sy-gap/2;d+=`M${x0+c*sx-gap/2},${y}h${sx}`}}return d};
  // ① decomposition layout: a scaled outline of the tensor with the cut lines, so it can be laid on the tensor
  const ty0=30, cc=Math.max(2,Math.min(18,Math.floor(320/cols),Math.floor(220/rows)));
  const ch=rank===1?Math.max(cc,Math.min(28,Math.floor(160/rows))):cc, tw0=cols*cc, th0=rows*ch;   // 1-D rows are not a real aspect: give them room
  s+=cap(0,14,"① Decomposition layout")+`<rect x="0.5" y="${ty0+.5}" width="${tw0}" height="${th0}" fill="var(--bg)" stroke="${ink}" stroke-width="1.2" rx="3"/>`;
  s+=`<path d="${cutPath(.5,ty0+.5,cc,ch,0)}" fill="none" stroke="${ink}" stroke-width="${cc>=6?1.4:.8}" stroke-dasharray="${cc>=6?"5 3":"2 2"}"/>`;
  Object.entries(bc).forEach(([pi,[c0,r0,c1,r1]])=>{const lab=pname(L.pids[+pi]);
    const fs=(r1-r0)*ch>=16?11:9.5;if((c1-c0)*cc>=lab.length*fs*0.62+4&&(r1-r0)*ch>=12)s+=`<text x="${(c0+c1)/2*cc}" y="${ty0+(r0+r1)/2*ch+fs*0.36}" font-size="${fs}" fill="${muted}" text-anchor="middle">${lab}</text>`});
  const rulesTxt=L.rules.length?L.rules.map((r,k)=>r.c.sizes&&r.c.sizes.length>12?`t${k}: UnevenBlock(axis=${r.axis}, ${r.c.sizes.length} runs)`:`t${k}: ${ruleRepr(r)}`):["one piece"];
  rulesTxt.forEach((t,k)=>s+=sub(0,ty0+th0+18+k*15,t));
  const tBottom=ty0+th0+18+(rulesTxt.length-1)*15;
  // ② the tensor: same cuts, colored by the owner PE of each piece; kept well apart from ①
  const TW=Math.max(tw0,200), X0=TW+130, AW=W-X0;
  const cs=Math.max(4,Math.min(30,Math.floor(AW/cols)-2)), gap=cs>8?2:1, step=cs+gap, tw=cols*step, th=rows*step;
  const y0=ty0, box={};
  s+=`<path d="M${tw0+28},${ty0+Math.min(th0,th)/2} H${X0-28}" stroke="${muted}" stroke-width="1.5" marker-end="url(#ahm)"/>`;
  s+=cap(X0,14,`② Tensor ${L.shape.join(" × ")}`)+(sr>1||sc>1?sub(X0+150,14,`sampled 1/${rank===1?sr*sc:[sr,sc].join(" × 1/")}`):"");
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const idx=at(r,c);if(!idx)continue;
    const pid=L.partOf(idx),pi=pidAt[r][c],own=L.peIx(L.primaryOf(pid));
    const x=X0+c*step,y=y0+r*step;const bb=box[pi]=box[pi]||[x,y,x,y];bb[0]=Math.min(bb[0],x);bb[1]=Math.min(bb[1],y);bb[2]=Math.max(bb[2],x+cs);bb[3]=Math.max(bb[3],y+cs);
    s+=`<rect class="el" data-i="${idx}" data-o="${own}" x="${x}" y="${y}" width="${cs}" height="${cs}" rx="${cs>10?3:1}" fill="${peColor(L,own)}"/>`}
  if(cs>=8)s+=`<path d="${cutPath(X0,y0,step,step,gap)}" fill="none" stroke="${ink}" stroke-width="1.4" stroke-dasharray="5 3" pointer-events="none"/>`;
  Object.entries(box).forEach(([pi,[x1,y1,x2,y2]])=>{const lab=pname(L.pids[+pi]),lw=lab.length*7+10;if(x2-x1>=lw+4&&y2-y1>=16){
    const cx=(x1+x2)/2,cy=(y1+y2)/2;
    s+=`<g pointer-events="none"><rect x="${cx-lw/2}" y="${cy-9}" width="${lw}" height="18" rx="9" fill="rgba(23,37,60,.78)"/><text x="${cx}" y="${cy+4}" font-size="11.5" font-weight="700" fill="#fff" text-anchor="middle">${lab}</text></g>`}});
  // ③ PE grid
  const G=L.grid, gr=G.length>1?G.slice(0,-1).reduce((a,b)=>a*b,1):1, gc=G[G.length-1], halo=h&&L.maxPieces<=1;
  const bw=Math.max(104,Math.min(160,Math.floor((W-8)/gc)-12)), bh=58, bg=halo?34:12, py=Math.max(y0+th,tBottom)+48;
  const rr=L.replicate.filter(g=>L.grid[g]>1);
  s+=cap(0,py-14,`③ Mapping layout · grid ${G.join(" × ")}`)+sub(220,py-14,[halo?`halo ${h}`:"",rr.length?`replicated over ${rr.map(g=>"g"+g).join(", ")}`:""].filter(Boolean).join(" · "));
  const pos={};let maxB=0,over=0;
  L.pes.forEach(pe=>{const i=L.peIx(pe),row=G.length>1?Math.floor(i/gc):0,col=i%gc,x=col*(bw+bg),y=py+row*(bh+bg);pos[i]=[x+bw/2,y+bh/2];
    const gh=L.spec.ghosts?L.spec.ghosts[i]:0;
    const ps=L.piecesAt(pe),pl=halo?plan(L,pe,h):null,b=((pl?vol(pl.req):ps.reduce((a,p)=>a+vol(L.region(p)),0))+gh)*bytes;maxB=Math.max(maxB,b);if(b>budget)over++;
    const frac=Math.min(1,b/budget),own=primaryIxOfPe(L,pe),col2=ps.length?peColor(L,own):"#B9C4D2";
    const label=ps.length===0?"empty":ps.length===1?pname(ps[0]):ps.length<=3?ps.map(pname).join(" "):`${ps.length} pieces`;
    const owned=ps.reduce((a,p)=>a+vol(L.region(p)),0);
    const regs=gh||ps.length>3?`${owned} owned${gh?` + ${gh} ghosts`:""}`:ps.length===0?"—":ps.length===1?rstr(L.region(ps[0])):ps.slice(0,2).map(p=>rstr(L.region(p))).join(" ")+(ps.length>2?" …":"");
    s+=`<g class="pe" data-o="${own}" data-pe="${i}"><rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="7" fill="${col2}" ${L.isPrimary(pe)?"":`fill-opacity=".55"`}/>`+
      `<text x="${x+8}" y="${y+16}" font-size="12" font-weight="700" fill="#fff">PE ${i} ← ${label}${L.isPrimary(pe)?"":" · replica"}</text>`+
      `<text x="${x+8}" y="${y+31}" font-size="10.5" fill="#fff">${regs}</text>`+
      `<rect x="${x+8}" y="${y+40}" width="${bw-52}" height="8" rx="3" fill="rgba(255,255,255,.35)"/>`+
      `<rect x="${x+8}" y="${y+40}" width="${(bw-52)*frac}" height="8" rx="3" fill="${b>budget?"#17253C":"#fff"}"/>`+
      `<text x="${x+bw-8}" y="${y+48}" font-size="10" font-weight="${b>budget?700:400}" fill="#fff" text-anchor="end">${b>budget?"✕ ":""}${fmtB(b)}</text>`+
      `<title>PE ${i} (${pe}) holds ${ps.map(p=>pname(p)+" "+rstr(L.region(p))).join(", ")||"nothing"}${pl?`, needs ${rstr(pl.req)}`:""}: ${fmtB(b)} of ${fmtB(budget)}</title></g>`});
  let arrows="";const pair={};
  if(halo)L.pes.forEach(pe=>{const dst=L.peIx(pe),pl=plan(L,pe,h);if(!pl)return;pl.pieces.filter(p=>!p.local).forEach(p=>{const a=Math.min(p.src,dst),b=Math.max(p.src,dst),k=a+"-"+b;
    pair[k]=pair[k]||{a,b,n:0,dirs:new Set()};pair[k].n+=vol(p.r);pair[k].dirs.add(p.src===a?"ab":"ba")})});
  if(!halo&&L.spec.imports)L.spec.imports.forEach(([dst,src,n])=>{const a=Math.min(src,dst),b=Math.max(src,dst),k=a+"-"+b;
    pair[k]=pair[k]||{a,b,n:0,dirs:new Set()};pair[k].n+=n;pair[k].dirs.add(src===a?"ab":"ba")});
  const edge=(x,y,dx,dy)=>{const t=Math.min(Math.abs(dx)>1e-9?(bw/2+4)/Math.abs(dx):1e9,Math.abs(dy)>1e-9?(bh/2+4)/Math.abs(dy):1e9);return [x+dx*t,y+dy*t]};
  const arcs=!halo&&!!L.spec.imports&&G.length===1;let arcDepth=0;
  if(arcs)Object.values(pair).forEach(({a,b,n,dirs})=>{const [x1,y1]=pos[a],[x2]=pos[b],yb=y1+bh/2+3,d=14+12*Math.abs(b-a);arcDepth=Math.max(arcDepth,d);
    arrows+=`<path d="M${x1},${yb} Q${(x1+x2)/2},${yb+2*d} ${x2},${yb}" fill="none" stroke="#D95455" stroke-width="${Math.min(3,1+n/12)}" ${dirs.has("ab")?'marker-end="url(#ah)"':""} ${dirs.has("ba")?'marker-start="url(#ah2)"':""}><title>PE ${a} ⇄ PE ${b}: ${n} ghost elements</title></path>`});
  else Object.values(pair).forEach(({a,b,n,dirs})=>{const [x1,y1]=pos[a],[x2,y2]=pos[b],dx=x2-x1,dy=y2-y1;
    const [sx,sy]=edge(x1,y1,dx,dy),[ex,ey]=edge(x2,y2,-dx,-dy),diag=Math.abs(dx)>1&&Math.abs(dy)>1;
    arrows+=`<line x1="${sx}" y1="${sy}" x2="${ex}" y2="${ey}" stroke="#D95455" stroke-width="${diag?1.2:2.2}" ${diag?'stroke-dasharray="4 3"':""} ${dirs.has("ab")?'marker-end="url(#ah)"':""} ${dirs.has("ba")?'marker-start="url(#ah2)"':""}><title>PE ${a} ⇄ PE ${b}: ${n} ${halo?"halo":"ghost"} elements</title></line>`});
  let H=py+gr*(bh+bg)+4+arcDepth, PW=Math.max(X0+tw,X0+330,gc*(bw+bg),TW);   // room for the captions
  // ④ graph view: mesh nodes at their positions, colored by owner; cut edges dashed red
  const gv=L.spec.graph;
  if(gv&&rank===1){const P=gv.pos,mx=Math.max(...P.map(p=>p[0])),my=Math.max(...P.map(p=>p[1])),sp=Math.max(14,Math.min(34,Math.floor((W-20)/(my+1)))),gy=H+46;
    const own=P.map((_,id)=>L.peIx(L.primaryOf(L.partOf([id])))),X=id=>10+P[id][1]*sp,Y=id=>gy+P[id][0]*sp;
    let cutN=0,es="";gv.edges.forEach(([u,v])=>{const cut=own[u]!==own[v];if(cut)cutN++;
      es+=`<line x1="${X(u)}" y1="${Y(u)}" x2="${X(v)}" y2="${Y(v)}" stroke="${cut?"#D95455":"var(--line)"}" stroke-width="${cut?1.6:1.2}" ${cut?'stroke-dasharray="4 3"':""}/>`});
    s+=cap(0,gy-24,"④ Graph")+sub(80,gy-24,`${cutN} cut edges`)+es;
    P.forEach((_,id)=>{s+=`<circle class="el" data-i="${id}" data-o="${own[id]}" cx="${X(id)}" cy="${Y(id)}" r="${Math.max(4,sp*0.28)}" fill="${peColor(L,own[id])}" stroke="var(--bg)" stroke-width="1.2"><title>id ${id} → PE ${own[id]}</title></circle>`});
    H=gy+mx*sp+20;PW=Math.max(PW,20+my*sp)}
  const mk=(id,d,rx)=>`<marker id="${id}" viewBox="0 0 10 10" refX="${rx}" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="${d}" fill="${id==="ahm"?"#52657B":"#D95455"}"/></marker>`;
  $("pic").innerHTML=`<div style="overflow-x:auto"><svg xmlns="http://www.w3.org/2000/svg" width="${PW}" height="${H}" viewBox="0 0 ${PW} ${H}"><defs>${mk("ah","M0,0L10,5L0,10z",8)}${mk("ah2","M10,0L0,5L10,10z",2)}${mk("ahm","M0,0L10,5L0,10z",8)}</defs>${s}${arrows}</svg></div>`;
  // legend & stats
  const byOwner={};L.pids.forEach(pid=>{const o=L.peIx(L.primaryOf(pid));(byOwner[o]=byOwner[o]||[]).push(pid)});
  $("legend").innerHTML=L.prim.slice(0,12).map(o=>{const ps=byOwner[o],hs=L.holdersOf(ps[0]).length;
    return `<span><i class="sw" style="background:${peColor(L,o)}"></i>PE ${o} ← ${ps.length<=3?ps.map(pname).join(", "):ps.length+" pieces"}${hs>1?` +${hs-1} replica${hs>2?"s":""}`:""}</span>`}).join("")+(L.prim.length>12?`<span>… ${L.prim.length-12} more</span>`:"")+(halo?`<span><i class="sw" style="background:#D95455"></i>halo import</span>`:"")+(L.spec.imports?`<span><i class="sw" style="background:#D95455"></i>ghost import (${L.spec.imports.reduce((a,x)=>a+x[2],0)} ids)</span>`:"");
  L.maxB=maxB;
  const fit=over?`<b style="color:var(--comm)">${over} PE${over>1?"s":""} over budget</b>`:`<b style="color:var(--ok)">fits</b>`;
  $("stat").innerHTML=`max ${fmtB(maxB)} / PE of ${fmtB(budget)} · ${fit}`+(pinned?` · vs A (${pinned.name}): ${fmtB(pinned.maxB)} / PE → ${maxB<=pinned.maxB?(pinned.maxB/maxB).toFixed(2)+"× less":(maxB/pinned.maxB).toFixed(2)+"× more"}`:"");
  const svgEl=$("pic").querySelector("svg");
  svgEl.querySelectorAll(".el").forEach(e=>{e.onmouseenter=()=>{hl(+e.dataset.o);$("formula").textContent=formula(L,e.dataset.i.split(",").map(Number))};e.onmouseleave=unhl});
  svgEl.querySelectorAll(".pe").forEach(e=>{e.onmouseenter=()=>hl(+e.dataset.o);e.onmouseleave=unhl});
  function hl(o){svgEl.querySelectorAll(".el,.pe").forEach(e=>e.style.opacity=+e.dataset.o===o?1:.2)}
  function unhl(){svgEl.querySelectorAll(".el,.pe").forEach(e=>e.style.opacity="");$("formula").textContent=hint(L)}
  $("formula").textContent=hint(L);
}

// ---------- UI ----------
function parseList(v){return v.split(/[x×,\s]+/).filter(Boolean).map(Number)}
function syncForm(){
  const box=$("cuts");box.innerHTML="";
  cur.shape.forEach((n,a)=>{const c=cur.cuts.find(c=>c.axis===a),kind=c?c.kind:"whole";
    box.insertAdjacentHTML("beforeend",`<div class="row3"><div><label>Tensor axis ${a} (${n})</label><select data-a="${a}">${["whole","block","uneven"].map(k=>`<option ${k===kind?"selected":""}>${k}</option>`).join("")}</select></div>`+
      `<div><label>pieces</label><input type="number" min="1" data-p="${a}" value="${c?c.parts:1}" ${c?"":"disabled"}></div></div>`+
      (kind==="uneven"?`<div class="row3"><div><label>sizes</label><input class="mono" data-s="${a}" placeholder="e.g. 3, 7" value="${c&&c.sizes?c.sizes.join(", "):""}"></div>`+
        `<div><label>policy</label><select data-pol="${a}" ${c&&c.sizes?"disabled":""}>${["balanced","tail"].map(k=>`<option ${k===(c.policy||"balanced")?"selected":""}>${k}</option>`).join("")}</select></div></div>`:""))});
  box.querySelectorAll("select").forEach(sel=>sel.onchange=()=>{const a=+sel.dataset.a,k=sel.value;cur.cuts=cur.cuts.filter(c=>c.axis!==a);
    if(k!=="whole"){const ncut=cur.cuts.length;cur.cuts.push({axis:a,kind:k,parts:cur.grid[Math.min(ncut,cur.grid.length-1)]||2,policy:"balanced"});cur.cuts.sort((x,y)=>x.axis-y.axis);
      const k2=cur.cuts.findIndex(c=>c.axis===a),g=Math.min(k2,cur.grid.length-1);if(!(cur.map[g]||"").trim())cur.map[g]="t"+k2}
    delete cur.mlir;delete cur.air;delete cur.imports;delete cur.ghosts;syncForm();render()});
  box.querySelectorAll("input[data-p]").forEach(inp=>inp.oninput=()=>{const c=cur.cuts.find(c=>c.axis===+inp.dataset.p);if(c){c.parts=+inp.value;delete c.sizes;
    const si=box.querySelector(`input[data-s="${c.axis}"]`);if(si)si.value="";const ps=box.querySelector(`select[data-pol="${c.axis}"]`);if(ps)ps.disabled=false;delete cur.mlir;delete cur.air;render()}});
  box.querySelectorAll("input[data-s]").forEach(inp=>inp.oninput=()=>{const c=cur.cuts.find(c=>c.axis===+inp.dataset.s);if(!c)return;
    const v=inp.value.split(/[,\s]+/).filter(Boolean).map(Number);const ps=box.querySelector(`select[data-pol="${c.axis}"]`);
    if(v.length){c.sizes=v;c.parts=v.length;const pi=box.querySelector(`input[data-p="${c.axis}"]`);if(pi)pi.value=v.length;if(ps)ps.disabled=true}else{delete c.sizes;if(ps)ps.disabled=false}
    delete cur.mlir;delete cur.air;render()});
  box.querySelectorAll("select[data-pol]").forEach(sel=>sel.onchange=()=>{const c=cur.cuts.find(c=>c.axis===+sel.dataset.pol);if(c){c.policy=sel.value;delete cur.mlir;delete cur.air;render()}});
  const mb=$("maps");mb.innerHTML="";
  if(cur.table){mb.innerHTML=`<label>Owner table: ${cur.table.length} runs</label>`;return}
  cur.grid.forEach((n,g)=>mb.insertAdjacentHTML("beforeend",`<label>Grid axis g${g} (${n} PEs)</label><input class="mono" data-g="${g}" value="${(cur.map[g]||"").replace(/"/g,"&quot;")}" placeholder="replicated">`));
  mb.querySelectorAll("input").forEach(inp=>inp.oninput=()=>{cur.map[+inp.dataset.g]=inp.value;delete cur.mlir;delete cur.air;render()});
}
function load(spec){cur=JSON.parse(JSON.stringify(spec));$("shape").value=cur.shape.join(" × ");$("grid").value=cur.grid.join(" × ");
  $("halo").value=cur.halo||0;$("bytes").value=cur.bytes||4;$("kib").value=cur.kib||48;syncForm();render()}
function render(){
  try{const L=build(cur);$("err").textContent="";draw(L);showCode(L);window.__last=L}
  catch(e){$("err").textContent=e.message}}
function showCode(L){
  const t={python:["Python",pyText(L)],attr:["#dist.layout",attrText(L)],air:["air.api",airText(L)],mlir:["MLIR",cur.mlir||"# MLIR is embedded when the page is rendered with MLIR-AIR installed\n# (python3 examples/gallery.py). Editing the layout drops it."]};
  $("tabs").innerHTML=Object.entries(t).map(([k,[n]])=>`<button data-t="${k}" class="${k===tab?"on":""}">${n}</button>`).join("");
  $("tabs").querySelectorAll("button").forEach(b=>b.onclick=()=>{tab=b.dataset.t;showCode(L)});$("code").textContent=t[tab][1]}
$("shape").onchange=()=>{cur.shape=parseList($("shape").value);cur.cuts=cur.cuts.filter(c=>c.axis<cur.shape.length).map(c=>{const d=Object.assign({},c);delete d.sizes;return d});delete cur.mlir;delete cur.air;syncForm();render()};
$("grid").onchange=()=>{cur.grid=parseList($("grid").value);cur.map=cur.grid.map((_,g)=>cur.map[g]||"");delete cur.mlir;delete cur.air;syncForm();render()};
["halo","bytes","kib"].forEach(id=>$(id).oninput=()=>{cur[id]=+$(id).value;if(id==="halo"){delete cur.air;delete cur.mlir}render()});
$("pin").onclick=()=>{if(window.__last){pinned={name:`${cur.title||"layout"} on ${cur.grid.join("×")}`,maxB:window.__last.maxB};render()}};
$("svg").onclick=()=>{const s=$("pic").querySelector("svg");if(!s)return;const css=getComputedStyle(document.documentElement);
  const v=(n,d)=>css.getPropertyValue(n).trim()||d;
  const txt=new XMLSerializer().serializeToString(s).replace(/var\(--muted\)/g,v("--muted","#52657B")).replace(/var\(--ink\)/g,v("--ink","#17253C")).replace(/var\(--bg\)/g,v("--bg","#FFFFFF"));
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([txt],{type:"image/svg+xml"}));a.download=(cur.title||"layout").replace(/\W+/g,"_")+".svg";a.click()};
const P=$("presets");PRESETS.forEach((p,i)=>{const b=document.createElement("button");b.textContent=p.title||`Layout ${i+1}`;
  b.onclick=()=>{P.querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on");load(p)};P.appendChild(b)});
if(PRESETS.length<2)P.style.display="none";
P.firstChild&&P.firstChild.classList.add("on");load(PRESETS[0]);
window.addEventListener("resize",()=>render());
window.distlayout={build,pyText,attrText,airText,plan,parseExpr,strE};
</script></body></html>"""
