"""Pictures of a layout: one self-contained HTML page per :class:`DistributedView`.

    from distlayout.viz import render_html
    open("block.html", "w").write(render_html(view, title="Block 100 over 4"))

The page has five linked panels:

1. **Input**: the logical tensor, each element colored by its primary owner.
2. **Mapping**: the PE grid; each PE shows the region it holds and whether it
   is the primary owner or a replica.
3. **Local views**: each PE's local buffer. With ``requests={pe: Box}`` the
   buffer is the *requested* region (owned block plus halo), cells colored by
   where they come from and ghost cells marked ``g``.
4. **Layout**: the Python spec that builds the view and its ``#dist.layout`` text.
5. **Generated code**: the ``air.api`` load with the affine offset the backend
   derives, a per-PE offset table, and optionally MLIR text (``mlir=``).

Hovering an element highlights every PE holding it; hovering a PE highlights
its region. No external assets: fonts fall back to the system's, scripts and
styles are inline, so the file opens anywhere and can be committed or mailed.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .decomposition import Block, UnevenBlock
from .region import Box, as_index
from .view import DistributedView

# Deck palette: PE 0..3 first, then extensions in the same family.
PE_COLORS = ["#3978CF", "#E59B38", "#249A82", "#A36BC6", "#D95455", "#52657B",
             "#2FA7C9", "#C58B1E", "#5E9E4A", "#7A5BC7", "#C2577F", "#3E8E8A",
             "#8C6A4A", "#4867B8", "#B8743A", "#6B7F99"]
MAX_CELLS = 4096


# -- symbolic coordinates for code text --------------------------------------

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


# -- text artifacts ------------------------------------------------------------

def spec_code(view: DistributedView, src: str = "buf") -> str:
    """Python that rebuilds ``view``'s layout over a handle named ``src``."""
    rules = ", ".join(repr(r) for r in view.decomposition.rules)
    m = view.mapping
    grid = f"Grid({tuple(m.grid.shape) if m.grid.rank > 1 else m.grid.shape[0]})"
    if not m.place and len(m.replicate) == m.grid.rank:
        mapping = f"Replicate({grid})"
    else:
        rep = f", replicate={list(m.replicate)}" if m.replicate else ""
        mapping = f"Map({grid}, {m.place_dict}{rep})"
    head = f"view = bind(\n    {src},\n    Decomposition({rules}),\n    {mapping},\n"
    if not view.source.is_global:
        return (f"source = DataSource({src}, global_shape={list(view.shape)}, "
                f"origin={list(view.source.origin)})\n" + head.replace(f"    {src},", "    source,", 1) + ")")
    return head + ")"


def air_code(view: DistributedView, src: str = "l2", dtype: str = "f32") -> Tuple[str, List[Tuple[Tuple[int, ...], Tuple[slice, ...]]]]:
    """(``air.api`` snippet, per-PE concrete slices). Uniform layouts only."""
    from .backends import air as backend

    names = ["tx", "ty", "tz"][: view.grid.rank] if view.grid.rank <= 3 else [f"t{i}" for i in range(view.grid.rank)]
    if not view.is_uniform():
        sizes = sorted({view.local_shape(pe) for pe in view.grid.coords()})
        return (f"# uneven layout: local shapes {sizes} differ per PE.\n"
                f"# air.api needs one static transfer size, so the AIR backend refuses this\n"
                f"# layout today. view.plan(pe) below gives each PE's exact region.", [])
    key = backend.slice_key(view, [_Sym.of(n) for n in names])
    sub = ", ".join(f"{k.start} : {k.stop}" for k in key)
    shape = list(view.max_local_shape())
    coords = ", ".join(names)
    rep = [names[g] for g in view.mapping.replicate]
    note = (f"\n    # {', '.join(rep)} not in the offset: every PE along it reads the same region (broadcast)"
            if rep else "")
    code = (f"with air.herd({_herd_text(view)}) as h:\n"
            f"  @h.body\n"
            f"  def _({coords}):\n"
            f"    a = air.alloc({shape}, {dtype}, scope=h.private())\n"
            f"    air.ops.load(a, {src}[{sub}]){note}\n"
            f"\n# from distlayout, the same thing:\n"
            f"#   a = air_backend.alloc_local(view, {dtype}, scope=h.private())\n"
            f"#   air_backend.load_local(view, a, ({coords}{',' if len(names) == 1 else ''}))")
    table = [(pe, backend.slice_key(view, pe)) for pe in view.grid.coords()]
    return code, table


def _herd_text(view: DistributedView) -> str:
    from .backends import air as backend
    rs = backend.herd_space(view)
    parts = [f"range({r.start}, {r.stop}, {r.step})" if r.step != 1 else f"range({r.stop})" for r in rs]
    return "[" + ", ".join(parts) + "]"


# -- model for the page --------------------------------------------------------

def _pe_index(view: DistributedView, pe: Tuple[int, ...]) -> int:
    idx = 0
    for c, s in zip(pe, view.grid.shape):
        idx = idx * s + c
    return idx


def _model(view: DistributedView, requests: Optional[Dict[Any, Box]]) -> Dict[str, Any]:
    shape = view.shape
    if len(shape) > 2:
        raise NotImplementedError("viz draws rank-1 and rank-2 tensors")
    pids = list(view.partitions())
    pid_ix = {p: i for i, p in enumerate(pids)}
    parts = []
    for p in pids:
        primary, holders = view.mapping.pes_of(p)
        parts.append({
            "region": str(view.partition_region(p)),
            "primary": _pe_index(view, primary),
            "holders": [_pe_index(view, h) for h in holders],
            "color": PE_COLORS[len(parts) % len(PE_COLORS)],
        })
    # tensor cells (rank-1 drawn as wrapped rows)
    if len(shape) == 1:
        n = shape[0]
        sizes = {view.local_shape(pe)[0] for pe in view.grid.coords()}
        b = max(sizes)
        width = b if b <= 32 and len(sizes) == 1 else min(n, 25 if n <= 400 else 32)
        rows, cols = -(-n // width), width
        coord = lambda r, c: (r * width + c,) if r * width + c < n else None
    else:
        rows, cols = shape
        coord = lambda r, c: (r, c)
    stride = 1
    while (rows // stride + 1) * (cols // stride + 1) > MAX_CELLS:
        stride += 1
    cells = []
    for r in range(0, rows, stride):
        row = []
        for c in range(0, cols, stride):
            idx = coord(r, c)
            row.append(None if idx is None else [list(idx), pid_ix[view.partition_of(idx)]])
        cells.append(row)
    pes = []
    for pe in view.grid.coords():
        reg = view.held_region(pe)
        pes.append({
            "coord": list(pe), "ix": _pe_index(view, pe), "part": pid_ix[view.partition_at(pe)],
            "primary": view.is_primary(pe), "region": str(reg), "shape": list(reg.size),
        })
    # local views: requested regions split by source
    local = []
    reqs = {as_index(k): v for k, v in (requests or {}).items()}
    for pe in view.grid.coords():
        req = reqs.get(pe)
        plan = view.plan(pe, req)
        pieces = []
        for t in plan.transfers:
            src = _pe_index(view, pe) if t.local else _pe_index(view, t.src_pe)
            part = pid_ix[view.partition_of(t.region.offset)]
            pieces.append({"region": str(t.region), "dst": list(t.dst_offset), "size": list(t.region.size),
                           "local": t.local, "src": src, "part": part})
        local.append({"pe": _pe_index(view, pe), "coord": list(pe), "request": str(plan.request),
                      "size": list(plan.request.size), "pieces": pieces, "halo": req is not None})
    return {"shape": list(shape), "grid": list(view.grid.shape), "stride": stride,
            "wrapped": len(shape) == 1, "cells": cells, "parts": parts, "pes": pes, "local": local,
            "colors": PE_COLORS}


# -- HTML ------------------------------------------------------------------------

def render_html(view: DistributedView, title: str = "Data layout", subtitle: str = "",
                src: str = "l2", requests: Optional[Dict[Any, Box]] = None,
                mlir: Optional[str] = None, dtype: str = "f32") -> str:
    """A standalone HTML page picturing ``view`` (see module docstring)."""
    model = _model(view, requests)
    code, table = air_code(view, src, dtype)
    rows = "".join(
        f"<tr><td>{html.escape(str(pe))}</td><td>{'primary' if view.is_primary(pe) else 'replica'}</td>"
        f"<td><code>{src}[{', '.join(f'{k.start}:{k.stop}' for k in key)}]</code></td></tr>"
        for pe, key in table[:64]
    )
    more = f"<p class='note'>{len(table) - 64} more PEs not shown.</p>" if len(table) > 64 else ""
    offsets = (f"<table class='tbl'><thead><tr><th>PE</th><th>role</th><th>region read</th></tr></thead>"
               f"<tbody>{rows}</tbody></table>{more}") if table else ""
    mlir_block = ("<div class='card'><h2><span class='num'>06</span>MLIR emitted by air.api</h2>"
                  "<p class='cap'>The real module, traced by MLIR-AIR from the code above. The herd's logical "
                  "tiles are strip-mined onto its cores, so the offset map can carry an extra loop term.</p>"
                  f"<pre class='code'>{html.escape(mlir)}</pre></div>" if mlir else "")
    sub = subtitle or f"tensor {tuple(view.shape)} on a {' × '.join(map(str, view.grid.shape))} PE grid"
    return (_PAGE
            .replace("%TITLE%", html.escape(title))
            .replace("%SUBTITLE%", html.escape(sub))
            .replace("%SPEC%", html.escape(spec_code(view, src)))
            .replace("%ATTR%", html.escape(view.to_attr()))
            .replace("%CODE%", html.escape(code))
            .replace("%OFFSETS%", offsets)
            .replace("%MLIR%", mlir_block)
            .replace("%MODEL%", json.dumps(model).replace("</", "<\\/")))


def render_index(pages: Sequence[Tuple[str, str, str]], title: str = "distlayout gallery") -> str:
    """An index page: ``pages`` is ``[(href, name, description), ...]``."""
    items = "".join(
        f"<a class='tile' href='{html.escape(h)}'><b>{html.escape(n)}</b><span>{html.escape(d)}</span></a>"
        for h, n, d in pages)
    return (_PAGE_INDEX.replace("%TITLE%", html.escape(title)).replace("%ITEMS%", items))


_CSS = """
:root{--ink:#17253C;--muted:#52657B;--faint:#8896A8;--card:#F6F9FD;--line:#DDE7F2;--bg:#FFFFFF;
--ghost:#A8C9F1;--comm:#D95455;--codebg:#F4F7FB;--chip:#E6F0FC}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ink:#E8EEF6;--muted:#A9B7C8;--faint:#7D8FA3;
--card:#18222F;--line:#2A3747;--bg:#0F1722;--ghost:#3C5A80;--codebg:#141D29;--chip:#1E2D40}}
:root[data-theme="dark"]{--ink:#E8EEF6;--muted:#A9B7C8;--faint:#7D8FA3;--card:#18222F;--line:#2A3747;--bg:#0F1722;
--ghost:#3C5A80;--codebg:#141D29;--chip:#1E2D40}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:Aptos,"Segoe UI",Inter,system-ui,-apple-system,sans-serif;line-height:1.45}
.wrap{max-width:1180px;margin:0 auto;padding:28px 16px 40px}
h1{font-size:28px;margin:0 0 4px;font-weight:700;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 18px;font-size:14px;padding-bottom:14px;border-bottom:1px solid var(--line)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:820px){.grid2{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px;margin-bottom:16px;min-width:0}
.card h2{font-size:17px;margin:0 0 2px;display:flex;gap:10px;align-items:center}
.num{font-size:11px;color:var(--faint);font-weight:600;border:1px solid var(--line);border-radius:6px;padding:1px 6px}
.cap{color:var(--muted);font-size:13px;margin:0 0 12px}
.scroll{overflow-x:auto}
.cells{display:inline-grid;gap:2px}
.cell{width:var(--cs);height:var(--cs);border-radius:3px;color:#fff;font-size:10px;display:flex;align-items:center;
justify-content:center;font-weight:600;cursor:default;transition:opacity .12s,transform .12s}
.cell.empty{background:transparent}
.dim #tensor .cell:not(.hl),.dim #pes .pe:not(.hl){opacity:.22}
#tensor .cell.hl{transform:scale(1.08)}
.pes{display:inline-grid;gap:8px}
.pe{border-radius:8px;padding:8px 10px;color:#fff;min-width:118px;font-size:12px;cursor:default;transition:opacity .12s}
.pe b{font-size:13px}.pe .r{opacity:.92;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px}
.pe.replica{background-image:repeating-linear-gradient(135deg,rgba(255,255,255,.22) 0 6px,transparent 6px 12px)}
.pe .tag{display:inline-block;font-size:10px;background:rgba(255,255,255,.25);border-radius:4px;padding:0 5px;margin-left:4px}
.locals{display:flex;flex-wrap:wrap;gap:14px}
.local{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:12px}
.local .t{color:var(--muted);margin-bottom:6px}
.legend{display:flex;flex-wrap:wrap;gap:14px;color:var(--muted);font-size:12px;margin-top:6px}
.sw{display:inline-block;width:12px;height:12px;border-radius:2px;vertical-align:-2px;margin-right:6px}
pre.code{background:var(--codebg);border:1px solid var(--line);border-radius:8px;padding:12px 14px;overflow-x:auto;
font:12.5px/1.5 ui-monospace,Menlo,Consolas,monospace;color:var(--ink);margin:0 0 10px;white-space:pre}
pre.wrap{white-space:pre-wrap;word-break:break-word}
h3{font-size:13px;margin:12px 0 6px;color:var(--muted);font-weight:600}
.tbl{border-collapse:collapse;font-size:12px;width:100%}.tbl th,.tbl td{text-align:left;padding:3px 8px;border-bottom:1px solid var(--line)}
.tbl th{color:var(--muted);font-weight:600}.tbl code{font-size:11.5px}
.note{color:var(--faint);font-size:12px}
.info{min-height:20px;font-size:12.5px;color:var(--muted);margin-top:10px;font-family:ui-monospace,Menlo,Consolas,monospace}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:14px}
.tile{display:block;background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;color:var(--ink);text-decoration:none}
.tile:hover{border-color:#3978CF}.tile span{display:block;color:var(--muted);font-size:13px;margin-top:4px}
"""

_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>%TITLE%</title>
<style>""" + _CSS + """</style></head><body><div class="wrap">
<h1>%TITLE%</h1><p class="sub">%SUBTITLE%</p>
<div class="card"><h2><span class="num">01</span>Input: logical tensor</h2>
<p class="cap">Each element is colored by its primary owner. Hover an element to see every PE that holds it.</p>
<div class="scroll"><div id="tensor" class="cells"></div></div><div id="tinfo" class="info"></div></div>
<div class="card"><h2><span class="num">02</span>Mapping: PE grid</h2>
<p class="cap">The region each PE holds. Striped cards are read-only replicas; the solid card along a replicated axis is the primary owner.</p>
<div class="scroll"><div id="pes" class="pes"></div></div><div id="legend" class="legend"></div></div>
<div class="card"><h2><span class="num">03</span>Local views</h2>
<p class="cap" id="lcap">Each PE's local buffer, in local coordinates.</p>
<div id="locals" class="locals"></div></div>
<div class="grid2">
<div class="card"><h2><span class="num">04</span>Layout</h2><p class="cap">Decomposition + mapping, bound to existing storage.</p>
<pre class="code wrap">%SPEC%</pre><h3>As an attribute (future <code>#dist.layout</code>)</h3><pre class="code wrap">%ATTR%</pre></div>
<div class="card"><h2><span class="num">05</span>Generated code</h2><p class="cap">What the AIR backend emits inside the herd body.</p>
<pre class="code">%CODE%</pre>%OFFSETS%</div>
</div>%MLIR%</div>
<script>
const M=%MODEL%;
const $=id=>document.getElementById(id), el=(t,c)=>{const e=document.createElement(t);if(c)e.className=c;return e};
const partColor=p=>M.parts[p].color;
// 01 tensor
const T=$("tensor"), cols=M.cells[0].length, cs=cols>48?9:cols>32?13:cols>16?20:26;
T.style.setProperty("--cs",cs+"px");T.style.gridTemplateColumns=`repeat(${cols},${cs}px)`;
M.cells.forEach(row=>row.forEach(c=>{const d=el("div","cell");
 if(!c){d.classList.add("empty");T.appendChild(d);return}
 const [idx,p]=c;d.style.background=partColor(p);d.dataset.part=p;
 if(cs>=20)d.textContent=M.parts[p].primary;
 d.onmouseenter=()=>focus(p,`element ${M.wrapped?idx[0]:"("+idx+")"}: partition ${M.parts[p].region}, primary PE ${M.parts[p].primary}, held by PE ${M.parts[p].holders.join(", ")}`);
 d.onmouseleave=unfocus;T.appendChild(d)}));
if(M.stride>1)$("tinfo").textContent=`showing every ${M.stride}th element per axis`;
// 02 PE grid
const P=$("pes"), gc=M.grid.length>1?M.grid[M.grid.length-1]:M.grid[0];
P.style.gridTemplateColumns=`repeat(${Math.min(gc,M.grid.length>1?gc:8)},auto)`;
M.pes.forEach(pe=>{const d=el("div","pe"+(pe.primary?"":" replica"));d.style.background=partColor(pe.part);
 d.dataset.part=pe.part;d.dataset.pe=pe.ix;
 d.innerHTML=`<b>PE ${pe.ix}</b> <span class="r">(${pe.coord})</span>${pe.primary?"":"<span class='tag'>replica</span>"}<div class="r">${pe.region}</div><div class="r">local [${pe.shape}]</div>`;
 d.onmouseenter=()=>focus(pe.part,`PE ${pe.ix} (${pe.coord}) holds ${pe.region} as ${pe.primary?"primary owner":"replica"}`);
 d.onmouseleave=unfocus;P.appendChild(d)});
$("legend").innerHTML=M.parts.slice(0,16).map(p=>`<span><i class="sw" style="background:${p.color}"></i>PE ${p.primary}${p.holders.length>1?` (+${p.holders.length-1} replica${p.holders.length>2?"s":""})`:""}</span>`).join("")+
 `<span><i class="sw" style="background:var(--ghost)"></i>ghost (halo) copy</span>`;
// 03 local views
const L=$("locals");let anyHalo=false;
M.local.slice(0,32).forEach(lv=>{anyHalo=anyHalo||lv.halo;const box=el("div","local");
 const sz=lv.size, r2=sz.length>1, R=r2?sz[0]:1, C=r2?sz[1]:sz[0], w=Math.min(C,32), rows=r2?R:Math.ceil(C/w), cols=r2?C:w;
 const lcs=cols>24?11:16;
 box.innerHTML=`<div class="t"><b style="color:var(--ink)">PE ${lv.pe}</b> needs ${lv.request} → local [${sz}]</div>`;
 const g=el("div","cells");g.style.setProperty("--cs",lcs+"px");g.style.gridTemplateColumns=`repeat(${Math.min(cols,64)},${lcs}px)`;
 const own=new Array(rows*cols).fill(null);
 lv.pieces.forEach(pc=>{const pr=r2?pc.size[0]:1,pcw=r2?pc.size[1]:pc.size[0];
  for(let a=0;a<pr;a++)for(let b=0;b<pcw;b++){let rr,cc;if(r2){rr=pc.dst[0]+a;cc=pc.dst[1]+b}else{const k=pc.dst[0]+b;rr=Math.floor(k/w);cc=k%w}
   if(rr<rows&&cc<cols)own[rr*cols+cc]=pc}});
 own.slice(0,64*64).forEach(pc=>{const d=el("div","cell");if(!pc){d.classList.add("empty")}else if(pc.local){d.style.background=partColor(pc.part)}
  else{d.style.background="var(--ghost)";d.style.color="#17253C";if(lcs>=14)d.textContent="g";d.title=`ghost from PE ${pc.src}: ${pc.region}`;d.style.outline=`2px solid ${partColor(pc.part)}`;d.style.outlineOffset="-2px"}
  g.appendChild(d)});
 box.appendChild(g);
 const rem=lv.pieces.filter(p=>!p.local);
 if(rem.length){const s=el("div","t");s.style.marginTop="6px";s.innerHTML=rem.map(p=>`<span style="color:var(--comm)">⇠</span> ${p.region} from PE ${p.src}`).join("<br>");box.appendChild(s)}
 L.appendChild(box)});
if(anyHalo)$("lcap").textContent="Each PE's requested region: owned cells in the owner's color, ghost cells (g) copied from the outlined neighbor. Ownership does not change.";
if(M.local.length>32)L.insertAdjacentHTML("beforeend",`<p class="note">${M.local.length-32} more PEs not shown.</p>`);
// linking
function focus(part,msg){document.body.classList.add("dim");
 document.querySelectorAll("[data-part]").forEach(e=>e.classList.toggle("hl",+e.dataset.part===part));$("tinfo").textContent=msg}
function unfocus(){document.body.classList.remove("dim");document.querySelectorAll(".hl").forEach(e=>e.classList.remove("hl"));
 $("tinfo").textContent=M.stride>1?`showing every ${M.stride}th element per axis`:""}
</script></body></html>"""

_PAGE_INDEX = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>%TITLE%</title>
<style>""" + _CSS + """</style></head><body><div class="wrap"><h1>%TITLE%</h1>
<p class="sub">Same global data, different ways to define ownership. Each page shows input, mapping, local views and generated code.</p>
<div class="tiles">%ITEMS%</div></div></body></html>"""
