"""Layout explorer: one interactive HTML page for picturing layouts.

    from distlayout.viz import render_html
    open("explorer.html", "w").write(render_html(view, title="My layout"))

or, for the built-in presets plus your own views::

    render_explorer([("Block", view1, {}), ("Halo", view2, {"halo": 1})])

The page is a small workbench, in the spirit of a layout playground:

* **presets** across the top;
* a **builder** on the left -- tensor shape, PE grid, and for each tensor axis
  "whole" or "block/uneven over grid axis g" (grid axes nothing is placed on
  are replicated), halo radius, element size and PE memory;
* **one picture** on the right: the tensor colored by owner (hover for the
  owner formula and the holders), the PE grid with memory-fit bars and red
  arrows for halo imports;
* **code tabs**: the Python spec, the ``#dist.layout`` text, the ``air.api``
  herd body, and the real MLIR when it was rendered with MLIR-AIR installed;
* **pin** a layout to compare its bytes/PE against the current one, and
  **export** the picture as SVG.

The layout rules run in the page (a direct port of the v0 algebra) so the
builder updates live; ``tests/test_viz.py`` checks the port against Python.
No network, fonts or libraries are needed: the file opens anywhere.
"""

from __future__ import annotations

import html
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .decomposition import Block, UnevenBlock
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
    return f"view = bind({src}, Decomposition({rules}), {mapping})"


def air_code(view: DistributedView, src: str = "l2", dtype: str = "f32") -> Tuple[str, List[Tuple[Tuple[int, ...], Tuple[slice, ...]]]]:
    """(``air.api`` herd body, per-PE concrete slices). Uniform layouts only."""
    from .backends import air as backend

    names = ["tx", "ty", "tz"][: view.grid.rank]
    if not view.is_uniform():
        return ("# uneven layout: PEs hold different sizes, and air.api needs one static\n"
                "# transfer size, so the AIR backend does not emit this layout yet.", [])
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


# -- view <-> page spec ---------------------------------------------------------

def to_spec(view: DistributedView, title: str = "", note: str = "", halo: int = 0,
            mlir: Optional[str] = None, src: str = "l2", elem_bytes: int = 4,
            pe_kib: float = 48) -> Dict[str, Any]:
    """The JSON the page understands. Raises for layouts the page cannot draw."""
    if len(view.shape) > 2:
        raise NotImplementedError("the explorer draws rank-1 and rank-2 tensors")
    placed = view.mapping.place_dict
    axes: List[Dict[str, Any]] = [{"kind": "whole"} for _ in view.shape]
    for k, rule in enumerate(view.decomposition.rules):
        g = placed[k]
        if isinstance(rule, Block):
            axes[rule.axis] = {"kind": "block", "g": g}
        elif isinstance(rule, UnevenBlock):
            a = {"kind": "uneven", "g": g, "policy": rule.policy}
            if rule.sizes_ is not None:
                a["sizes"] = list(rule.sizes_)
            axes[rule.axis] = a
        else:
            raise NotImplementedError(f"the explorer cannot draw {rule!r}")
    if not view.source.is_global:
        raise NotImplementedError("the explorer draws global sources")
    spec = {"title": title, "note": note, "shape": list(view.shape), "grid": list(view.grid.shape),
            "axes": axes, "halo": halo, "bytes": elem_bytes, "kib": pe_kib, "src": src}
    if mlir:
        spec["mlir"] = mlir
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
.wrap{max-width:1240px;margin:0 auto;padding:20px 16px 32px}
h1{font-size:24px;margin:0 0 12px;font-weight:700}
.presets{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px}
.presets button,.btn{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--ink);
border-radius:8px;padding:6px 12px;cursor:pointer}
.presets button.on{background:var(--accent);border-color:var(--accent);color:#fff}
.main{display:grid;grid-template-columns:330px 1fr;gap:16px;align-items:start}
@media(max-width:860px){.main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-width:0}
.card h2{font-size:15px;margin:0 0 10px}
label{display:block;color:var(--muted);font-size:12px;margin:10px 0 4px}
input,select{font:inherit;font-size:13px;width:100%;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--ink)}
.row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.err{color:var(--comm);font-size:12.5px;min-height:18px;margin-top:8px}
.tabs{display:flex;gap:4px;margin:14px 0 0;border-bottom:1px solid var(--line)}
.tabs button{font:inherit;font-size:12.5px;border:0;background:none;color:var(--muted);padding:6px 10px;cursor:pointer;border-bottom:2px solid transparent}
.tabs button.on{color:var(--ink);border-bottom-color:var(--accent)}
pre{background:var(--code);border:1px solid var(--line);border-top:0;border-radius:0 0 8px 8px;margin:0;padding:10px 12px;
font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;white-space:pre-wrap;word-break:break-word;max-height:340px;overflow:auto}
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
  <div id="axes"></div>
  <div class="row"><div><label>Halo radius</label><input id="halo" type="number" min="0" value="0"></div>
   <div><label>Bytes / element · PE KiB</label><div class="row"><input id="bytes" type="number" min="1"><input id="kib" type="number" min="1"></div></div></div>
  <div class="err" id="err"></div>
  <div class="tabs" id="tabs"></div><pre id="code"></pre>
 </div>
 <div class="card">
  <p class="note" id="note"></p>
  <div id="pic"></div>
  <div class="formula" id="formula"></div>
  <div class="legend" id="legend"></div>
  <div class="bar"><button class="btn" id="pin">Pin as A</button><button class="btn" id="svg">Export SVG</button><span class="stat" id="stat"></span></div>
 </div>
</div></div>
<script>
const PRESETS=%PRESETS%, COLORS=%COLORS%;
const $=id=>document.getElementById(id);
let cur=null, pinned=null, tab="python";

// ---------- layout algebra (port of distlayout v0) ----------
function sizesOf(ax,extent,parts){
  if(ax.kind==="block"){if(extent%parts)throw new Error(`axis of ${extent} is not divisible into ${parts} blocks (${extent} % ${parts} = ${extent%parts}); choose uneven`);return Array(parts).fill(extent/parts)}
  if(ax.sizes){const s=ax.sizes.reduce((a,b)=>a+b,0);if(s!==extent||ax.sizes.length!==parts)throw new Error(`uneven sizes [${ax.sizes}] must be ${parts} parts summing to ${extent}`);return ax.sizes.slice()}
  if((ax.policy||"balanced")==="balanced"){const q=Math.floor(extent/parts),r=extent%parts;return Array.from({length:parts},(_,k)=>k<r?q+1:q)}
  const b=Math.ceil(extent/parts);let left=extent;return Array.from({length:parts},()=>{const v=Math.min(b,left);left-=v;return v})}
function build(spec){
  const shape=spec.shape, grid=spec.grid, rank=shape.length, G=grid.length;
  if(!rank||rank>2)throw new Error("tensor must be 1-D or 2-D");
  if(!G||G>3)throw new Error("grid must be 1-D to 3-D");
  if(shape.concat(grid).some(v=>!(v>=1)||!Number.isInteger(v)))throw new Error("shapes must be positive integers");
  const rules=[];spec.axes.forEach((ax,a)=>{if(ax.kind!=="whole"){
    if(!(ax.g>=0&&ax.g<G))throw new Error(`axis ${a}: grid axis g${ax.g} does not exist`);
    if(rules.some(r=>r.g===ax.g))throw new Error(`two tensor axes placed on grid axis g${ax.g}`);
    const sz=sizesOf(ax,shape[a],grid[ax.g]);if(sz.some(s=>s<1))throw new Error(`axis ${a}: ${grid[ax.g]} parts leave an empty part`);
    const st=[0];sz.forEach(s=>st.push(st[st.length-1]+s));rules.push({axis:a,g:ax.g,ax,sizes:sz,starts:st})}});
  const replicate=[...Array(G).keys()].filter(g=>!rules.some(r=>r.g===g));
  const counts=rules.map(r=>r.sizes.length);
  const pids=prod(counts.map(n=>[...Array(n).keys()]));
  const region=pid=>{const lo=shape.map(()=>0),hi=shape.slice();rules.forEach((r,k)=>{lo[r.axis]=r.starts[pid[k]];hi[r.axis]=r.starts[pid[k]+1]});return{lo,hi}};
  const partOf=idx=>rules.map(r=>{let p=0;while(idx[r.axis]>=r.starts[p+1])p++;return p});
  const pidIx=pid=>pid.reduce((a,p,k)=>a*counts[k]+p,0);
  const peIx=c=>c.reduce((a,v,g)=>a*grid[g]+v,0);
  const pes=prod(grid.map(n=>[...Array(n).keys()]));
  const partAt=pe=>rules.map(r=>pe[r.g]);
  const isPrimary=pe=>replicate.every(g=>pe[g]===0);
  const primaryOf=pid=>{const c=grid.map(()=>0);rules.forEach((r,k)=>c[r.g]=pid[k]);return c};
  const holdersOf=pid=>pes.filter(pe=>rules.every((r,k)=>pe[r.g]===pid[k]));
  const uniform=rules.every(r=>new Set(r.sizes).size===1);
  return {spec,shape,grid,rules,replicate,pids,region,partOf,pidIx,peIx,pes,partAt,isPrimary,primaryOf,holdersOf,uniform};
}
function colorOf(L,pi){
  if(L.pids.length<=COLORS.length)return COLORS[pi%COLORS.length];
  // many partitions: hue by the first split component, shade by the rest
  const pid=L.pids[pi], rest=pid.slice(1), counts=L.rules.slice(1).map(r=>r.sizes.length);
  const ri=rest.reduce((a,v,k)=>a*counts[k]+v,0), rn=counts.reduce((a,b)=>a*b,1);
  return mix(COLORS[pid[0]%COLORS.length], rn>1?ri/(rn-1)*0.5:0)}
function mix(hex,t){const n=parseInt(hex.slice(1),16),c=[n>>16,(n>>8)&255,n&255].map(v=>Math.round(v+(255-v)*t));return "#"+c.map(v=>v.toString(16).padStart(2,"0")).join("")}
function prod(lists){return lists.reduce((acc,l)=>acc.flatMap(a=>l.map(x=>a.concat([x]))),[[]])}
function inter(a,b){const lo=a.lo.map((v,i)=>Math.max(v,b.lo[i])),hi=a.hi.map((v,i)=>Math.min(v,b.hi[i]));return hi.every((v,i)=>v>lo[i])?{lo,hi}:null}
function contains(a,b){return b.lo.every((v,i)=>v>=a.lo[i]&&b.hi[i]<=a.hi[i])}
function vol(r){return r.lo.reduce((a,v,i)=>a*(r.hi[i]-v),1)}
function rstr(r){return "["+r.lo.map((v,i)=>v+":"+r.hi[i]).join(", ")+")"}
function plan(L,pe,h){
  const held=L.region(L.partAt(pe));
  const req={lo:held.lo.map(v=>Math.max(0,v-h)),hi:held.hi.map((v,i)=>Math.min(L.shape[i],v+h))};
  const pieces=[];L.pids.forEach(pid=>{const p=inter(L.region(pid),req);if(p)pieces.push({r:p,local:contains(held,p),src:L.peIx(L.primaryOf(pid)),part:L.pidIx(pid)})});
  return {held,req,pieces};
}

// ---------- text ----------
function pyTuple(a){return a.length===1?`(${a[0]},)`:`(${a.join(", ")})`}
function ruleRepr(r){const ax=r.ax;if(ax.kind==="block")return `Block(axis=${r.axis}, parts=${r.sizes.length})`;
  if(ax.sizes)return `UnevenBlock(axis=${r.axis}, sizes=${pyTuple(ax.sizes)})`;
  return `UnevenBlock(axis=${r.axis}, parts=${r.sizes.length}, policy='${ax.policy||"balanced"}')`}
function gridRepr(g){return g.length>1?`Grid(${pyTuple(g)})`:`Grid(${g[0]})`}
function pyText(L){
  const rules=L.rules.map(ruleRepr).join(", ");
  let m;if(!L.rules.length)m=`Replicate(${gridRepr(L.grid)})`;
  else{const place="{"+L.rules.map((r,k)=>`${k}: ${r.g}`).join(", ")+"}";m=`Map(${gridRepr(L.grid)}, ${place}${L.replicate.length?`, replicate=[${L.replicate.join(", ")}]`:""})`}
  return `view = bind(${L.spec.src||"l2"}, Decomposition(${rules}), ${m})`}
function attrText(L){
  const rr=L.rules.map(r=>r.ax.kind==="block"?`block<axis = ${r.axis}, parts = ${r.sizes.length}>`:
    r.ax.sizes?`uneven<axis = ${r.axis}, sizes = [${r.ax.sizes.join(", ")}]>`:`uneven<axis = ${r.axis}, parts = ${r.sizes.length}, policy = ${r.ax.policy||"balanced"}>`).join(", ");
  return `#dist.layout<shape = [${L.shape.join(", ")}], grid = [${L.grid.join(", ")}], decomp = [${rr}], place = [${L.rules.map((r,k)=>`${k} -> g${r.g}`).join(", ")}], replicate = [${L.replicate.map(g=>"g"+g).join(", ")}]>`}
function airText(L){
  if(!L.uniform)return "# uneven layout: PEs hold different sizes, and air.api needs one static\n# transfer size, so the AIR backend does not emit this layout yet.";
  const names=["tx","ty","tz"].slice(0,L.grid.length), src=L.spec.src||"l2";
  const sub=L.shape.map((n,a)=>{const r=L.rules.find(r=>r.axis===a);if(!r)return `0 : ${n}`;const b=r.sizes[0],c=names[r.g];return `${c}*${b} : ${c}*${b} + ${b}`}).join(", ");
  const herd=L.grid.map((n,g)=>{const r=L.rules.find(r=>r.g===g);return r?`range(0, ${L.shape[r.axis]}, ${r.sizes[0]})`:`range(${n})`}).join(", ");
  const loc=L.shape.map((n,a)=>{const r=L.rules.find(r=>r.axis===a);return r?r.sizes[0]:n});
  const rep=L.replicate.map(g=>names[g]);
  return `with air.herd([${herd}]) as h:\n  @h.body\n  def _(${names.join(", ")}):\n    a = air.alloc([${loc.join(", ")}], f32, scope=h.private())\n    air.ops.load(a, ${src}[${sub}])`+
    (rep.length?`\n    # ${rep.join(", ")} not in the offset: every PE along it reads the same region (broadcast)`:"")+
    `\n\n# with distlayout:\n#   a = air_backend.alloc_local(view, f32, scope=h.private())\n#   air_backend.load_local(view, a, (${names.join(", ")}${names.length===1?",":""}))`}
function formula(L,idx){
  const pid=L.partOf(idx);
  const parts=L.rules.map((r,k)=>r.ax.kind==="block"?`p${k} = i${r.axis} // ${r.sizes[0]} = ${pid[k]}`:
    `p${k} = part holding i${r.axis} in [${r.sizes.join(", ")}] = ${pid[k]}`);
  const pe=L.primaryOf(pid), hs=L.holdersOf(pid).map(L.peIx);
  return `element (${idx.join(", ")}): ${parts.join(";  ")||"whole tensor"}  →  owner PE ${L.peIx(pe)} (${pe.join(", ")})`+(hs.length>1?`, held by ${hs.length} PEs`:"");
}

// ---------- drawing (SVG) ----------
function draw(L){
  const W=Math.min(820,$("pic").clientWidth||820), h=+L.spec.halo||0, bytes=+L.spec.bytes||4, budget=(+L.spec.kib||48)*1024;
  const rank=L.shape.length;
  let rows,cols,at0;
  if(rank===1){const n=L.shape[0],b=L.rules.length?Math.max(...L.rules[0].sizes):n;const w=b<=64?b:64;cols=Math.min(n,w);rows=Math.ceil(n/cols);at0=(r,c)=>r*cols+c<n?[r*cols+c]:null}
  else{rows=L.shape[0];cols=L.shape[1];at0=(r,c)=>[r,c]}
  // sample big tensors: at most 64 columns x 48 rows of cells
  const sr=Math.ceil(rows/48), sc=Math.ceil(cols/64), at=(r,c)=>at0(r*sr,c*sc);
  rows=Math.ceil(rows/sr);cols=Math.ceil(cols/sc);
  const cs=Math.max(4,Math.min(26,Math.floor(W/cols)-2)), gap=cs>8?2:1, tw=cols*(cs+gap), th=rows*(cs+gap);
  let s=`<text x="0" y="13" font-size="12" fill="var(--muted)">Tensor ${L.shape.join(" × ")} — each element colored by its owner${sr>1||sc>1?` (showing every ${rank===1?sr*sc:[sr,sc].join(" × ")}th)`:""}</text>`;
  const y0=22;
  for(let r=0;r<rows;r++)for(let c=0;c<cols;c++){const idx=at(r,c);if(!idx)continue;const pid=L.partOf(idx),pi=L.pidIx(pid),pe=L.peIx(L.primaryOf(pid));
    s+=`<rect class="el" data-i="${idx}" data-p="${pi}" x="${c*(cs+gap)}" y="${y0+r*(cs+gap)}" width="${cs}" height="${cs}" rx="${cs>10?3:1}" fill="${colorOf(L,pi)}"/>`;
    if(cs>=18)s+=`<text pointer-events="none" x="${c*(cs+gap)+cs/2}" y="${y0+r*(cs+gap)+cs/2+4}" font-size="10" font-weight="600" fill="#fff" text-anchor="middle">${pe}</text>`}
  // PE grid
  const G=L.grid, gr=G.length>1?G.slice(0,-1).reduce((a,b)=>a*b,1):1, gc=G[G.length-1];
  const bw=Math.max(92,Math.min(150,Math.floor((W-8)/gc)-12)), bh=58, bg=h?34:12, py=y0+th+40;
  s+=`<text x="0" y="${py-12}" font-size="12" fill="var(--muted)">PE grid ${G.join(" × ")} — region held, memory used${h?` (with halo ${h})`:""}</text>`;
  const pos={};let maxB=0,over=0;
  L.pes.forEach(pe=>{const i=L.peIx(pe),row=G.length>1?Math.floor(i/gc):0,col=i%gc,x=col*(bw+bg),y=py+row*(bh+bg);pos[i]=[x+bw/2,y+bh/2];
    const pid=L.partAt(pe),pi=L.pidIx(pid),pl=plan(L,pe,h),b=vol(pl.req)*bytes;maxB=Math.max(maxB,b);if(b>budget)over++;
    const frac=Math.min(1,b/budget),col2=colorOf(L,pi);
    s+=`<g class="pe" data-p="${pi}" data-pe="${i}"><rect x="${x}" y="${y}" width="${bw}" height="${bh}" rx="7" fill="${col2}" ${L.isPrimary(pe)?"":`fill-opacity=".55"`}/>`+
      `<text x="${x+8}" y="${y+16}" font-size="12" font-weight="700" fill="#fff">PE ${i}${L.isPrimary(pe)?"":" · replica"}</text>`+
      `<text x="${x+8}" y="${y+31}" font-size="11" fill="#fff">${rstr(pl.held)}</text>`+
      `<rect x="${x+8}" y="${y+40}" width="${bw-16}" height="8" rx="3" fill="rgba(255,255,255,.35)"/>`+
      `<rect x="${x+8}" y="${y+40}" width="${(bw-16)*frac}" height="8" rx="3" fill="${b>budget?"#D95455":"#fff"}"/>`+
      `<title>PE ${i} (${pe}) holds ${rstr(pl.held)}${h?`, needs ${rstr(pl.req)}`:""}: ${fmtB(b)} of ${fmtB(budget)}</title></g>`});
  // import arrows (halo)
  let arrows="";const pair={};
  if(h)L.pes.forEach(pe=>{const dst=L.peIx(pe);plan(L,pe,h).pieces.filter(p=>!p.local).forEach(p=>{const a=Math.min(p.src,dst),b=Math.max(p.src,dst),k=a+"-"+b;
    pair[k]=pair[k]||{a,b,n:0,dirs:new Set()};pair[k].n+=vol(p.r);pair[k].dirs.add(p.src===a?"ab":"ba")})});
  const edge=(x,y,dx,dy)=>{const t=Math.min(Math.abs(dx)>1e-9?(bw/2+4)/Math.abs(dx):1e9,Math.abs(dy)>1e-9?(bh/2+4)/Math.abs(dy):1e9);return [x+dx*t,y+dy*t]};
  Object.values(pair).forEach(({a,b,n,dirs})=>{const [x1,y1]=pos[a],[x2,y2]=pos[b],dx=x2-x1,dy=y2-y1;
    const [sx,sy]=edge(x1,y1,dx,dy),[ex,ey]=edge(x2,y2,-dx,-dy),diag=Math.abs(dx)>1&&Math.abs(dy)>1;
    arrows+=`<line x1="${sx}" y1="${sy}" x2="${ex}" y2="${ey}" stroke="#D95455" stroke-width="${diag?1.2:2.2}" ${diag?'stroke-dasharray="4 3"':""} ${dirs.has("ab")?'marker-end="url(#ah)"':""} ${dirs.has("ba")?'marker-start="url(#ah2)"':""}><title>PE ${a} ⇄ PE ${b}: ${n} halo elements</title></line>`});
  const ph=(gr)*(bh+bg), H=py+ph+4, PW=Math.max(tw,gc*(bw+bg));
  const svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${PW}" height="${H}" viewBox="0 0 ${PW} ${H}"><defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0,0L10,5L0,10z" fill="#D95455"/></marker><marker id="ah2" viewBox="0 0 10 10" refX="2" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M10,0L0,5L10,10z" fill="#D95455"/></marker></defs>${s}${arrows}</svg>`;
  $("pic").innerHTML=`<div style="overflow-x:auto">${svg}</div>`;
  // legend & stats
  $("legend").innerHTML=L.pids.slice(0,12).map(pid=>{const pi=L.pidIx(pid),hs=L.holdersOf(pid).length;return `<span><i class="sw" style="background:${colorOf(L,pi)}"></i>${rstr(L.region(pid))} → PE ${L.peIx(L.primaryOf(pid))}${hs>1?` +${hs-1}`:""}</span>`}).join("")+(L.pids.length>12?`<span>… ${L.pids.length-12} more</span>`:"")+(h?`<span><i class="sw" style="background:#D95455"></i>halo import</span>`:"");
  L.maxB=maxB;
  const fit=over?`<b style="color:var(--comm)">${over} PE${over>1?"s":""} over budget</b>`:`<b style="color:var(--ok)">fits</b>`;
  $("stat").innerHTML=`max ${fmtB(maxB)} / PE of ${fmtB(budget)} · ${fit}`+(pinned?` · vs A (${pinned.name}): ${fmtB(pinned.maxB)} / PE → ${maxB<=pinned.maxB?(pinned.maxB/maxB).toFixed(2)+"× less":(maxB/pinned.maxB).toFixed(2)+"× more"}`:"");
  // hover
  const svgEl=$("pic").querySelector("svg");
  svgEl.querySelectorAll(".el").forEach(e=>{e.onmouseenter=()=>{hl(+e.dataset.p);$("formula").textContent=formula(L,e.dataset.i.split(",").map(Number))};e.onmouseleave=unhl});
  svgEl.querySelectorAll(".pe").forEach(e=>{e.onmouseenter=()=>hl(+e.dataset.p);e.onmouseleave=unhl});
  function hl(p){svgEl.querySelectorAll(".el,.pe").forEach(e=>e.style.opacity=+e.dataset.p===p?1:.2)}
  function unhl(){svgEl.querySelectorAll(".el,.pe").forEach(e=>e.style.opacity="");$("formula").textContent=hint(L)}
  $("formula").textContent=hint(L);
}
function hint(L){return L.rules.length?"hover an element for its owner formula; "+L.rules.map((r,k)=>r.ax.kind==="block"?`p${k} = i${r.axis} // ${r.sizes[0]}`:`p${k} from sizes [${r.sizes.join(", ")}]`).join(", ")+(L.replicate.length?`; replicated over ${L.replicate.map(g=>"g"+g).join(", ")}`:""):"every PE holds the whole tensor; owner is PE 0"}
function fmtB(b){return b>=1024?(b/1024).toFixed(b%1024?1:0)+" KiB":b+" B"}

// ---------- UI ----------
function parseList(v){return v.split(/[x×,\s]+/).filter(Boolean).map(Number)}
function axisOptions(G){const o=[["whole","whole (not split)"]];for(let g=0;g<G;g++){o.push([`block:${g}`,`block → g${g}`]);o.push([`uneven:${g}`,`uneven → g${g}`])}return o}
function syncAxes(spec){
  const G=spec.grid.length, box=$("axes");box.innerHTML="";
  spec.shape.forEach((n,a)=>{const ax=spec.axes[a]||{kind:"whole"},v=ax.kind==="whole"?"whole":`${ax.kind}:${ax.g}`;
    box.insertAdjacentHTML("beforeend",`<label>Tensor axis ${a} (${n})</label><select data-a="${a}">${axisOptions(G).map(([k,t])=>`<option value="${k}" ${k===v?"selected":""}>${t}</option>`).join("")}</select>`)});
  box.querySelectorAll("select").forEach(s=>s.onchange=()=>{const [k,g]=s.value.split(":");cur.axes[+s.dataset.a]=k==="whole"?{kind:"whole"}:{kind:k,g:+g,policy:"balanced"};render()});
}
function load(spec){cur=JSON.parse(JSON.stringify(spec));$("shape").value=cur.shape.join(" × ");$("grid").value=cur.grid.join(" × ");
  $("halo").value=cur.halo||0;$("bytes").value=cur.bytes||4;$("kib").value=cur.kib||48;$("note").textContent=cur.note||"";syncAxes(cur);render()}
function render(){
  try{const L=build(cur);$("err").textContent="";draw(L);showCode(L);window.__last=L}
  catch(e){$("err").textContent=e.message}}
function showCode(L){
  const t={python:["Python",pyText(L)],attr:["#dist.layout",attrText(L)],air:["air.api",airText(L)],mlir:["MLIR",cur.mlir||"# MLIR is embedded when the page is rendered with MLIR-AIR installed\n# (python3 examples/gallery.py). Editing the layout drops it."]};
  $("tabs").innerHTML=Object.entries(t).map(([k,[n]])=>`<button data-t="${k}" class="${k===tab?"on":""}">${n}</button>`).join("");
  $("tabs").querySelectorAll("button").forEach(b=>b.onclick=()=>{tab=b.dataset.t;showCode(L)});$("code").textContent=t[tab][1]}
["shape","grid"].forEach(id=>$(id).onchange=()=>{const v=parseList($(id).value);cur[id]=v;delete cur.mlir;
  if(id==="shape")cur.axes=v.map((_,a)=>cur.axes[a]||{kind:"whole"});
  cur.axes=cur.axes.map(ax=>ax.kind!=="whole"&&ax.g>=cur.grid.length?{kind:"whole"}:ax).map(ax=>{const c=Object.assign({},ax);delete c.sizes;return c});
  syncAxes(cur);render()});
["halo","bytes","kib"].forEach(id=>$(id).oninput=()=>{cur[id]=+$(id).value;render()});
$("pin").onclick=()=>{if(window.__last){pinned={name:`${cur.title||"layout"} on ${cur.grid.join("×")}`,maxB:window.__last.maxB};render()}};
$("svg").onclick=()=>{const s=$("pic").querySelector("svg");if(!s)return;const css=getComputedStyle(document.documentElement);
  let txt=new XMLSerializer().serializeToString(s).replace(/var\(--muted\)/g,css.getPropertyValue("--muted").trim()||"#52657B");
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([txt],{type:"image/svg+xml"}));a.download=(cur.title||"layout").replace(/\W+/g,"_")+".svg";a.click()};
const P=$("presets");PRESETS.forEach((p,i)=>{const b=document.createElement("button");b.textContent=p.title||`Layout ${i+1}`;
  b.onclick=()=>{P.querySelectorAll("button").forEach(x=>x.classList.remove("on"));b.classList.add("on");load(p)};P.appendChild(b)});
if(PRESETS.length<2)P.style.display="none";
P.firstChild&&P.firstChild.classList.add("on");load(PRESETS[0]);
window.addEventListener("resize",()=>render());
window.distlayout={build,pyText,attrText,airText,plan};
</script></body></html>"""
