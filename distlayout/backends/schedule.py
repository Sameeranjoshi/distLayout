"""Inverting an owner map: which pieces does PE ``x`` hold, in closed form?

A layout says *piece t goes to PE owner(t)*. A herd body runs on a PE, so code
generation needs the inverse: *I am PE x; my s-th piece is t(x, s)*. This module
solves that for the quasi-affine owner maps of category 2 and returns a
:class:`Schedule`: a loop nest (uniform trip counts, the same on every PE) and,
for each piece coordinate, a recipe in the herd coordinates ``x`` and the loop
variables ``s``. Recipes are evaluated with ordinary ``+ - * // %``, so the same
schedule yields integers (for checking), printable text, or ``air.api``
``IndexExpr`` offsets.

Supported, per grid axis ``g`` with coordinate ``x_g``:

=========================================  =========================================
owner expression on ``g``                  solved as
=========================================  =========================================
``t_k``                                     ``t_k = x_g``
``t_k floordiv q`` and ``t_k mod q``        ``t_k = x_g1 * q + x_g2``  (refine)
``t_k floordiv q`` alone                    ``t_k = x_g * q + s``,  ``s < q``
``(a·t + c) mod P``, a unit coefficient     ``t_j = (±(x_g - c - Σ a_k t_k)) mod P + P*s``
on some ``t_j``                             (``n_j`` divisible by ``P``)
=========================================  =========================================

Piece coordinates no constraint fixes become full loops. Anything else raises
``NotImplementedError``. Every schedule is checked by enumeration against the
view before it is returned, so a wrong inversion can never be emitted.

The tile stack's slot order is the loop nest's row-major order. For cyclic,
block-cyclic and skew it equals the view's piece order; for some maps (tile
split) it does not, and :meth:`Schedule.slots` says which piece each slot holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..expr import Add, Const, Expr, FloorDiv, Mod, Mul, Var
from ..mapping import as_placement


def _linear(e: Expr) -> Optional[Tuple[Dict[int, int], int]]:
    """``e`` as ``Σ a_k t_k + c``, or ``None`` if it is not linear."""
    if isinstance(e, Var):
        return {e.k: 1}, 0
    if isinstance(e, Const):
        return {}, e.k
    if isinstance(e, Add):
        a, b = _linear(e.a), _linear(e.b)
        if a is None or b is None:
            return None
        t = dict(a[0])
        for k, v in b[0].items():
            t[k] = t.get(k, 0) + v
        return {k: v for k, v in t.items() if v}, a[1] + b[1]
    if isinstance(e, Mul):
        a = _linear(e.a)
        return None if a is None else ({k: v * e.k for k, v in a[0].items()}, a[1] * e.k)
    return None


@dataclass
class Schedule:
    """Loop nest + recipes. ``loops[i]`` is the extent of loop variable ``s_i``."""

    loops: List[int]
    recipes: Dict[int, Tuple]          # piece coordinate k -> recipe
    ncomp: int

    @property
    def slots(self) -> int:
        n = 1
        for e in self.loops:
            n *= e
        return n

    def pieces(self, x: Sequence[Any], s: Sequence[Any]) -> Tuple[Any, ...]:
        """Piece coordinates ``(t_0, ...)`` at herd coordinates ``x``, loop variables ``s``."""
        memo: Dict[int, Any] = {}

        def t(k: int) -> Any:
            if k in memo:
                return memo[k]
            r = self.recipes[k]
            kind = r[0]
            if kind == "x":
                v = x[r[1]]
            elif kind == "pair":                     # t = x_g1 * q + x_g2
                _, g1, q, g2 = r
                v = x[g1] * q + x[g2]
            elif kind == "group":                    # t = x_g * q + s
                _, g, q, i = r
                v = x[g] * q + s[i]
            elif kind == "loop":
                v = s[r[1]]
            elif kind == "pivot":                    # t_j = (sign*(x_g - c - Σ a_k t_k)) mod P + P*s
                _, g, coeffs, c, sign, P, i = r
                acc = x[g] - c if c else x[g]
                for kk, a in coeffs.items():
                    acc = acc - t(kk) * a
                v = ((acc if sign == 1 else acc * -1) % P) + s[i] * P
            else:  # pragma: no cover
                raise AssertionError(r)
            memo[k] = v
            return v

        return tuple(t(k) for k in range(self.ncomp))

    def slot(self, s: Sequence[Any]) -> Any:
        """Row-major slot index of loop variables ``s`` in the local tile stack."""
        v: Any = None
        for si, e in zip(s, self.loops):
            v = si if v is None else v * e + si
        return 0 if v is None else v

    def concrete(self, pe: Sequence[int]) -> List[Tuple[int, Tuple[int, ...]]]:
        """``[(slot, piece), ...]`` at PE ``pe``, in slot order."""
        out = []
        for s in product(*(range(e) for e in self.loops)):
            out.append((self.slot(s) if self.loops else 0, self.pieces(pe, s)))
        return out


def solve(view) -> Schedule:
    """The schedule for ``view``'s mapping (placement or owner map); raises if unsupported."""
    num = view.decomposition.num_parts(view.shape)
    ncomp = len(num)
    exprs = dict(view.mapping.exprs())
    recipes: Dict[int, Tuple] = {}
    loops: List[int] = []
    pending = []

    def new_loop(extent: int) -> int:
        loops.append(extent)
        return len(loops) - 1

    # bare coordinates and floordiv/mod pairs first
    fdiv = {g: e for g, e in exprs.items() if isinstance(e, FloorDiv) and isinstance(e.a, Var)}
    mods = {g: e for g, e in exprs.items() if isinstance(e, Mod) and isinstance(e.a, Var)}
    used = set()
    for g, e in exprs.items():
        if isinstance(e, Var):
            if e.k in recipes:
                raise NotImplementedError(f"t{e.k} drives two grid axes")
            recipes[e.k] = ("x", g)
            used.add(g)
    for g1, e1 in fdiv.items():
        k, q = e1.a.k, e1.k
        g2 = next((g for g, e in mods.items() if e.a.k == k and e.k == q and g not in used), None)
        if k in recipes:
            raise NotImplementedError(f"t{k} constrained twice")
        if g2 is not None:
            recipes[k] = ("pair", g1, q, g2)
            used |= {g1, g2}
        else:
            recipes[k] = ("group", g1, q, None)
            used.add(g1)
    # remaining expressions must be (linear) mod P with a unit pivot
    for g, e in sorted(exprs.items()):
        if g in used:
            continue
        lin = _linear(e.a) if isinstance(e, Mod) else None
        if lin is None:
            raise NotImplementedError(f"cannot invert g{g} = {e} (supported: t, t floordiv q, (linear) mod P)")
        coeffs, c = lin
        P = e.k
        cand = [k for k, a in coeffs.items() if a in (1, -1) and k not in recipes]
        if not cand:
            raise NotImplementedError(f"cannot invert g{g} = {e}: no free coordinate with coefficient ±1")
        j = max(cand)                                   # the last free coordinate: keeps row-major order
        if num[j] % P:
            raise NotImplementedError(
                f"g{g} = {e}: t{j} has {num[j]} values, not a multiple of {P}, so PEs would hold "
                f"different numbers of pieces (air.api needs the same trip count on every core)")
        pending.append((g, j, {k: a for k, a in coeffs.items() if k != j}, c, coeffs[j], P))
        recipes[j] = ("pivot-pending",)
        used.add(g)
    # loops: one per coordinate that is free or a pivot, in coordinate order
    loop_of: Dict[int, int] = {}
    for k in range(ncomp):
        r = recipes.get(k)
        if r is None:
            loop_of[k] = new_loop(num[k])
            recipes[k] = ("loop", loop_of[k])
        elif r[0] == "group":
            loop_of[k] = new_loop(r[2])
            recipes[k] = ("group", r[1], r[2], loop_of[k])
        elif r[0] == "pivot-pending":
            g, j, rest, c, a, P = next(p for p in pending if p[1] == k)
            loop_of[k] = new_loop(num[k] // P)
            recipes[k] = ("pivot", g, rest, c, a, P, loop_of[k])
    sched = Schedule(loops, recipes, ncomp)
    _verify(view, sched)
    return sched


def _verify(view, sched: Schedule) -> None:
    """Every PE gets exactly its pieces, once each, and slots are 0 .. slots-1."""
    for pe in view.grid.coords():
        got = sched.concrete(pe)
        pids = [p for _, p in got]
        want = sorted(view.pieces_at(pe))
        num = view.decomposition.num_parts(view.shape)
        if any(not all(0 <= v < n for v, n in zip(p, num)) for p in pids) or sorted(pids) != want:
            raise NotImplementedError(
                f"owner map {view.mapping!r} could not be inverted exactly at PE {pe} "
                f"(schedule gives {sorted(pids)[:6]}, layout has {want[:6]})")
        if sorted(s for s, _ in got) != list(range(sched.slots)):
            raise AssertionError("slots are not a bijection")


def is_placement(view) -> bool:
    return as_placement(view.mapping, view.decomposition.num_parts(view.shape)) is not None


# -- printing -------------------------------------------------------------------

class Text:
    """Printable stand-in for an index value: builds Python source like ``(tx - s0) % 4 + s1*4``."""

    def __init__(self, terms: Dict[str, int], const: int = 0, opaque: Optional[str] = None) -> None:
        self.terms = {k: v for k, v in terms.items() if v}
        self.const = const
        self.opaque = opaque                 # a non-linear term (a mod / floordiv) printed as-is

    @staticmethod
    def var(name: str) -> "Text":
        return Text({name: 1})

    def _lift(self, o: Any) -> "Text":
        return o if isinstance(o, Text) else Text({}, int(o))

    def _lin(self) -> "Text":
        return Text({self.opaque: 1}) if self.opaque is not None else self

    def __add__(self, o: Any) -> "Text":
        a, b = self._lin(), self._lift(o)._lin()
        t = dict(a.terms)
        for k, v in b.terms.items():
            t[k] = t.get(k, 0) + v
        return Text(t, a.const + b.const)

    __radd__ = __add__

    def __sub__(self, o: Any) -> "Text":
        return self + self._lift(o) * -1

    def __mul__(self, o: Any) -> "Text":
        k = int(o)
        a = self._lin()
        return Text({n: c * k for n, c in a.terms.items()}, a.const * k)

    __rmul__ = __mul__

    def __mod__(self, k: int) -> "Text":
        return Text({}, 0, f"{self._paren()} % {k}")

    def __floordiv__(self, k: int) -> "Text":
        return Text({}, 0, f"{self._paren()} // {k}")

    def _paren(self) -> str:
        s = str(self)
        return s if (s.isidentifier() or s.isdigit()) else f"({s})"

    def __str__(self) -> str:
        if self.opaque is not None:
            return self.opaque
        parts = []
        for n, c in self.terms.items():
            term = n if abs(c) == 1 else (f"({n})*{abs(c)}" if " " in n else f"{n}*{abs(c)}")
            parts.append(("- " if c < 0 else "+ ") + term)
        if self.const:
            parts.append(("- " if self.const < 0 else "+ ") + str(abs(self.const)))
        if not parts:
            return "0"
        s = " ".join(parts)
        return s[2:] if s.startswith("+ ") else "-" + s[2:]
