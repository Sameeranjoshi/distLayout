"""Quasi-affine expressions over piece coordinates ``t0, t1, ...``.

The owner map of category-2 layouts: which grid coordinate a tile goes to,
as an expression in the tile's coordinates. The language is what MLIR's affine
maps accept: ``+``, ``-``, multiplication by a constant, and ``floordiv`` /
``mod`` by a positive constant::

    from distlayout.expr import t, parse
    t(0) % 4                       # cyclic:        t0 mod 4
    (t(0) + t(1)) % 4              # skew:          (t0 + t1) mod 4
    (t(0) + 2 * t(1)) % 4          # tile split:    (t0 + t1 * 2) mod 4
    parse("(t0 floordiv 2) mod 4") # also accepts // and %

``str(e)`` prints MLIR affine syntax, so the expression can become a
``#dist.layout`` / ``affine_map`` attribute unchanged later.
"""

from __future__ import annotations

import re
from typing import Sequence, Union

Number = int


class Expr:
    def eval(self, t: Sequence[int]) -> int:
        raise NotImplementedError

    def vars(self) -> set:
        raise NotImplementedError

    # -- operators ---------------------------------------------------------
    def __add__(self, o: "ExprLike") -> "Expr":
        return Add(self, lift(o))

    def __radd__(self, o: "ExprLike") -> "Expr":
        return Add(lift(o), self)

    def __sub__(self, o: "ExprLike") -> "Expr":
        o = lift(o)
        return Add(self, Const(-o.k) if isinstance(o, Const) else Mul(o, -1))

    def __rsub__(self, o: "ExprLike") -> "Expr":
        return Add(lift(o), Mul(self, -1))

    def __neg__(self) -> "Expr":
        return Mul(self, -1)

    def __mul__(self, o: "ExprLike") -> "Expr":
        o = lift(o)
        if isinstance(o, Const):
            return Mul(self, o.k)
        if isinstance(self, Const):
            return Mul(o, self.k)
        raise TypeError(f"non-affine: {self} * {o} (multiply by a constant only)")

    __rmul__ = __mul__

    def __floordiv__(self, k: int) -> "Expr":
        return FloorDiv(self, _pos(k, "floordiv"))

    def __mod__(self, k: int) -> "Expr":
        return Mod(self, _pos(k, "mod"))

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Expr) and str(self) == str(o)

    def __hash__(self) -> int:
        return hash(str(self))

    def __repr__(self) -> str:
        return f"expr({str(self)!r})"


ExprLike = Union[Expr, int]


class Var(Expr):
    def __init__(self, k: int) -> None:
        if k < 0:
            raise ValueError(f"t{k}: negative index")
        self.k = k

    def eval(self, t: Sequence[int]) -> int:
        if self.k >= len(t):
            raise IndexError(f"t{self.k} used, but pieces have {len(t)} coordinates")
        return int(t[self.k])

    def vars(self) -> set:
        return {self.k}

    def __str__(self) -> str:
        return f"t{self.k}"


class Const(Expr):
    def __init__(self, k: int) -> None:
        self.k = int(k)

    def eval(self, t: Sequence[int]) -> int:
        return self.k

    def vars(self) -> set:
        return set()

    def __str__(self) -> str:
        return str(self.k)


class Add(Expr):
    def __init__(self, a: Expr, b: Expr) -> None:
        self.a, self.b = a, b

    def eval(self, t: Sequence[int]) -> int:
        return self.a.eval(t) + self.b.eval(t)

    def vars(self) -> set:
        return self.a.vars() | self.b.vars()

    def __str__(self) -> str:
        b = self.b
        if isinstance(b, Const) and b.k < 0:
            return f"{self.a} - {-b.k}"
        if isinstance(b, Mul) and b.k < 0:
            return f"{self.a} - {Mul(b.a, -b.k)}"
        return f"{self.a} + {b}"


class Mul(Expr):
    def __init__(self, a: Expr, k: int) -> None:
        self.a, self.k = a, int(k)

    def eval(self, t: Sequence[int]) -> int:
        return self.a.eval(t) * self.k

    def vars(self) -> set:
        return self.a.vars()

    def __str__(self) -> str:
        if self.k == 1:
            return str(self.a)
        if self.k == -1:
            return f"-{_atom(self.a)}"
        return f"{_atom(self.a)} * {self.k}"


class FloorDiv(Expr):
    def __init__(self, a: Expr, k: int) -> None:
        self.a, self.k = a, k

    def eval(self, t: Sequence[int]) -> int:
        return self.a.eval(t) // self.k

    def vars(self) -> set:
        return self.a.vars()

    def __str__(self) -> str:
        return f"{_atom(self.a)} floordiv {self.k}"


class Mod(Expr):
    def __init__(self, a: Expr, k: int) -> None:
        self.a, self.k = a, k

    def eval(self, t: Sequence[int]) -> int:
        return self.a.eval(t) % self.k

    def vars(self) -> set:
        return self.a.vars()

    def __str__(self) -> str:
        return f"{_atom(self.a)} mod {self.k}"


def t(k: int) -> Var:
    """The k-th piece coordinate."""
    return Var(k)


def lift(o: ExprLike) -> Expr:
    if isinstance(o, Expr):
        return o
    if isinstance(o, int):
        return Const(o)
    raise TypeError(f"not an expression: {o!r}")


def _atom(e: Expr) -> str:
    return str(e) if isinstance(e, (Var, Const)) else f"({e})"


def _pos(k: int, what: str) -> int:
    if not isinstance(k, int) or k <= 0:
        raise TypeError(f"{what} needs a positive integer constant, got {k!r}")
    return k


# -- parser ---------------------------------------------------------------------

_TOKEN = re.compile(r"\s*(?:(\d+)|(t\d+)|(floordiv|mod|//|[%*+()-]))")


def parse(text: Union[str, Expr]) -> Expr:
    """Parse ``"(t0 + t1) mod 4"``; ``//`` and ``%`` are accepted for floordiv and mod."""
    if isinstance(text, Expr):
        return text
    toks, pos, s = [], 0, text.strip()
    while pos < len(s):
        m = _TOKEN.match(s, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot parse {text!r} at {s[pos:]!r}")
        toks.append(m.group(1) or m.group(2) or m.group(3))
        pos = m.end()
    i = 0

    def peek():
        return toks[i] if i < len(toks) else None

    def take():
        nonlocal i
        if i >= len(toks):
            raise ValueError(f"unexpected end of {text!r}")
        i += 1
        return toks[i - 1]

    def additive() -> Expr:
        e = multiplicative()
        while peek() in ("+", "-"):
            op = take()
            r = multiplicative()
            e = e + r if op == "+" else e - r
        return e

    def multiplicative() -> Expr:
        e = unary()
        while peek() in ("*", "floordiv", "//", "mod", "%"):
            op = take()
            r = unary()
            if op == "*":
                e = e * r
            else:
                if not isinstance(r, Const):
                    raise ValueError(f"{op} needs a constant right-hand side in {text!r}")
                e = e // r.k if op in ("floordiv", "//") else e % r.k
        return e

    def unary() -> Expr:
        if peek() == "-":
            take()
            return -unary()
        return atom()

    def atom() -> Expr:
        tok = take() if peek() is not None else None
        if tok is None:
            raise ValueError(f"unexpected end of {text!r}")
        if tok == "(":
            e = additive()
            if take() != ")":
                raise ValueError(f"missing ')' in {text!r}")
            return e
        if tok.isdigit():
            return Const(int(tok))
        if tok.startswith("t"):
            return Var(int(tok[1:]))
        raise ValueError(f"unexpected {tok!r} in {text!r}")

    e = additive()
    if i != len(toks):
        raise ValueError(f"trailing input in {text!r}: {' '.join(toks[i:])}")
    return e


def substitute(e: Expr, k: int, repl: Expr) -> Expr:
    """``e`` with every ``t{k}`` replaced by ``repl``."""
    if isinstance(e, Var):
        return repl if e.k == k else e
    if isinstance(e, Const):
        return e
    if isinstance(e, Add):
        return Add(substitute(e.a, k, repl), substitute(e.b, k, repl))
    if isinstance(e, Mul):
        return Mul(substitute(e.a, k, repl), e.k)
    if isinstance(e, FloorDiv):
        return FloorDiv(substitute(e.a, k, repl), e.k)
    if isinstance(e, Mod):
        return Mod(substitute(e.a, k, repl), e.k)
    raise TypeError(f"unknown expression {e!r}")
