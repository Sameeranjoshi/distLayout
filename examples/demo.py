"""The five design cases, printed. No MLIR-AIR needed: ``python3 examples/demo.py``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distlayout import Block, Box, Decomposition, Grid, Map, Replicate, UnevenBlock, bind  # noqa: E402


class Buf:
    def __init__(self, *shape):
        self.shape = shape


cases = {
    "block 100 over 4x1": bind(Buf(100), Decomposition(Block(0, 4)), Map(Grid((4, 1)), {0: 0}, replicate=[1])),
    "block + replicate 100 on 4x4": bind(Buf(100), Decomposition(Block(0, 4)), Map(Grid((4, 4)), {0: 0}, replicate=[1])),
    "uneven 99 over 4x1": bind(Buf(99), Decomposition(UnevenBlock(0, parts=4)), Map(Grid((4, 1)), {0: 0}, replicate=[1])),
    "replicate 100 on 4x4": bind(Buf(100), Decomposition(), Replicate(Grid((4, 4)))),
    "2-D block 16x32 on 2x2": bind(Buf(16, 32), Decomposition(Block(0, 2), Block(1, 2)), Map(Grid((2, 2)), {0: 0, 1: 1})),
}

for name, v in cases.items():
    v.validate()
    print(f"== {name}\n{v.to_attr()}\n{v.table()}\n")

v = cases["block + replicate 100 on 4x4"]
print("primary_owner(55) =", v.primary_owner(55))
print("holders(55)       =", v.holders(55))
print("owned_region((2,1)) =", v.owned_region((2, 1)), " held_region((2,1)) =", v.held_region((2, 1)))

v = cases["block 100 over 4x1"]
print("\nplan for PE (1,0) needing [24, 51) (own block + one halo element each side):")
for t in v.plan((1, 0), Box((24,), (27,))).transfers:
    src = "local" if t.local else f"from PE {t.src_pe}"
    print(f"  {t.region} -> local[{t.dst_offset[0]}:]  {src}")
