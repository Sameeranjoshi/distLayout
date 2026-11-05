import json
import re

from distlayout import Block, Box, Decomposition, Grid, Map, Replicate, UnevenBlock, bind
from distlayout.viz import air_code, render_html, render_index, spec_code


class Buf:
    def __init__(self, *shape):
        self.shape = shape


def model_of(page):
    return json.loads(re.search(r"const M=(\{.*?\});\n", page, re.S).group(1).replace("<\\/", "</"))


def test_block_replicate_page_model():
    v = bind(Buf(100), Decomposition(Block(0, 4)), Map(Grid((4, 4)), {0: 0}, replicate=[1]))
    page = render_html(v, title="t")
    m = model_of(page)
    assert len(m["parts"]) == 4 and m["parts"][2]["holders"] == [8, 9, 10, 11]
    assert [p["color"] for p in m["parts"]] == ["#3978CF", "#E59B38", "#249A82", "#A36BC6"]
    assert sum(not pe["primary"] for pe in m["pes"]) == 12
    assert "l2[tx*25 : tx*25 + 25]" in page            # the derived affine window
    assert "<!doctype html>" in page and "http" not in page.split("<body>")[0]


def test_halo_local_view_marks_ghosts():
    v = bind(Buf(48), Decomposition(Block(0, 4)), Map(Grid(4), {0: 0}))
    m = model_of(render_html(v, requests={1: Box((11,), (14,))}))
    pieces = m["local"][1]["pieces"]
    assert [(p["local"], p["src"]) for p in pieces] == [(False, 0), (True, 1), (False, 2)]


def test_uneven_page_explains_refusal():
    v = bind(Buf(99), Decomposition(UnevenBlock(0, parts=4)), Map(Grid(4), {0: 0}))
    code, table = air_code(v)
    assert "uneven" in code and table == []
    assert "uneven" in render_html(v)


def test_spec_code_round_trips():
    from distlayout import DataSource  # noqa: F401  (names used by the generated code)
    for v in [bind(Buf(16, 32), Decomposition(Block(0, 2), Block(1, 2)), Map(Grid((2, 2)), {0: 0, 1: 1})),
              bind(Buf(100), Decomposition(), Replicate(Grid((4, 4)))),
              bind(Buf(99), Decomposition(UnevenBlock(0, parts=4)), Map(Grid((4, 1)), {0: 0}, replicate=[1]))]:
        env = {"buf": Buf(*v.shape)}
        exec("from distlayout import *\n" + spec_code(v), env)
        assert env["view"].to_attr() == v.to_attr()


def test_index_page():
    assert "block.html" in render_index([("block.html", "Block", "d")])
