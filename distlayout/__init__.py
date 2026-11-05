"""distlayout: distributed data layouts over existing tensors and buffers.

    Data source + Decomposition + Mapping  --bind-->  DistributedView
    DistributedView + requested region     --plan-->  MaterializationPlan

The core has no dependencies. ``distlayout.backends.air`` lowers a view to
``air.api`` calls and imports MLIR-AIR only when used.
"""

from .decomposition import AxisRule, Block, Decomposition, UnevenBlock
from .expr import parse, t
from .irregular import (CSRPattern, GhostSchedule, bfs_partition, block_partition, grid_graph, inspect,
                        owner_array, random_partition, renumber)
from .layouts import block_cyclic, cyclic, refine, skew
from .mapping import Grid, Map, Mapping, OwnerMap, Replicate, TableMap
from .region import Box, IndexSet
from .view import DataSource, DistributedView, MaterializationPlan, Transfer, bind

__all__ = [
    "AxisRule", "Block", "UnevenBlock", "Decomposition",
    "Grid", "Mapping", "Map", "Replicate", "OwnerMap", "t", "parse",
    "cyclic", "block_cyclic", "skew", "refine",
    "TableMap", "owner_array", "CSRPattern", "GhostSchedule", "inspect",
    "block_partition", "random_partition", "bfs_partition", "renumber", "grid_graph",
    "Box", "IndexSet",
    "DataSource", "DistributedView", "MaterializationPlan", "Transfer", "bind",
]
