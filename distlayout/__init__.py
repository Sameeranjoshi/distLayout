"""distlayout: distributed data layouts over existing tensors and buffers.

    Data source + Decomposition + Mapping  --bind-->  DistributedView
    DistributedView + requested region     --plan-->  MaterializationPlan

The core has no dependencies. ``distlayout.backends.air`` lowers a view to
``air.api`` calls and imports MLIR-AIR only when used.
"""

from .decomposition import AxisRule, Block, Decomposition, UnevenBlock
from .mapping import Grid, Map, Mapping, Replicate
from .region import Box
from .view import DataSource, DistributedView, MaterializationPlan, Transfer, bind

__all__ = [
    "AxisRule", "Block", "UnevenBlock", "Decomposition",
    "Grid", "Mapping", "Map", "Replicate",
    "Box",
    "DataSource", "DistributedView", "MaterializationPlan", "Transfer", "bind",
]
