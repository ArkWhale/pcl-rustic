"""
高性能Python点云运算库 - pcl-rustic

基于Burn张量库的批量张量运算，支持LAZ/LAS/Parquet/CSV多格式I/O
"""

from ._core import (
    DownsampleStrategy,
    NormalSearch,
    Octree,
    PointCloud,
    has_wgpu_device,
    rayon_current_num_threads,
    registration,
)

__version__ = "0.1.0"
__all__ = [
    "PointCloud",
    "DownsampleStrategy",
    "NormalSearch",
    "Octree",
    "has_wgpu_device",
    "rayon_current_num_threads",
    "registration",
]
