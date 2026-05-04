# Downsample

`PointCloud.voxel_downsample(voxel_size, strategy, seed=None)` groups points by
voxel and emits one point per occupied voxel.

## Strategies

| Strategy | Behavior |
|---|---|
| `DownsampleStrategy.RANDOM_SEEDED` | Deterministically samples one existing point per voxel from `seed`. |
| `DownsampleStrategy.NEAREST_TO_CENTROID` | Picks the existing point nearest to the voxel centroid. |
| `DownsampleStrategy.AVERAGE` | Emits a synthetic averaged point. Float attributes are averaged, integer attributes use mode, and bool attributes use majority vote. |

The legacy `RANDOM` and `CENTROID` aliases were removed during the RFC-0002 API reset.

## Example

```python
import numpy as np
from pcl_rustic import DownsampleStrategy, PointCloud

pc = PointCloud.from_xyz(np.random.randn(100_000, 3).astype(np.float32))

preview = pc.voxel_downsample(
    0.5,
    DownsampleStrategy.RANDOM_SEEDED,
    seed=42,
)

geometry = pc.voxel_downsample(
    0.15,
    DownsampleStrategy.NEAREST_TO_CENTROID,
)

averaged = pc.voxel_downsample(
    0.15,
    DownsampleStrategy.AVERAGE,
)
```

`voxel_size` uses the coordinate unit of the input cloud. For LAS data that is
normally meters.
