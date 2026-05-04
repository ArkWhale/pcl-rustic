# PointCloud

`PointCloud` is the main NumPy-facing API. Coordinates are stored as `float32`
XYZ. `from_xyz()` accepts `float32`, `float64`, `int32`, and `int64` arrays and
autocasts to `float32`.

Attributes preserve their NumPy dtype for `float32`, `float64`, `uint8`,
`uint16`, `uint32`, `int32`, `int64`, and `bool`. Packed covariance attributes
use shape `[N, 6]` with `float32`.

## Construction

```python
import numpy as np
from pcl_rustic import PointCloud

xyz = np.random.randn(1000, 3).astype(np.float32)
pc = PointCloud.from_xyz(xyz)

pc = PointCloud.from_numpy({
    "xyz": xyz,
    "classification": np.random.randint(0, 8, len(xyz), dtype=np.uint8),
    "gps_time": np.arange(len(xyz), dtype=np.float64),
})
```

## Attributes

```python
pc.set_attribute("classification", np.array([2, 6], dtype=np.uint8))
pc.set_intensity(np.array([0.3, 0.8], dtype=np.float32))

classification = pc.get_attribute("classification")
names = pc.attribute_names()
info = pc.attribute_info()  # (name, length, dtype)
```

Reserved LAS-oriented names include `intensity`, `red`, `green`, `blue`,
`classification`, `return_number`, `number_of_returns`, and `gps_time`.

## Selection

```python
ground = pc.select_by_classification([2])
first_returns = pc.select_return_number(1)
bright = pc.select_intensity_range(0.4, 1.0)
low = pc.select_elevation_range(-2.0, 1.0)
tile = pc.crop_aabb([0, 0, -5], [10, 10, 5])
```

`select(mask)` accepts a boolean NumPy mask. `select_indices(indices)` preserves
input order.

## Concatenation

```python
merged = PointCloud.concatenate([pc1, pc2], policy="strict")
merged = PointCloud.concatenate([pc1, pc2], policy="union")
merged = PointCloud.concatenate([pc1, pc2], policy="intersection")
```

`strict` requires the same attribute names and dtypes. `union` zero-fills missing
attributes. `intersection` keeps only attributes present in every input.

## Neighbors And Normals

```python
from pcl_rustic import NormalSearch

indices, distances = pc.knn(query_xyz, k=16)
hits = pc.radius_search(query_xyz, radius=0.5)

pc.estimate_normals(NormalSearch.knn(30))
normals = np.column_stack([
    pc.get_attribute("nx"),
    pc.get_attribute("ny"),
    pc.get_attribute("nz"),
])

pc.estimate_covariances(knn=30)
cov = pc.get_attribute("covariance")  # shape [N, 6]
```

## Outlier Removal

```python
clean, kept_mask = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
clean, kept_mask = pc.remove_radius_outlier(nb_points=5, radius=0.1)
```

The mask is a boolean array with one entry per input point.

## Device

```python
cpu_cloud = pc.to("cpu")
gpu_cloud = pc.to("gpu")
print(pc.device())
```

Tensor-backed coordinate operations use Burn devices. Attribute vectors remain
host-side to preserve typed LAS attributes.

