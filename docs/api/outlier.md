# Outlier Removal

Outlier removal uses the KD-tree neighbor infrastructure and returns both the
filtered cloud and a boolean kept mask.

## Statistical Outlier Removal

```python
clean, kept = pc.remove_statistical_outlier(
    nb_neighbors=20,
    std_ratio=2.0,
)
```

For each point, SOR computes the mean distance to its nearest neighbors. Points
whose mean distance is greater than global mean plus `std_ratio` standard
deviations are removed.

Lower `std_ratio` removes more points. `nb_neighbors` must be at least 2.

## Radius Outlier Removal

```python
clean, kept = pc.remove_radius_outlier(
    nb_points=5,
    radius=0.1,
)
```

ROR keeps points that have at least `nb_points` other points within `radius`.
Use this when the expected local density is known.

