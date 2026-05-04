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

Lower `std_ratio` removes more points. `std_ratio=2.0` is a good first pass for
clearly isolated noise because it rejects points beyond roughly two standard
deviations of the mean-neighbor-distance distribution. `std_ratio=3.0` is more
conservative and is often better for preserving sparse edges, rooflines, wires,
or other legitimate low-density structures. `nb_neighbors` must be at least 2;
larger values smooth the density estimate but can hide small isolated clusters.

## Radius Outlier Removal

```python
clean, kept = pc.remove_radius_outlier(
    nb_points=5,
    radius=0.1,
)
```

ROR keeps points that have at least `nb_points` other points within `radius`.
Use this when the expected local density is known.

ROR is usually easier to tune when scan spacing is known. Start with `radius`
around one to two times the expected point spacing, then increase `nb_points`
until isolated speckles disappear without eroding thin structures.

## Masks And Attributes

Both functions return `(clean_cloud, kept_mask)`, where `kept_mask` is a boolean
NumPy array with one entry per input point. The mask is useful when paired data
needs the same filter:

```python
clean, kept = pc.remove_statistical_outlier(20, 2.0)
labels_clean = labels[kept]
```

All typed attributes on the input cloud are selected with the same mask, so
`intensity`, `classification`, `gps_time`, RGB channels, and custom attributes
remain aligned with the filtered XYZ coordinates.
