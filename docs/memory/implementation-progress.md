# Implementation Progress — RFC 0002-0007

## Current Status: RFC-0002/0003/0005/0006/0007 substantially implemented

Last updated: 2026-05-04.

### Commits Made This Session

1. `59d5d5a Implement neighbor outlier and registration core`
   - Added KD-tree wrapper, octree, normal/covariance estimation, SOR/ROR, and registration core.
2. `2509bef Expose RFC APIs to Python tests and CI`
   - Exposed Python APIs, updated stubs/exports, added pytest coverage, and changed CI workflows to run `just ci`.

### Verified

- `cargo test` passes: 16/16 Rust unit tests.
- `uv run maturin develop` succeeded after cache permission escalation.
- `uv run pytest tests/test_point_cloud.py -v` passes: 40/40 Python integration tests.

Note: local shell does not have `just` installed, so `just test` / `just ci` could not be invoked directly. Equivalent steps were run manually. CI installs `just` before invoking `just ci`.

## Implemented By RFC

### RFC-0002 — API Reset & Typed Attributes

- `AttributeValue` supports `F32`, `F64`, `U8`, `U16`, `U32`, `I32`, `I64`, `Bool`.
- Added `F32x6` for RFC-0007 packed covariance attribute storage.
- XYZ accepts `float32`, `float64`, `int32`, `int64` NumPy inputs and stores `float32`.
- Attributes preserve dtype through `set_attribute()` / `get_attribute()`.
- Voxel strategies exposed as:
  - `RANDOM_SEEDED`
  - `NEAREST_TO_CENTROID`
  - `AVERAGE`
- Legacy aliases `RANDOM` and `CENTROID` remain.
- Python `.pyi` and `__init__.py` updated for current API.

### RFC-0003 — Coordinate Ops & Selection

- Implemented and exposed:
  - `select(mask)`
  - `select_indices(indices)`
  - `select_by_classification(codes)`
  - `select_return_number(n)`
  - `select_intensity_range(lo, hi)`
  - `select_elevation_range(lo, hi)`
  - `crop_aabb(min, max)`
  - `aabb()`
  - `PointCloud.concatenate(clouds, policy)`
  - `translate`, `scale`, `rotate`, `rigid_transform`, `transform`
- Added examples:
  - `examples/split_grid_downsample_concat.py`
  - `examples/classification_aware_downsample.py`

### RFC-0004 — GPU Hot Path

- Added device hooks:
  - Rust `HighPerformancePointCloud::to_device(...)`
  - Python `pc.to("cpu" | "gpu")`
  - Python `pc.device()`
- Transform path continues to use Burn tensors.
- Full RFC-0004 tensor-native voxel binning / segment-reduce rewrite is **not complete**. Current voxel, selection, and neighbor algorithms remain CPU/reference implementations for correctness.

### RFC-0005 — KD-tree, Octree, Normals

- Added `src/neighbors/`:
  - `kdtree.rs`
  - `octree.rs`
  - `normals.rs`
- Python APIs:
  - `pc.knn(query, k) -> (indices, distances)`
  - `pc.radius_search(query, radius) -> list[np.ndarray]`
  - `pc.octree(max_depth)`
  - `octree.range_search(center, radius)`
  - `octree.voxel_centers()`
  - `pc.estimate_normals(NormalSearch.knn/radius/hybrid(...))`
- KD-tree uses `kiddo` when geometry is suitable.
- Degenerate geometry fallback is documented in RFC-0008.

### RFC-0006 — Outlier Removal

- Implemented:
  - `remove_statistical_outlier(nb_neighbors, std_ratio) -> (cloud, kept_mask)`
  - `remove_radius_outlier(nb_points, radius) -> (cloud, kept_mask)`
- Rust and Python tests cover synthetic outliers and mask propagation.

### RFC-0007 — ICP/GICP Registration

- Added `src/registration.rs`.
- Python submodule `pcl_rustic.registration` exposes:
  - `ICPConvergenceCriteria`
  - `TransformationEstimation.point_to_point()`
  - `TransformationEstimation.point_to_plane()`
  - `TransformationEstimation.generalized(epsilon=1e-3)`
  - `RegistrationResult`
  - `registration.icp(...)`
  - `registration.evaluate(...)`
- Point-to-point ICP is implemented and tested.
- Point-to-plane validates required target normals.
- GICP validates required source/target packed covariance attributes.
- `estimate_covariances(knn)` writes packed `covariance` as `float32[N, 6]`.
- Full covariance-weighted GICP update is staged; see RFC-0008.

## Docs Updated

- Added:
  - `docs/api/outlier.md`
  - `docs/api/registration.md`
  - `docs/plans/rfc-0008-2026-05-04-kdtree-fallback-and-gicp-staging.md`
- Rewrote:
  - `docs/api/pointcloud.md`
  - `docs/api/downsample.md`
- Updated:
  - `docs/api/overview.md`
  - `docs/getting-started/examples.md`
  - `mkdocs.yml` nav includes RFCs and new API pages.
- Replaced most stale `YOUR_USERNAME` and deprecated downsample strategy references in docs/README.

## Important Open Work

1. Finish RFC-0004 tensor-native voxel binning and GPU benchmark matrix.
2. Replace staged GICP update with covariance-weighted plane-to-plane solve.
3. Implement `crop_obb` / `obb()` if strict RFC-0003 completion is required.
4. Run full `just ci` in an environment with `just` installed.
5. Run full docs build after docs dependency sync.

## Architecture Decisions Made

| Decision | Rationale |
|---|---|
| `AttributeValue` uses `Vec<T>` for typed attrs | Burn Router backend does not carry all LAS attribute dtypes directly. |
| Added `F32x6` packed covariance variant | RFC-0007 needs `[N, 6]` covariance storage while preserving typed attribute boundary. |
| KD-tree falls back to brute-force on degenerate axis buckets | `kiddo` can panic on many identical values along an axis; fallback preserves correctness. |
| GICP API staged behind covariance validation | Keeps RFC API usable while avoiding an unverified covariance-weighted solver. |
| `just ci` no longer depends on pre-commit | CI should use pytest/cargo directly per user instruction; pre-commit remains separately available. |

