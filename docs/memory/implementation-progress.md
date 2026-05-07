# Implementation Progress - RFC 0002-0010

## Current Status: API Surface Broadly Present; Acceptance Criteria Still Partial

Last updated: 2026-05-07.

This review used read-only subagent audits split across RFC-0002 through RFC-0005
and RFC-0006 through RFC-0009, plus a local pass over source, tests, docs,
examples, and automation. RFC-0010 was added on 2026-05-07 as a storage
amendment to RFC-0002.

| RFC | Status | Summary |
|---|---|---|
| RFC-0002 API Reset & Typed Attributes | Partial, amended by RFC-0010 | Typed attributes, host typed storage, and new downsample strategies exist; XYZ getter optimization / getter benchmarks are not implemented. |
| RFC-0003 Coordinate Ops & Selection | Partial | Selection, generic attribute filters, AABB/OBB crop, concat, transform wrappers, examples, and LAS standard-attribute round-trip coverage exist; fixture-backed examples and device-native selection are still missing. |
| RFC-0004 GPU Hot Path | Mostly missing | Device transfer hooks exist, but voxel downsample, selection, concat, and benchmarks remain CPU/reference paths. |
| RFC-0005 KD-tree, Octree, Normals | Partial | KD-tree, fallback, octree, Python APIs, normals, covariance support, 10k brute-force oracle tests, 10k plane-normal coverage, and KD-tree cache behavior tests exist; pruning, parallel normals, and benchmarks are still missing. |
| RFC-0006 Outlier Removal | Partial, amended by RFC-0010 | SOR/ROR APIs, host masks, docs, synthetic acceptance tests, typed attribute propagation, and LAS standard-attribute propagation coverage exist; custom LAS ExtraBytes propagation and 10M benchmark are missing. |
| RFC-0007 ICP/GICP Registration | Partial | Registration API, point-to-point ICP, evaluate, covariance storage, and prerequisite validation exist; point-to-plane/GICP solvers are staged, not complete. |
| RFC-0008 KD-tree Fallback & GICP Staging | Mostly implemented | Fallback behavior and staged GICP decision are implemented; full covariance-weighted GICP remains open by design. |
| RFC-0009 Large-Scale Benchmark Suite | Mostly implemented | Smoke/standard/full modes, concat/downsample matrices, typed attrs, CSV output, just recipes, CI smoke job, and docs regeneration support are implemented; standard/full benchmark runs remain unexecuted. |
| RFC-0010 Host Typed Attribute Storage Amendment | Accepted | Host typed attribute storage is documented as the accepted RFC-0002 storage model, with cross-RFC amendments for selection, GPU scope, outlier masks, and covariance storage. |

## Implemented Evidence

### RFC-0002 - API Reset & Typed Attributes

- `src/point_cloud/attribute_value.rs` defines `AttributeValue` for `F32`,
  `F64`, `U8`, `U16`, `U32`, `I32`, `I64`, `Bool`, plus `F32x6`.
- Intensity and RGB are standard attributes; `HighPerformancePointCloud` has
  `xyz`, `attributes`, and `kdtree_cache` rather than dedicated intensity/RGB
  tensor fields.
- `PointCloud.from_xyz` accepts `float32`, `float64`, `int32`, and `int64`
  NumPy inputs and stores XYZ as `float32`.
- `set_attribute()` / `get_attribute()` preserve NumPy dtypes for covered
  scalar attributes.
- `DownsampleStrategy.RANDOM_SEEDED`, `NEAREST_TO_CENTROID`, and `AVERAGE`
  are exposed, legacy `RANDOM` / `CENTROID` aliases are removed, and seeded
  determinism is tested.
- Python CSV/Parquet/load/save wrappers are exposed for the existing Rust table
  I/O implementation.
- 1D NumPy attribute inputs now accept strided views, so common expressions such
  as `rgb[:, 0]` work without caller-side copies.
- Attribute length validation now applies to zero-point clouds, so empty clouds
  reject non-empty typed attributes instead of accepting impossible schemas.
- RFC-0010 accepts host `Vec<T>` typed attribute storage as the RFC-0002
  storage contract while leaving XYZ tensor-backed.
- `pyproject.toml` now points to `README.md`, and RFC docs are in MkDocs nav.

### RFC-0003 - Coordinate Ops & Selection

- Implemented and exposed:
  - `select(mask)`
  - `select_indices(indices)`
  - `select_where(name, op, values, inclusive=True)`
  - `select_by_classification(codes)`
  - `select_return_number(n)`
  - `select_intensity_range(lo, hi)`
  - `select_elevation_range(lo, hi)`
  - `crop_aabb(min, max)`
  - `aabb()`
  - `crop_obb(center, extents, rotation)`
  - `obb()`
  - `PointCloud.concatenate(clouds, policy)`
  - `translate`, `scale`, `rotate`, `rigid_transform`, `transform`
- `PointCloud.concatenate` supports `strict`, `union`, and `intersection`.
- Tests cover single-cloud concat, intersection policy, missing-attribute strict
  rejection, and dtype-mismatch strict rejection.
- Empty selections now preserve typed attribute schemas, zero-point XYZ tensors
  use the CPU backend to avoid WGPU zero-size resource panics, and strict concat
  can round-trip an empty selected partition with a non-empty cloud.
- LAS export now explicitly finalizes the writer header, and tests cover
  `select_by_classification([2, 6]) -> to_las -> from_las` with classification,
  return-number, GPS time, and RGB preservation.
- Added examples:
  - `examples/split_grid_downsample_concat.py`
  - `examples/classification_aware_downsample.py`
- `docs/getting-started/examples.md` references both examples.

### RFC-0004 - GPU Hot Path

- Added device hooks:
  - Rust `HighPerformancePointCloud::to_device(...)`
  - Python `pc.to("cpu" | "gpu")`
  - Python `pc.device()`
- Burn Router backend is configured with WGPU and CPU support.
- Transform still uses Burn tensors, but the main hot paths are not yet
  tensor-native.

### RFC-0005 - KD-tree, Octree, Normals

- Added neighbor modules:
  - `src/neighbors/kdtree.rs`
  - `src/neighbors/octree.rs`
  - `src/neighbors/normals.rs`
- Python APIs:
  - `pc.knn(query, k) -> (indices, distances)`
  - `pc.radius_search(query, radius) -> list[np.ndarray]`
  - `pc.octree(max_depth)`
  - `octree.range_search(center, radius)`
  - `octree.voxel_centers()`
  - `pc.estimate_normals(NormalSearch.knn/radius/hybrid(...))`
- KD-tree uses `kiddo` when geometry is suitable and falls back to deterministic
  brute-force queries for degenerate axis buckets.
- `PointCloud` has a lazy `OnceCell` KD-tree cache.
- KD-tree kNN/radius search and octree range search now have deterministic
  10k-point random-cloud oracle tests against brute force.
- KD-tree cache behavior is covered: repeated immutable access reuses the
  cached index, and mutable XYZ access invalidates stale cache state before the
  next query.
- Empty KD-tree/octree input errors and non-finite KD-tree input errors are
  covered.
- Normal estimation coverage now uses a 10k-point synthetic plane and checks
  unit-length normals within `1e-5` with at least 99% alignment to the plane
  normal.

### RFC-0006 - Outlier Removal

- Implemented:
  - `remove_statistical_outlier(nb_neighbors, std_ratio) -> (cloud, kept_mask)`
  - `remove_radius_outlier(nb_points, radius) -> (cloud, kept_mask)`
- Python bindings return NumPy boolean masks.
- Attribute propagation is supported through `AttributeValue::select_mask`.
- Tests cover empty-input errors, injected isolated SOR outliers, mask sum
  consistency, and typed attribute propagation through ROR.
- LAS round-trip coverage verifies that standard LAS attributes survive
  selection, radius outlier removal, LAS export, and reload.
- `docs/api/outlier.md` exists.
- `examples/classification_aware_downsample.py` includes SOR and ROR as optional
  cleaning steps.

### RFC-0007 - ICP/GICP Registration

- Added `src/registration.rs`.
- Python submodule `pcl_rustic.registration` exposes:
  - `ICPConvergenceCriteria`
  - `TransformationEstimation.point_to_point()`
  - `TransformationEstimation.point_to_plane()`
  - `TransformationEstimation.generalized(epsilon=1e-3)`
  - `RegistrationResult`
  - `registration.icp(...)`
  - `registration.evaluate(...)`
- Point-to-point ICP is implemented and tested for identity and known
  translation cases, plus a 10k-point known-rotation recovery case.
- ICP now recomputes correspondences after each accepted delta before returning
  iteration metrics, so `fitness`, `inlier_rmse`, and correspondence sets
  describe the returned transform.
- Point-to-plane validates required target normals.
- GICP validates required source/target packed covariance attributes.
- `estimate_covariances(knn)` writes packed `covariance` as `float32[N, 6]`.
- `docs/api/registration.md` documents the staged GICP behavior.

### RFC-0008 - KD-tree Fallback & GICP Staging

- `KdTreeIndex::build` detects axis buckets over the configured limit and skips
  `kiddo` for those clouds.
- `knn` and `radius_search` use sorted brute-force queries under fallback.
- Dedicated fallback tests assert deterministic `knn` and `radius_search`
  results for degenerate axis-bucket geometry.
- Plane normal estimation no longer panics on the covered axis-aligned plane
  case.
- GICP prerequisite errors are implemented and covered by Python tests.

### RFC-0009 - Large-Scale Benchmark Suite

- `tests/test_benchmark.py` implements RFC-0009 benchmark modes:
  - smoke: C20 with 20 clouds x 1M points, plus D20 scaled to 2M points.
  - standard: full C20 target plus D20/D50/D100 downsampling.
  - full: C20-C200 concat matrix plus D20-D400 downsampling matrix.
- Full concat matrix covers 20, 40, 80, 120, 160, and 200 clouds with 10M
  points per cloud.
- Full downsampling matrix covers 20M, 50M, 100M, 200M, and 400M points with
  voxel sizes `0.05`, `0.15`, and `0.50` across `NEAREST_TO_CENTROID`,
  `AVERAGE`, and `RANDOM_SEEDED`.
- Synthetic benchmark data includes RFC-required typed attributes:
  `intensity: float32`, `classification: uint8`, `return_number: uint8`, and
  `gps_time: float64`.
- Benchmark runs write fresh CSV output under `reports/benchmarks/` with the
  RFC-0009 section 3.3 columns.
- Added `just benchmark-smoke`, `just benchmark-standard`,
  `just benchmark-full`, and `just benchmark-docs`; `just benchmark` remains a
  smoke alias.
- `.github/workflows/test.yml` runs `just benchmark-smoke` in CI and requires an
  explicit `allow_expensive_benchmarks=true` input for standard/full manual
  dispatches.
- `tools/render_benchmark_docs.py` regenerates `docs/performance/benchmarks.md`
  from recorded CSV output without fabricating performance numbers.

## Important Gaps By RFC

This section is the current implementation backlog. It intentionally separates
code gaps, acceptance-test gaps, and benchmark/evidence gaps so RFC status is
not upgraded based on API presence alone.

### RFC-0002

- Code gap: `get_xyz()` clones/materializes arrays; the XYZ getter optimization
  path is not implemented.
- Evidence gap: no 10M-point XYZ getter benchmark or documented speedup.
- External-doc gap: `multica-home/knowledge/projects/pcl-rustic.md` is not
  present in this repo.

### RFC-0003

- Code gap: selection and concat still need XYZ device-residency optimization
  before RFC-0004 can claim a GPU hot path.
- Test gap: selectors beyond `select_by_classification` still lack LAS
  fixture-backed coverage.
- Fixture gap: no reusable `tests/data` LAS fixture exists.
- Example gap: end-to-end examples are not fixture-backed and do not assert
  expected point-count reductions.

### RFC-0004

- Code gap: voxel downsample materializes host XYZ and groups points with
  `HashMap`; tensor-native voxel binning is not implemented.
- Code gap: `select` and `concatenate` do not use Burn `nonzero`,
  `select_dim`, or `cat`.
- Test gap: no GPU device-residency pipeline test.
- Test gap: no CPU/GPU golden tests for point counts, centroids, or seeded
  randomness.
- Benchmark gap: the benchmark harness is not the RFC backend matrix.
- Evidence gap: no 50M LAZ GPU-vs-CPU 3x speedup result.

### RFC-0005

- Code gap: octree `range_search` brute-forces over all XYZ rather than pruning
  via octree cells, though correctness is covered by a 10k brute-force oracle.
- Code gap: normal estimation is not parallelized with `rayon`.
- Benchmark gap: no 10M-point kNN result is recorded in
  `docs/performance/benchmarks.md`.

### RFC-0006

- Code/test gap: LAS ExtraBytes/custom-attribute propagation is still missing.
- Benchmark gap: no SOR 10M benchmark evidence.

### RFC-0007

- Code gap: point-to-plane and GICP are API-staged; they validate
  prerequisites but reuse the point-to-point closed-form update.
- Code gap: full covariance-weighted GICP plane-to-plane solve is not
  implemented.
- Test gap: Open3D Bunny comparison test is missing.
- Benchmark gap: 500k-vs-500k registration benchmark is missing.
- Documentation gap: registration docs describe staged GICP behavior, but full
  estimator examples and Open3D migration notes remain incomplete until the
  solvers are real.

### RFC-0008

- Follow-up gap: the full covariance-weighted GICP solver remains open by
  design and belongs with RFC-0007 solver completion.

### RFC-0009

- Evidence gap: standard and full benchmark modes have not been executed
  locally; they require high-memory benchmark hardware.
- Documentation gap: the docs renderer exists, but release benchmark results
  still need to be produced on recorded hardware before publishing measured
  performance rows.

## 2026-05-04 Cleanup Pass

- Removed the legacy Python `DownsampleStrategy.RANDOM` and
  `DownsampleStrategy.CENTROID` aliases from the PyO3 class and `.pyi` stub.
- Updated tests and examples to use `RANDOM_SEEDED`,
  `NEAREST_TO_CENTROID`, and explicit NumPy arrays where the API requires them.
- Exposed Python wrappers for the existing Rust table I/O implementation:
  `from_csv`, `to_csv`, `from_parquet`, `to_parquet`, `load_from_file`, and
  `save_to_file`.
- Added round-trip coverage for CSV/Parquet table I/O and an assertion that the
  old downsample aliases are no longer exported.
- Updated docs to stop claiming current zero-copy getters, clarify current
  standard-attribute table export behavior, and point README roadmap readers to
  the RFC index.
- Updated RFC status headers and acceptance checkboxes:
  - RFC-0001 is now marked `Active roadmap`.
  - RFC-0002, RFC-0003, RFC-0005, RFC-0006, and RFC-0007 are marked `Partial`.
  - RFC-0008 is marked `Implemented (staged GICP follow-up open)`.
  - RFC-0009 is marked `Implemented`.

## Verification Snapshot

Fresh verification from the 2026-05-07 RFC-0005 KD-tree cache pass:

- Red check before implementation: `cargo test xyz_mut_invalidates_cached_kdtree --lib`
  failed because kNN after `xyz_mut()` still answered from the stale cached
  index (`left: 1`, `right: 0`).
- Green focused checks after implementation:
  - `cargo test xyz_mut_invalidates_cached_kdtree --lib` passed.
  - `cargo test repeated_kdtree_access_reuses_cached_index --lib` passed.
  - `cargo test neighbors::kdtree::tests --lib` passed: 7/7 KD-tree tests.
  - `cargo test --lib` passed: 25/25 Rust unit tests.

Fresh verification from the 2026-05-05 RFC-0003 empty-selection pass:

- Red check before implementation: `uv run pytest tests/test_point_cloud.py::TestSelectionAndConcatenation::test_empty_selection_preserves_attribute_schema_for_strict_concat -q --no-cov` failed with a WGPU `0 size resources are not yet supported` panic.
- Green focused checks after implementation:
  - `uv run pytest tests/test_point_cloud.py::TestSelectionAndConcatenation::test_empty_selection_preserves_attribute_schema_for_strict_concat -q --no-cov` passed.
  - `uv run ruff check tests/test_point_cloud.py` passed.
  - `cargo test --lib` passed: 17/17 Rust unit tests. Rust emitted existing dead-code warnings.
  - `uv run pytest tests/test_point_cloud.py::TestSelectionAndConcatenation -q --no-cov` passed: 4/4 Python selection/concat tests.

Fresh verification from the 2026-05-05 RFC-0002 empty-attribute validation pass:

- Red check before implementation: `uv run pytest tests/test_point_cloud.py::TestPointCloudProperties::test_empty_point_cloud_rejects_non_empty_attribute -q --no-cov` failed because no `ValueError` was raised.
- Green focused check after implementation: `uv run pytest tests/test_point_cloud.py::TestPointCloudProperties::test_empty_point_cloud_rejects_non_empty_attribute -q --no-cov` passed.

Fresh verification from the 2026-05-05 RFC-0007 ICP metric consistency pass:

- Red check before implementation: `cargo test single_iteration_metrics_match_returned_transform --lib` failed because one-iteration ICP returned non-zero RMSE for the already-updated transform.
- Green checks after implementation:
  - `cargo test single_iteration_metrics_match_returned_transform --lib` passed.
  - `cargo test registration::tests --lib` passed: 4/4 registration unit tests.
  - `cargo test --lib` passed: 18/18 Rust unit tests. Rust emitted existing dead-code warnings.

Fresh verification from the 2026-05-05 RFC-0003/RFC-0006 LAS round-trip pass:

- Red check before implementation: `uv run pytest tests/test_point_cloud.py::TestTableIo::test_las_classification_selection_and_outlier_round_trip -q --no-cov` failed because `from_las` reloaded zero points from a file whose writer header had not been finalized.
- Green focused check after implementation: `uv run pytest tests/test_point_cloud.py::TestTableIo::test_las_classification_selection_and_outlier_round_trip -q --no-cov` passed.

Fresh verification from the 2026-05-05 RFC-0005 neighbor-oracle pass:

- First check after adding tests: `cargo test random_10k_queries_match_bruteforce_oracle --lib` and `cargo test random_10k_range_search_matches_bruteforce_oracle --lib` initially failed to compile because `unwrap_err()` required debug formatting for success types. The tests were adjusted to pattern-match errors directly.
- Green focused checks after test fix:
  - `cargo test random_10k_queries_match_bruteforce_oracle --lib` passed.
  - `cargo test random_10k_range_search_matches_bruteforce_oracle --lib` passed.

Fresh verification from the 2026-05-05 RFC-0007 known-rotation pass:

- `cargo test known_rotation_10k_is_recovered --lib` passed.

Fresh verification from the 2026-05-05 RFC-0005 normal-estimation pass:

- `cargo test plane_normals_are_unit_and_axis_aligned --lib` passed against a 10k-point synthetic plane.

Fresh verification from the 2026-05-04 cleanup pass:

- `cargo test --lib` passed: 17/17 Rust unit tests. Rust emitted existing
  dead-code warnings for unused helpers and staged fields.
- `uv run ruff check tests/test_point_cloud.py examples/basic_usage.py` passed.
- `uv run pytest tests/test_point_cloud.py -q --no-cov` passed: 46/46 Python
  integration tests.
- `uv run --project /Users/lz/Codes/github/pcl-rustic python
  /Users/lz/Codes/github/pcl-rustic/examples/basic_usage.py` completed from
  `/private/tmp`.
- `uv run mkdocs build` could not start because the docs group was not
  installed in the active environment. `uv run --group docs mkdocs build`
  downloaded dependencies but did not complete before the session was closed at
  user request.

Previous session evidence recorded:

- `cargo test` passed: 16/16 Rust unit tests.
- `uv run maturin develop` succeeded after cache permission escalation.
- `uv run pytest tests/test_point_cloud.py -v` passed: 40/40 Python integration
  tests.

The RFC-0009 implementation slice also ran focused verification:

- `uv run ruff format tests/test_benchmark.py tests/conftest.py
  tools/render_benchmark_docs.py`
- `uv run ruff check tests/test_benchmark.py tests/conftest.py
  tools/render_benchmark_docs.py`
- `uv tool run --from rust-just just --summary`
- `uv tool run --from rust-just just --dry-run benchmark-smoke`
- `uv run pytest tests/test_benchmark.py --no-cov` confirmed benchmark tests
  remain skipped without `--run-slow`.
- A matrix sanity script confirmed RFC-0009 smoke, standard, and full case
  definitions.
- `tools/render_benchmark_docs.py` was exercised against an empty CSV directory.
- RFC-0003 selection/OBB focused checks:
  - `cargo fmt`
  - `uv run ruff format tests/test_point_cloud.py`
  - `cargo test selection --lib`
  - `uv run pytest tests/test_point_cloud.py::TestSelectionAndConcatenation::test_feature_and_spatial_selection -v --no-cov`
- RFC-0006 outlier focused checks:
  - `uv run ruff format tests/test_point_cloud.py
    examples/classification_aware_downsample.py`
  - `uv run pytest tests/test_point_cloud.py::TestNeighborsNormalsOutliersRegistration::test_outlier_empty_input_errors tests/test_point_cloud.py::TestNeighborsNormalsOutliersRegistration::test_statistical_outlier_removes_injected_outliers tests/test_point_cloud.py::TestNeighborsNormalsOutliersRegistration::test_outlier_preserves_typed_attributes -v --no-cov`
  - `uv run python examples/classification_aware_downsample.py`
- RFC-0008 fallback focused checks:
  - `cargo fmt`
  - `cargo test degenerate_axis_uses_deterministic_fallback --lib`
- RFC-0003 concat focused checks:
  - `uv run ruff format tests/test_point_cloud.py`
  - `uv run pytest tests/test_point_cloud.py::TestSelectionAndConcatenation::test_concatenate_policies tests/test_point_cloud.py::TestSelectionAndConcatenation::test_concatenate_edge_cases -v --no-cov`

Note: a prior local shell did not have `just` installed, so `just test` /
`just ci` could not be invoked directly in that session. Equivalent steps were
run manually then. CI installs `just` before invoking `just ci`.

## Next Implementation Priorities

1. RFC-0004: implement tensor-native XYZ-heavy voxel binning, selection, concat,
   and the CPU/GPU golden tests needed before claiming a GPU hot path.
2. RFC-0007/RFC-0008: replace staged point-to-plane/GICP updates with actual
   solvers, then add Open3D comparison and registration benchmark coverage.
3. RFC-0002: implement the remaining XYZ getter optimization path and benchmark
   evidence if getter performance remains a release criterion.
4. RFC-0003: add a small reusable LAS fixture, run fixture-backed examples, and
   cover remaining LAS selectors.
5. RFC-0005/RFC-0006: add octree pruning, LAS ExtraBytes propagation, and the
   required 10M kNN/SOR benchmark evidence.
6. RFC-0009: run standard/full benchmark modes on recorded high-memory hardware
   and regenerate benchmark docs from the resulting CSV files.

## Architecture Decisions Recorded

| Decision | Rationale |
|---|---|
| `AttributeValue` uses host `Vec<T>` for typed attrs | Accepted by RFC-0010 so LAS/NumPy dtypes remain exact while XYZ stays tensor-backed. |
| Added `F32x6` packed covariance variant | RFC-0007 needs `[N, 6]` covariance storage while preserving the RFC-0010 host typed attribute boundary. |
| KD-tree falls back to brute-force on degenerate axis buckets | `kiddo` can panic on many identical values along an axis; fallback preserves correctness. |
| GICP API staged behind covariance validation | Keeps RFC API usable while avoiding an unverified covariance-weighted solver. |
| `just ci` no longer depends on pre-commit | CI should use pytest/cargo directly per user instruction; pre-commit remains separately available. |
