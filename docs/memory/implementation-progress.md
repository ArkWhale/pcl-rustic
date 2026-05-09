# Implementation Progress - RFC 0002-0018

## Current Status: Repo-Local RFCs Implemented; External Evidence Gates Remain

Last updated: 2026-05-09.

This review used read-only subagent audits split across RFC-0002 through RFC-0005
and RFC-0006 through RFC-0009, plus a local pass over source, tests, docs,
examples, and automation. RFC-0010 was added on 2026-05-07 as a storage
amendment to RFC-0002. RFC-0011 was added on 2026-05-07 to separate
repo-local implementation completion from external evidence gates such as
high-memory benchmarks, cross-platform hosted CI, GPU speedup measurements, and
external workspace documents. External evidence remains unclaimed until a
recorded artifact exists. RFC-0012 was added and accepted on 2026-05-07 to
amend RFC-0004's repo-local GPU hot-path scope to device-preserving XYZ results
with host-side planning and explicit external benchmark evidence gates.
RFC-0013 was added and accepted on 2026-05-07 to require LAS 1.4 point format
10 support with optional public attributes and precision-preserving standard
dimensions. RFC-0014 was added and accepted on 2026-05-08 to pin the explicit
point format 10 export API and waveform metadata drop/reject policy. Commits
`fda5273` and `7a46e3a` implement and test RFC-0013/RFC-0014; the worktree was
clean after those commits. A 2026-05-08 RFC-0002 through RFC-0012 loop found no
remaining repo-local implementation gaps, removed redundant internal helpers in
commit `5cb3284`, and left only the RFC-0011 external evidence gates open.
RFC-0015 was drafted and implemented on 2026-05-08 to add pytest-benchmark
based Open3D comparison benchmarks and Plotly chart rendering from recorded
benchmark artifacts. Measured Open3D comparison results remain external
evidence until generated and recorded under RFC-0011 rules. RFC-0016 was added
and accepted on 2026-05-09 to migrate the tensor backend adapter from Burn
Router to Burn Dispatch while preserving GPU-first behavior and preparing for
Burn's planned backend-parameter removal. RFC-0017 was accepted and then
superseded before implementation by RFC-0018. RFC-0018 amends RFC-0016 to avoid
WGPU/Vulkan for large tensor workloads and use non-WGPU Dispatch backends,
starting with CUDA on Linux plus CPU fallback.

| RFC | Status | Summary |
|---|---|---|
| RFC-0002 API Reset & Typed Attributes | Partial (external evidence open), amended by RFC-0010/RFC-0011 | Typed attributes, host typed storage, dtype-preserving typed getters, and new downsample strategies exist; external Multica docs, cross-platform CI evidence, and XYZ getter benchmark evidence remain open. |
| RFC-0003 Coordinate Ops & Selection | Implemented | Selection, generic attribute filters, AABB/OBB crop, concat, transform wrappers, LAS fixture-backed selector coverage, fixture-backed examples, and LAS standard-attribute round-trip coverage exist. |
| RFC-0004 GPU Hot Path | Implemented (external evidence open), amended by RFC-0010/RFC-0012/RFC-0016 | Device transfer hooks exist; selection, concat, voxel downsample, and transforms now preserve source/common XYZ device with host-side planning; 50M speedup and backend benchmark artifacts remain open. |
| RFC-0005 KD-tree, Octree, Normals | Implemented (external evidence open) | KD-tree, fallback, octree cell-pruned range search, Python APIs, normals, covariance support, 10k brute-force oracle tests, 10k plane-normal coverage, and KD-tree cache behavior tests exist; 10M benchmark evidence remains open. |
| RFC-0006 Outlier Removal | Implemented (external evidence open), amended by RFC-0010/RFC-0011 | SOR/ROR APIs, host masks, docs, synthetic acceptance tests, typed attribute propagation, LAS standard-attribute propagation, and custom ExtraBytes propagation coverage exist; 10M benchmark evidence remains open. |
| RFC-0007 ICP/GICP Registration | Implemented (external evidence open) | Registration API, point-to-point ICP, point-to-plane update, covariance-weighted GICP update, evaluate, covariance storage, and prerequisite validation exist; Open3D comparison and 500k benchmark evidence remain open. |
| RFC-0008 KD-tree Fallback & GICP Staging | Implemented | Fallback behavior and the covariance-weighted GICP follow-up are implemented. |
| RFC-0009 Large-Scale Benchmark Suite | Implemented (external evidence open) | Smoke/standard/full modes, concat/downsample matrices, typed attrs, CSV output, just recipes, CI smoke job, and docs regeneration support are implemented; standard/full benchmark artifacts remain unrecorded. |
| RFC-0010 Host Typed Attribute Storage Amendment | Accepted | Host typed attribute storage is documented as the accepted RFC-0002 storage model, with cross-RFC amendments for selection, GPU scope, outlier masks, and covariance storage. |
| RFC-0011 Completion Evidence Gates | Accepted | RFC tracking now distinguishes repo-local gaps from external evidence gaps and forbids benchmark claims without recorded artifacts. |
| RFC-0012 RFC-0004 Device Residency Scope | Accepted | RFC-0004 repo-local completion now requires source/common-device XYZ results for selection, concat, voxel downsample, and transforms, while measured GPU speedup remains external evidence. |
| RFC-0013 LAS Point Format 10 Precision Support | Implemented | LAS 1.4 point format 10 read/write, optional standard attributes, uint16 RGB/NIR, waveform metadata default/reject/drop behavior, raw coordinate precision sidecars, standard ExtraBytes collision handling, and uint64/int16 typed storage are implemented and tested. |
| RFC-0014 LAS Point Format 10 Export API And Payload Policy | Implemented | Python/Rust-compatible `to_las` policy, `point_format=10`, `las_version`, `drop_waveform`, partial RGB defaults, version validation, unsupported format errors, and waveform metadata policy are implemented and tested. |
| RFC-0015 Open3D Comparison Benchmark Charts | Implemented (external evidence open) | pytest-benchmark based pcl-rustic/Open3D comparison harness, optional benchmark dependency group, just recipes, parser tests, and Plotly chart renderer exist; measured comparison artifacts remain unrecorded. |
| RFC-0016 Burn Dispatch Backend Migration | Implemented, amended by RFC-0018 | Burn 0.21 Dispatch backend is configured, Router is removed, tensor construction helpers are centralized in `src/utils/tensor.rs`, and `.to("gpu")` now reports unavailable GPU as a Python error. |
| RFC-0017 GPU Benchmark Allocation Preflight | Superseded by RFC-0018 | WGPU allocation preflight was not implemented; RFC-0018 replaces it by removing WGPU/Vulkan from the default large-tensor backend path. |
| RFC-0018 Non-WGPU Dispatch Backend Selection | Implemented | Default Linux backend features use Burn Dispatch CUDA + CPU, not WGPU/Vulkan; RFC-0009 smoke rows must record the actual backend before supporting any benchmark claim. |

## Implemented Evidence

### RFC-0002 - API Reset & Typed Attributes

- `src/point_cloud/attribute_value.rs` defines `AttributeValue` for `F32`,
  `F64`, `U8`, `U16`, `U32`, `U64`, `I16`, `I32`, `I64`, `Bool`, plus
  `F32x6`.
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
- Added Python `has_wgpu_device()` before RFC-0018; it now behaves as a
  compatibility accelerator-availability check until a rename RFC replaces it.
- Burn Dispatch backend is configured with explicit `dispatch`, `cuda`, and
  `cpu` features; Router and WGPU/Vulkan default features are removed per
  RFC-0018.
- `src/utils/tensor.rs` owns backend/device aliases and tensor construction
  helpers, keeping Dispatch-specific code out of feature modules.
- `.to("gpu")` surfaces a Python error when no configured accelerator is
  available rather than silently falling back to CPU.
- RFC-0012 accepts host-side planning for the repo-local implementation while
  requiring derived XYZ tensors to preserve the source/common device.
- Implemented source/common-device XYZ result preservation for:
  - `select(mask)`
  - `select_indices(indices)`
  - `PointCloud.concatenate(...)`
  - `voxel_downsample(...)` across all strategies
  - `transform`, `transform_3x3`, `translate`, `scale`, `rotate`, and
    `rigid_transform`
- CPU and WGPU-gated Python tests cover the
  `select_by_classification -> voxel_downsample -> transform -> concatenate`
  pipeline, dtype-preserving attributes, and seeded deterministic downsampling.

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
- `docs/api/registration.md` documents the covariance-weighted GICP behavior.

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

### RFC-0013 / RFC-0014 - LAS Point Format 10

- `PointCloud.to_las(..., point_format=10, las_version="1.4")` explicitly
  writes LAS 1.4 point format 10 while preserving existing inferred
  `to_las(path, compress=False)` behavior for older formats.
- Python stubs and bindings expose keyword-only `point_format`, `las_version`,
  and `drop_waveform` options.
- Point format 10 standard attributes are optional at the public API boundary;
  missing GPS time, RGB, NIR, and waveform option groups are materialized as
  LAS defaults during export.
- Preferred precision dtypes are supported and tested: `uint16` intensity/RGB
  /NIR, raw LAS 1.4 `int16` scan angle, `uint64` waveform offset, `uint32`
  waveform size, `float32` waveform location/vector fields, and `float64`
  GPS time.
- Legacy `float32` intensity and `uint8` RGB remain accepted on write.
- Non-default waveform point metadata is rejected unless `drop_waveform=True`;
  drop mode writes no-waveform defaults and does not emit waveform attributes
  as ExtraBytes.
- Uncompressed LAS imports keep raw integer `X/Y/Z` sidecars plus scale/offset
  metadata. Sidecars gather through selection and compatible concat, and are
  dropped by coordinate-mutating operations.
- Standard LAS ExtraBytes VLRs are decoded for supported scalar dtypes; names
  colliding with point format 10 standard dimensions are prefixed with
  `extra_`.
- Tests cover Rust PF10 fixtures, a laspy-authored PF10 fixture, standard
  ExtraBytes collision behavior, Python dtype propagation, version/format
  validation, partial RGB defaults, and waveform policy.

### RFC-0015 - Open3D Comparison Benchmark Charts

- `pyproject.toml` defines a benchmark-only dependency group for `open3d`,
  `plotly`, and `pytest-benchmark`; these are not runtime dependencies.
- `tests/test_open3d_benchmark.py` implements the RFC-0015 comparison harness
  using pytest-benchmark when installed, with collection-safe skips otherwise.
- The harness covers construction, transform, voxel downsample, warm kNN, warm
  radius search, normal estimation, SOR/ROR, and point-to-point/point-to-plane
  ICP across smoke/standard/full point-count modes.
- Benchmark metadata records library, operation, case, point count, output
  count, dependency versions, platform, git SHA, and operation parameters.
- Timed callables separate setup/mutation/cache policy where needed, including
  fresh Open3D clouds for mutating operations and explicit warm-index neighbor
  cases.
- `tools/render_open3d_benchmark_charts.py` reads pytest-benchmark JSON,
  writes `open3d-comparison-summary.csv`, and renders interactive Plotly HTML
  without running benchmarks.
- Tests cover parser behavior, summary speedup rows, non-comparable exclusions,
  `extra_info` metadata parsing, mode matrices, operation coverage, and
  metadata contract.
- Added `just benchmark-compare-smoke`, `benchmark-compare-standard`,
  `benchmark-compare-full`, and `benchmark-compare-charts`.
- `docs/performance/open3d-comparison.md` documents commands, modes, outputs,
  optional dependency isolation, and RFC-0011 evidence rules.

## Important Gaps By RFC

This section is the current implementation backlog. Per RFC-0011, repo-local
code/test/docs gaps keep an RFC `Partial`; external evidence gaps remain
unchecked and visible but do not by themselves imply missing repo-local
implementation.

### RFC-0002

- External evidence gap: no 10M-point XYZ getter benchmark or documented
  speedup artifact is recorded.
- External-doc gap: `multica-home/knowledge/projects/pcl-rustic.md` is not
  present in this repo.
- External evidence gap: no recorded Linux/macOS/Windows `just ci` matrix is
  attached to this repo-local status pass.

### RFC-0003

- No repo-local implementation gap remains. Device-residency optimization moved
  to RFC-0004 and does not block M2 completion.

### RFC-0004

- External evidence gap: no recorded 50M LAZ GPU-vs-CPU 3x speedup result.
- External evidence gap: no recorded RFC-0004 backend benchmark matrix artifact.
- Future optimization gap: tensor-native voxel binning remains future work under
  RFC-0012, not a repo-local completion blocker.

### RFC-0005

- External evidence gap: no 10M-point kNN result is recorded in
  `docs/performance/benchmarks.md`.

### RFC-0006

- External evidence gap: no SOR 10M benchmark evidence is recorded.

### RFC-0007

- External evidence gap: Open3D Bunny comparison artifact is not recorded.
- External evidence gap: 500k-vs-500k registration benchmark artifact is not
  recorded.

### RFC-0008

- No repo-local implementation gap remains; RFC-0007 now contains a
  covariance-weighted GICP update.

### RFC-0009

- External evidence gap: standard and full benchmark modes have not been
  executed on recorded high-memory benchmark hardware.
- Documentation evidence gap: the docs renderer exists, but release benchmark
  results still need to be produced on recorded hardware before publishing
  measured performance rows.

### RFC-0011

- No implementation gap remains in the tracking RFC itself.
- Ongoing requirement: every future benchmark or external-evidence claim must
  cite an artifact with the RFC-0011 evidence fields.

### RFC-0013

- No repo-local implementation gap remains. LAS point format 10 read/write,
  laspy fixture import, standard dimension dtype preservation, raw integer XYZ
  sidecars, standard ExtraBytes parsing, waveform default/reject/drop behavior,
  and docs/tests are complete.
- Intentional limitation: full waveform descriptor/payload preservation is not
  implemented. Non-default waveform point metadata is rejected unless callers
  explicitly set `drop_waveform=True` to write no-waveform defaults.

### RFC-0014

- No repo-local implementation gap remains. The explicit `point_format=10`
  export API policy is implemented with backward-compatible existing LAS calls,
  version validation, partial RGB defaults, and waveform metadata policy tests.

### RFC-0015

- No repo-local implementation gap remains. The comparison harness and chart
  renderer exist.
- External evidence gap: no recorded Open3D comparison benchmark JSON, summary
  CSV, or Plotly HTML artifact has been generated on benchmark hardware.

## External Evidence Register

| RFC | Criterion | Artifact | Date | Git SHA | Hardware / Dataset | Status | Notes |
|---|---|---|---|---|---|---|---|
| RFC-0002 | `multica-home/knowledge/projects/pcl-rustic.md` written | unrecorded | — | — | Multica workspace | open | Outside this repository. |
| RFC-0002 | `just ci` green on Linux, macOS, Windows | unrecorded | — | — | Hosted CI matrix | open | Requires recorded cross-platform CI run. |
| RFC-0002 | XYZ getter 10M-point benchmark evidence | unrecorded | — | — | Reference benchmark hardware | open | Performance artifact not recorded. |
| RFC-0004 | 50M LAZ GPU-vs-CPU speedup | unrecorded | — | — | Reference GPU machine and LAZ fixture | open | Repo-local device-residency behavior is implemented; measured GPU speedup artifact is absent. |
| RFC-0005 | 10M-point kNN benchmark | unrecorded | — | — | Reference benchmark hardware | open | Repo-local neighbor API, KD-tree cache, octree pruning, and normal-estimation criteria are implemented. |
| RFC-0006 | 10M-point SOR benchmark | unrecorded | — | — | Reference benchmark hardware | open | Repo-local SOR/ROR behavior and LAS standard/custom attribute propagation are implemented. |
| RFC-0007 | Open3D Bunny comparison and 500k registration benchmark | unrecorded | — | — | Bundled Bunny fixture / reference CPU | open | Repo-local point-to-point, point-to-plane, and GICP solvers are implemented. |
| RFC-0009 | Standard/full benchmark CSVs | unrecorded | — | — | High-memory benchmark hardware | open | Harness exists; measured artifacts are absent. |
| RFC-0015 | Open3D comparison benchmark JSON/CSV/HTML artifacts | unrecorded | — | — | Benchmark machine with Open3D and pytest-benchmark | open | Repo-local harness and Plotly renderer exist; measured comparison artifacts are absent. |

## 2026-05-08 RFC-0002 Through RFC-0012 Loop

- Re-audited RFC-0002 through RFC-0012 acceptance checkboxes. The remaining
  unchecked items are external evidence/doc gates: Multica workspace note,
  hosted Linux/macOS/Windows `just ci`, high-memory benchmark runs, GPU speedup
  artifact, Open3D comparison artifact, and benchmark docs generated from
  recorded CSVs.
- Removed redundant internal helpers after implementation review:
  `#![allow(dead_code)]`, unused tensor conversion/validation helpers, unused
  attribute selection helpers, unused KD-tree accessors, and unused point-cloud
  getters.
- Narrowed `HighPerformancePointCloud::xyz_mut()` to test builds because its
  only current use is the KD-tree cache invalidation test.
- Cleanup commit: `5cb3284 refactor: remove redundant internal helpers`.

Fresh verification from the 2026-05-08 RFC-0002 through RFC-0012 cleanup loop:

- `rtk cargo fmt` passed.
- `rtk cargo test --lib` passed: 40/40 Rust unit tests.
- `rtk cargo clippy -- -D warnings` passed with no issues found.
- `rtk uv run pytest tests/test_point_cloud.py -q --no-cov` passed: 60/60
  Python tests.

## 2026-05-08 LAS Point Format 10 Completion Pass

- Added and accepted RFC-0014 to specify the point format 10 writer API and
  waveform metadata policy.
- Implemented typed host attributes for `uint64` and `int16` across NumPy
  import/export, selection, concat, voxel averaging/mode behavior, and LAS
  ExtraBytes.
- Implemented explicit PF10 LAS export via
  `to_las(path, compress=False, *, point_format=10, las_version="1.4",
  drop_waveform=False)`, with Rust `to_las(path, compress)` kept as the
  backward-compatible wrapper.
- Implemented PF10 standard field read/write coverage for intensity, packed
  return/classification flags, scan angle, point source/user fields, GPS time,
  16-bit RGB, NIR, and waveform point metadata defaults.
- Added raw LAS integer XYZ sidecars plus scale/offset metadata for uncompressed
  LAS imports. Sidecars gather through selection and compatible concat, and are
  dropped by coordinate-mutating operations.
- Added LAS standard ExtraBytes VLR parsing for scalar dtypes needed by PF10,
  including collision prefixing to `extra_...` when ExtraBytes names conflict
  with canonical standard dimensions.
- Added laspy as a dev dependency and introduced a laspy-authored PF10 fixture
  test.
- Updated `docs/api/io.md`, `docs/plans/rfc-0013...`, and
  `docs/plans/rfc-0014...` to document and mark the PF10 work complete.

Fresh verification from the 2026-05-08 PF10 completion pass:

- `rtk cargo fmt --check` passed.
- `rtk cargo clippy -- -D warnings` passed with no issues found.
- `rtk cargo test --lib` passed: 40/40 Rust unit tests.
- `rtk uv run pytest tests/test_point_cloud.py -q --no-cov` passed: 60/60
  Python tests.

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
  - RFC-0008 was initially marked `Implemented (GICP follow-up open)`; later
    RFC-0007 work implemented the covariance-weighted GICP follow-up.
  - RFC-0009 is marked `Implemented`.

## 2026-05-07 RFC-0011 Evidence-Gate Pass

- Added RFC-0011 to define repo-local implementation completion separately from
  external evidence completion.
- Two independent subagents reviewed RFC-0011. Their required changes were
  incorporated: mandatory `Implemented (external evidence open)` suffixes,
  stronger benchmark artifact fields, preserved LAS-fixture requirements,
  narrowed third-party comparison wording, and a canonical evidence register.
- Marked RFC-0010 acceptance criteria complete based on existing docs and tests.
- Updated RFC-0002 and RFC-0009 statuses to expose external evidence gaps
  without claiming unrecorded benchmark or cross-platform results.

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

1. RFC-0002: either produce the external Multica project note and
   cross-platform `just ci` evidence, or explicitly move those artifacts out of
   the repo-local release gate.
2. RFC-0004/RFC-0009: run the recorded backend benchmark matrix and 50M LAZ
   GPU-vs-CPU pipeline on reference hardware before making GPU speedup claims.
3. RFC-0005/RFC-0006: run and publish the required 10M kNN and 10M SOR
   benchmark artifacts on reference hardware.
4. RFC-0007: record the Open3D Bunny comparison and 500k-vs-500k registration
   benchmark artifacts.
5. RFC-0009: run standard/full benchmark modes on recorded high-memory hardware
   and regenerate benchmark docs from the resulting CSV files.

## Architecture Decisions Recorded

| Decision | Rationale |
|---|---|
| `AttributeValue` uses host `Vec<T>` for typed attrs | Accepted by RFC-0010 so LAS/NumPy dtypes remain exact while XYZ stays tensor-backed. |
| Added `F32x6` packed covariance variant | RFC-0007 needs `[N, 6]` covariance storage while preserving the RFC-0010 host typed attribute boundary. |
| KD-tree falls back to brute-force on degenerate axis buckets | `kiddo` can panic on many identical values along an axis; fallback preserves correctness. |
| GICP uses covariance-weighted delta with prerequisite validation | Keeps the public API explicit about required covariance data and avoids silent fallback to a different estimator. |
| `just ci` no longer depends on pre-commit | CI should use pytest/cargo directly per user instruction; pre-commit remains separately available. |
| LAS point format 10 waveform payloads are reject/drop only | Point-record waveform metadata references descriptor/payload data that is not preserved yet; non-default metadata errors unless `drop_waveform=True` writes no-waveform defaults. |
| LAS raw integer XYZ sidecars are internal metadata | Compute XYZ stays `float32`, while unchanged uncompressed LAS imports preserve raw `X/Y/Z` plus scale/offset for precision round-trips. |
