# RFC-0021: Overall Speed Parity

- **Status:** Accepted
- **Date:** 2026-05-13
- **Author:** Codex
- **Tracking issue:** Pending; open before implementation work starts.
- **Related:** RFC-0001, RFC-0004, RFC-0009, RFC-0011, RFC-0015, RFC-0016, RFC-0018, RFC-0020

## 1. Summary

Make pcl-rustic competitive with Open3D across the full comparison suite, not
only in isolated hot paths. This RFC defines a benchmark-driven optimization
program for construction, voxel downsampling, normal estimation, outlier
removal, registration, and benchmark reporting. The implementation must improve
speed by changing algorithm and memory behavior, not by weakening correctness,
dropping typed attributes, or hiding slow operations from comparison charts.

## 2. Motivation

The Open3D comparison harness now provides enough evidence to show that
pcl-rustic is not yet a credible Open3D replacement for speed-sensitive users.
The latest local standard-mode summary at
`reports/benchmarks/last-benchmark-summary.csv` records these speedup ratios
where values above `1.0` mean pcl-rustic is faster than Open3D:

| Operation | 10k | 100k | 1M | Current finding |
|---|---:|---:|---:|---|
| `construction` | 0.28x | 0.45x | 1.76x | Small/medium inputs pay too much fixed overhead. |
| `transform` | 1.36x | 4.78x | 81.83x | Tensor path is strong at scale. |
| `voxel_downsample` | 0.09x | 0.23x | 0.19x | Current grouping path is the biggest broad regression. |
| `estimate_normals` | 0.06x | 0.07x | 0.09x | Point-by-point normal path is still far behind Open3D. |
| `remove_radius_outlier` | 0.65x | 0.65x | 1.19x | RFC-0020 helps large rows; small/medium overhead remains. |
| `remove_statistical_outlier` | 0.59x | 0.78x | 1.01x | Close at large scale; small/medium still slower. |
| `registration_point_to_point` | 0.36x | 0.32x | 0.36x | Case labels scale, but the current harness caps measured registration inputs at 10k points. |
| `registration_point_to_plane` | 0.58x | 0.44x | 0.26x | Case labels scale, but the current harness caps measured registration inputs at 10k points. |
| warm `knn` / `radius_search` | 2.36x-11.36x | | | Reduced neighbor APIs are already a strength. |

The problem is therefore not one missing acceleration backend. The current
codebase has several different speed losses:

- `src/point_cloud/voxel.rs` groups every point into
  `HashMap<[i32; 3], Vec<usize>>`, then revisits grouped vectors for selection
  or aggregation.
- `src/neighbors/normals.rs` builds a normal-specific index and loops over
  points sequentially, allocating a neighbor vector per point.
- construction paths cross Python, NumPy, Rust, and Burn tensor boundaries even
  when input is already `float32` and contiguous.
- registration still does per-iteration host work and small allocations even
  after RFC-0020 correspondence acceleration.
- registration benchmark rows currently record the requested case point count
  even when the operation uses the 10k `REGISTRATION_POINTS_CAP`, so their
  standard 100k and 1M rows are not valid evidence for 100k or 1M registration
  throughput.
- benchmark reports show speed gaps, but there is no project-level performance
  budget that stops future changes from regressing operations that are already
  faster than Open3D.

This RFC turns the comparison suite into an optimization contract.

## 3. Detailed Design

### 3.1 Performance Budget

Add an explicit speed budget for comparable Open3D rows. A row is in scope when
RFC-0015 marks it `comparison_status=comparable` in the rendered summary, or
when the raw pytest-benchmark metadata records `comparable=True` before summary
rendering.

Initial targets for standard mode:

| Milestone | Required result |
|---|---|
| P0 | No comparable standard row with `measured_point_count >= 1_000_000` is below `0.80x` Open3D unless listed in §3.9. |
| P1 | Geometric mean speedup across comparable standard rows with `measured_point_count >= 1_000_000` is at least `1.25x`. |
| P2 | `voxel_downsample` and `estimate_normals` each reach at least `1.00x` at measured 1M, and registration reaches at least `1.00x` at its measured 10k cap or the benchmark cap is removed and replaced with real measured-scale rows. Follow-up RFCs may defer a P2 row only with measured blocker evidence. |
| P3 | No row that is currently above `1.00x` regresses by more than 10% against the previous accepted artifact. |

Why this, not a single "be faster than Open3D everywhere" rule: some rows are
small enough that Python/PyO3 fixed overhead dominates, while other rows are
algorithmic. The 1M row is large enough to represent production point-cloud
workloads and small enough to run routinely on benchmark hosts.

### 3.2 Benchmark Gap Gate

Add a benchmark analysis tool that reads the generated summary CSV and fails
when the accepted budget is violated.

```bash
uv run python tools/check_open3d_speed_budget.py \
  reports/benchmarks/last-benchmark-summary.csv \
  --mode standard \
  --min-1m-speedup 0.80 \
  --min-1m-geomean 1.25 \
  --baseline reports/benchmarks/accepted-open3d-baseline-summary.csv \
  --max-regression-ratio 0.10
```

The script compares by `(operation, case_id, requested_point_count,
measured_point_count)` and emits a table with:

- current pcl-rustic mean
- current Open3D mean
- speedup
- accepted baseline speedup, when present
- requested point count from the benchmark case
- measured point count used inside the timed operation
- pass/fail reason

The benchmark renderer and summary CSV must add `requested_point_count` and
`measured_point_count`. For uncapped rows both values are identical. For
registration rows, `requested_point_count` records the RFC-0015 comparison case
and `measured_point_count` records the post-cap `limited_case.point_count`.
The speed-budget checker must not treat capped registration rows as 1M evidence.

Why this, not manually reading Plotly charts: charts are useful for diagnosis,
but a budget gate gives reviewers a deterministic pass/fail signal.

### 3.3 Voxel Downsample Rewrite

Replace the current vector-per-voxel grouping with an allocation-bounded
builder that performs one pass over points and stores per-voxel accumulator
state directly.

```rust
pub(crate) enum VoxelAccumulator {
    RandomSeeded {
        chosen_index: usize,
        seen: u32,
    },
    NearestToCentroid {
        count: u32,
        sum: [f32; 3],
        candidates: Vec<usize>,
    },
    Average {
        count: u32,
        xyz_sum: [f64; 3],
        attrs: AttributeAccumulatorSet,
    },
}

pub(crate) fn voxel_downsample_fast(
    cloud: &HighPerformancePointCloud,
    voxel_size: f32,
    strategy: &DownsampleStrategy,
) -> Result<HighPerformancePointCloud>;
```

Implementation requirements:

- Use `rustc_hash::FxHashMap` or `hashbrown::HashMap` after benchmark evidence
  shows faster insertion than `std::collections::HashMap` for `[i32; 3]` keys.
- Encode voxel keys as a compact `VoxelKey` with explicit overflow checks.
- Avoid storing `Vec<usize>` for `RandomSeeded` and `Average`.
- Keep `NearestToCentroid` exact. It may store candidates per voxel in the
  first implementation, but a two-pass CSR-style layout is preferred for large
  clouds.
- Parallelize by chunking input points into thread-local maps, then merge maps
  deterministically.
- Preserve typed attributes and existing output behavior for all strategies.
- The RFC-0015 Open3D comparison `voxel_downsample` row currently measures
  `NearestToCentroid`, so the benchmark speed gate is not satisfied until that
  strategy improves even if `RandomSeeded` or `Average` improve first.

Why this, not GPU voxelization first: the current CPU path is more than 5x
slower than Open3D at 1M in standard mode. Fixing allocation and hash behavior
is lower risk and benefits every host before adding GPU-specific complexity.

### 3.4 Normal Estimation Rewrite

Move normal estimation onto the shared neighbor infrastructure from RFC-0020
and parallelize the per-point covariance solve.

```rust
pub(crate) trait NormalNeighborhoods {
    fn knn_indices_for_all_points(&self, k_without_self: usize) -> Result<Vec<SmallNeighbors>>;
    fn radius_indices_for_all_points(&self, radius: f32) -> Result<Vec<SmallNeighbors>>;
    fn hybrid_indices_for_all_points(
        &self,
        radius: f32,
        max_neighbors: usize,
    ) -> Result<Vec<SmallNeighbors>>;
}

pub(crate) struct SmallNeighbors {
    pub len: u16,
    pub inline: [u32; 32],
    pub spill: Option<Vec<u32>>,
}
```

Implementation requirements:

- Build one neighbor index per operation and reuse it for all points.
- Use `rayon::par_iter` for per-point covariance and eigen solve.
- Avoid allocating a new `Vec<usize>` for every point when the neighbor count
  fits inline.
- Keep projected-2D and brute-force degeneracy handling from the current
  implementation.
- Write `nx`, `ny`, and `nz` as three preallocated vectors, then insert
  attributes once.
- Share the same neighborhood batch path with `estimate_covariances`.

Why this, not call Open3D through FFI: pcl-rustic needs independent Rust
implementations for typed attributes, backend metadata, and distribution. FFI
would hide the bottleneck and add a runtime dependency.

### 3.5 Construction And Interop Fast Paths

Reduce fixed overhead in Python construction and array export. The public API
does not change.

```rust
impl HighPerformancePointCloud {
    pub fn from_contiguous_f32_xyz_slice(
        xyz: &[f32],
        rows: usize,
        device: &tensor::BackendDevice,
    ) -> Result<Self>;

    pub fn from_contiguous_f32_xyz_cpu_first(xyz: &[f32], rows: usize) -> Result<Self>;
}
```

Implementation requirements:

- Keep the current `float64` and integer autocast behavior.
- For `float32` C-contiguous NumPy arrays, avoid intermediate
  `Vec<[f32; 3]>` materialization.
- Benchmark whether small/medium construction should default to CPU tensors
  and move to GPU only after `.to("gpu")`, or whether Dispatch CUDA creation is
  already cheap enough at 1M.
- Reuse a single flattening/allocation path for `from_xyz`, `from_numpy`,
  CSV/Parquet loading, and LAS loading where practical.
- Add benchmark metadata for `storage_device` and `construction_backend`.

Why this, not zero-copy NumPy ownership: the tensor backend owns its storage and
point clouds must outlive Python views. This RFC optimizes copies first without
changing lifetime semantics.

### 3.6 Registration Loop Cleanup

Keep RFC-0020's accelerated correspondence path and remove the remaining
per-iteration overhead from ICP.

```rust
pub(crate) struct IcpWorkspace {
    transformed_source: Vec<[f32; 3]>,
    correspondences: Vec<Correspondence>,
    normal_equations: NormalEquationScratch,
}

impl IcpWorkspace {
    pub fn resize_for(&mut self, source_points: usize);
    pub fn transform_source_in_place(&mut self, source: &[[f32; 3]], transform: &Matrix4<f32>);
}
```

Implementation requirements:

- Reuse `transformed_source` and correspondence buffers across iterations.
- Add parallel reductions for point-to-plane and GICP estimator accumulation,
  not only point-to-point.
- Precompute target normal arrays once for point-to-plane ICP.
- Preserve current convergence semantics and tolerances.
- Record iteration count, correspondence count, and estimator type in
  benchmark metadata.

Why this, not increase benchmark cap or reduce iterations: the benchmark should
measure the same algorithmic work as Open3D. Speed must come from less
allocation and better reductions, not an easier workload.

### 3.7 Backend Policy

This RFC keeps GPU work narrow. CUDA/Burn tensor paths are already strong for
large transforms, but voxel, normals, and registration still need better CPU
algorithms before GPU kernels are credible.

Rules:

- CPU optimizations must record `execution_backend=cpu_rayon` when they use
  Rayon.
- GPU speed claims require operation-specific metadata such as
  `execution_backend=cuda_voxel` or `cuda_exact_neighbors`.
- A cloud stored on CUDA is not enough to claim GPU execution.
- Do not add approximate algorithms to beat Open3D unless a later RFC defines
  the accuracy/recall contract and exposes the approximation clearly.

Why this, not force every slow row onto CUDA immediately: Open3D's CPU
implementations are the current comparison target. A faster CPU baseline makes
future GPU work easier to verify and lowers risk for users without CUDA.

### 3.8 Implementation Slices

Implementation should land in reviewable slices:

1. Add benchmark summary metadata for `requested_point_count` and
   `measured_point_count`; fix capped registration rows so downstream gates do
   not treat the 10k cap as 100k or 1M evidence.
2. Add `tools/check_open3d_speed_budget.py`, accepted baseline summary support,
   and documentation for running the speed gate.
3. Rewrite voxel downsample for `RandomSeeded` and `Average`, with exact typed
   attribute tests and standard-mode benchmark evidence.
4. Finish exact `NearestToCentroid` acceleration with deterministic merge
   behavior.
5. Move normal estimation and covariance estimation onto batch neighborhood
   APIs with Rayon.
6. Add construction fast paths for contiguous `float32` arrays and shared table
   loading.
7. Add ICP workspace reuse and point-to-plane/GICP parallel reductions.
8. Refresh `docs/performance/open3d-comparison.md` and
   `docs/performance/benchmarks.md` with accepted benchmark artifacts.

### 3.9 Accepted Decisions And Exceptions

The draft open-question defaults are accepted as part of this revision:

- The first hard budget target is `0.80x` for comparable measured 1M standard
  rows. P2 still requires `1.00x` for voxel downsample and normal estimation at
  measured 1M, and for registration at the current measured 10k cap or a
  replacement uncapped benchmark.
- 10k and 100k construction rows are advisory fixed-overhead metrics for this
  RFC. The hard gate applies to measured 1M construction.
- `NearestToCentroid` tie-breaking must remain deterministic and compatible
  with current behavior unless a later RFC changes the policy.
- Curated benchmark summaries may be committed under `docs/performance/`.
  Raw benchmark JSON should stay in generated reports or durable external
  artifacts unless a release process explicitly snapshots it.

No speed-budget exceptions are accepted at RFC acceptance time. Any future
exception must name the operation, case id, measured point count, threshold,
benchmark artifact, and follow-up RFC or issue that owns removal of the
exception.

## 4. Acceptance Criteria

- [ ] `docs/plans/README.md` lists RFC-0021 with status `Accepted`.
- [ ] Benchmark summaries include both `requested_point_count` and
      `measured_point_count`.
- [ ] Capped registration rows record `measured_point_count=10000` when
      `REGISTRATION_POINTS_CAP` is active, even for larger requested comparison
      cases.
- [ ] A speed-budget checker exists and reads
      `reports/benchmarks/last-benchmark-summary.csv`.
- [ ] The checker fails when any comparable standard row with
      `measured_point_count >= 1_000_000` is below the configured minimum
      speedup and is not listed in §3.9.
- [ ] The checker reports geometric mean speedup across comparable standard
      rows with `measured_point_count >= 1_000_000`.
- [ ] An accepted baseline summary artifact is recorded or the checker supports
      an explicit `--no-baseline` mode for first adoption. When `--no-baseline`
      is used, the checker must skip P3 regression enforcement and state that
      regression protection starts after the first accepted baseline is
      recorded.
- [ ] `voxel_downsample` no longer builds
      `HashMap<[i32; 3], Vec<usize>>` for `RandomSeeded` or `Average`.
- [ ] `voxel_downsample` preserves existing output behavior and typed
      attributes for all three strategies.
- [ ] `voxel_downsample` standard 1M speedup improves from the recorded
      `0.190900x` to at least `1.00x`, or a follow-up RFC records why Open3D
      parity requires GPU voxel kernels. The first implementation slice may
      land at `0.80x` only if the speed-budget checker records it as a P0 pass
      and P2 remains open.
- [ ] `estimate_normals` uses batched shared neighborhood APIs instead of a
      point-by-point normal-specific query loop.
- [ ] `estimate_normals` standard 1M speedup improves from the recorded
      `0.085009x` to at least `1.00x`, or a follow-up RFC records why parity
      requires a different neighbor backend. The first implementation slice may
      land at `0.80x` only if the speed-budget checker records it as a P0 pass
      and P2 remains open.
- [ ] `from_xyz` and `from_numpy` avoid intermediate `Vec<[f32; 3]>`
      materialization for contiguous `float32` NumPy inputs.
- [ ] `from_xyz` and `from_numpy` preserve the current behavior for contiguous
      `float32`, `float64` autocast, and integer autocast inputs.
- [ ] Non-contiguous NumPy XYZ behavior is explicit in tests: either the API
      rejects non-contiguous arrays with the existing error text or copies them
      deliberately through a documented slow path.
- [ ] Point-to-point and point-to-plane ICP reuse workspace buffers across
      iterations.
- [ ] Registration benchmark metadata records estimator, iteration count, and
      correspondence count.
- [ ] Registration rows either reach at least `1.00x` Open3D at
      `measured_point_count=10000`, or the benchmark cap is removed and the
      speed-budget checker gates registration by real measured point count.
- [ ] After the first accepted baseline exists, current faster rows, including
      standard 1M `transform`, `knn_warm`, and `radius_search_warm`, do not
      regress by more than 10% against that baseline.
- [ ] `rtk cargo test --lib` passes.
- [ ] `rtk cargo clippy -- -D warnings` passes.
- [ ] `rtk just build` passes.
- [ ] `rtk uv run --group benchmark pytest tests/test_open3d_benchmark.py -v -s --run-slow --benchmark-mode=standard --benchmark-compare-open3d --benchmark-json=reports/benchmarks/rfc0021-standard.json --no-cov`
      completes on benchmark hardware.
- [ ] `rtk uv run python tools/render_open3d_benchmark_charts.py reports/benchmarks/rfc0021-standard.json --html-output reports/benchmarks/rfc0021-standard.html --summary-output reports/benchmarks/rfc0021-standard-summary.csv`
      completes.
- [ ] On first adoption,
      `rtk uv run python tools/check_open3d_speed_budget.py reports/benchmarks/rfc0021-standard-summary.csv --mode standard --min-1m-speedup 0.80 --min-1m-geomean 1.25 --no-baseline`
      passes or prints only explicitly accepted exceptions from §3.9.
- [ ] After an accepted baseline exists,
      `rtk uv run python tools/check_open3d_speed_budget.py reports/benchmarks/rfc0021-standard-summary.csv --mode standard --min-1m-speedup 0.80 --min-1m-geomean 1.25 --baseline reports/benchmarks/accepted-open3d-baseline-summary.csv --max-regression-ratio 0.10`
      passes or prints only explicitly accepted exceptions from §3.9.

## 5. Risks And Mitigations

| Risk | Trigger | Mitigation |
|---|---|---|
| Optimizing for synthetic data hurts real LAS workloads | Speed improves on RFC-0015 fixtures but not on real files | Add at least one recorded LAS-derived benchmark artifact before claiming user-facing speed parity. |
| Parallel voxel map merge changes deterministic output | Different thread counts produce different selected indices | Require deterministic merge tests under `RAYON_NUM_THREADS=1` and `RAYON_NUM_THREADS=4`. |
| Faster hashing changes collision or overflow behavior | Compact voxel key overflows for large coordinates or tiny voxel sizes | Add checked key encoding and tests for negative coordinates, large offsets, and tiny voxel sizes. |
| Inline neighbor storage is too small for radius normals | Radius queries spill frequently and lose speed | Record spill counts in debug metadata and tune inline capacity from benchmark evidence. |
| Construction fast path breaks Python lifetime safety | Rust stores borrowed NumPy memory beyond the call | Keep owned tensor storage; optimize intermediate copies only. |
| Speed budget is flaky across machines | CI hardware differs from benchmark host | Gate only on dedicated benchmark artifacts; ordinary CI runs smoke wiring and unit tests. |
| CUDA work expands the RFC too far | Voxel or normal CPU parity stalls | Open a focused GPU RFC with operation-specific exactness tests and metadata labels. |

## 6. Out Of Scope

- Approximate nearest-neighbor search.
- Changing public Python method names or return types.
- Dropping typed attributes to match Open3D's simpler data model.
- Removing correctness or comparability checks from RFC-0015 benchmarks.
- Replacing Burn Dispatch as the tensor backend.
- Publishing new performance claims without RFC-0011 evidence fields.
- Full GPU voxelization, GPU normal estimation, or CUDA exact neighbor kernels.
- Optimizing LAS/LAZ compression speed beyond construction and table-loading
  paths shared with point-cloud creation.

## 7. Open Questions

None. The draft open-question defaults were accepted and are recorded in §3.9.

## 8. References

- RFC-0015: Open3D Comparison Benchmark Charts.
- RFC-0020: Neighbor Hot-Path Acceleration.
- `docs/performance/open3d-comparison.md`.
- `reports/benchmarks/last-benchmark-summary.csv`, generated 2026-05-12 on
  local benchmark hardware.
- Current implementation files:
  - `src/point_cloud/voxel.rs`
  - `src/neighbors/normals.rs`
  - `src/interop/numpy.rs`
  - `src/registration.rs`
