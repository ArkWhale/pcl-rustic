# RFC-0020: Neighbor Hot-Path Acceleration

- **Status:** Implemented
- **Date:** 2026-05-12
- **Author:** Codex
- **Tracking issue:** Not assigned; open before implementation work starts.
- **Related:** RFC-0004, RFC-0005, RFC-0006, RFC-0007, RFC-0009, RFC-0011, RFC-0016, RFC-0018

## 1. Summary

Make neighbor-heavy pcl-rustic operations use explicit acceleration backends
instead of hidden single-threaded host loops. The first accepted implementation
must add CPU multicore execution and allocation-bounded query APIs for outlier
removal and point-to-point ICP correspondence search. CUDA neighbor kernels are
out of scope for this RFC; this RFC only reserves benchmark metadata and policy
so future GPU work cannot be mistaken for the current Kiddo-backed CPU path.

## 2. Motivation

The benchmark rows `pcl_rustic-remove_radius_outlier-standard-1000000` and
`pcl_rustic-registration_point_to_point-standard-10000` currently behave like
single-core CPU workloads even when pcl-rustic is built with Burn Dispatch CUDA.
The cause is architectural, not a benchmark harness accident:

- `remove_radius_outlier` pulls all XYZ data to a host `Vec` and calls
  `self.kdtree()?.radius_search(&xyz, radius)` for every point.
- `remove_statistical_outlier` follows the same host path through batched
  `knn`.
- `registration::icp` pulls source and target XYZ to host vectors once, then
  transforms source points, finds correspondences, computes deltas, and
  computes metrics on host vectors for every iteration.
- `KdTreeIndex::knn` and `KdTreeIndex::radius_search` use sequential
  `.iter().map(...).collect()` over all query points.
- The batched KD-tree APIs allocate `Vec<Vec<NeighborHit>>`, even when callers
  only need a count, one nearest neighbor, or a small reduction.
- No code under `src/` imports `rayon::prelude` or calls `par_iter`, despite
  RFC-0006 and RFC-0007 specifying Rayon for outliers and ICP.
- The CUDA-enabled Burn tensor backend lives in `src/utils/tensor.rs`, but
  Kiddo and nalgebra operate only on host data. A cloud being GPU-resident
  therefore does not make these algorithms GPU-resident.

This contradicts the intent of prior RFCs:

- RFC-0006 said SOR/ROR should use Rayon and short-circuit ROR counts through
  Kiddo's unsorted radius iterator.
- RFC-0007 said ICP correspondence search should be a parallel Rayon loop over
  source points and that per-iteration transform should use the existing
  GPU-aware transform path.

### 2.1 Amendments To Prior RFCs

This RFC amends RFC-0006 by making the Rayon and short-circuit ROR design a
current implementation requirement rather than a historical note.

This RFC amends RFC-0007 by narrowing the immediate ICP acceleration contract:
until correspondence search is GPU-resident, point-to-point ICP may use a
parallel host transform inside the ICP loop to avoid device-host round trips.
The public `PointCloud::transform` API remains GPU-aware; only the ICP internal
loop is amended.

The current benchmark harness also makes the outlier rows stricter than a warm
query microbenchmark: `benchmark_fresh_input` builds a fresh point cloud for
each round, then the operation builds the cached KD-tree inside the measured
operation. That is acceptable if documented, but it means the benchmark is an
end-to-end operation benchmark and the implementation must avoid unnecessary
host allocations in both index construction and query phases.

## 3. Detailed Design

### 3.1 Backend Boundary

Introduce an internal neighbor-query boundary in `src/neighbors/` that exposes
the operations the high-level algorithms actually need. Do not expose this as a
Python API in this RFC.

```rust
pub(crate) struct KnnHit {
    pub index: u64,
    pub distance_sq: f32,
}

pub(crate) trait NeighborIndex {
    fn knn_one_within(
        &self,
        query: &[[f32; 3]],
        max_distance_sq: f32,
    ) -> Result<Vec<Option<KnnHit>>>;

    fn knn_mean_distances_for_index_points(
        &self,
        k_without_self: usize,
    ) -> Result<Vec<f32>>;

    fn radius_self_has_at_least(
        &self,
        radius: f32,
        min_neighbors_without_self: usize,
    ) -> Result<Vec<bool>>;
}
```

Why this, not more `Vec<Vec<NeighborHit>>`: the algorithms do not need every
neighbor row as an owned vector. The boundary lets each backend short-circuit,
stream, or reduce without storing all intermediate hits.

Initial implementation target:

- `CpuKdTreeIndex` backed by Kiddo and Rayon.
- Self-query APIs operate over the index's own host points and exclude self by
  stored point item id, not by distance. Duplicate coordinates must not be
  treated as self.
- `knn_mean_distances_for_index_points` returns mean Euclidean distance, not
  mean squared distance, so SOR masks remain compatible with current behavior.
- `knn_one_within` returns squared distance for ICP filtering and RMSE
  accumulation, where square roots are unnecessary.
- Reduced APIs must work for both Kiddo-backed indices and the current
  degenerate-geometry brute-force fallback.
- It is exact and preserves existing user-visible behavior.
- It is the default backend for all clouds, including GPU-resident clouds,
  until a GPU backend has its own acceptance evidence.

Future implementation target:

- A later RFC may add `GpuNeighborIndex` for exact radius count and exact ICP
  correspondence on CUDA-capable hosts.
- RFC-0020 does not select a GPU neighbor backend and does not require CUDA
  code.

### 3.2 CPU Multicore Outlier Removal

Rewrite outlier removal to consume reduced neighbor-query APIs directly:

```rust
impl HighPerformancePointCloud {
    pub fn remove_statistical_outlier(
        &self,
        nb_neighbors: usize,
        std_ratio: f32,
    ) -> Result<(Self, Vec<bool>)>;

    pub fn remove_radius_outlier(
        &self,
        nb_points: usize,
        radius: f32,
    ) -> Result<(Self, Vec<bool>)>;
}
```

`remove_statistical_outlier`:

1. Convert XYZ to host once.
2. Build or reuse the CPU KD-tree.
3. Use `rayon::par_iter` over query points to compute one mean distance per
   point.
4. Compute mean and variance deterministically over the resulting `Vec<f32>`.
5. Build `kept_mask` and call `select_mask`.

`remove_radius_outlier`:

1. Convert XYZ to host once.
2. Build or reuse the CPU KD-tree.
3. Use `rayon::par_iter` over query points.
4. For each point, count neighbors within `radius` and stop once
   `nb_points` non-self neighbors have been found.
5. Return the mask and selected cloud.

ROR may use Kiddo's `within_unsorted_iter` on architectures where Kiddo exposes
it. On other targets, it may use `within_unsorted` or brute-force counting.
Every path must preserve exact masks and must not allocate all radius hit rows
for the common Kiddo-backed case.

Why this, not a GPU first rewrite: it closes the current single-core regression
without changing public behavior or relying on unproven GPU nearest-neighbor
kernels. It also creates the exact operation-level interface a GPU backend must
implement later.

### 3.2.1 Host XYZ Ownership

The implementation must avoid duplicate host transfers for the same cloud in a
single operation. Today `remove_radius_outlier` obtains `xyz` with
`get_xyz_vec()` and `KdTreeIndex::build` obtains another host copy through
`pc.get_xyz_vec()`. ICP similarly holds `target_xyz` while correspondence search
uses `target.kdtree()`.

Add one of these internal mechanisms:

- `KdTreeIndex::build_from_xyz(xyz_host: Vec<[f32; 3]>) -> Result<Self>`, with
  callers using the index's owned points for self-query reductions; or
- an internal host-XYZ accessor that returns the same host buffer used by the
  cached KD-tree.

For ICP, target XYZ used by estimators must come from the same host buffer as
the target index. Source XYZ may still be a separate host buffer because it is
transformed every iteration.

### 3.3 CPU Multicore Point-To-Point ICP

Keep the public registration API unchanged:

```rust
pub fn icp(
    source: &HighPerformancePointCloud,
    target: &HighPerformancePointCloud,
    max_correspondence_distance: f32,
    init: Matrix4<f32>,
    estimation: TransformationEstimation,
    criteria: ICPConvergenceCriteria,
) -> Result<RegistrationResult>;
```

Change the internals:

1. Build the target neighbor index once per ICP call.
2. For point-to-point ICP, transform source points with a parallel host path
   for the CPU backend.
3. Use `knn_one_within` to find all correspondences in parallel.
4. Compute fitness and RMSE from squared distances without redundant square
   roots.
5. Use Rayon reductions for point-to-point centroids and cross-covariance.
6. Keep the small point-to-point SVD solve on CPU through nalgebra.

Why this, not call `PointCloud::transform` each iteration: the current ICP
already owns host `source_xyz` and needs host points for correspondence and
small-matrix estimators. Calling the tensor transform would add device-host
round trips unless the full ICP correspondence path is also GPU-resident.

Point-to-plane and GICP benefit from the shared correspondence path, but their
normal-equation accumulation is not required for the first RFC-0020
implementation. Those estimator reductions should be handled in a follow-up
RFC or implementation slice after point-to-point evidence exists.

### 3.4 Future GPU Policy

This RFC does not permit pretending Kiddo-backed work is GPU-accelerated. GPU
acceleration requires a separate RFC or implementation plan with exactness
tests and benchmark evidence.

Reserve these backend labels for metadata:

```rust
pub(crate) enum NeighborExecutionBackend {
    CpuRayon,
    CudaExact,
}
```

RFC-0020 implementation rules:

- Default to `CpuRayon`.
- Do not construct or select `CudaExact`.
- Record `CudaExact` only in future work that implements exact CUDA neighbor
  queries and passes CPU-vs-GPU exactness tests.
- If a cloud's XYZ tensor is CUDA-resident but the neighbor algorithm runs
  through Kiddo and Rayon, metadata must say `CpuRayon`.

Why this, not a full GPU KD-tree now: exact GPU KD-tree construction and query
would be a separate project. The immediate bottlenecks can be accelerated with
parallel CPU now, without blocking on unproven CUDA kernels.

### 3.5 Benchmark Metadata

Extend the Open3D comparison benchmark metadata for pcl-rustic neighbor-heavy
rows:

```python
{
    "execution_backend": "cpu_rayon" | "cuda_exact",
    "thread_count": int,
    "rayon_current_num_threads": int,
    "gpu_accelerated": bool,
    "storage_device": str,
}
```

Measured rows must not imply GPU acceleration unless `execution_backend` is
`cuda_exact`. `storage_device` records `PointCloud.device_name()` for tensor
residency, but it is not enough to prove neighbor-search residency.

Why this, not infer from `PointCloud.device()`: these algorithms can store XYZ
on CUDA while querying Kiddo on CPU. Storage device and algorithm backend are
separate facts.

### 3.6 Tests

Add Rust tests that prove behavior and backend shape. Use dependency injection
or a test-only instrumented backend instead of timing assertions:

- `radius_outlier_uses_count_backend_without_materializing_rows` proves ROR
  calls `radius_self_has_at_least` and does not call the old batched
  `radius_search` path.
- `point_to_point_icp_uses_parallel_correspondence_backend` proves
  point-to-point ICP calls `knn_one_within` and does not call the old batched
  `knn` path.
- `cpu_rayon_matches_existing_outlier_masks` compares masks against the current
  implementation on deterministic fixtures.
- `point_to_point_correspondence_matches_existing_no_ties` compares
  correspondences on fixtures with no equidistant nearest-neighbor ties.

Timing checks belong in benchmark acceptance, not unit tests.

### 3.7 Implementation Slices

Implementation must be split so each slice is reviewable:

1. Add the reduced CPU neighbor-query APIs plus test-only instrumentation while
   preserving existing `knn` and `radius_search` public Rust methods.
2. Add a compile-time `assert_sync::<KdTreeIndex>()`; if it fails, choose an
   immutable/thread-safe index representation before moving algorithm code.
3. Move ROR onto `radius_self_has_at_least` and Rayon.
4. Move SOR onto `knn_mean_distances_for_index_points` and Rayon.
5. Move point-to-point ICP correspondence and host transform onto
   `knn_one_within` and Rayon.
6. Add point-to-point ICP metric and reduction cleanup where it is directly
   exercised by the benchmark row.
7. Add benchmark metadata for algorithm execution backend and Rayon thread
   count.
8. Leave CUDA exact backend for a follow-up RFC after CPU evidence exists.

## 4. Acceptance Criteria

- [x] `src/point_cloud/outlier.rs` no longer calls batched
      `radius_search(&xyz, radius)` and then materializes all radius hits for
      ROR.
- [x] `remove_radius_outlier` uses a count-or-short-circuit API and allocates
      `O(n)` mask/output state, not `O(n * neighbors_in_radius)` hit rows.
- [x] `remove_statistical_outlier` computes per-point mean neighbor distance
      through a Rayon-backed reduced query API and preserves current mean
      Euclidean distance semantics.
- [x] Self-exclusion in SOR and ROR is by stored point id for self-query APIs,
      not by zero distance or coordinate equality.
- [x] Outlier removal avoids duplicate host XYZ extraction for the same cloud
      inside one operation.
- [x] `src/registration.rs` no longer calls a sequential batched
      `target.kdtree()?.knn(transformed_source, 1)` for ICP correspondences.
- [x] ICP target XYZ used by point-to-point estimators comes from the same host
      target buffer as the target neighbor index.
- [x] Point-to-point ICP correspondence search uses a Rayon-backed exact
      nearest-neighbor query path and produces the same correspondence set as
      the current implementation for deterministic test fixtures with no
      equidistant nearest-neighbor ties.
- [x] Point-to-point centroid and cross-covariance accumulation use parallel
      reductions when the correspondence count is large enough to amortize
      Rayon overhead.
- [x] Existing public Python signatures for outlier removal and registration
      are unchanged.
- [x] Existing unit tests for outlier removal and registration continue to pass.
- [x] New tests prove the reduced neighbor-query APIs are used by ROR and
      point-to-point ICP.
- [x] A compile-time test proves the chosen CPU neighbor index is `Sync`; if it
      is not `Sync`, the implementation must use a thread-safe index
      representation before satisfying `execution_backend=cpu_rayon`.
- [x] `pcl_rustic-remove_radius_outlier-standard-1000000` completes without
      single-core-only execution on a multicore CPU host, verified by benchmark
      metadata recording `execution_backend=cpu_rayon`,
      `rayon_current_num_threads > 1`, `storage_device`, and the benchmark
      artifact path.
- [x] `pcl_rustic-registration_point_to_point-standard-10000` completes with
      `execution_backend=cpu_rayon`, `rayon_current_num_threads > 1`,
      `storage_device`, and the benchmark artifact path.
- [x] The benchmark JSON/CSV metadata distinguishes storage device from
      algorithm execution backend.
- [x] No benchmark row is described as GPU-accelerated unless it records
      `execution_backend=cuda_exact`.

## 5. Verification

Required local commands:

```bash
rtk cargo fmt --check
rtk cargo test --lib
rtk cargo clippy -- -D warnings
rtk just build
rtk uv run --group benchmark pytest 'tests/test_open3d_benchmark.py::test_open3d_comparison_benchmark[pcl_rustic-remove_radius_outlier-standard-1000000]' -q -s --run-slow --benchmark-mode=standard --benchmark-min-rounds=1 --benchmark-max-time=1 --benchmark-json=reports/benchmarks/open3d-rfc0020-ror-standard-1000000.json --no-cov
rtk uv run --group benchmark pytest 'tests/test_open3d_benchmark.py::test_open3d_comparison_benchmark[pcl_rustic-registration_point_to_point-standard-10000]' -q -s --run-slow --benchmark-mode=standard --benchmark-min-rounds=1 --benchmark-max-time=1 --benchmark-json=reports/benchmarks/open3d-rfc0020-icp-p2p-standard-10000.json --no-cov
```

The benchmark JSON artifacts must include `execution_backend`,
`rayon_current_num_threads`, `gpu_accelerated`, and `storage_device` metadata
for the pcl-rustic rows.

### External Evidence

| Criterion | Artifact | Date | Git SHA | Hardware / Dataset | Status | Notes |
|---|---|---|---|---|---|---|
| ROR standard 1M row records `execution_backend=cpu_rayon` and `rayon_current_num_threads > 1` | `reports/benchmarks/open3d-rfc0020-ror-standard-1000000.json` | 2026-05-12 | 0d1660a + RFC-0020 working tree | 32-thread x86_64 Linux host / generated standard 1M outlier cloud | recorded | Mean 1.1442 s; metadata recorded `execution_backend=cpu_rayon`, `rayon_current_num_threads=32`, `gpu_accelerated=False`, and `storage_device=Cuda(Cuda(0))`. |
| Point-to-point ICP standard 10k row records `execution_backend=cpu_rayon` and `rayon_current_num_threads > 1` | `reports/benchmarks/open3d-rfc0020-icp-p2p-standard-10000.json` | 2026-05-12 | 0d1660a + RFC-0020 working tree | 32-thread x86_64 Linux host / generated standard 10k plane registration cloud | recorded | Mean 8.7062 ms; metadata recorded `execution_backend=cpu_rayon`, `rayon_current_num_threads=32`, `gpu_accelerated=False`, and `storage_device=Cuda(Cuda(0))`. |

## 6. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Kiddo's tree type is not `Sync` for shared Rayon queries | Add a compile-time assertion in tests. If it fails, use an immutable/thread-safe index representation or block the Rayon acceptance slice. A mutex around Kiddo queries does not satisfy `execution_backend=cpu_rayon`. |
| Parallel floating-point reductions change last-bit results | For masks and transforms, compare within existing tolerances; keep SOR global mean/variance deterministic by reducing the per-point vector sequentially unless benchmark evidence requires parallel reduction. |
| ROR short-circuit changes behavior when the point itself is returned first | Count only non-self neighbors and add explicit tests where self appears in the radius result. |
| Multicore CPU speedup hides that GPU is still absent | Benchmark metadata must record `execution_backend=cpu_rayon`; RFC-0011 evidence gates reject GPU claims from CPU rows. |
| CUDA exact backend becomes too large for one implementation PR | CUDA exact backend is out of scope for RFC-0020 implementation; require a follow-up RFC or implementation plan. |
| Benchmarks include KD-tree build time because inputs are fresh per round | Keep that behavior, but report backend metadata and optimize both build and query allocations. Do not switch to warm-cache semantics without a separate benchmark RFC. |

## 7. Out Of Scope

- Public Python API changes for choosing neighbor backends.
- Approximate nearest-neighbor search.
- Replacing Kiddo with a different CPU KD-tree crate.
- Full GPU KD-tree construction and arbitrary-size GPU kNN.
- CUDA exact neighbor kernels.
- Point-to-plane and GICP estimator reduction rewrites, beyond improvements
  that fall out of the shared correspondence path.
- Changing the Open3D comparison benchmark point-count matrix.
- Claiming new performance numbers in docs without RFC-0011 benchmark
  artifacts.

## 8. Open Questions

1. Should the CUDA exact backend use CubeCL custom kernels or Burn tensor
   primitives?
   Proposed default: do not decide in RFC-0020. Require a focused GPU RFC or
   implementation plan before adding CUDA code.

2. Should benchmark metadata go only into pytest-benchmark JSON, or also into a
   derived CSV summary?
   Proposed default: JSON first, because `tests/test_open3d_benchmark.py`
   already attaches `extra_info`; add CSV only if chart rendering needs it.

3. Should users be able to force a backend through an environment variable?
   Proposed default: no public control in this RFC. Add internal test hooks
   only; expose user controls after at least two backends are production-ready.

## 9. References

- RFC-0006: Outlier Removal, especially the intended Rayon and ROR
  short-circuit design.
- RFC-0007: ICP & GICP Registration, especially the intended Rayon
  correspondence search.
- RFC-0011: Completion Evidence Gates.
- Current evidence paths:
  - `src/point_cloud/outlier.rs`
  - `src/neighbors/kdtree.rs`
  - `src/registration.rs`
  - `tests/test_open3d_benchmark.py`
