# RFC-0007: ICP & GICP Registration (M6)

- **Status:** Partial
- **Date:** 2026-04-30
- **Author:** Master PM (agent)
- **Tracking issue:** LEO-36 (parent), per-milestone LEO issue TBD
- **Related:** RFC-0001 (roadmap), RFC-0005 (prerequisite — KD-tree, normals), RFC-0003 (selection / transform wrappers)

## 1. Summary

Close LEO-36 priority #8 by shipping an ICP-family registration pipeline with three transformation estimators — Point-to-Point, Point-to-Plane, and Generalized ICP — backed by RFC-0005's KD-tree for correspondence search and RFC-0002's typed attributes for per-point covariance / normal storage. The public API mirrors Open3D's `registration_icp` / `registration_generalized_icp` so users migrating from Open3D find familiar shapes.

## 2. Motivation

Registration is the single largest feature gap vs. Open3D in the LEO-36 priorities, and it's the reason several internal scanning workflows still pull in Open3D despite our LAS/LAZ advantage. Without ICP, pcl-rustic is a data-prep library; with it, it becomes a standalone alternative for the common scan-alignment pipelines.

Shipping three estimators together (not just point-to-point) is a deliberate choice: the infrastructure cost (correspondence solver + Gauss-Newton step + convergence loop) is shared, and the incremental cost of Point-to-Plane and GICP on top of Point-to-Point is small compared to adding them in a follow-up RFC.

## 3. Detailed design

### 3.1 Top-level API

```rust
pub mod registration {
    use nalgebra::Matrix4;

    pub struct ICPConvergenceCriteria {
        pub max_iteration: usize,
        /// Convergence threshold on relative fitness change between iterations.
        /// `f32` precision is sufficient for ICP convergence decisions.
        pub relative_fitness: f32,
        /// Convergence threshold on relative RMSE change between iterations.
        /// `f32` precision is sufficient for ICP convergence decisions.
        pub relative_rmse: f32,
    }

    pub enum TransformationEstimation {
        PointToPoint,
        PointToPlane,
        /// GICP plane-to-plane regularization strength. `f32` is sufficient.
        Generalized { epsilon: f32 },
    }

    pub struct RegistrationResult {
        pub transformation: Matrix4<f32>,
        pub fitness: f32,
        pub inlier_rmse: f32,
        pub correspondence_set: Vec<(u64, u64)>,  // (source_idx, target_idx)
    }

    pub fn icp(
        source: &HighPerformancePointCloud,
        target: &HighPerformancePointCloud,
        max_correspondence_distance: f32,
        init: Matrix4<f32>,
        estimation: TransformationEstimation,
        criteria: ICPConvergenceCriteria,
    ) -> Result<RegistrationResult>;

    pub fn evaluate(
        source: &HighPerformancePointCloud,
        target: &HighPerformancePointCloud,
        max_correspondence_distance: f32,
        transformation: Matrix4<f32>,
    ) -> Result<RegistrationResult>;  // one-shot, no iteration
}
```

Python:

```python
from pcl_rustic import registration as reg
import numpy as np

result = reg.icp(
    source, target,
    max_correspondence_distance=0.05,
    init=np.eye(4, dtype=np.float32),
    estimation=reg.TransformationEstimation.point_to_plane(),
    criteria=reg.ICPConvergenceCriteria(max_iteration=30, relative_fitness=1e-6, relative_rmse=1e-6),
)
print(result.transformation, result.fitness, result.inlier_rmse)
```

### 3.2 Correspondence search

Per ICP iteration:

1. Transform `source` by the current cumulative `T`.
2. For each transformed source point, find the nearest target point via `target.kdtree()` (RFC-0005).
3. Drop pairs with distance > `max_correspondence_distance`.
4. Compute `fitness = |inliers| / |source|` and `inlier_rmse = sqrt(mean(d²))`.

`target.kdtree()` is built once per ICP call (cached on the target cloud). Source KD-tree is not needed.

### 3.3 Estimators

**Point-to-Point** — closed-form SVD of cross-covariance (Horn, 1987; Besl–McKay). Implementation via `nalgebra::linalg::SVD` on a 3×3 matrix — trivial.

**Point-to-Plane** — linearized Gauss-Newton with per-iteration target-normal projections. Requires `target.estimate_normals(...)` (RFC-0005) already called; verified at entry and a clear error if normals are missing.

**Generalized ICP** — plane-to-plane formulation (Segal, Haehnel, Thrun 2009). Requires both source *and* target covariances per point. Each covariance is the upper-triangle of a symmetric 3×3 matrix packed into a 6-element row, stored as a single `AttributeValue` named `"covariance"` using a 2D tensor of shape `[N, 6]`. The six column indices encode the following components (docstring on `estimate_covariances`):

```
index 0 → cov_00  (variance along X)
index 1 → cov_01  (XY covariance)
index 2 → cov_02  (XZ covariance)
index 3 → cov_11  (variance along Y)
index 4 → cov_12  (YZ covariance)
index 5 → cov_22  (variance along Z)
```

This maps to the full matrix as `[[00, 01, 02], [01, 11, 12], [02, 12, 22]]`. Storing as a 2D tensor keeps the typed-attribute design aligned (extending `AttributeValue` with a `Tensor2` variant as needed in RFC-0002). `estimate_covariances(knn)` computes and writes this attribute; M6 does not use separate scalar attributes for covariance components. Solve each iteration as a sum of Mahalanobis residuals via Gauss-Newton; reuse the linear solver from Point-to-Plane.

`epsilon` on the GICP variant regularizes the covariances toward planarity (Open3D exposes it as `epsilon=1e-3` by default) — surface through the enum constructor.

### 3.4 Convergence loop

Iterate up to `max_iteration`. Between iterations, compare:

- `|fitness_k - fitness_{k-1}| < relative_fitness`
- `|rmse_k - rmse_{k-1}| < relative_rmse`

Exit on either. Return the last `RegistrationResult`.

Early-exit diagnostic: log at `info` level which criterion triggered termination so users can tune.

### 3.5 Thread & device strategy

- Correspondence search: parallel `rayon` loop over source points, each doing one `kdtree.knn(1)`.
- Per-iteration transform: uses the existing `PointCloud::transform(matrix)` — already GPU-aware (RFC-0004).
- Linear solve: CPU, `nalgebra`, small systems (3×3 for P2P, 6×6 for P2Plane and GICP). Not a bottleneck.

### 3.6 Supporting work

- New method `PointCloud::estimate_covariances(knn: usize)` — writes a 2D tensor attribute `"covariance"` of shape `[N, 6]` (upper-triangle packed, see §3.3 for component layout). Shares the kNN infra from RFC-0005.
- `registration::evaluate` is a one-shot variant with no iteration, useful for acceptance testing and for users who want to score a known transform.

## 4. Acceptance criteria

- [ ] `registration::icp` implemented with all three estimators; signatures match §3.1.
- [ ] `registration::evaluate` implemented.
- [ ] `PointCloud::estimate_covariances(knn)` implemented and tested.
- [ ] Unit tests:
  - Identity registration: source = target = random cloud, `icp` with init = identity converges in one iteration, `transformation ≈ I`, `fitness == 1`.
  - Known-rotation recovery: apply a known 4×4 `T_gt` to a 10k-point fixture, register back with init = noisy identity, assert `‖T_recovered · T_gt − I‖_F < 1e-3`. Covered for the current point-to-point solver with identity init.
  - Point-to-Plane requires normals: without `estimate_normals` first, returns a clear error.
  - GICP requires covariances: without `estimate_covariances` first, returns a clear error.
- [ ] Comparison test against Open3D: on a bundled 5000-point Bunny fixture, `fitness` and `inlier_rmse` agree with Open3D's `registration_icp` within 1% after 30 iterations (same `max_correspondence_distance`, same init, same estimator).
- [ ] Benchmark: 500k-vs-500k point alignment completes in < 20 s (CPU, 30 iterations) on the reference machine.
- [ ] Documentation: `docs/api/registration.md` with full examples for each estimator, plus a migration note for Open3D users.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| GICP numerical instability on near-degenerate covariances | `epsilon` regularization (§3.3); fall back to Point-to-Plane if the linear system is ill-conditioned beyond a threshold; emit a warning. |
| Normal sign ambiguity from RFC-0005 produces wrong Point-to-Plane residuals | Add an "orient toward source" step at registration entry, disabled by a flag if the user pre-oriented. |
| Correspondence search dominates run time | Benchmark first; only optimize if the §4 target misses. The first lever is batched kNN; the second is RFC-0005's stretch GPU NN path. |
| `nalgebra` SVD instability on pathological cross-covariance | Check for NaN / non-finite results, bail with a clear error naming the likely cause (degenerate geometry). |
| Color-based / Doppler ICP users assume feature-complete parity | Call out in the migration doc that these estimators are explicitly deferred. |

## 6. Out of scope

- **Colored ICP** — Open3D-only feature; defer until a user asks.
- **Multi-scale ICP** — a thin wrapper once the core loop is stable; ship as a future RFC if users want it.
- **RANSAC global registration** and **Fast Global Registration** — explicitly deferred per RFC-0001.
- **FPFH** or other feature descriptors needed for global registration — blocked by the above.
- **GPU-resident correspondence search** — depends on RFC-0005 §3.5 stretch.

## 7. Open questions

- Covariance storage policy: persist to LAS export, or strip on serialize? Proposal: strip — covariances are cheap to recompute and bloat LAS attributes.
- Default `max_iteration` — Open3D uses 30. Proposal: match Open3D; users who want more pay for it explicitly.
- Whether `TransformationEstimation::Generalized { epsilon }` should expose `max_correspondence_distance` separately. Proposal: no — the top-level `max_correspondence_distance` parameter is shared; adding per-estimator variants increases the API surface without a clear user need.
