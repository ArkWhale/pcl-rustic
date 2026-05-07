# RFC-0006: Outlier Removal — SOR & ROR (M5)

- **Status:** Partial (amended by RFC-0010)
- **Date:** 2026-04-30
- **Author:** Master PM (agent)
- **Tracking issue:** LEO-36 (parent), per-milestone LEO issue TBD
- **Related:** RFC-0001 (roadmap), RFC-0005 (prerequisite — KD-tree), RFC-0010 (host typed attribute storage amendment)

## 1. Summary

Close LEO-36 priority #7 with `remove_statistical_outlier(nb_neighbors, std_ratio)` and `remove_radius_outlier(nb_points, radius)`. Both reuse the KD-tree from RFC-0005 and return `(filtered_cloud, kept_mask)`, where `kept_mask` is a boolean per-input-point mask (`true` = kept). CPU implementation; the neighbor query is the hot path and RFC-0005's CPU KD-tree is sufficient for realistic input sizes.

## 2. Motivation

LEO-36 priority #7 is a direct Open3D-parity ask. Both algorithms are well-defined and small (< 200 LOC total once RFC-0005 lands). Shipping them turns pcl-rustic into a viable cleaner for raw LAS/LAZ data — the most common reason Open3D shows up in our pipelines.

## 3. Detailed design

### 3.1 Statistical outlier removal (SOR)

For each point `p_i`, compute the mean distance μ_i to its k nearest neighbors (excluding self). Compute the distribution (μ̄, σ) over all points. Reject points with `μ_i > μ̄ + std_ratio · σ`.

```rust
impl HighPerformancePointCloud {
    pub fn remove_statistical_outlier(
        &self,
        nb_neighbors: usize,
        std_ratio: f32,
    ) -> Result<(Self, Vec<bool>)>;
}
```

Returns the filtered cloud and the per-input-point `kept` mask (`true` = kept).
RFC-0010 accepts host typed attributes and host masks for the current selection
contract, so Rust returns `Vec<bool>` and Python returns a bool NumPy array. A
same-device tensor mask is deferred until device-resident attributes are
revisited.

Algorithm detail:

1. `tree = self.kdtree()` (RFC-0005 — cached).
2. Per-point mean-neighbor distance via `rayon` parallel iterator over points, `tree.knn(point, nb_neighbors + 1)` (skip the self-hit).
3. Compute `μ̄`, `σ` in one pass with Welford's online algorithm.
4. Build the mask; call `select(mask)` (RFC-0003).

### 3.2 Radius outlier removal (ROR)

For each point `p_i`, count neighbors within `radius`. Reject if `count < nb_points`.

```rust
impl HighPerformancePointCloud {
    pub fn remove_radius_outlier(
        &self,
        nb_points: usize,
        radius: f32,
    ) -> Result<(Self, Vec<bool>)>;
}
```

Implementation via `tree.radius_search(point, radius)` — we only need the count, so we can use `kiddo`'s `within_unsorted_iter` to short-circuit once the count reaches `nb_points`.

### 3.3 Python API

```python
pc_clean, kept_mask = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
pc_clean, kept_mask = pc.remove_radius_outlier(nb_points=5, radius=0.1)

# kept_mask is a bool NDArray[(N,)] — useful for replicating the filter on paired data
xyz_clean = pc.get_xyz()[kept_mask]  # equivalent to pc_clean.get_xyz()
```

`.pyi` stubs updated to reflect the tuple return.

### 3.4 Defaults

Open3D ships no defaults (all args required). We match that — no silent defaults, because bad values here corrupt downstream pipelines quietly.

### 3.5 Threading & determinism

`rayon` parallel iteration writes per-point mean-distances into a pre-allocated `Vec<f32>` indexed by point ID, then reduces globally. The two-pass structure (parallel per-point distances, then sequential Welford over the `Vec`) produces **nearly** deterministic results: the per-point distances are exact (independent lookups), but the global `μ̄`/`σ` reduction may exhibit tiny floating-point differences across runs due to `rayon`'s non-deterministic scheduling order. Such differences are accepted — they can only move a point across the threshold if it is extremely close to `μ̄ + std_ratio · σ`, and the practical effect on cleaned point clouds is negligible.

> **Implementation note:** Document in the code with a comment on the `μ̄`/`σ` computation that parallel reduction order is non-deterministic and that tiny FP differences are expected and intentional. If strict bit-identical output is ever required (e.g. for regression testing), a `remove_statistical_outlier_deterministic` variant using single-threaded Welford over the pre-computed `Vec<f32>` can be added with the same public signature — add it only if the performance trade-off is acceptable and a user surfaces the need.

## 4. Acceptance criteria

- [x] Both methods implemented with the Python signatures in §3.3. Rust returns a host `Vec<bool>` mask per RFC-0010.
- [x] Unit tests:
  - SOR on a synthetic cloud = Gaussian blob + injected outliers; assert ≥ 95% of injected outliers removed at `std_ratio=2.0`.
  - ROR on a 1D line of points with one isolated extra point; assert the extra is removed at `nb_points=2, radius=step_size*1.5`.
  - Empty input → error, not panic.
  - `kept_mask.sum() == pc_clean.point_count()`.
- [ ] Benchmark: SOR on a 10M-point cloud with `nb_neighbors=20` completes in < 30 s on the reference machine (dominated by kNN queries).
- [ ] Attribute propagation: intensity, RGB, classification, custom attributes all survive correctly (tested via a LAS round-trip + outlier step + comparison). Synthetic typed-attribute propagation coverage exists, and LAS round-trip coverage now covers standard LAS attributes; custom attributes via LAS ExtraBytes remain open.
- [x] Documentation page `docs/api/outlier.md` ships with worked examples and parameter-tuning guidance (what `std_ratio=2.0` vs `3.0` does).
- [x] Both functions included in the classification-aware pipeline example from RFC-0003 as an optional cleaning step.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Neighborhood count = 1 (`nb_neighbors=1`) produces undefined μ̄, σ | Reject at the API boundary; require `nb_neighbors ≥ 2`. |
| `std_ratio` very small → too many points removed, silently | No silent recovery; document the behavior and leave the user to pick. |
| RFC-0005's KD-tree cache gets invalidated mid-run | Not a concern — SOR/ROR are read-only on the cloud. |
| GPU-resident clouds pay a host transfer | Documented; the RFC-0007 ICP inner loop pays this *once* per iteration, which is acceptable. If hot, RFC-0005's §3.5 stretch (GPU NN) unblocks it. |

## 6. Out of scope

- Bilateral filter / Guided filter — not in LEO-36 priorities.
- Noise removal via DBSCAN-cluster-size thresholds — deferred.
- Denoising with learned models — out of scope of this library.

## 7. Open questions

- Should the API also expose `remove_statistical_outlier_inplace`? Proposal: no — our clouds are cheap to clone post-RFC-0002, and the functional style composes better with pipeline examples.
- Do we return the *removed* indices in addition to the kept mask? Open3D returns the kept indices tuple `(pcd, ind)`. We return the mask, which is equivalent and more ergonomic; if a user surfaces who wants the indices list, we add `mask.nonzero()` as sugar.
