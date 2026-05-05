# RFC-0005: Neighborhood Infra — KD-tree, Octree & Normal Estimation (M4)

- **Status:** Partial
- **Date:** 2026-04-30
- **Author:** Master PM (agent)
- **Tracking issue:** LEO-36 (parent), per-milestone LEO issue TBD
- **Related:** RFC-0001 (roadmap), RFC-0002 (typed attrs), blocks RFC-0006 and RFC-0007

## 1. Summary

Add the nearest-neighbor infrastructure that outlier removal (RFC-0006) and ICP-family registration (RFC-0007) depend on: a fast KD-tree on the CPU via the `kiddo` crate, exposed as `knn(k)` and `radius_search(r)` on `PointCloud`, plus `estimate_normals(radius | knn)` that writes `nx/ny/nz` as typed attributes. An octree implementation is also included — either via a suitable crate or a purpose-built `pcl-rustic-octree` crate if no adequate crate exists. GPU neighbor search is an explicit stretch goal; both CPU index structures are committed deliverables.

## 2. Motivation

- LEO-36 priority #7 (SOR / ROR) requires kNN or radius queries — no way to compute distances without an index.
- Priority #8 (ICP / GICP) needs correspondence search on every iteration, and GICP additionally needs per-point covariance, which requires normals or at least local neighborhoods.
- Open3D exposes `KDTreeFlann` (+ `PointCloud.estimate_normals`); users migrating expect the same primitives.

Building KD-tree once, correctly, unblocks two milestones worth of algorithm work.

## 3. Detailed design

### 3.1 CPU KD-tree via `kiddo`

`kiddo 4.x` is the current best-in-class Rust KD-tree: SIMD-friendly, SoA layout, sub-microsecond lookups on 10M-point clouds. Add as a dep:

```toml
kiddo = "4"
```

Wrapper lives in `src/neighbors/kdtree.rs`:

```rust
pub struct KdTree {
    inner: kiddo::KdTree<f32, u64, 3, 32, u32>,
    // keep the host-side xyz copy for distance checks; GPU clouds materialize once on index build
    xyz_host: Vec<[f32; 3]>,
}

impl KdTree {
    pub fn build(pc: &HighPerformancePointCloud) -> Result<Self>;
    pub fn knn(&self, query: &[[f32; 3]], k: usize) -> Vec<Vec<NeighborHit>>;
    pub fn radius_search(&self, query: &[[f32; 3]], r: f32) -> Vec<Vec<NeighborHit>>;
}

pub struct NeighborHit { pub index: u64, pub distance: f32 }
```

Queries accept either a slice of `[f32; 3]` host points or a `&HighPerformancePointCloud` (device points are transferred once per call; caller pays the cost only if their source is GPU-resident).

### 3.2 Cache on `PointCloud`

Building a KD-tree on 50M points is non-trivial (~seconds). Cache it:

```rust
use once_cell::sync::OnceCell;

pub struct HighPerformancePointCloud {
    // existing fields …
    // Thread-safe lazy cache so `kdtree()` can be called from `rayon` parallel code.
    kdtree_cache: OnceCell<KdTree>,
}

impl HighPerformancePointCloud {
    /// Returns a reference to the cached KD-tree, building it on first call.
    /// Returns `Err` if the build fails (e.g. empty cloud or invalid geometry)
    /// so callers can propagate errors rather than panic.
    pub fn kdtree(&self) -> Result<&KdTree> {
        self.kdtree_cache.get_or_try_init(|| KdTree::build(self))
    }
    // Initialize with `OnceCell::new()` in constructors so `HighPerformancePointCloud`
    // remains `Sync` when `KdTree` is `Sync`.
    // Invalidate on any mutation (transform/select/concat/downsample)
    fn invalidate_kdtree(&mut self) { self.kdtree_cache.take(); }
}
```

Every mutating op in M2–M3 has to call `invalidate_kdtree()`. Add a `cargo clippy` lint or a runtime debug assertion that panics in tests if a stale tree is returned after a mutation.

### 3.3 Normal estimation

```rust
pub enum NormalSearch {
    Knn(usize),             // fixed neighbor count
    Radius(f32),            // fixed radius
    Hybrid(f32, usize),     // radius-limited, at most k neighbors
}

impl HighPerformancePointCloud {
    pub fn estimate_normals(&mut self, search: NormalSearch) -> Result<()>;
}
```

Algorithm: for each point, gather k (or radius-bounded) neighbors, compute the 3×3 covariance, take the smallest-eigenvector as the normal. Eigendecomposition via `nalgebra::SymmetricEigen` (small fixed-size matrices, no heavy deps). Results stored as three `AttributeValue::F32` attributes `nx`, `ny`, `nz` (per RFC-0002 typed attrs). Parallelized with `rayon` (already a dep).

Normal orientation (Open3D calls these `orient_normals_*`) is **deferred** — only GICP needs oriented normals and it can orient-toward-viewer locally.

### 3.4 Python API

```python
from pcl_rustic import PointCloud, NormalSearch

pc = PointCloud.from_las("scan.laz")

# Neighbor queries
query = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
indices, distances = pc.knn(query, k=16)           # each: (Q, k) int64 / (Q, k) f32
indices, distances = pc.radius_search(query, r=0.5)  # variable-length per query (list of arrays)

# Normals (mutates the cloud — adds nx/ny/nz attributes)
pc.estimate_normals(NormalSearch.knn(30))
normals = np.stack([pc.get_attribute("nx"), pc.get_attribute("ny"), pc.get_attribute("nz")], axis=1)
```

`radius_search` returns a Python `list[np.ndarray]` (ragged), not a 2D array — variable-length neighborhoods are unavoidable.

### 3.5 Octree

An octree provides efficient range queries for spatially-clustered data and is complementary to the KD-tree. Implementation strategy:

1. Survey available crates at implementation time (e.g. `octree`, `octomap-rs`). Use the most suitable one.
2. If no suitable crate exists, create a `pcl-rustic-octree` sub-crate with a minimal but correct implementation (insert, range query, depth-limited traversal).

```rust
pub struct Octree { /* opaque */ }

impl Octree {
    pub fn build(pc: &HighPerformancePointCloud, max_depth: u8) -> Result<Self>;
    pub fn range_search(&self, center: &[f32; 3], radius: f32) -> Vec<u64>;
    pub fn voxel_centers(&self) -> Vec<[f32; 3]>;
}

impl HighPerformancePointCloud {
    pub fn octree(&self, max_depth: u8) -> Result<Octree>;
}
```

`octree()` is not cached (it is parameterized by `max_depth`) — callers construct and hold it explicitly.

### 3.6 GPU neighbor search (stretch)

Optional. Two candidate approaches if the CPU path shows up as a bottleneck in RFC-0007's ICP inner loop:

1. **Hash-grid** — for radius queries with bounded neighborhood size. Implementable as tensor ops; hooks into the Morton binning from RFC-0004.
2. **Bitonic-sort k-select** — for kNN. Heavier; only worth it if hash-grid doesn't suffice.

Neither is committed for M4. If adopted, keep the CPU `kiddo` path as the correctness reference.

## 4. Acceptance criteria

- [x] `KdTree::build` / `knn` / `radius_search` implemented and tested against a brute-force oracle on 10k-point random clouds (exact match for kNN, setwise match for radius).
- [ ] `PointCloud.kdtree()` cache works: tested that repeated `knn` calls on an unchanged cloud do not rebuild; tested that any mutating op invalidates.
- [x] `PointCloud.kdtree()` returns `Err` (not panic) on empty or invalid input; tested.
- [ ] `Octree::build` / `range_search` implemented and tested against the same brute-force oracle; `PointCloud.octree(max_depth)` returns `Err` on empty input. Correctness and empty-input coverage exist; `range_search` still brute-forces over all XYZ rather than pruning via octree cells.
- [ ] `estimate_normals(Knn | Radius | Hybrid)` ships; normals are unit-length within 1e-5; smoke test against a synthetic plane produces normals aligned to ±plane-normal for ≥ 99% of points.
- [ ] Python API exposed per §3.4; `.pyi` stubs updated.
- [ ] Benchmark: 10M-point kNN(k=16) on the reference machine completes in < 10 s build + < 2 s query; documented in `docs/performance/benchmarks.md`.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `kiddo` API changes between 4.x minors | Pin exact minor; compatibility test runs on every `cargo update`. |
| KD-tree cache goes stale after a user mutates internal state via an escape hatch | Mark the escape hatches `#[doc(hidden)]`; document that all public mutators invalidate. |
| Normal sign ambiguity bites ICP downstream | Document in RFC-0007 that point-to-plane / GICP must orient toward source-sensor position; no fix here. |
| Eigendecomposition numerical instability on degenerate covariances (e.g., collinear neighbors) | Detect rank deficiency, return a flagged normal (`nx=nan`) or the largest eigenvector direction. Document the failure mode. |

## 6. Out of scope

- Oriented normals (`orient_normals_*`) — not required for M5/M6.
- Feature descriptors (FPFH, SHOT) — deferred, no RFC in this batch.
- GPU neighbor search — stretch (§3.6).

## 7. Open questions

- Should `estimate_normals` return the normals or mutate in place (current proposal)? Mutation matches Open3D's legacy behavior. Some users may want a non-mutating variant — add `estimated_normals() -> PointCloud` later if asked.
- Radius-search Python return shape — list of arrays vs. flat array + offsets ("CSR-style"). Proposal: both, with the CSR pair available via `knn_csr(r)` for bulk workflows that want to stay on NumPy. Decide once a user surfaces.
