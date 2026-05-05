# RFC-0003: Coordinate Ops & Selection (M2)

- **Status:** Partial
- **Date:** 2026-04-30
- **Author:** Master PM (agent)
- **Tracking issue:** LEO-36 (parent), per-milestone LEO issue TBD
- **Related:** RFC-0001 (roadmap), RFC-0002 (prerequisite), RFC-0004 (GPU lift follows)

## 1. Summary

Close LEO-36 priority #3 by shipping `concatenate`, a single `select(mask)` tensor primitive, and layered feature-/region-based selectors (`select_where`, `select_by_classification`, `select_intensity_range`, `crop_aabb`, `crop_obb`). Ship priority #9's end-to-end example in two variants (spatial-grid split, classification-aware split). CPU-first implementation — RFC-0004 moves the hot path onto the GPU.

## 2. Motivation

Priority #3 covers transform, concatenate, and selection (region-level) per LEO-36, and was expanded via the 2026-04-30 follow-up comment to include **feature-based selection** (classification labels, intensity ranges, any attribute predicate). Without these:

- The end-to-end example in priority #9 can't be written.
- Every other M3+ milestone has to rebuild its own point-masking helpers.
- LAS workflows (ground-only, building-only, intensity-thresholded) have no native expression.

## 3. Detailed design

### 3.1 Core primitive: `select`

One kernel everything else lowers to:

```rust
impl HighPerformancePointCloud {
    pub fn select(&self, mask: &Tensor1<Backend, bool>) -> Result<Self>;
    pub fn select_indices(&self, indices: &Tensor1<Backend, i64>) -> Result<Self>;
}
```

- `select(mask)` does a Burn `select` / `masked_select` over XYZ and every attribute. Attribute types flow through unchanged (RFC-0002 prerequisite).
- `select_indices(i)` is the gather form; `select(mask)` is sugar that computes `mask.nonzero()` first.
- Both preserve point order (stable index).
- Both live on the tensor device — once RFC-0004 lands, no host round-trip on GPU.

### 3.2 Feature-based selectors

Sugar over `select(mask)`:

```rust
pub enum AttrPredicate<T> {
    Eq(T),
    In(Vec<T>),
    Range { lo: T, hi: T, inclusive: bool },
    Gt(T), Ge(T), Lt(T), Le(T),
}

impl HighPerformancePointCloud {
    pub fn select_where<T>(&self, name: &str, pred: AttrPredicate<T>) -> Result<Self>
        where T: BurnElement + PartialOrd;

    // LAS-aware convenience
    pub fn select_by_classification(&self, codes: &[u8]) -> Result<Self>;
    pub fn select_return_number(&self, n: u8) -> Result<Self>;
    pub fn select_intensity_range(&self, lo: f32, hi: f32) -> Result<Self>;
    pub fn select_elevation_range(&self, lo: f32, hi: f32) -> Result<Self>;
}
```

`select_where` dispatches on the attribute's runtime dtype. Type mismatch between `T` and the stored dtype is a clean error ("attribute 'classification' is u8, not i32") — no silent casts.

Python ergonomics:

```python
# Primitive — accepts any boolean NumPy mask
ground = pc.select(pc.get_attribute("classification") == 2)

# Sugar
ground_and_buildings = pc.select_by_classification([2, 6])
first_returns        = pc.select_return_number(1)
bright               = pc.select_intensity_range(0.4, 1.0)
```

### 3.3 Spatial selectors

```rust
pub struct Aabb { pub min: [f32; 3], pub max: [f32; 3] }
pub struct Obb  { pub center: [f32; 3], pub extents: [f32; 3], pub rotation: [[f32; 3]; 3] }

impl HighPerformancePointCloud {
    pub fn crop_aabb(&self, bbox: &Aabb) -> Result<Self>;
    pub fn crop_obb (&self, obb:  &Obb)  -> Result<Self>;
    pub fn aabb(&self) -> Aabb;                    // returns the cloud's AABB
    pub fn obb(&self)  -> Obb;                     // principal-axis OBB
}
```

Both lower to `select(mask)` on the fly. No `BoundingVolume` trait — two concrete types is enough; add a trait only when a third implementer appears.

### 3.4 Concatenate

```rust
pub enum ConcatPolicy {
    Strict,      // all clouds must carry the same attribute set and dtypes
    Union,       // missing attributes zero-filled with a warning
    Intersection // only attributes present in every cloud are kept
}

impl HighPerformancePointCloud {
    pub fn concatenate(clouds: &[&Self], policy: ConcatPolicy) -> Result<Self>;
}
```

XYZ is concatenated on axis 0. Intensity/RGB are standard attributes now (RFC-0002), so they ride the attribute path. Default policy in the Python binding: `Strict` — forces authors to be explicit. `Union` is the ergonomic choice when concatenating slices of the *same* source cloud (the priority #9 scenario), and is picked automatically in the end-to-end example.

### 3.5 Convenience transform wrappers

```rust
pub fn translate(&self, t: [f32; 3]) -> Result<Self>;
pub fn scale    (&self, s: f32, center: Option<[f32; 3]>) -> Result<Self>;
pub fn rotate   (&self, r: [[f32; 3]; 3], center: Option<[f32; 3]>) -> Result<Self>;
```

Each composes a 4×4 matrix and calls the existing `transform`. Matches Open3D's `PointCloud.translate/scale/rotate` signatures so users migrating from Open3D don't have to re-learn.

### 3.6 End-to-end example (priority #9)

Ship two runnable scripts under `examples/`:

- `examples/split_grid_downsample_concat.py` — divide the cloud's AABB into an N×M×1 grid, `crop_aabb` each cell, `voxel_downsample` each with a cell-dependent `voxel_size` (e.g. denser near sensor), `concatenate(ConcatPolicy::Union)`. Documents the spatial partitioning pattern.
- `examples/classification_aware_downsample.py` — for each LAS classification code present, `select_classifications([code])`, apply a class-specific `voxel_size` (ground 0.5 m, vegetation 0.2 m, building 0.05 m, noise dropped entirely), `concatenate`. Documents the semantic partitioning pattern and is the realistic LAS workflow.

Both examples load a test LAS file from `tests/data/` (add a small 10k-point fixture if none exists today).

## 4. Acceptance criteria

- [ ] `select(mask)`, `select_indices(indices)` implemented and tested; operate in place on the Burn tensor device. Current implementation is host-backed, not device-native.
- [ ] `select_where`, `select_by_classification`, `select_return_number`, `select_intensity_range`, `select_elevation_range` implemented and tested with LAS fixtures. Synthetic tests exist, and LAS standard-attribute round-trip coverage now covers `select_by_classification`; full fixture coverage for all selectors is still open.
- [x] `crop_aabb`, `crop_obb`, `aabb()`, `obb()` implemented and tested.
- [x] `concatenate(&[&Self], ConcatPolicy)` implemented and tested for all three policies.
- [x] `translate`, `scale`, `rotate` match Open3D semantics; doc page at `docs/api/transform.md` updated.
- [ ] Both end-to-end example scripts run to completion against the test fixture, produce expected point-count reductions, and are referenced from `docs/getting-started/examples.md`. The scripts are present and documented, but fixture-backed acceptance is still open.
- [ ] New tests cover: empty selection / zero-point strict concat, single-cloud concat, dtype-mismatch `ConcatPolicy::Strict` rejection, classification round-trip (LAS → select_by_classification([2]) → LAS → read back → same subset). Empty selection, concat edge cases, and LAS classification round-trip are covered; fixture-backed examples are still open.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Burn `masked_select` missing for some dtypes | Fall back to `nonzero` + `gather`; already supported across backends in 0.20. |
| `select_where<T>` monomorphization bloat | Generic only over the six numeric dtypes; lean on `AttrPredicate::erase()` internally. |
| ConcatPolicy::Union surprises users | Hard-log the synthesized zero-fills; Python warns via `UserWarning`. |

## 6. Out of scope

- FPS downsampling, uniform-every-k downsampling — defer; priority #6 names random + voxel only.
- Geometry primitives beyond AABB/OBB (convex hulls, oriented polygons) — deferred.
- Tensor-native (GPU) voxel binning — moved entirely to RFC-0004. M2's voxel path is unchanged from M1.

## 7. Open questions

- Should `select_where(name, pred)` return `(Self, mask)` like outlier-removal APIs in RFC-0006, or just `Self`? Proposal: just `Self`. Selection is not a probabilistic operation; callers who want the mask can compute it themselves.
- Units of `voxel_size` on `crop_aabb` example — same as `voxel_downsample`, unspecified (implicitly meters for LAS, arbitrary otherwise). Flag in docstring; revisit if users report confusion.
