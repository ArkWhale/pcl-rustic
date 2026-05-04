# Implementation Progress — RFC 0002-0007

## Current Status: M1 (RFC-0002) COMPLETE (Rust side), Python build pending

### What's Done

**M1 (RFC-0002): API Reset & Typed Attributes — RUST COMPLETE**

All Rust code compiles and passes tests. Waiting for `uv` upgrade to build the Python extension.

Key changes made:

1. **`src/point_cloud/attribute_value.rs`** (NEW)
   - `AttributeValue` enum with variants: F32, F64, U8, U16, U32, I32, I64, Bool
   - `AttrDType` enum for runtime type tagging
   - Methods: `len()`, `dtype()`, `gather()`, `select_mask()`, `concatenate()`, `zeros()`, `to_f32_vec()`, `memory_usage()`
   - Accessors: `as_f32()`, `as_u8()`, etc.

2. **`src/point_cloud/core.rs`** (REWRITTEN)
   - `HighPerformancePointCloud` now has: `xyz: Tensor2` + `attributes: HashMap<String, AttributeValue>`
   - Removed dedicated `intensity`, `rgb_r/g/b` fields — all stored as named attributes
   - Reserved attribute names: `intensity` (f32), `red/green/blue` (u8), `classification` (u8), `return_number` (u8), `number_of_returns` (u8), `gps_time` (f64)
   - New constructors: `from_xyz_vec(Vec<[f32;3]>)`, `from_tensor_xyz(Tensor2)`
   - Selection primitives: `select_mask(&[bool])`, `select_indices(&[usize])`

3. **`src/point_cloud/voxel.rs`** (REWRITTEN)
   - New `DownsampleStrategy` enum: `RandomSeeded { seed }`, `NearestToCentroid`, `Average`
   - `RandomSeeded` uses `ChaCha8Rng` for deterministic sampling (sorted voxel keys for reproducibility)
   - `Average` computes mean for f32/f64, mode for integer types, majority vote for bool
   - All strategies use `select_indices` internally (clean attribute propagation)

4. **`src/point_cloud/transform.rs`** (REWRITTEN)
   - Direct impl methods: `transform(&[[f32;4];4])`, `transform_3x3(&[[f32;3];3])`
   - New: `translate([f32;3])`, `scale(f32, center)`, `rotate(&[[f32;3];3], center)`
   - `rigid_transform(&[[f32;3];3], [f32;3])` — rotation + translation

5. **`src/point_cloud/selection.rs`** (NEW)
   - Feature selectors: `select_by_classification(&[u8])`, `select_intensity_range(lo, hi)`, `select_return_number(n)`, `select_elevation_range(lo, hi)`
   - Spatial: `crop_aabb(min, max)`, `aabb() -> (min, max)`
   - `concatenate(&[&Self], ConcatPolicy)` with Strict/Union/Intersection policies

6. **`src/interop/numpy.rs`** (REWRITTEN)
   - `from_xyz_array` accepts f32, f64, i32, i64 numpy arrays (autocast to f32)
   - `read_attribute_from_pyany` detects and preserves: f32, f64, u8, u16, u32, i32, i64, bool
   - `attribute_to_numpy` outputs correct dtype
   - `from_numpy(dict)` reads all keys as typed attributes

7. **`src/io/las_laz.rs`** (REWRITTEN)
   - Reads LAS fields as typed attributes: classification (u8), return_number (u8), gps_time (f64), etc.
   - Writes with full attribute fidelity
   - Supports format 0-3 selection based on available attributes

8. **`src/io/table.rs`** (REWRITTEN)
   - Updated to use new AttributeValue API (no old trait references)

9. **`src/lib.rs`** (REWRITTEN)
   - Complete Python bindings for: PointCloud, DownsampleStrategy
   - New strategy constants: RANDOM_SEEDED=0, NEAREST_TO_CENTROID=1, AVERAGE=2 (old RANDOM/CENTROID kept as aliases)
   - Selection methods exposed: `select(mask)`, `select_indices`, `select_by_classification`, `select_intensity_range`
   - `concatenate(clouds, policy)` as static method
   - Transform methods: `translate`, `scale`, `rotate`
   - `voxel_downsample(voxel_size, strategy, seed=None)`

10. **Removed old code:**
    - `src/traits/` directory (DownsampleStrategy trait, PointCloudCore trait, etc.)
    - `src/utils/reflect.rs` (old voxel grouping)
    - `src/point_cloud/attributes.rs` (merged into core)

11. **`Cargo.toml`** updated:
    - Added: `rand = "0.8"`, `rand_chacha = "0.3"`, `kiddo = "4"`, `nalgebra = "0.33"`, `once_cell = "1"`

12. **`pyproject.toml`** fixed:
    - `readme` now points to `README.md` (was `ai_doc/README.md`)

### Rust Tests (6/6 passing)
- `test_translate`
- `test_scale`
- `test_rigid_transform`
- `test_voxel_downsample_nearest_to_centroid`
- `test_voxel_downsample_random_seeded_deterministic`
- `test_voxel_downsample_average`

---

## What's Next (after uv upgrade)

1. **Build Python extension** with `maturin develop --uv`
2. **Update Python tests** (`tests/test_point_cloud.py`) for new API
3. **Update `.pyi` stubs** and `__init__.py`
4. **Commit M1**
5. **Proceed to M4** (RFC-0005: KD-tree, normals) — M2 selection is already mostly done in M1
6. **M5** (RFC-0006: Outlier removal)
7. **M6** (RFC-0007: ICP/GICP)
8. **M3** (RFC-0004: GPU hot-path) — partial, depends on GPU hardware

---

## Architecture Decisions Made

| Decision | Rationale |
|----------|-----------|
| `AttributeValue` uses `Vec<T>` not Burn tensors | Burn Router backend doesn't support u8/u16/u32/f64 element types; Vec gives correct dtype preservation |
| XYZ stays as Burn `Tensor2` | GPU acceleration via Burn for transform/matmul ops |
| Intensity/RGB/classification in unified attribute map | Simplifies selection, concatenation, and I/O; no special-case code |
| Voxel keys sorted before RNG iteration | Ensures seeded determinism regardless of HashMap order |
| `ConcatPolicy` as enum in `lib.rs` | Used by both `selection.rs` and Python bindings |
| `select_indices` is the single primitive | All other selection ops lower to it; clean attribute propagation |
