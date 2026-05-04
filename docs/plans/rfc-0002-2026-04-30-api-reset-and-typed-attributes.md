# RFC-0002: API Reset & Typed Attributes (M1)

- **Status:** Partial
- **Date:** 2026-04-30
- **Author:** Master PM (agent)
- **Tracking issue:** LEO-36 (parent), per-milestone LEO issue TBD
- **Related:** RFC-0001 (roadmap), RFC-0003 (unblocks feature selection)

## 1. Summary

Reset the public API before the feature work starts. Introduce a typed `AttributeValue` enum so LAS fields round-trip losslessly, make NumPy getters true zero-copy tensor views, accept `f64`/`i32`/`i64` input with explicit autocast, fix the miscategorized `RandomSampleStrategy`, and clean up repo hygiene. No new point-cloud algorithms; this RFC is entirely about making M2–M6 cheap to build.

## 2. Motivation

The audit on LEO-36 (2026-04-30 comment) flagged these substantive issues:

1. **f32-only attribute storage.** `HashMap<String, Tensor1<f32>>` loses data every time we read LAS classification (u8), return numbers (u8), GPS time (f64), or source ID (u16). Feature-based selection (RFC-0003) cannot do exact equality on floats, so this is a hard blocker.
2. **Getters rebuild `Array2` from `Vec<Vec<f32>>`.** Every `get_xyz()` call materializes the tensor, flattens it, and re-shapes. On a 10M-point cloud this is gigabytes of needless traffic.
3. **`RandomSampleStrategy` is not random** — `src/point_cloud/voxel.rs:18` returns `indices[indices.len() / 2]`. Users relying on it for subsampling get biased results.
4. **Legacy `list[list[float]]` inputs** are still accepted in `examples/basic_usage.py`; the stub files advertise NumPy only. Pick one and enforce it.
5. Repo hygiene: `YOUR_USERNAME` placeholders in README, `pyproject.toml` `readme = "ai_doc/README.md"` points to a stale doc that contradicts the root README.

Per LEO-36 ("forward/backward compatibility is not a constraint"), this is the right moment to break everything at once.

## 3. Detailed design

### 3.1 `AttributeValue` enum

```rust
// src/point_cloud/attribute_value.rs (new)
pub enum AttributeValue {
    F32(Tensor1<Backend, f32>),
    F64(Tensor1<Backend, f64>),
    U8 (Tensor1<Backend, u8>),
    U16(Tensor1<Backend, u16>),
    U32(Tensor1<Backend, u32>),
    I32(Tensor1<Backend, i32>),
    I64(Tensor1<Backend, i64>),
    Bool(Tensor1<Backend, bool>),
}

impl AttributeValue {
    pub fn len(&self) -> usize { /* dispatch */ }
    pub fn dtype(&self) -> AttrDType { /* tag */ }
    pub fn gather(&self, indices: &Tensor1<Backend, i64>) -> Self { /* M2 hook */ }
}
```

`HighPerformancePointCloud.attributes` becomes `HashMap<String, AttributeValue>`. Intensity and RGB become *standard attributes* stored in the same map under reserved names (`"intensity"`, `"red"`, `"green"`, `"blue"`); the dedicated `Option<Tensor1>` fields go away. Convenience accessors (`has_intensity`, `set_intensity`, …) remain on `PointCloud` but are thin wrappers over the attribute map.

Rationale: unifies the storage model, lets M2 express "select where `attributes["classification"] == 2`" without a special case, and removes the RGB f32/u8 inconsistency.

### 3.2 Zero-copy NumPy getters

Current path (`src/lib.rs::get_xyz`): `tensor2_to_vec` → `Vec<Vec<f32>>` → flatten → `Array2` → `PyArray2`. Replace with direct `Tensor -> TensorData -> PyArray` on CPU devices. On GPU devices, implement a one-shot `tensor.to_data()` that transfers once, caches nothing, and returns a NumPy array backed by the CPU copy (zero-copy from user's POV, one device→host copy only).

`src/interop/numpy.rs` becomes the single boundary:

```rust
pub fn tensor2_to_numpy<T: BurnElement + ToPyObject>(
    py: Python<'_>, tensor: &Tensor2<Backend, T>,
) -> PyResult<Bound<'_, PyArray2<T>>> { ... }
```

with a blanket `impl` for every dtype the `AttributeValue` enum carries.

### 3.3 Input dtype handling

`PointCloud.from_xyz(xyz)` currently requires `dtype=float32`. Relax to:

- `float32` → zero-copy.
- `float64` → autocast to `float32` with `log::debug!` noting the cast; emit a `UserWarning` on the Python side once per process.
- `int32` / `int64` → autocast to `float32`.
- Anything else → `TypeError` with an actionable message.

Attributes keep their native dtype end-to-end. `set_attribute("classification", np.asarray([2, 6], dtype=np.uint8))` round-trips as `AttributeValue::U8`.

### 3.4 Voxel downsample strategies

Rename and fix:

- `DownsampleStrategy.RANDOM` → `DownsampleStrategy.RANDOM_SEEDED(seed: u64)`. Use `rand_chacha::ChaCha8Rng::seed_from_u64` for deterministic-by-seed sampling. Add `rand = "0.9"` + `rand_chacha = "0.9"` to `Cargo.toml`.
- `DownsampleStrategy.CENTROID` → `DownsampleStrategy.NEAREST_TO_CENTROID` (clearer; it's not averaging, it's picking the nearest existing point).
- Add `DownsampleStrategy.AVERAGE` — compute the actual centroid (average of XYZ and of every attribute) and emit a synthetic point. This matches Open3D's `voxel_down_sample` semantics.

Python-side:

```python
class DownsampleStrategy:
    RANDOM_SEEDED: int      # arg: seed (see voxel_downsample(voxel, strategy, seed=...))
    NEAREST_TO_CENTROID: int
    AVERAGE: int
```

### 3.5 Repo hygiene

- `sed -i 's/YOUR_USERNAME/ArkWhale/g' README.md mkdocs.yml .github/workflows/*.yml`.
- Point `pyproject.toml::readme` to `README.md`. Delete `ai_doc/` contents that are out-of-date summaries (`ai_doc/PROJECT_SUMMARY.md`, `ai_doc/CHECKLIST.md`, etc.); keep `ai_doc/README.md` only if it offers material not in `README.md` (it doesn't today). These `ai_doc/*` files are AI-generated stubs with incorrect CI and file-count claims.
- Add `docs/plans/` to the mkdocs nav so RFCs render on the site.
- Update the root README's `## 📈 路线图` section to link to RFC-0001 instead of the current ad-hoc bullet list.

### 3.6 Workspace registration — done

Per LEO-36 comment 2026-04-30, the repo is now registered with the Leo Multica workspace. No further action inside the repo, but `multica-home/knowledge/projects/pcl-rustic.md` needs the summary (handled by RFC-0001's author before closing M1 — see Acceptance Criteria).

## 4. API surface after M1

Nothing in priorities #3, #6, #7, #8 changes behavior yet — but the signatures that downstream RFCs rely on are now:

```python
PointCloud.from_xyz(xyz: NDArray)                                  # accepts f32/f64/i32/i64
PointCloud.from_numpy(dict_of_arrays: dict[str, NDArray])          # replaces from_dict, dtype-aware
pc.get_attribute(name: str) -> NDArray                             # native dtype
pc.set_attribute(name: str, data: NDArray) -> None                 # keeps caller's dtype
pc.voxel_downsample(voxel_size: float, strategy: int, *, seed: int | None = None) -> PointCloud
```

`pc.get_xyz()`, `pc.get_intensity()`, `pc.get_rgb()` survive as convenience wrappers.

## 5. Acceptance criteria

- [ ] `AttributeValue` enum implemented with `Tensor1<_, T>` for the eight listed dtypes. The enum exists for the listed dtypes, but current storage is host `Vec<T>`, not typed Burn tensors.
- [x] Intensity and RGB are stored as standard attributes; the legacy `Option<Tensor1>` fields removed.
- [ ] `get_xyz`, `get_intensity`, `get_rgb`, `get_attribute` use the new zero-copy path; benchmark shows ≥ 10× speedup on the getter for a 10M-point cloud vs. the current `Vec<Vec<f32>>` materialization.
- [x] `PointCloud.from_xyz` accepts f32/f64/i32/i64 NumPy; `tests/test_point_cloud.py` adds `test_autocast_f64_input`, `test_autocast_int_input`, `test_reject_string_input`.
- [x] `DownsampleStrategy` exposes `RANDOM_SEEDED`, `NEAREST_TO_CENTROID`, `AVERAGE`; the legacy middle-index bug is gone; seeded determinism is tested with two runs under the same seed yielding identical outputs.
- [x] Repo hygiene items in §3.5 shipped; `pyproject.toml::readme` points to `README.md`, RFCs are in MkDocs nav, and the README now points to RFC-0001 for roadmap tracking.
- [ ] `multica-home/knowledge/projects/pcl-rustic.md` written.
- [ ] All existing tests pass; added tests above pass; `just ci` is green on Linux, macOS, Windows.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Burn lacks direct `Tensor1<Backend, u8>` constructors for all needed dtypes | Verify with a spike in the first 2 days; if blocked, lower u8/u16 to `Tensor1<_, i32>` internally and expose the original dtype via a sidecar `AttrDType` tag. Do NOT silently upcast in the Python getter. |
| Existing users rely on the middle-index "random" | No users yet (per LEO-36); call out in the release notes. |
| `ai_doc/` deletion loses historical AI-generated context | Keep one snapshot at `docs/plans/archive/ai_doc-2026-01-31-snapshot.md` in case future agents want to inspect. |

## 7. Out of scope (deferred to later RFCs)

- New selection / concatenate ops → RFC-0003.
- Tensor-native voxel binning → RFC-0004.
- KD-tree, normal estimation, outlier removal, ICP → RFC-0005 / RFC-0006 / RFC-0007.

## 8. Open questions

- Do we expose `AttributeValue` in Python as a first-class type, or only via NumPy dtypes? Proposal: NumPy dtypes only — users get arrays, not a wrapper. Decided unless M2 needs otherwise.
- Should `AVERAGE` downsample emit averaged attributes for non-numeric types (e.g. mode for u8 classification)? Proposal: yes — mode for integer attributes, mean for float. Flag in docstring.
