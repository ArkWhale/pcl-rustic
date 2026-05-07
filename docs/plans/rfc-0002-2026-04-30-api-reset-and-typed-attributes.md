# RFC-0002: API Reset & Typed Attributes (M1)

- **Status:** Partial (amended by RFC-0010)
- **Date:** 2026-04-30
- **Author:** Master PM (agent)
- **Tracking issue:** LEO-36 (parent), per-milestone LEO issue TBD
- **Related:** RFC-0001 (roadmap), RFC-0003 (unblocks feature selection), RFC-0010 (host typed attribute storage amendment)

## 1. Summary

Reset the public API before the feature work starts. Introduce a typed `AttributeValue` enum so LAS fields round-trip losslessly, preserve NumPy getter dtypes, accept `f64`/`i32`/`i64` input with explicit autocast, fix the miscategorized `RandomSampleStrategy`, and clean up repo hygiene. No new point-cloud algorithms; this RFC is entirely about making M2–M6 cheap to build.

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

**Amendment:** RFC-0010 supersedes the storage detail in this section. The
accepted implementation stores typed attributes as host `Vec<T>` values inside
`AttributeValue`, while XYZ remains tensor-backed. The public dtype-preserving
contract remains unchanged.

```rust
// src/point_cloud/attribute_value.rs
pub enum AttributeValue {
    F32(Vec<f32>),
    F64(Vec<f64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
    F32x6(Vec<[f32; 6]>),
}

impl AttributeValue {
    pub fn len(&self) -> usize { /* dispatch */ }
    pub fn dtype(&self) -> AttrDType { /* tag */ }
    pub fn gather(&self, indices: &[usize]) -> Result<Self> { /* host typed gather */ }
}
```

`HighPerformancePointCloud.attributes` becomes `HashMap<String, AttributeValue>`. Intensity and RGB become *standard attributes* stored in the same map under reserved names (`"intensity"`, `"red"`, `"green"`, `"blue"`); the dedicated `Option<Tensor1>` fields go away. Convenience accessors (`has_intensity`, `set_intensity`, …) remain on `PointCloud` but are thin wrappers over the attribute map.

Rationale: unifies the storage model, lets M2 express "select where `attributes["classification"] == 2`" without a special case, and removes the RGB f32/u8 inconsistency. RFC-0010 accepts host typed storage so the public dtype contract stays exact for LAS/NumPy attributes.

### 3.2 NumPy getters

Current path (`src/lib.rs::get_xyz`): `tensor2_to_vec` → `Vec<Vec<f32>>` → flatten → `Array2` → `PyArray2`. XYZ getter optimization remains open: replace this with a direct `Tensor -> TensorData -> PyArray` path on CPU devices, and a one-shot `tensor.to_data()` transfer on GPU devices.

Typed attribute getters (`get_attribute`, `get_intensity`, `get_rgb`) may return owned NumPy arrays materialized from host typed vectors. Their normative contract is dtype preservation, not zero-copy.

`src/interop/numpy.rs` becomes the single boundary:

```rust
pub fn attribute_to_numpy(py: Python<'_>, attr: &AttributeValue) -> Result<Py<PyAny>>;
```

with explicit dispatch for every dtype the `AttributeValue` enum carries.

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

- [x] `AttributeValue` enum implemented with exact typed host storage for the listed dtypes, per RFC-0010.
- [x] Intensity and RGB are stored as standard attributes; the legacy `Option<Tensor1>` fields removed.
- [ ] Typed attribute getters preserve dtype. XYZ getter performance and 10M-point benchmark evidence remain tracked as a separate optimization item.
- [x] `PointCloud.from_xyz` accepts f32/f64/i32/i64 NumPy; `tests/test_point_cloud.py` adds `test_autocast_f64_input`, `test_autocast_int_input`, `test_reject_string_input`.
- [x] `DownsampleStrategy` exposes `RANDOM_SEEDED`, `NEAREST_TO_CENTROID`, `AVERAGE`; the legacy middle-index bug is gone; seeded determinism is tested with two runs under the same seed yielding identical outputs.
- [x] Repo hygiene items in §3.5 shipped; `pyproject.toml::readme` points to `README.md`, RFCs are in MkDocs nav, and the README now points to RFC-0001 for roadmap tracking.
- [ ] `multica-home/knowledge/projects/pcl-rustic.md` written.
- [ ] All existing tests pass; added tests above pass; `just ci` is green on Linux, macOS, Windows.

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Future device-resident attributes may not support all LAS dtypes cleanly | RFC-0010 keeps typed attributes as host vectors unless a future RFC proves exact dtype round-trip, mask/gather, export, and benchmark behavior. |
| Existing users rely on the middle-index "random" | No users yet (per LEO-36); call out in the release notes. |
| `ai_doc/` deletion loses historical AI-generated context | Keep one snapshot at `docs/plans/archive/ai_doc-2026-01-31-snapshot.md` in case future agents want to inspect. |

## 7. Out of scope (deferred to later RFCs)

- New selection / concatenate ops → RFC-0003.
- Tensor-native voxel binning → RFC-0004.
- KD-tree, normal estimation, outlier removal, ICP → RFC-0005 / RFC-0006 / RFC-0007.

## 8. Open questions

- Do we expose `AttributeValue` in Python as a first-class type, or only via NumPy dtypes? Proposal: NumPy dtypes only — users get arrays, not a wrapper. Decided unless M2 needs otherwise.
- Should `AVERAGE` downsample emit averaged attributes for non-numeric types (e.g. mode for u8 classification)? Proposal: yes — mode for integer attributes, mean for float. Flag in docstring.
