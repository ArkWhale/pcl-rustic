# RFC-0010: Host Typed Attribute Storage Amendment

- **Status:** Accepted
- **Date:** 2026-05-07
- **Author:** Codex
- **Tracking issue:** RFC-0002 follow-up
- **Related:** RFC-0002, RFC-0003, RFC-0004, RFC-0006, RFC-0007

## 1. Summary

Amend RFC-0002 so typed point attributes are stored as host `Vec<T>` values
inside `AttributeValue`, not as Burn `Tensor1<Backend, T>` values. XYZ remains
the tensor-backed coordinate hot path. Attributes remain exact typed arrays for
LAS and NumPy interoperability, and device-native attribute gather/cat is
deferred until Burn can support the required dtype and boolean-mask surface
without sidecar conversions.

This RFC converts the current implementation from a documented deviation into
the accepted storage contract.

## 2. Motivation

RFC-0002 originally required `AttributeValue::{F32,F64,U8,...}` to wrap typed
Burn tensors. Implementation work found that this design fights the current
backend constraints:

1. LAS attributes require exact `uint8`, `uint16`, `uint32`, `int32`, `int64`,
   `float64`, and bool round-trips.
2. Burn Router is already useful for XYZ math, but not for all LAS-oriented
   attribute dtypes and masks without widening or sidecar dtype tags.
3. Widening small integer attributes to tensor-supported numeric types would
   preserve values only if every getter/export path remembered the original
   dtype, which makes the storage model harder to reason about.
4. Most current attribute operations are semantic filters, LAS/table I/O,
   selection propagation, and concat policy checks. These are correctness-bound
   today, not GPU-bound.

Current evidence covers native NumPy dtype preservation, standard LAS
attributes, packed covariance rows, selection propagation, concat policies, and
outlier propagation. This RFC makes the storage contract explicit so future work
does not treat the host typed vectors as an unresolved deviation.

## 3. Detailed Design

### 3.1 Storage Contract

`HighPerformancePointCloud` stores:

```rust
xyz: Tensor2,
attributes: HashMap<String, AttributeValue>,
kdtree_cache: OnceCell<KdTreeIndex>,
```

`AttributeValue` stores typed host vectors:

```rust
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
```

`F32x6` remains the packed covariance representation used by registration.

### 3.2 Device Semantics

`PointCloud.to("gpu")` moves XYZ tensors to the target device. `pc.device()`
reports XYZ tensor residency only. Attribute vectors remain host-side and keep
their dtype. APIs that currently operate on attributes may materialize host
masks or host gather results.

This means RFC-0003 and RFC-0004 must not claim a fully device-resident
attribute pipeline while selection, concat, outlier masks, and attribute-aware
voxel downsampling still touch host attributes. RFC-0004 can still optimize
XYZ-heavy stages independently.

### 3.3 NumPy Getter Semantics

`get_attribute`, `get_intensity`, and `get_rgb` may materialize NumPy arrays
from host vectors. This is no longer a violation of RFC-0002. The required
guarantee is dtype-preserving output with one owned NumPy array per call.

XYZ getter optimization remains valuable, but it is tracked as an independent
performance improvement rather than a prerequisite for typed attribute storage.

### 3.4 RFC-0002 Amendment

RFC-0002 acceptance criteria are amended as follows:

- Replace "AttributeValue enum implemented with `Tensor1<_, T>`" with
  "AttributeValue enum implemented with exact typed host storage for the listed
  dtypes."
- Replace "all typed getters use the new zero-copy path" with "typed attribute
  getters preserve dtype; XYZ getter performance improvements are tracked by a
  separate benchmark-backed optimization item."
- Keep the existing public API, dtype round-trip, strategy rename, and repo
  hygiene criteria unchanged.

### 3.5 Cross-RFC Amendments

RFC-0010 also amends downstream RFC wording where it assumes device-resident
typed attributes:

- RFC-0003: selection and concat may use host masks/gather for typed
  attributes. Device-native selection remains future optimization work for XYZ
  and any future device-resident attribute subset.
- RFC-0004: GPU hot-path criteria apply to XYZ-heavy tensor work and must not
  require all typed attribute gather/cat paths to stay on device. CPU/GPU golden
  tests must still validate point counts, XYZ centroids, seeded randomness, and
  dtype-preserving attribute propagation.
- RFC-0006: outlier masks may be host `Vec<bool>` / NumPy bool arrays. A
  same-device boolean tensor mask is deferred until the selection contract is
  revisited.
- RFC-0007: covariance storage is `AttributeValue::F32x6(Vec<[f32; 6]>)`, not a
  separate 2D tensor attribute. The public NumPy shape remains `[N, 6]`.

### 3.6 Future Device Attribute Work

A future RFC may reintroduce device-resident attributes only if it can prove:

1. exact dtype round-trip for all public attribute dtypes,
2. boolean-mask gather and concat for every dtype,
3. LAS/table export without dtype sidecars leaking into user behavior,
4. CPU/GPU golden tests for masks, concat policies, and voxel strategy results,
5. benchmark evidence that the added complexity improves real workloads.

## 4. API Surface

No Python signature changes.

The observable contract is:

```python
pc.set_attribute("classification", np.array([2, 6], dtype=np.uint8))
assert pc.get_attribute("classification").dtype == np.uint8
```

Device movement continues to expose:

```python
gpu_pc = pc.to("gpu")
gpu_pc.device()
```

but this reports XYZ tensor residency, not attribute-vector residency.

## 5. Acceptance Criteria

- [ ] RFC-0002 is updated to reference this amendment and mark host typed
  attribute storage as accepted.
- [ ] RFC-0003, RFC-0004, RFC-0006, and RFC-0007 are updated where their
  storage, mask, selection, concat, or covariance wording conflicts with this
  amendment.
- [ ] `docs/api/pointcloud.md` explicitly states that XYZ is tensor-backed and
  typed attributes are host vectors.
- [ ] `docs/memory/implementation-progress.md` removes the RFC-0002 storage
  scope-decision gap and keeps any remaining getter/benchmark gaps separate.
- [ ] Focused Python dtype round-trip coverage exists for every public
  `AttributeValue` dtype: `float32`, `float64`, `uint8`, `uint16`, `uint32`,
  `int32`, `int64`, `bool`, and packed `float32[N, 6]`.
- [ ] Existing LAS standard-attribute propagation, concat policy, outlier
  propagation, and covariance tests still pass.
- [ ] `cargo test --lib` and the focused Python typed-attribute tests pass.

## 6. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Users assume `to("gpu")` moves attributes too | Document that device movement currently applies to XYZ tensors only. |
| RFC-0004 GPU work overclaims device residency | Keep RFC-0004 acceptance criteria tied to explicit CPU/GPU golden tests and benchmark evidence. |
| Host attribute gather becomes a bottleneck | Add benchmark evidence before changing storage again. |
| Future Burn support makes this decision obsolete | Require a new RFC with dtype, mask, export, and benchmark proof before migration. |

## 7. Out Of Scope

- Tensor-native voxel binning and GPU selection for RFC-0004.
- Zero-copy XYZ getter benchmarking.
- Full covariance-weighted GICP.
- Large-scale benchmark execution on high-memory hardware.

## 8. Open Questions

None. This RFC intentionally resolves the RFC-0002 storage ambiguity.
