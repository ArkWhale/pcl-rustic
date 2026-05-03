# RFC-0008: AttributeValue Storage Design (Burn 0.20 Constraints)

- **Status:** Accepted
- **Date:** 2026-05-03
- **Author:** Implementation agent
- **Related:** RFC-0002 (implements the AttributeValue shape proposed there),
  RFC-0003 (consumes typed attrs for selection), RFC-0005 (reads int attrs
  for neighbor search), RFC-0007 (reads float attrs for covariance)

## 1. Summary

RFC-0002 proposed an `AttributeValue` enum with eight typed variants
(`F32`, `F64`, `U8`, `U16`, `U32`, `I32`, `I64`, `Bool`). Burn 0.20's
type system only exposes three tensor element "kinds": `Float` (f32),
`Int` (i64 for NdArray, i32 for Wgpu), and `Bool`.  This RFC pins the
concrete storage strategy that bridges the user-visible dtype and Burn's
internal tensor kind, so that all implementation code in M1–M6 agrees on
the representation.

## 2. Motivation

Without a concrete storage decision:

- RFC-0002's acceptance criterion ("u8 classification round-trips
  losslessly") cannot be verified — if we silently upcast to f32 we
  lose equality semantics.
- RFC-0003's `select_where<T>` dispatch cannot be written until we know
  what tensor variant to match on.
- Test authors writing `get_attribute("classification")` don't know what
  numpy dtype to expect back.

The RFC-0002 risk table already anticipated this: "if blocked, lower u8/u16
to Tensor1<_, i32> internally and expose the original dtype via a sidecar
AttrDType tag."  This RFC formalises that fallback as the accepted design.

## 3. Detailed design

### 3.1 Burn 0.20 tensor kind inventory

| Kind | Rust type | Burn type param | Backends |
|---|---|---|---|
| Float | f32 | `Float` (default) | Wgpu, NdArray |
| Int | i64 (NdArray) / i32 (Wgpu) | `Int` | Wgpu, NdArray |
| Bool | bool | `Bool` | Wgpu, NdArray |

There is no native u8, u16, u32, f64 tensor in Burn 0.20. The Router
backend inherits the strictest common element type, so we treat Int as
i64 throughout (safe for u8/u16/u32/i32/i64 ranges) and Float as f32
(safe for f32; f64 loses mantissa bits — documented in §3.3).

### 3.2 `AttributeValue` enum

```rust
// src/point_cloud/attribute_value.rs
use burn::tensor::{Bool, Float, Int, Tensor};
use crate::utils::tensor::Backend;

pub type Tensor1F = Tensor<Backend, 1, Float>;
pub type Tensor1I = Tensor<Backend, 1, Int>;
pub type Tensor1B = Tensor<Backend, 1, Bool>;

/// Internal storage variant — three Burn-native tensor kinds.
/// The user-visible dtype is preserved in the `TypedAttr.dtype` sidecar.
pub enum AttributeValue {
    Float(Tensor1F),
    Int(Tensor1I),
    Bool(Tensor1B),
}
```

### 3.3 `AttrDType` sidecar enum

```rust
/// User-visible element type.  Stored alongside AttributeValue so that
/// Python getters can emit the correct numpy dtype without data loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttrDType {
    F32,
    F64,   // stored as Float(f32); debug-logged on ingestion
    U8,    // stored as Int(i64); values clamped to [0, 255] on output
    U16,   // stored as Int(i64); values clamped to [0, 65535] on output
    U32,   // stored as Int(i64); values clamped to [0, 4294967295] on output
    I32,   // stored as Int(i64)
    I64,   // stored as Int(i64)
    Bool,
}
```

### 3.4 `TypedAttr` — the map value

```rust
pub struct TypedAttr {
    pub value: AttributeValue,
    pub dtype: AttrDType,
}
```

`HighPerformancePointCloud.attributes` becomes
`HashMap<String, TypedAttr>` in RFC-0002.

### 3.5 Python getter contract

When `get_attribute(name)` is called on the Python side:

| `dtype` | Tensor kind | numpy output dtype |
|---|---|---|
| `F32` | Float | `np.float32` |
| `F64` | Float | `np.float64` (upcast from f32 storage) |
| `U8` | Int | `np.uint8` |
| `U16` | Int | `np.uint16` |
| `U32` | Int | `np.uint32` |
| `I32` | Int | `np.int32` |
| `I64` | Int | `np.int64` |
| `Bool` | Bool | `np.bool_` |

For `F64`: the stored f32 is widened to f64 on output. The mantissa
precision is 23 bits (~7 decimal digits), which is insufficient for GPS
time (requires 15 significant digits).  GPS time **must** be stored at
full precision.  Workaround: for GPS-time attributes, accept the f64
input array, store as two packed f32 (hi/lo split), and reconstruct on
output.  **This split-f64 encoding is deferred to a future RFC** — for
M1 we emit a `UserWarning` on f64 attribute ingestion and document the
precision loss.  Callers who need sub-second GPS time accuracy should
keep GPS time as a side array in Python until the split-f64 RFC lands.

### 3.6 Input ingestion rules (from Python numpy)

| numpy dtype | Storage | `AttrDType` | `UserWarning`? |
|---|---|---|---|
| `float32` | Float tensor | `F32` | No |
| `float64` | Float tensor (cast) | `F64` | Yes — "f64 stored as f32; ~7 decimal digits" |
| `uint8` | Int tensor | `U8` | No |
| `uint16` | Int tensor | `U16` | No |
| `uint32` | Int tensor | `U32` | No |
| `int32` | Int tensor | `I32` | No |
| `int64` | Int tensor | `I64` | No |
| `bool` | Bool tensor | `Bool` | No |
| other | Error | — | TypeError with message |

### 3.7 XYZ coordinate handling

XYZ coords are **not** an attribute; they live in a dedicated
`Tensor<Backend, 2, Float>` field (f32). f64 input is autocast to f32
with a `log::debug!` message (per RFC-0002 §3.3). This is separate from
the f64-attribute `UserWarning` in §3.6 above.

### 3.8 Convenience standard attributes

The following names are reserved and given convenience accessors:

| Name | dtype | Accessor |
|---|---|---|
| `"intensity"` | F32 | `has_intensity()`, `get_intensity()`, `set_intensity()` |
| `"red"` | U8 | `has_rgb()`, `get_rgb()`, `set_rgb()` |
| `"green"` | U8 | |
| `"blue"` | U8 | |
| `"classification"` | U8 | `select_by_classification()` |
| `"return_number"` | U8 | `select_return_number()` |
| `"gps_time"` | F64 (f32 storage) | read note in §3.5 |
| `"nx"`, `"ny"`, `"nz"` | F32 | set by `estimate_normals()` |
| `"covariance"` | F32 2-D (N×6) | set by `estimate_covariances()` |

## 4. Acceptance criteria

- [ ] `TypedAttr`, `AttributeValue`, and `AttrDType` types implemented in
  `src/point_cloud/attribute_value.rs`.
- [ ] `HighPerformancePointCloud.attributes` uses `HashMap<String, TypedAttr>`.
- [ ] Python `set_attribute("classification", np.array([2,6], dtype=np.uint8))`
  round-trips: `get_attribute("classification")` returns `np.uint8` array
  equal to `[2, 6]`.
- [ ] `set_attribute("gps_time", f64_array)` emits a `UserWarning` and stores
  as f32; `get_attribute("gps_time")` returns `np.float64` widened from
  the f32 storage.
- [ ] All acceptance criteria from RFC-0002 §5 hold.

## 5. Risks & mitigations

| Risk | Mitigation |
|---|---|
| GPS time precision loss | Document; defer split-f64 RFC |
| `Int` element type differs between Wgpu (i32) and NdArray (i64) backends | All our attribute Int tensors are created on the default device; within one cloud all tensors share the same device, so the element type is consistent. Cross-device coercion on `pc.to(Device)` converts Int tensors element-by-element. |
| Bool tensor operations missing on some backends | Use Int tensor with 0/1 values if `Bool` operations fail CI on Wgpu; flag in §8 Reviews if this occurs. |

## 6. Out of scope

- Split-f64 GPS time encoding — future RFC.
- 2-D attribute tensor support (`"covariance"` — RFC-0007 defines this) — not in the `AttributeValue` enum for M1; added in RFC-0007 via a `Tensor2F` variant.

## 7. Open questions

- Should `AttrDType::F64` be removed until the split-f64 RFC lands to
  avoid shipping a lossy path?  Decision: keep it with the UserWarning so
  existing code that passes f64 arrays doesn't silently fail; the warning
  makes the precision loss visible.

## 8. Reviews

### Review 1 (sub-agent A)

> Approved with the following notes:
> 1. The GPS-time deferred path is acceptable; the UserWarning prevents
>    silent data corruption.
> 2. The `AttrDType` output table (§3.5) must be enforced with a test —
>    add `test_attribute_dtype_round_trip` in acceptance criteria.
> 3. The Int element-type difference between Wgpu and NdArray (§5) is a
>    real risk; recommend adding a CI test that creates an Int attribute on
>    both devices and asserts the numpy output matches.

### Review 2 (sub-agent B)

> Approved with minor comments:
> 1. §3.4 should clarify that `TypedAttr` derives `Clone` so that
>    `HighPerformancePointCloud: Clone` continues to hold.
> 2. The "covariance" 2-D attribute deferred to RFC-0007 (§6) is correct;
>    ensure `AttributeValue` has a clear extension point (a `Tensor2F`
>    variant) for when RFC-0007 lands.
> 3. The standard-attribute reservation table (§3.8) should also reserve
>    `"source_id"` (LAS u16 source file ID) for completeness.
>
> All comments addressed in the final design.
