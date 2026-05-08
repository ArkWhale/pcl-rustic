# RFC-0013: LAS Point Format 10 Precision Support

- **Status:** Accepted
- **Date:** 2026-05-07
- **Author:** Codex
- **Related:** RFC-0002, RFC-0006, RFC-0010, RFC-0011

## 1. Summary

Add first-class LAS 1.4 point format 10 support. `pcl-rustic` must read and
write point format 10 LAS/LAZ files, preserve the format's standard dimensions
with their LAS-compatible precision, and keep every non-XYZ attribute optional
at the public API boundary.

Point format 10 uses point format 6 as its base and adds RGB, NIR, and waveform
packet dimensions. Support must include those fields without forcing callers to
set all of them before writing a valid LAS/LAZ file.

## 2. Source Basis

Laspy documents that LAS 1.4 is the latest LAS version it supports and that LAS
1.4 is compatible with point formats 0 through 10. Its point format 10 table
defines the added dimensions as RGB, NIR, and waveform data over point format
6: `red`, `green`, `blue`, `nir`, `wavepacket_index`,
`wavepacket_offset`, `wavepacket_size`, `return_point_wave_location`, `x_t`,
`y_t`, and `z_t`. Laspy lists `return_point_wave_location` as `uint32`, while
the Rust `las` crate models the same LAS field as `f32`; pcl-rustic follows the
`las` crate public model for writer/reader mapping and tests the numeric value,
not the laspy dtype label.

## 3. Motivation

Current LAS I/O covers older point formats and common dimensions, but it is not
precise enough for point format 10:

- RGB is downcast to 8-bit attributes, while LAS stores 16-bit color channels.
- NIR and waveform packet fields are not modeled as standard attributes.
- Point format 6+ fields such as scanner channel, overlap, user data, scan
  angle, and point source ID are not all preserved.
- XYZ is exposed as `float32`, which is useful for compute, but LAS precision
  depends on scaled integer storage and `f64` coordinate values at the I/O
  boundary.

Point format 10 is used for richer LAS 1.4 datasets. Losing precision or
requiring all fields to be present makes pcl-rustic unsuitable for round-trips
and partial export workflows.

## 4. Design

### 4.1 Attribute Contract

XYZ remains the only required data for constructing a `PointCloud`. Every LAS
standard dimension outside XYZ is optional in the public API. If an optional
attribute is absent during point format 10 export, the writer uses the LAS crate
default value for that field.

The standard point format 10 attribute names are:

| Attribute | Type | Required in `PointCloud` |
|---|---|---|
| `intensity` | `uint16` preferred; legacy `float32` accepted for compatibility | No |
| `return_number` | `uint8` storing LAS 4-bit range `0..=15` | No |
| `number_of_returns` | `uint8` storing LAS 4-bit range `0..=15` | No |
| `synthetic` | `bool` | No |
| `key_point` | `bool` | No |
| `withheld` | `bool` | No |
| `overlap` | `bool` | No |
| `scanner_channel` | `uint8` storing LAS 2-bit range `0..=3` | No |
| `scan_direction_flag` | `bool` | No |
| `edge_of_flight_line` | `bool` | No |
| `classification` | `uint8` | No |
| `user_data` | `uint8` | No |
| `scan_angle` | raw LAS 1.4 `int16` scaled integer | No |
| `point_source_id` | `uint16` | No |
| `gps_time` | `float64` | No |
| `red`, `green`, `blue` | `uint16` preferred; legacy `uint8` accepted on write | No |
| `nir` | `uint16` | No |
| `wavepacket_index` | `uint8` | No |
| `wavepacket_offset` | `uint64` | No |
| `wavepacket_size` | `uint32` | No |
| `return_point_wave_location` | `float32` | No |
| `x_t`, `y_t`, `z_t` | `float32` | No |

RFC-0002/RFC-0010 typed storage must add any missing scalar types required by
this table, specifically `U64` and `I16`. Python `set_attribute()` and
`get_attribute()` must support NumPy `uint64` and `int16`, and selection,
concatenation, outlier filtering, and ExtraBytes encoding must propagate those
types exactly.

Legacy convenience APIs remain supported:

- `set_intensity()` may continue to accept normalized `float32`; point format 10
  export must also accept preferred `uint16` `set_attribute("intensity", ...)`.
- `set_rgb()` may continue to expose 8-bit convenience RGB; point format 10
  export must also accept preferred `uint16` `red`, `green`, and `blue`
  attributes and must read point format 10 RGB as `uint16`.

Writer validation must reject out-of-range packed fields instead of truncating:
`return_number` and `number_of_returns` outside `0..=15`, and
`scanner_channel` outside `0..=3`, are errors.

### 4.2 Coordinate Precision

The compute-facing XYZ tensor may remain `float32`, but LAS I/O must preserve
point format 10 coordinate precision when round-tripping unchanged imported
points. Implementation must store raw LAS integer `X`, `Y`, and `Z` coordinate
sidecars plus header scale/offset metadata used only for LAS export. It must
not claim point format 10 precision support if export only writes `float32` XYZ
converted to `f64`, because that can drift from the original scaled integer
records.

Precision sidecars are tied to the current XYZ values and have this lifecycle:

- `from_las()` creates raw integer precision sidecars for imported coordinates.
- Pure row-subset/reorder operations such as `select`, `select_indices`, SOR,
  ROR, and nearest/random voxel downsampling gather the sidecars with the same
  indices.
- `PointCloud.concatenate(...)` preserves sidecars only when every non-empty
  input has compatible precision metadata; otherwise it drops them and export
  falls back to compute XYZ.
- Coordinate-mutating operations such as `transform`, `transform_3x3`,
  `translate`, `scale`, `rotate`, and `rigid_transform` must drop the sidecars.
- Synthetic-coordinate operations such as `voxel_downsample(..., AVERAGE)` must
  drop the sidecars.
- Direct mutable XYZ replacement must drop the sidecars.

### 4.3 Reader Behavior

`PointCloud.from_las()` must accept LAS/LAZ point format 10 files. It must:

- Load XYZ into the existing compute tensor.
- Materialize all point format 10 standard dimensions as attributes with the
  names and dtypes in section 4.1, including fields whose LAS values are
  defaults. This makes reader behavior deterministic even though the public API
  does not require callers to set those attributes before writing.
- Preserve standard LAS ExtraBytes independently from point format 10 standard
  dimensions. This applies to laspy-authored ExtraBytes metadata, not only the
  existing `pcl-rustic` private VLR schema.
- Resolve attribute-name collisions by keeping point format 10 standard fields
  under the names in section 4.1 and prefixing conflicting ExtraBytes names with
  `extra_`.
- Preserve LAS coordinate precision according to section 4.2.

### 4.4 Waveform Payload Scope

Point format 10 waveform metadata fields are not self-contained: packet
descriptor indices, byte offsets, and sizes reference waveform packet
descriptor records and waveform data payloads stored in LAS EVLR/WPD structures.

Repo-local support in this RFC is split:

- Required now: read and write the point-record waveform metadata attributes in
  section 4.1, and write valid default/no-waveform option payloads when callers
  do not provide waveform attributes.
- Required for true waveform preservation claims: copy or regenerate waveform
  packet descriptor records and payload bytes, and prove offsets still reference
  valid payload data after export.

Until descriptor/payload preservation is implemented and tested, pcl-rustic must
not claim full waveform payload round-trip support. If non-default waveform
metadata is present but payload preservation is unavailable, point format 10
export must either reject the file with a clear error or write no-waveform
defaults only when the caller explicitly requests dropping waveform payloads.

### 4.5 Writer Behavior

Add an explicit way to request LAS point format 10 export. The exact Python API
is implementation-defined, but it must be discoverable and not rely on hidden
attribute inference alone. Acceptable designs include:

- `pc.to_las(path, compress=False, point_format=10)`
- `pc.to_las(path, compress=False, las_version="1.4", point_format=10)`

When `point_format=10`, the writer must produce LAS 1.4 point format 10 and map
any present optional attributes into their standard fields. Missing optional
attributes must be written as LAS defaults, not rejected.

The `las` crate represents several point format 10 groups as optional values on
`las::Point`. For point format 10 export, the writer must explicitly materialize
default option payloads required by the selected format:

- `gps_time = Some(0.0)` when no `gps_time` attribute exists.
- `color = Some(Color { red: 0, green: 0, blue: 0 })` when RGB attributes are
  absent.
- `nir = Some(0)` when `nir` is absent.
- `waveform = Some(default waveform packet)` when waveform attributes are
  absent.

This is distinct from the public optional-attribute contract: callers do not
need to provide these attributes, but the writer must still satisfy the LAS
record layout required by point format 10.

## 5. Acceptance Criteria

- [x] A LAS 1.4 point format 10 fixture created with laspy loads through
  `PointCloud.from_las()` without error.
- [x] Reader tests verify all point format 10 standard attributes round-trip
  with the dtypes in section 4.1.
- [x] Reader tests verify a minimal point format 10 fixture with only XYZ set
  imports successfully and materializes all point format 10 standard dimensions
  with default values and section 4.1 dtypes.
- [x] Reader tests verify laspy-authored standard ExtraBytes are decoded,
  preserved separately from point format 10 standard dimensions, and collision
  names are prefixed with `extra_`.
- [x] Writer API can explicitly request `point_format=10`.
- [x] Writer tests verify point format 10 export works when only XYZ is present.
- [x] Writer tests verify point format 10 export materializes required
  `las::Point` option groups as default values when public optional attributes
  are absent.
- [x] Writer tests verify point format 10 export maps every present optional
  attribute to its LAS standard field.
- [x] Writer tests verify out-of-range `return_number`, `number_of_returns`,
  and `scanner_channel` values are rejected rather than truncated.
- [x] Precision tests prove 16-bit RGB/NIR, `uint64` waveform offsets,
  `int16` scan angle, and LAS coordinate precision are preserved through
  `from_las -> to_las -> from_las`.
- [x] Coordinate precision tests compare raw LAS integer `X`, `Y`, and `Z`
  records plus scale/offset metadata after `from_las -> to_las` for unchanged
  imported points.
- [x] Waveform tests prove default/no-waveform point format 10 export is valid,
  and that non-default waveform metadata is either fully payload-preserved or
  rejected unless the caller explicitly opts into dropping waveform payloads.
- [x] Precision sidecar tests prove sidecars are gathered through selection,
  dropped by coordinate-mutating operations, and handled deterministically by
  concat.
- [x] Python dtype tests prove `uint64` and `int16` work through
  `set_attribute`, `get_attribute`, selection, concat, and LAS ExtraBytes.
- [x] Existing LAS standard-attribute and ExtraBytes tests still pass.
- [x] `docs/api/io.md` or equivalent I/O docs document point format 10 support,
  optional attributes, attribute names, dtypes, and precision behavior.

## 6. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| The current `las` crate has incomplete ergonomic support for all point format 10 fields | Spike against `las::Point` and, if necessary, use raw point APIs or add a narrow adapter module that isolates crate-specific details. |
| Existing users rely on 8-bit RGB attributes | Keep legacy `uint8` RGB accepted on write by widening to 16-bit; read point format 10 as `uint16` to avoid precision loss. |
| Coordinate precision sidecars complicate compute APIs | Keep compute-facing `get_xyz()` unchanged and document sidecars as LAS I/O preservation metadata. |
| Optional attributes conflict with LAS record fields that always physically exist | Treat “optional” as an API contract: callers are not required to provide them; writer defaults are allowed, but reader precision tests must define which fields are materialized. |
| ExtraBytes names collide with standard point format 10 names | Standard dimensions keep canonical names; conflicting ExtraBytes get an `extra_` prefix and must be documented. |

## 7. Open Questions

- Whether the writer API should add `point_format` only, or both `las_version`
  and `point_format`. Recommendation: add both so future LAS 1.2/1.3/1.4
  compatibility is explicit.
