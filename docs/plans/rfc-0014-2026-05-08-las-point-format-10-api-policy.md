# RFC-0014: LAS Point Format 10 Export API And Payload Policy

- **Status:** Accepted
- **Date:** 2026-05-08
- **Author:** Codex
- **Related:** RFC-0013

## 1. Summary

Implement RFC-0013 point format 10 export with an explicit Python API and a
backward-compatible Rust API extension:

```python
pc.to_las(path, compress=False, *, point_format=None, las_version="1.4", drop_waveform=False)
```

`point_format=None` preserves the existing inference behavior for point formats
0 through 3. `point_format=10` writes LAS 1.4 point format 10 and treats every
non-XYZ standard dimension as optional at the public API boundary.

## 2. Decision

Point format 10 export uses these policy rules:

- Rust keeps the existing `to_las(path, compress)` behavior and adds a
  `to_las_with_options(path, compress, point_format, las_version,
  drop_waveform)` entry point for explicit export.
- Python keeps `path` and `compress` positional-compatible. New arguments are
  keyword-only to avoid changing the meaning of existing positional calls.
- `point_format=None` ignores `las_version` and preserves existing inference for
  formats 0 through 3.
- Explicit `point_format=0..3` writes that older point format with the requested
  LAS version if supported by the `las` crate.
- `point_format=10` requires `las_version="1.4"`.
- Other point formats are unsupported until separately implemented.
- Missing `gps_time`, RGB, NIR, and waveform option fields are materialized as
  LAS defaults so XYZ-only clouds can be exported as valid point format 10.
- Present standard attributes are mapped to LAS standard fields, not ExtraBytes.
- Legacy `float32` intensity and `uint8` RGB remain accepted on write, while
  preferred point format 10 precision uses `uint16`.
- Partial RGB is allowed: any present `red`, `green`, or `blue` attribute creates
  a LAS `Color`; missing channels default to zero.
- Non-default waveform metadata is rejected by default because waveform packet
  descriptor records and payload bytes are not yet preserved.
- `drop_waveform=True` permits writing no-waveform defaults by discarding all
  waveform point attributes and descriptor/payload references.

Waveform metadata state is normative:

| Point attributes | `drop_waveform=False` | `drop_waveform=True` |
|---|---|---|
| No waveform attributes | Write `Waveform::default()` | Write `Waveform::default()` |
| Any waveform attributes present and all values are default | Write `Waveform::default()` | Write `Waveform::default()` |
| Any non-default waveform value | Error | Discard waveform metadata and write `Waveform::default()` |

The waveform point attributes are `wavepacket_index`, `wavepacket_offset`,
`wavepacket_size`, `return_point_wave_location`, `x_t`, `y_t`, and `z_t`. A
partial waveform group is allowed only when every present waveform value is the
default value; any partial non-default group follows the non-default row above.
The default values are `wavepacket_index=0`, `wavepacket_offset=0`,
`wavepacket_size=0`, `return_point_wave_location=0.0`, `x_t=0.0`, `y_t=0.0`,
and `z_t=0.0`.

## 3. Rationale

RFC-0013 intentionally left the exact writer API implementation-defined. Adding
both `point_format` and `las_version` avoids hidden inference for an advanced
LAS 1.4 format and leaves room for future explicit LAS version behavior.

The waveform policy is intentionally conservative. Writing non-default waveform
metadata without descriptor and payload preservation could create references to
missing or invalid waveform data. Reject-by-default is safer than silently
emitting corrupt metadata, while `drop_waveform_payload=True` gives callers an
explicit lossy export path.

## 4. Acceptance Criteria

- [ ] Python stubs and bindings expose the decided `to_las` signature.
- [ ] Existing Python calls keep working: `to_las(path)`, `to_las(path, True)`,
  `to_las(path, compress=True)`, and `save_to_file()` for `.las`/`.laz` retain
  current inference and compression behavior when `point_format=None`.
- [ ] Rust keeps backward-compatible `to_las(path, compress)` and exposes
  explicit point format 10 options internally.
- [ ] `point_format=10` writes LAS 1.4 point format 10.
- [ ] Preferred PF10 precision attributes `uint16` intensity/RGB/NIR, `int16`
  scan angle, and all waveform metadata names map to LAS standard fields, not
  ExtraBytes.
- [ ] Legacy `float32` intensity and `uint8` RGB still write correctly.
- [ ] XYZ-only point format 10 export succeeds by materializing default option
  groups.
- [ ] Version mismatch and unsupported explicit point formats return clear
  errors.
- [ ] Partial RGB is accepted and missing channels default to zero.
- [ ] Non-default waveform metadata is rejected unless `drop_waveform=True`.
- [ ] Drop-mode output writes no-waveform defaults and does not emit waveform
  point attributes as ExtraBytes.
- [ ] RFC-0013 implementation and docs refer to this policy.
