# RFC-0012: RFC-0004 Device Residency Scope

- **Status:** Accepted
- **Date:** 2026-05-07
- **Author:** Codex
- **Related:** RFC-0004, RFC-0010, RFC-0011

## 1. Summary

RFC-0004 remains the GPU hot-path target, but its repo-local completion
contract is amended to match the current Burn 0.20 and RFC-0010 architecture.
XYZ tensors must preserve the input device across selection, concatenation,
voxel downsampling, and coordinate transforms. Typed attributes stay host-side
and dtype-preserving. Large GPU speedup claims remain external evidence gates
under RFC-0011 until a recorded benchmark artifact exists.

## 2. Motivation

RFC-0004 originally described a fully tensor-native voxel pipeline using
argsort, segmented reductions, and device-side gathers. The current codebase
uses Burn `Router<(Wgpu, NdArray)>` for XYZ storage and math, while typed LAS
attributes are host `Vec<T>` values per RFC-0010. Burn 0.20 does not provide a
small, stable point-cloud segment-reduction abstraction in this repo that can
replace the existing deterministic CPU grouping without a custom kernel effort.

The practical gap users can observe today is device drift: host-planned
selection, concatenation, voxel downsampling, and some transform temporaries
recreate XYZ tensors on the default device instead of the source device. That
breaks the narrowed RFC-0010 contract even before benchmark speedups can be
evaluated.

## 3. Decision

Implement RFC-0004 repo-local completion in two layers:

1. Device residency: every RFC-0004 hot-path operation that derives a new cloud
   from existing XYZ data must create the result XYZ tensor on the operation's
   source device. For multi-cloud concatenation, all non-empty inputs must
   share the same device and the result uses that device.
2. Host planning: masks, voxel groups, random choices, and typed attribute
   gather/cat/mode/mean operations may run on the host. This must be explicit
   in docs and must not be described as a measured GPU acceleration.
3. Correctness parity: CPU and GPU/device-preserving paths must retain the
   existing semantic guarantees for point count, nearest-centroid coordinates,
   seeded determinism, and dtype-preserving attribute propagation.

The future fully tensor-native voxel pipeline remains an optimization target,
not a blocker for marking RFC-0004 repo-local implementation complete.

## 4. RFC-0004 Amendment Map

| RFC-0004 criterion | RFC-0012 amendment |
|---|---|
| Voxel downsample keeps XYZ-heavy binning/reduction on the Burn tensor device where practical. | Amended: voxel planning may run on the host in this repo-local implementation, but the resulting XYZ tensor must be rebuilt on the source device. Fully tensor-native binning remains future work. |
| `select(mask)`, `select_indices(indices)`, `concatenate` keep result XYZ tensors on the input device. | Preserved and expanded: transform, translate, scale, rotate, rigid transform, and all voxel strategies also preserve source/common XYZ device. |
| Benchmark matrix runs in CI against `wgpu-vulkan` and `ndarray-cpu`. | Split: RFC-0009 smoke benchmark CI remains repo-local; measured GPU backend benchmark results remain external evidence under RFC-0011 until a recorded GPU artifact exists. |
| README performance table replaced with generated table and CPU-only row labeled. | Preserved as documentation work: remove or relabel stale CPU-only claims now; publish generated measured rows only from benchmark artifacts. |
| 50M LAZ pipeline shows >= 3x GPU speedup. | External evidence gate. Do not mark checked until a benchmark artifact records hardware, backend, dataset, command, commit, and result. |
| Golden tests for CPU/GPU point count, centroid tolerance, dtype propagation, and seeded determinism. | Preserved: repo-local tests must cover CPU semantics and attempt GPU coverage when WGPU is available, skipping cleanly otherwise. |
| `docs/performance/optimization.md` updated with device-selection guidance. | Preserved and expanded: docs must state current device-residency guarantees and external evidence limits. |

## 5. Implementation Requirements

- Add a tensor helper that builds `Tensor2` on a caller-provided
  `BackendDevice`.
- Update `HighPerformancePointCloud::select_indices` so non-empty and empty
  selections keep the source XYZ device.
- Update `HighPerformancePointCloud::concatenate` so the result keeps the
  common input XYZ device and rejects mixed-device non-empty inputs. If all
  inputs are empty, use the first input device; if there are no inputs, return
  an empty CPU cloud.
- Update `voxel_downsample` outputs so all strategies keep the source XYZ
  device.
- Update `transform`, `transform_3x3`, `translate`, `scale`, `rotate`, and
  `rigid_transform` so all temporary tensors are created on the source XYZ
  device.
- Add tests that assert `select`, `select_indices`, `concatenate`, and the
  three voxel strategies preserve CPU device residency and existing semantic
  outputs. Add a GPU test gated by a Rust helper equivalent to
  `WgpuDevice::device_count_total() > 0`; when the helper reports no adapter,
  the test must report a pytest skip (or Rust `#[ignore]`/early return with an
  explicit skip message if implemented as a Rust-only test), not a pass that
  silently omits the GPU path.
- Update performance docs to state that RFC-0004 currently guarantees
  device-preserving XYZ results, not recorded GPU speedups.

## 6. Acceptance Criteria

- [x] `select(mask)` and `select_indices(indices)` preserve the source XYZ
  device for empty and non-empty results.
- [x] `concatenate` preserves the common source XYZ device and errors on
  mixed-device non-empty inputs; all-empty input uses the first cloud's device.
- [x] `voxel_downsample` preserves the source XYZ device for
  `RANDOM_SEEDED`, `NEAREST_TO_CENTROID`, and `AVERAGE`.
- [x] `transform`, `transform_3x3`, `translate`, `scale`, `rotate`, and
  `rigid_transform` preserve the source XYZ device.
- [x] Golden tests cover point count parity, nearest-centroid coordinates within
  `1e-5`, dtype-preserving attribute propagation, and same-backend
  `RANDOM_SEEDED` determinism.
- [x] CPU device-residency tests cover selection, concatenation, and the full
  `select_by_classification -> voxel_downsample -> transform -> concatenate`
  pipeline.
- [x] GPU residency coverage is attempted when `WgpuDevice::device_count_total()
  > 0`; when unavailable, the test runner reports an explicit skip naming WGPU
  adapter unavailability.
- [x] README and `docs/performance/optimization.md` document the current device
  scope and external benchmark evidence requirements without stale CPU-only
  performance claims.

## 7. External Evidence Register

| Criterion | Artifact | Date | Git SHA | Hardware / Dataset | Status | Notes |
|---|---|---|---|---|---|---|
| RFC-0004 50M LAZ GPU-vs-CPU speedup target | None recorded | 2026-05-07 | None recorded | Requires hardware profile, OS/driver, backend/device, dataset identity or fixture hash, voxel size/strategy | Open | Artifact must record command, CPU wall time, GPU wall time, speedup, and output point count. |
| RFC-0004 backend benchmark matrix | None recorded | 2026-05-07 | None recorded | Requires benchmark runner identity and backend labels for WGPU and NdArray paths | Open | Artifact must record benchmark mode, point counts, voxel sizes, strategies, CSV path, and generated docs path. |
| README generated performance rows | None recorded | 2026-05-07 | None recorded | Requires source benchmark CSV and renderer environment | Open | Publish measured rows only after recording source CSV path, render command, commit SHA, measured rows, reviewer, and date. |

These items may remain unchecked while RFC-0004 is marked
`Implemented (external evidence open)` under RFC-0011.

## 8. Rejected Alternatives

| Alternative | Reason rejected |
|---|---|
| Keep RFC-0004 blocked until a full tensor-native voxel implementation exists | This would leave observable device drift unfixed and conflate repo-local correctness with a larger custom-kernel optimization project. |
| Claim GPU acceleration from device-preserving result tensors | RFC-0011 forbids performance claims without recorded artifacts. Device residency is necessary but not sufficient evidence of speedup. |
| Move typed attributes back to tensors for GPU gather/cat | RFC-0010 accepted host typed storage to preserve exact LAS and NumPy dtypes. Reversing it needs a separate storage RFC. |

## 9. Status Changes On Acceptance

- Mark this RFC `Accepted`.
- Update RFC-0004 to show it is amended by RFC-0012.
- When implementation and repo-local tests pass, mark RFC-0004
  `Implemented (external evidence open)` instead of `Proposed`.
- Keep the external evidence register open until benchmark artifacts exist.

## 10. Future Work

- Introduce a tensor-native voxel implementation once Burn exposes stable
  sort/segment/scatter primitives suitable for this workload, or after a future
  RFC accepts custom WGPU kernels.
- Add a dedicated GPU benchmark artifact for the original RFC-0004 50M LAZ
  speedup target on the reference machine.
