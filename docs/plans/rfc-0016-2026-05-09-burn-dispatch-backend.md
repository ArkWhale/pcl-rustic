# RFC-0016: Burn Dispatch Backend Migration

- **Status:** Implemented
- **Date:** 2026-05-09
- **Author:** Codex
- **Related:** RFC-0004, RFC-0009, RFC-0011, RFC-0012

## 1. Summary

Migrate pcl-rustic's tensor backend adapter from Burn's `Router<(Wgpu,
NdArray)>` backend to Burn's `Dispatch` backend after upgrading to the first
Burn release that includes the merged Dispatch backend work from
https://github.com/tracel-ai/burn/pull/4508.

The migration must keep pcl-rustic GPU-first for benchmark and production hot
paths. CPU fallback remains acceptable only when no GPU device exists or when a
caller explicitly requests CPU through the public `.to("cpu")` API. High-scale
benchmark failures must not be solved by silently forcing CPU execution.

The implementation must also prepare for the future Burn Tensor API proposed in
https://github.com/tracel-ai/burn/pull/4717, where the backend type parameter
is removed. All backend-specific type aliases and device constructors must
therefore remain isolated in `src/utils/tensor.rs`.

## 2. Motivation

RFC-0004 and RFC-0012 require GPU-resident hot paths where possible. The
current `Router<(Wgpu, NdArray)>` setup allows automatic CPU fallback, but it
also exposes router-specific device details throughout the codebase and leaves
large WGPU allocation behavior tied to Router internals.

Burn PR #4508 adds a Dispatch backend that is intended to select backend
implementations dynamically without requiring the current Router backend
composition. Moving to Dispatch should simplify runtime backend selection while
preserving the GPU-first contract.

Burn PR #4717 is still open, but its direction is clear enough to shape this
change: pcl-rustic should not spread `Tensor<Backend, D>` and backend device
types across implementation modules more than necessary. The current code
already centralizes most aliases in `src/utils/tensor.rs`; this RFC makes that
boundary explicit.

## 3. Detailed Design

### 3.1 Burn Upgrade

Update Burn dependencies from `0.20.1` to a compatible stable release that
contains `burn::{Dispatch, DispatchDevice}`. Burn `0.21.0` and
`burn-dispatch 0.21.0` are available in the public registry and are the initial
target.

Acceptance requires source compilation against the release crate rather than a
git dependency. If the release is unavailable in the configured registry, this
RFC remains accepted but unimplemented until the dependency can be resolved
without vendoring or pinning a temporary Git SHA.

### 3.2 Dispatch Feature Mapping

Dispatch WGPU variants are feature selected. The dependency configuration must
enable `dispatch` plus exactly one WGPU target feature for the supported
platform, and must remove `router`.

The default Linux development target is:

```toml
burn = { version = "0.21.0", features = ["std", "dispatch", "vulkan", "ndarray", "cpu"] }
```

`metal` may replace `vulkan` for macOS, and `webgpu` may replace it only for a
WebGPU target. Multiple WGPU target features must not be enabled together.

### 3.3 Tensor Adapter Boundary

`src/utils/tensor.rs` remains the only module that imports Burn backend types
directly.

It owns these aliases and constructors:

- `Backend`
- `BackendDevice`
- `Tensor2`
- `default_device()`
- `cpu_device()`
- `gpu_device()`
- `has_wgpu_device()`
- `tensor2_from_slice(...)`
- `tensor2_from_slice_on_device(...)`
- `tensor1_from_slice_on_device(...)`

Other modules may use these aliases and functions, but must not import
`burn::backend::{Router, Wgpu, NdArray}` or Dispatch-specific types directly.

Feature modules may use adapter aliases such as `Tensor2`, but should not spell
`Tensor<Backend, D>` directly. Existing direct uses, such as transform helper
tensors, should move behind adapter helper functions during this migration.

### 3.4 GPU-First Runtime Contract

`default_device()` must choose GPU when Burn reports an available WGPU device.
It may choose CPU only when no WGPU device exists.

`gpu_device()` must return `Result<BackendDevice>` and surface a clean Python
error through `.to("gpu")` if no GPU exists. It must not silently construct a
CPU device and should not rely on panic-style failure for normal user input.

`cpu_device()` remains available for explicit user calls and tests. It must not
be used as an implicit fix for benchmark scalability.

### 3.5 Benchmark Failure Handling

RFC-0009 `standard` and `full` modes must keep the default backend behavior.
They must not set an environment variable that forces CPU execution.

If high-scale WGPU allocation still fails after Dispatch migration, the next
fix must address GPU allocation strategy directly, such as chunked
concatenation, streaming benchmark data, or backend-specific tensor creation.
That follow-up requires either a separate RFC or an amendment to this one.

When a benchmark run is meant to support a GPU performance claim, the recorded
rows must show a Dispatch GPU device name. CPU-only Dispatch runs may validate
harness wiring but must not be used as GPU benchmark evidence under RFC-0011.

### 3.6 Future Tensor API Readiness

The codebase should be ready for the direction of PR #4717 by avoiding new
public references to backend-generic tensor signatures outside the tensor
adapter. The implementation should prefer local aliases such as `Tensor2`
instead of spelling `Tensor<Backend, 2>` in feature modules.

When Burn removes the backend parameter, the expected migration should be
mostly contained to `src/utils/tensor.rs` and its direct tests.

### 3.7 RFC And Memory Updates

This RFC amends RFC-0004 and RFC-0012 by replacing Router as the intended
runtime backend abstraction with Dispatch. On implementation, update:

- `docs/plans/README.md`
- RFC-0001 roadmap backend references, if they still name Router as current
  backend policy
- RFC-0004/RFC-0012 backend references where they name Router rather than the
  GPU-residency behavior
- `docs/memory/implementation-progress.md`

## 4. Acceptance Criteria

- [x] Burn is upgraded to `0.21.0` or a later compatible stable release that
      includes Dispatch backend support.
- [x] Cargo features enable `dispatch` plus exactly one WGPU target feature
      (`vulkan`, `metal`, or `webgpu`) and remove `router`.
- [x] `src/utils/tensor.rs` uses Dispatch aliases and Dispatch devices instead
      of `Router<(Wgpu, NdArray)>` and router `MultiDevice`.
- [x] `default_device()` remains GPU-first and does not force CPU for
      RFC-0009 benchmark modes.
- [x] `gpu_device()` returns `Result<BackendDevice>` and `.to("gpu")` surfaces a
      clean Python error when no GPU exists.
- [x] No non-adapter module imports Router, Dispatch, Wgpu backend, or NdArray
      backend types directly.
- [x] No non-adapter module constructs `Tensor<Backend, D>` directly; helper
      tensor construction goes through `src/utils/tensor.rs`.
- [x] Existing device-preservation tests for selection, concat, voxel
      downsample, and transforms continue to pass.
- [x] Device edge cases remain covered: source-derived empty selections preserve
      the source device, all-empty concat preserves the common source device,
      no-input concat returns a valid CPU empty cloud, and zero-size resources do
      not force GPU allocation failures.
- [x] RFC-0009 benchmark harness does not set `PCL_RUSTIC_DEFAULT_DEVICE` or
      any equivalent CPU-forcing environment variable.
- [x] GPU benchmark evidence is accepted only when benchmark rows record a
      Dispatch GPU device name.
- [x] RFC/backend references and `docs/memory/implementation-progress.md` are
      updated to reflect Dispatch as the accepted backend adapter.
- [x] The implementation documents any remaining WGPU single-buffer limitation
      as an unresolved GPU allocation problem, not as a CPU fallback policy.

## 5. Verification

Required local commands:

```bash
rtk cargo fmt --check
rtk cargo clippy -- -D warnings
rtk cargo test --lib
rtk uv run maturin develop
rtk uv run pytest tests/test_point_cloud.py -q --no-cov -k "device or concat or voxel or transform"
rtk uv run pytest tests/test_benchmark.py -q --run-slow --benchmark-mode=smoke --no-cov
```

If the benchmark smoke command is too slow for an ordinary development loop,
run the focused RFC-0009 harness tests first and run the full smoke command
before final commit.

## 6. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Dispatch is not available in the configured registry yet | Keep RFC accepted but implementation blocked; do not pin an unreleased Git dependency. |
| Dispatch API names differ from the draft PR | Update only `src/utils/tensor.rs`; do not leak Dispatch details across feature modules. |
| Incorrect feature flags produce CPU-only Dispatch | Require `dispatch` plus exactly one WGPU target feature and verify GPU device reporting where GPU evidence is claimed. |
| GPU allocation still fails for RFC-0009 full mode | Track as GPU allocation/chunking work, not as a CPU fallback policy. |
| PR #4717 changes the Tensor API after this migration | Keep backend-generic usage behind aliases so the follow-up edit is localized. |

## 7. Out Of Scope

- Chunked or streaming implementation of RFC-0009 full concatenation.
- Publishing new performance numbers.
- Changing the public Python device API beyond preserving `.to("cpu")`,
  `.to("gpu")`, `device()`, and `has_wgpu_device()`.
- Adding CUDA, ROCm, or other non-WGPU backends.

## 8. Remaining Limitations

- If Dispatch still hits WGPU buffer limits, should RFC-0009 full mode move to
  chunked GPU concatenation or split benchmark rows by chunk size? This remains
  unresolved GPU allocation work and must not be treated as a reason to force
  RFC-0009 `standard` or `full` runs onto CPU.
