# RFC-0018: Non-WGPU Dispatch Backend Selection

- **Status:** Implemented
- **Date:** 2026-05-09
- **Author:** Codex
- **Related:** RFC-0009, RFC-0011, RFC-0016, RFC-0017

## 1. Summary

Amend RFC-0016 to remove WGPU/Vulkan from pcl-rustic's default tensor backend
stack. Use Burn Dispatch with native CUDA when CUDA hardware is available,
LibTorch MPS on Apple Silicon/macOS when configured, and CPU otherwise.

RFC-0017 is superseded by this RFC. The correct fix for the observed full
benchmark failure is not a WGPU allocation preflight; it is avoiding WGPU for
large point-cloud tensors because wgpu currently exposes storage-buffer limits
around `i32::MAX` bytes even on GPUs with much more VRAM.

## 2. Motivation

The original `benchmark-full` failure attempted to allocate a 2.4 GB XYZ tensor
for C20 full concatenation:

```text
200_000_000 points * 3 coordinates * 4 bytes = 2,400,000,000 bytes
```

That is below the available 24 GB VRAM on the local RTX 3090, but above the
`2,147,483,647` byte class of WGPU/WebGPU binding limits documented in
https://github.com/gfx-rs/wgpu/issues/8105.

RFC-0016 correctly moved pcl-rustic to Burn Dispatch, but selected Dispatch
Vulkan. That still keeps the code on the wgpu path and therefore does not
remove the buffer-size blocker. pcl-rustic should select non-WGPU backends for
large tensor work:

- CUDA on Linux/NVIDIA machines.
- MPS through LibTorch on Apple Silicon/macOS when that backend is configured.
- CPU only when no supported accelerator is available or when the caller
  explicitly asks for `.to("cpu")`.

## 3. Detailed Design

### 3.1 Cargo Backend Features

The default Linux build enables Burn Dispatch with native CUDA and CPU:

```toml
burn = { version = "0.21.0", default-features = false, features = ["std", "dispatch", "cuda", "cpu"] }
```

WGPU/Vulkan/Metal/WebGPU features must not be enabled by default.

MPS is treated as a follow-up platform/backend configuration rather than a
Linux default. On macOS, a later feature may enable Burn's `tch` feature and
select `DispatchDevice::LibTorch(LibTorchDevice::Mps)` when `tch` reports MPS
availability. Burn-tch MPS requires a PyTorch-backed setup such as
`LIBTORCH_USE_PYTORCH=1`; this RFC does not require enabling LibTorch/MPS in
the Linux implementation.

### 3.2 Runtime Device Selection

`src/utils/tensor.rs` owns backend selection:

1. Try `DispatchDevice::Cuda(CudaDevice { index: 0 })` when compiled with CUDA
   support and a CUDA device can pass a tensor smoke check.
2. Try LibTorch MPS when compiled with the MPS/LibTorch backend on macOS and a
   minimal tensor smoke check passes.
3. Fall back to Dispatch CPU.

Selection must be explicit. Do not use `DispatchDevice::default()` for
accelerator selection, because its compile-time priority may not reflect actual
runtime availability.

The accelerator smoke check must exercise the same class of operations used by
point-cloud construction: host upload with `TensorData`, reshape, a simple
tensor operation, backend sync, and host readback.

The public API remains:

- `has_wgpu_device()` is deprecated in behavior and should be renamed in a
  later API RFC; for compatibility it may report whether any accelerator is
  available until a replacement `has_accelerator_device()` API is added.
- `.to("gpu")` maps to the selected accelerator device, not WGPU.
- `.to("cpu")` remains explicit CPU transfer.
- `device()` records the actual Dispatch device, such as `Cuda(Cuda(0))`,
  `LibTorch(Mps)`, or `Cpu(...)`.

### 3.3 Benchmark Policy

RFC-0009 standard/full benchmark modes must not force CPU and must not rely on
WGPU allocation preflight as the primary fix. Measured benchmark rows are valid
only if their `device_name` records the backend that actually ran.

If CUDA/MPS still cannot allocate a full benchmark tensor despite sufficient
hardware memory, that is a separate native-backend allocation bug or chunked
storage requirement and needs a follow-up RFC.

RFC-0011 still controls performance evidence. Smoke rows can verify that CUDA
is selected locally, but standard/full performance claims remain open external
evidence until recorded artifacts include the required hardware, command, mode,
case, operation, counts, device/backend, git SHA, wall time, throughput, and
run date metadata.

### 3.4 RFC-0017 Supersession

RFC-0017 is marked Superseded. Its preflight approach may still be useful as a
defensive harness feature in the future, but it is not the chosen fix for the
current buffer-limit failure.

RFC-0016 is amended by replacing its `dispatch` + `vulkan` implementation
policy with `dispatch` + `cuda` + `cpu` for Linux. RFC-0016, the RFC index, and
implementation memory must be updated so the implemented backend policy is not
self-contradictory.

## 4. Acceptance Criteria

- [x] Default Linux Cargo features use Burn Dispatch with `cuda` and `cpu`,
      not `vulkan`, `wgpu`, `metal`, `webgpu`, or `ndarray`.
- [x] `src/utils/tensor.rs` constructs `DispatchDevice::Cuda` for the default
      accelerator when CUDA is compiled in and a smoke check passes.
- [x] Accelerator selection does not use `DispatchDevice::default()`.
- [x] The CUDA smoke check covers host upload, reshape, a simple op, sync, and
      host readback.
- [x] CPU fallback is used only when no configured accelerator smoke check
      passes or the caller explicitly requests CPU.
- [x] Public `.to("gpu")` maps to the selected accelerator and returns a Python
      error when no accelerator exists.
- [x] RFC-0009 smoke benchmark rows record a non-WGPU accelerator device on
      accelerator hardware.
- [x] RFC-0017 is marked Superseded and not implemented as the current fix.
- [x] RFC-0016, `docs/plans/README.md`, and docs/memory record that RFC-0016 is
      amended by this non-WGPU backend policy.
- [x] RFC-0011 external evidence gates remain open for standard/full benchmark
      claims until measured artifacts are recorded.
- [x] MPS/LibTorch support is documented as a `tch`/PyTorch-backed
      follow-up/non-blocking path for this Linux CUDA implementation.

## 5. Verification

```bash
rtk cargo fmt --check
rtk cargo clippy -- -D warnings
rtk cargo test --lib
rtk uv run maturin develop
rtk uv run pytest tests/test_point_cloud.py -q --no-cov -k "device or concat or voxel or transform"
rtk uv run pytest tests/test_benchmark.py -q --run-slow --benchmark-mode=smoke --no-cov
```

On a CUDA host, the RFC-0009 smoke CSV must include `Cuda` in `device_name`.

## 6. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| CUDA feature requires CUDA runtime/toolchain availability | Verify in CI or local CUDA runner before marking implemented; keep CPU fallback for non-CUDA machines. |
| LibTorch MPS adds heavyweight dependencies on non-macOS targets | Keep MPS behind platform/backend configuration instead of enabling it in Linux default builds. |
| Existing `has_wgpu_device()` name becomes misleading | Preserve for compatibility now; add a follow-up API rename RFC. |
| CUDA still has large-allocation limits | Treat as native backend/chunked-storage work, not a WGPU preflight issue. |

## 7. Out Of Scope

- Chunked point-cloud storage.
- New public accelerator-detection API naming.
- Publishing new performance numbers.
- Enabling ROCm.
