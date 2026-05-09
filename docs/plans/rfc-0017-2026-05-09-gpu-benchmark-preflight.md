# RFC-0017: GPU Benchmark Allocation Preflight

- **Status:** Superseded by RFC-0018
- **Date:** 2026-05-09
- **Author:** Codex
- **Related:** RFC-0009, RFC-0011, RFC-0016

## 1. Summary

Add a GPU allocation preflight to the RFC-0009 benchmark harness so high-scale
benchmark cases skip before creating WGPU tensors that exceed a configured
single-allocation limit.

**Supersession note:** RFC-0018 supersedes this RFC before implementation. The
accepted fix for the reported failure is to avoid WGPU for large tensor
benchmarks and use non-WGPU Dispatch backends, starting with CUDA on Linux.

This is not a CPU fallback. Standard and full benchmark modes remain GPU-first
under RFC-0016. If a case exceeds the configured conservative single-tensor
threshold, the benchmark run must report a clean pytest skip instead of
triggering the known large-allocation Rust panic that poisons later tests.

## 2. Motivation

RFC-0016 moved pcl-rustic from Burn Router to Burn Dispatch and explicitly
rejected solving full benchmark failures by forcing CPU execution. The original
reported failure still has a separate root cause: RFC-0009 full C20
concatenation materializes a single `200_000_000 x 3` `float32` XYZ tensor,
which is about 2.4 GB before attributes and other intermediates.

Some WGPU devices reject single buffers around this size even when total system
RAM or VRAM is larger. When Burn panics during that allocation, the Python test
process can see a `pyo3_runtime.PanicException`, and subsequent tests can fail
with poisoned shared state. That is an invalid benchmark failure mode.

## 3. Detailed Design

### 3.1 Preflight Scope

The preflight runs inside `tests/test_benchmark.py` before synthetic data is
allocated for a case. The benchmark harness must parametrize concat and
downsample cases so pytest can report a skip for one matrix case without
aborting unrelated cases in the same test method.

It estimates the largest single XYZ tensor allocation required by the case:

- PointCloud construction: `point_count * 3 * sizeof(float32)`.
- Concatenation result: `input_points * 3 * sizeof(float32)`.
- Downsampling input: `input_points * 3 * sizeof(float32)`.

The guard intentionally targets single tensor/buffer allocation size, not total
peak process memory. Total-memory benchmarking remains part of RFC-0009 output.

### 3.2 Limit Configuration

The default maximum single XYZ tensor allocation is a conservative 2 GiB:

```text
PCL_RUSTIC_BENCHMARK_MAX_XYZ_TENSOR_BYTES=2147483648
```

The environment variable may raise or lower the limit on dedicated benchmark
machines. Invalid or non-positive values fail fast with a pytest error.

This threshold is not proof of backend compatibility. Some devices may reject
smaller allocations, and some may support larger ones. It is a local harness
safety valve for the known 2.4 GB C20 concat result allocation, not a complete
WGPU capability query.

### 3.3 Skip Semantics

If a case exceeds the configured single-allocation limit, pytest skips that
parametrized case before any point cloud is created. The skip reason must
include:

- case id
- operation family
- estimated XYZ tensor bytes
- configured limit
- the environment variable used to override the limit

The skip must happen before WGPU allocation, so it must not poison the process
or cause follow-up benchmark tests to fail.

Skipped cases must not write rows into RFC-0009 CSV outputs. RFC-0009 CSV rows
remain measured benchmark rows only.

### 3.4 Evidence Policy

Skipped high-scale cases are not benchmark evidence under RFC-0011. They only
prove the harness avoided an unsafe allocation path. Published performance
claims still require recorded CSV rows from cases that actually ran.

Skipped standard/full cases do not close the RFC-0009 external evidence
register. The register remains open until measured CSV rows record the case,
commit SHA, device, command, and benchmark environment under RFC-0011.

## 4. Acceptance Criteria

- [ ] RFC-0009 benchmark cases are parametrized so pytest can skip one case
      without aborting unrelated cases in the same matrix.
- [ ] RFC-0009 benchmark tests preflight large single XYZ tensor allocations
      before creating point clouds.
- [ ] The preflight uses
      `PCL_RUSTIC_BENCHMARK_MAX_XYZ_TENSOR_BYTES` with a 2 GiB default.
- [ ] Cases that exceed the limit skip cleanly before allocation for the known
      over-limit paths without Rust panic or poisoned follow-up tests.
- [ ] No code path forces CPU execution for standard/full benchmarks.
- [ ] Tests cover the skip path with a small configured limit.
- [ ] Skipped cases write no RFC-0009 CSV rows.
- [ ] Docs/memory record that skipped cases are not RFC-0011 benchmark evidence
      and do not close the RFC-0009 external evidence register.

## 5. Verification

Local implementation checks:

```bash
PCL_RUSTIC_BENCHMARK_MAX_XYZ_TENSOR_BYTES=1000 rtk uv run pytest tests/test_benchmark.py -q --run-slow --benchmark-mode=full --no-cov
rtk uv run pytest tests/test_benchmark.py -q --run-slow --benchmark-mode=smoke --no-cov
rtk uv run ruff check tests/test_benchmark.py
```

Optional benchmark-run safety check on dedicated hardware:

```bash
rtk uv run pytest tests/test_benchmark.py -q --run-slow --benchmark-mode=full --no-cov
```

The optional full-mode command may produce skips or measured rows depending on
the configured threshold. It does not create performance evidence unless the
measured CSV rows are preserved with RFC-0011 metadata.

## 6. Out Of Scope

- Chunked point-cloud storage.
- Streaming GPU concatenation.
- Publishing new performance numbers for skipped cases.
- Changing public `PointCloud.concatenate(...)` semantics.

## 7. Follow-Up

If full RFC-0009 cases must run on GPUs with smaller single-buffer limits, add a
separate RFC for chunked XYZ tensor storage or streaming GPU concatenation.
