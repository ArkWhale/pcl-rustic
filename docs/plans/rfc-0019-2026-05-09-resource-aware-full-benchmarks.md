# RFC-0019: Resource-Aware Full Benchmark Execution

- **Status:** Implemented
- **Date:** 2026-05-09
- **Author:** Codex
- **Related:** RFC-0009, RFC-0011, RFC-0015, RFC-0018

## 1. Summary

Amend RFC-0009 full benchmark execution so measured rows remain GPU-backed
while the harness avoids native-backend allocation panics on hardware that
cannot safely run every full matrix case.

The WGPU/WebGPU `i32` buffer limit is resolved by RFC-0018, but full-mode C20
now exposes a separate native CUDA/CubeCL allocation issue: the
concatenate-only C20 row succeeds on `Cuda(Cuda(0))`, then
concatenate-and-voxelize attempts a single CUDA allocation of about
6,318,129,152 bytes and destabilizes the CubeCL worker. This RFC treats that
as a benchmark harness/resource-design issue, not permission to fall back to
CPU.

## 2. Motivation

The full benchmark matrix is intended for high-memory benchmark machines. On a
24 GB RTX 3090, the current full run proves that the previous WGPU limit is no
longer the blocker:

```text
C20 concatenate: 200,000,000 points, device_name=Cuda(Cuda(0)), row written
```

The next row fails while voxelizing the merged C20 result:

```text
cubecl-cuda: can't allocate buffer of size: 6318129152
cubecl-runtime: attempt to add with overflow
```

The current concat voxel size (`0.15`) preserves almost every point in the
synthetic C20 cloud, so the benchmark measures a pathological near-identity
voxelization and forces large output/staging allocations. Larger full-matrix
cases such as C40-C200 and D200-D400 can also exceed local hardware limits even
when the backend is native CUDA.

## 3. Detailed Design

### 3.1 Measured Rows Must Stay GPU-Backed

The benchmark harness must never use CPU fallback to make a GPU benchmark row
pass. For any row that is measured as a pcl-rustic benchmark:

- `device_name` must be the actual runtime device.
- GPU-required rows must require `Cuda`, `Mps`, or another configured
  accelerator in `device_name`.
- A row that cannot run on the current accelerator memory budget must be
  skipped with an explicit reason rather than measured on CPU.

### 3.2 Resource Budget Preflight

`tests/test_benchmark.py` will estimate each case's peak working set and
largest single device allocation before allocating large arrays. The budget
check must run before synthetic NumPy data generation, `PointCloud.from_numpy`,
`PointCloud.concatenate`, or `voxel_downsample`.

Byte accounting uses explicit constants:

- `xyz_bytes_per_point = 12`
- `non_xyz_attribute_bytes_per_point = 14`
- `total_input_bytes_per_point = 26`

The budget should use:

- input point count and total bytes per point;
- expected output point upper bound;
- largest expected single CUDA/MPS staging allocation;
- a configurable safety fraction of detected accelerator memory;
- a conservative host-memory estimate for synthetic data generation.

Memory detection source order:

1. `PCL_RUSTIC_BENCH_DEVICE_MEMORY_BYTES` override.
2. `nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits` for
   CUDA hosts.
3. Unknown memory budget.

If memory cannot be detected for standard/full GPU-required rows, skip the row
before allocation and record the skip artifact. Do not guess and do not fall
back to CPU.

Rows above budget are recorded as skipped through a durable skip artifact, not
as successful benchmark measurements. RFC-0011 still forbids performance claims
from skipped or smoke-only rows.

### 3.3 Concat Voxelization Must Be Meaningful

The concat+voxelize workload should downsample the merged cloud enough to
exercise voxel grouping without preserving nearly every point. The concat path
may use a full-mode-specific voxel size distinct from the downsample matrix
voxel sizes, because RFC-0009 requires concat+voxelize timing but does not pin
the concat voxel size.

Acceptance target on a 24 GB CUDA host:

- C20 concatenate writes a CUDA row.
- C20 concatenate+voxelize writes a CUDA row.
- The C20 voxelized output point count is greater than zero and less than or
  equal to 50% of the input point count.
- Rows that exceed the local budget are skipped before large allocation.

### 3.4 CSV Semantics

Measured CSV rows remain unchanged. Skipped rows must not look like successful
measurements. Standard/full modes must write a separate skip artifact:

```text
reports/benchmarks/rfc0009-{mode}-skips.csv
```

The skip artifact records case, operation, parameters, reason, estimated host
bytes, estimated peak device bytes, estimated max single allocation bytes,
detected device memory bytes, detected device, git SHA, and run date.

Do not add skipped rows with fake timings to the benchmark result CSV.

### 3.5 Row-Level Continuation

The harness currently loops over many cases inside each pytest item. RFC-0019
does not require a full parametrization rewrite, but it does require per-row
skip recording and continuation. One over-budget row must not abort unrelated
rows in the same matrix.

Unexpected allocator panics after a row passes preflight should still fail the
test. `xfail` is not used for resource insufficiency; resource insufficiency is
a skip artifact, while unexpected runtime failure is a failure.

## 4. Acceptance Criteria

- [x] Full-mode C20 concatenate succeeds on CUDA and writes `device_name`
      containing `Cuda`.
- [x] Full-mode C20 concatenate+voxelize succeeds on CUDA and writes
      `device_name` containing `Cuda`.
- [x] Concat+voxelize output for C20 is greater than zero and less than or
      equal to 50% of the 200M-point input.
- [x] Rows that exceed detected accelerator or host-memory budget are skipped
      before generating synthetic NumPy arrays or point clouds.
- [x] Resource preflight checks both estimated peak working set and estimated
      maximum single allocation/staging buffer.
- [x] If standard/full mode cannot detect accelerator memory, rows are skipped
      before allocation and recorded in the skip artifact.
- [x] No full-mode row is measured on CPU when an accelerator benchmark row was
      requested.
- [x] Benchmark rows assert accepted accelerator device names before writing
      measured CSV results in standard/full mode.
- [x] Skipped rows are not written as successful timing rows in
      `reports/benchmarks/rfc0009-full.csv`.
- [x] Standard/full skipped rows are written to
      `reports/benchmarks/rfc0009-{mode}-skips.csv` with case, operation,
      parameters, reason, estimates, detected memory, device, git SHA, and run
      date.
- [x] One skipped row does not abort unrelated rows in the same matrix.
- [x] `rtk uv run pytest tests/test_benchmark.py -q --run-slow
      --benchmark-mode=full --no-cov` no longer panics on the 24 GB CUDA host.
- [x] Smoke benchmark behavior remains unchanged except for shared preflight
      helpers.
- [x] RFC-0011 external evidence gates remain open for skipped full-matrix
      cases.

## 5. Verification

```bash
rtk cargo fmt --check
rtk cargo clippy -- -D warnings
rtk cargo test --lib
rtk uv run maturin develop
rtk uv run pytest tests/test_benchmark.py -q --run-slow --benchmark-mode=smoke --no-cov
rtk uv run pytest tests/test_benchmark.py -q --run-slow --benchmark-mode=full --no-cov
```

The full-mode command must not panic. It may report skips for rows that exceed
the detected memory budget.

Verified on 2026-05-09 on an RTX 3090:

- `rfc0009-full.csv` recorded 11 measured rows, all `Cuda(Cuda(0))`.
- C20 concat+voxelize reduced 200,000,000 points to 799,036 points.
- `rfc0009-full-skips.csv` recorded 46 skipped oversized rows.
- The command completed in 548.01 seconds with 2 pytest tests passed.

## 6. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| Skipping oversized full rows hides missing performance evidence | RFC-0011 continues to mark skipped rows as external evidence gaps. |
| Larger concat voxel size makes results incomparable with older partial CSVs | Record `voxel_size` in every CSV row and do not compare rows with different parameters without labeling them. |
| Memory estimates are too optimistic | Use conservative safety factors and skip before allocation when uncertain. |
| Memory estimates are too conservative | Allow benchmark-machine overrides through environment variables. |

## 7. Out Of Scope

- Implementing chunked tensor storage in the core point-cloud type.
- Claiming full C40-C200 or D200-D400 performance on the local 24 GB GPU.
- CPU fallback for GPU benchmark rows.
