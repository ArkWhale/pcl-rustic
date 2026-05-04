# RFC-0009: Large-Scale Benchmark Suite

- **Status:** Implemented
- **Date:** 2026-05-04
- **Author:** Codex implementation agent
- **Related:** RFC-0003, RFC-0004, RFC-0005

## 1. Summary

Define the large-scale benchmark suite for pcl-rustic's core batch workflows:
concatenating many 10M-point clouds and voxelizing/downsampling very large point
clouds. The suite exists to make performance claims measurable and repeatable,
especially around RFC-0003 concatenation and RFC-0004 GPU hot-path work.

Primary target workload:

- Concatenate **20 clouds × 10M points** and voxelize the merged result.

Benchmark coverage:

- Concatenation: **20, 40, 80, 120, 160, 200** input clouds, each with **10M
  points**.
- Downsampling: single-cloud input sizes **20M, 50M, 100M, 200M, 400M** points.

## 2. Motivation

Small benchmarks hide the real bottlenecks in point-cloud pipelines. The target
workload stresses:

1. Attribute propagation through `PointCloud.concatenate`.
2. Memory behavior while merging many large clouds.
3. Voxel grouping cost at 200M+ total points.
4. CPU vs GPU viability for RFC-0004.
5. Reproducibility of `RANDOM_SEEDED` and point-count stability across
   strategies.

The suite should be large enough to expose allocator pressure, host/device copy
cost, and pathological hash/grouping behavior before users hit those cases in
production.

## 3. Detailed Design

### 3.1 Benchmark Matrix

#### Concatenation Matrix

Each input cloud contains exactly 10M points and the same attribute schema:

- `xyz`: `float32[N, 3]`
- `intensity`: `float32[N]`
- `classification`: `uint8[N]`
- `return_number`: `uint8[N]`
- `gps_time`: `float64[N]`

| Case | Clouds | Total points | Required operation |
|---|---:|---:|---|
| C20 | 20 | 200M | concatenate only; concatenate + voxelize |
| C40 | 40 | 400M | concatenate only; concatenate + voxelize |
| C80 | 80 | 800M | concatenate only; concatenate + voxelize |
| C120 | 120 | 1.2B | concatenate only; concatenate + voxelize |
| C160 | 160 | 1.6B | concatenate only; concatenate + voxelize |
| C200 | 200 | 2.0B | concatenate only; concatenate + voxelize |

The primary acceptance target is C20: concatenate 20 × 10M point clouds and run
voxel downsampling on the merged 200M-point cloud.

#### Downsampling Matrix

| Case | Input points | Voxel sizes | Strategies |
|---|---:|---|---|
| D20 | 20M | 0.05, 0.15, 0.50 | `NEAREST_TO_CENTROID`, `AVERAGE`, `RANDOM_SEEDED` |
| D50 | 50M | 0.05, 0.15, 0.50 | same |
| D100 | 100M | 0.05, 0.15, 0.50 | same |
| D200 | 200M | 0.05, 0.15, 0.50 | same |
| D400 | 400M | 0.05, 0.15, 0.50 | same |

### 3.2 Dataset Generation

Benchmarks use deterministic synthetic data by default so CI and developer runs
do not require large checked-in fixtures.

Generator requirements:

- Fixed seed per case.
- Configurable spatial extent and cluster count.
- Classification values drawn from LAS-style codes.
- Return numbers constrained to `1..=5`.
- GPS time monotonic within each cloud.
- Optional on-disk cache under `target/bench-data/`, never committed.

The benchmark harness may also accept real LAS/LAZ inputs, but synthetic data is
the correctness and CI baseline.

### 3.3 Metrics

Each benchmark emits one CSV row per case:

- `case_id`
- `operation`
- `backend`
- `input_clouds`
- `input_points`
- `output_points`
- `voxel_size`
- `strategy`
- `wall_time_s`
- `throughput_points_s`
- `peak_rss_bytes`
- `estimated_input_bytes`
- `estimated_output_bytes`
- `device_name`
- `git_sha`

`docs/performance/benchmarks.md` should be generated from the CSV for release
runs. The raw CSV should be stored under `reports/benchmarks/`.

### 3.4 Execution Modes

The suite has three modes:

| Mode | Scope | Trigger |
|---|---|---|
| smoke | C20 with 1M-point scaled inputs; D20 scaled to 2M | pull requests |
| standard | C20 full target; D20, D50, D100 | manual CI / nightly |
| full | C20-C200 and D20-D400 | release and dedicated benchmark machines |

Smoke mode must run on ordinary CI without exhausting memory. Standard and full
modes may require self-hosted runners.

### 3.5 Justfile Interface

The benchmark suite is run through `just`, not ad hoc shell scripts:

```bash
just benchmark-smoke
just benchmark-standard
just benchmark-full
```

Each recipe should call pytest with explicit markers or environment variables,
for example:

```bash
uv run pytest tests/test_benchmark.py -v -s --benchmark-mode=smoke
```

## 4. Acceptance Criteria

- [x] `tests/test_benchmark.py` supports smoke, standard, and full modes.
- [x] The concat matrix covers 20-200 clouds of 10M points each in full mode.
- [x] The downsampling matrix covers 20M-400M points in full mode.
- [x] C20 target workload reports concatenate-only and concatenate+voxelize
      timings.
- [x] Benchmark output writes CSV to `reports/benchmarks/`.
- [x] Release benchmark output can regenerate `docs/performance/benchmarks.md`.
- [x] CI runs smoke mode through `just benchmark-smoke`.
- [x] Full mode is documented as requiring a high-memory benchmark runner.
- [x] Benchmarks include typed attributes listed in §3.1, not XYZ-only data.

## 5. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Full concat matrix can exceed memory on normal workstations | Gate full mode behind explicit command and document runner memory requirements. |
| Synthetic data is too regular and flatters voxel grouping | Use clustered distributions plus uniform noise; allow real LAS/LAZ override. |
| Benchmark code becomes a second implementation path | Benchmarks must use public Python API except for measurement helpers. |
| CI time explodes | PR CI runs smoke mode only; standard/full run manually or on benchmark runners. |
| GPU results vary by driver/backend | Include backend, device name, and git SHA in every row. |

## 6. Out Of Scope

- Open3D comparison benchmarks. Those belong in a separate migration benchmark
  RFC once pcl-rustic's own workload measurements are stable.
- Visualization benchmarks.
- Registration benchmarks beyond the cost of preprocessing/voxelization.

## 7. Open Questions

- Exact reference hardware for the C20 target should be recorded before setting
  a hard pass/fail throughput threshold.
- Whether full-mode generated synthetic data should be cached between runs or
  regenerated every time for I/O isolation.
- Whether peak memory should use platform-specific RSS sampling or an optional
  Python dependency such as `psutil`.
