# Open3D Comparison Benchmarks

RFC-0015 adds a comparison benchmark suite for pcl-rustic and Open3D. It uses
pytest-benchmark for measurement and Plotly only for rendering charts from
recorded benchmark JSON artifacts.

## Scope

The suite compares operations that have an honest Open3D equivalent:

| Family | pcl-rustic | Open3D |
|---|---|---|
| Construction | `PointCloud.from_xyz(...)` | `open3d.geometry.PointCloud` from NumPy vectors |
| Transform | `transform(...)` | `PointCloud.transform(...)` |
| Downsample | `voxel_downsample(...)` | `voxel_down_sample(...)` |
| Neighbors | warm `knn` / `radius_search` | warm `KDTreeFlann` queries |
| Normals | `estimate_normals(...)` | `estimate_normals(...)` |
| Outliers | SOR / ROR | SOR / ROR |
| Registration | point-to-point and point-to-plane ICP | matching Open3D ICP estimators |

Features without a one-to-one Open3D equivalent, such as typed strict concat or
LAS point-format-10 policy behavior, are excluded from speedup-ratio charts.

## Commands

Install the optional benchmark dependency group before running comparison
benchmarks:

```bash
uv sync --group benchmark
```

Run the suites through `just`:

```bash
just benchmark-compare-smoke
just benchmark-compare-standard
just benchmark-compare-full
```

The recipes write pytest-benchmark JSON under `reports/benchmarks/`:

```text
reports/benchmarks/open3d-comparison-smoke.json
reports/benchmarks/open3d-comparison-standard.json
reports/benchmarks/open3d-comparison-full.json
```

Generate charts from existing JSON artifacts:

```bash
just benchmark-compare-charts
```

The renderer writes:

```text
reports/benchmarks/open3d-comparison.html
reports/benchmarks/open3d-comparison-summary.csv
```

## Modes

| Mode | Point counts | Use |
|---|---|---|
| `smoke` | 1k, 10k | Validates benchmark wiring and chart generation. |
| `standard` | 10k, 100k, 1M | Routine comparison on benchmark-capable machines. |
| `full` | 10k, 100k, 1M, 10M | Release comparison on recorded benchmark hardware. |

Normal tests do not require Open3D, Plotly, or pytest-benchmark. The benchmark
module is marked `slow`, skips without `--run-slow`, and skips with clear
reasons when optional dependencies are absent.

## Evidence Rules

Reports under `reports/benchmarks/` are generated local artifacts and are not
committed. A chart becomes release evidence only after it is linked from
`docs/performance/benchmarks.md` or another durable artifact location with the
RFC-0011 fields: command, hardware, dataset parameters, dependency versions,
date, git SHA, and artifact path.

No recorded Open3D comparison performance artifacts are published yet.
