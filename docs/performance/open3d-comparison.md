# Unified Benchmarks

The benchmark suite uses one pytest-benchmark JSON format for both pcl-rustic
only runs and Open3D comparison runs. Plotly renders charts from the same latest
artifact in either mode.

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

Install the optional benchmark dependency group before running benchmarks:

```bash
uv sync --group benchmark
```

Run the suite through `just`:

```bash
just benchmark mode=fast compare=false
just benchmark mode=fast compare=true
just benchmark mode=slow compare=false
just benchmark mode=slow compare=true
```

The recipe writes the latest pytest-benchmark JSON to one stable path:

```text
reports/benchmarks/last-benchmark.json
```

Generate charts from the latest JSON artifact:

```bash
just benchmark-visualize
```

The renderer writes:

```text
reports/benchmarks/last-benchmark.html
reports/benchmarks/last-benchmark-summary.csv
```

The HTML chart uses one subplot per point-count scale. Each subplot's x-axis is
the operation name, and library results are shown as colored point markers with
error bars.

## Modes

| Just mode | Pytest mode | Point counts | Use |
|---|---|---|---|
| `fast` | `smoke` | 1k, 10k | Validates benchmark wiring and chart generation. |
| `slow` | `standard` | 10k, 100k, 1M | Routine benchmark runs on benchmark-capable machines. |

Normal tests do not require Open3D, Plotly, or pytest-benchmark. `compare=false`
does not import Open3D. `compare=true` adds Open3D rows and skips with a clear
reason if the optional dependency is absent.

## Evidence Rules

Reports under `reports/benchmarks/` are generated local artifacts and are not
committed. A chart becomes release evidence only after it is linked from
`docs/performance/benchmarks.md` or another durable artifact location with the
RFC-0011 fields: command, hardware, dataset parameters, dependency versions,
date, git SHA, and artifact path.

No recorded Open3D comparison performance artifacts are published yet.
