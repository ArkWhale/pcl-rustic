"""Render RFC-0009 benchmark documentation from recorded CSV output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

CSV_COLUMNS = [
    "case_id",
    "operation",
    "backend",
    "input_clouds",
    "input_points",
    "output_points",
    "voxel_size",
    "strategy",
    "wall_time_s",
    "throughput_points_s",
    "peak_rss_bytes",
    "estimated_input_bytes",
    "estimated_output_bytes",
    "device_name",
    "git_sha",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("reports/benchmarks"),
        help="Directory containing rfc0009-*.csv benchmark output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/performance/benchmarks.md"),
        help="Markdown file to write.",
    )
    return parser.parse_args()


def read_rows(csv_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(csv_dir.glob("rfc0009-*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row = {column: row.get(column, "") for column in CSV_COLUMNS}
                row["source_csv"] = path.name
                rows.append(row)
    return rows


def fmt_int(value: str) -> str:
    try:
        return f"{int(float(value)):,}"
    except (TypeError, ValueError):
        return value


def fmt_float(value: str, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return value


def render_results(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return [
            "## Recorded Results",
            "",
            "No recorded benchmark CSV files were found under `reports/benchmarks/`.",
            "Run `just benchmark mode=fast compare=false`, then run",
            "`just benchmark-visualize` to render the latest measured output.",
            "",
        ]

    lines = [
        "## Recorded Results",
        "",
        "The rows below are generated from CSV files under `reports/benchmarks/`.",
        "",
        "| Source | Case | Operation | Inputs | Output | Voxel | Strategy | Time (s) | Throughput (pts/s) | Device | Git SHA |",
        "|---|---|---|---:|---:|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {source} | {case} | {op} | {inputs} | {output} | {voxel} | {strategy} | {time} | {throughput} | {device} | {sha} |".format(
                source=row["source_csv"],
                case=row["case_id"],
                op=row["operation"],
                inputs=fmt_int(row["input_points"]),
                output=fmt_int(row["output_points"]),
                voxel=row["voxel_size"],
                strategy=row["strategy"],
                time=fmt_float(row["wall_time_s"]),
                throughput=fmt_int(row["throughput_points_s"]),
                device=row["device_name"],
                sha=row["git_sha"],
            )
        )
    lines.append("")
    return lines


def render_document(rows: list[dict[str, str]]) -> str:
    lines = [
        "# Performance Benchmarks",
        "",
        "This page documents the RFC-0009 large-scale benchmark suite. It describes how",
        "to run the generated suite and how to read its CSV output. It intentionally",
        "does not publish measured performance numbers unless they come from a",
        "recorded benchmark run.",
        "",
        "## Suite Scope",
        "",
        "The suite covers two batch workflows:",
        "",
        "- Concatenating many point clouds with typed attributes.",
        "- Voxel downsampling large point clouds across multiple strategies.",
        "",
        "Every synthetic cloud uses this schema:",
        "",
        "| Field | dtype | Shape |",
        "|---|---|---|",
        "| `xyz` | `float32` | `[N, 3]` |",
        "| `intensity` | `float32` | `[N]` |",
        "| `classification` | `uint8` | `[N]` |",
        "| `return_number` | `uint8` | `[N]` |",
        "| `gps_time` | `float64` | `[N]` |",
        "",
        "Synthetic data is deterministic. It uses clustered spatial coordinates,",
        "LAS-style classification codes, return numbers in `1..5`, and monotonic GPS",
        "time per generated cloud.",
        "",
        "## Modes",
        "",
        "Run benchmarks through `just`:",
        "",
        "```bash",
        "just benchmark mode=fast compare=false",
        "just benchmark mode=slow compare=false",
        "just benchmark mode=slow compare=true",
        "```",
        "",
        "`mode=fast` maps to the smoke workload. `mode=slow` maps to the standard",
        "workload. Set `compare=true` to add Open3D rows to the same output artifact.",
        "",
        "| Mode | Purpose | Workload |",
        "|---|---|---|",
        "| `smoke` | Pull requests and ordinary CI | C20 with 20 clouds x 1M points, plus D20 scaled to 2M points across the full strategy/voxel schema. |",
        "| `standard` | Manual or scheduled runs on benchmark hardware | C20 full concat target plus D20, D50, and D100 full downsampling targets. |",
        "| `full` | Release and dedicated benchmark machines | Full concat and downsampling matrices. |",
        "",
        "The pytest option behind the recipes is:",
        "",
        "```bash",
        "uv run --group benchmark pytest tests/test_open3d_benchmark.py -v -s --run-slow --benchmark-mode=smoke --benchmark-json=reports/benchmarks/last-benchmark.json --no-cov",
        "```",
        "",
        "`--benchmark-mode` accepts `smoke`, `standard`, or `full` and defaults to",
        "`smoke`; the public `just` entry intentionally exposes only `fast` and `slow`.",
        "Benchmark tests are still marked `slow`, so normal test runs continue to skip",
        "them unless `--run-slow` or a benchmark recipe is used.",
        "",
        "## Matrices",
        "",
        "### Concatenation",
        "",
        "Full mode concatenates clouds with 10M points each:",
        "",
        "| Case | Clouds | Points per cloud | Total points | Operations |",
        "|---|---:|---:|---:|---|",
        "| C20 | 20 | 10M | 200M | concatenate; concatenate + voxelize |",
        "| C40 | 40 | 10M | 400M | concatenate; concatenate + voxelize |",
        "| C80 | 80 | 10M | 800M | concatenate; concatenate + voxelize |",
        "| C120 | 120 | 10M | 1.2B | concatenate; concatenate + voxelize |",
        "| C160 | 160 | 10M | 1.6B | concatenate; concatenate + voxelize |",
        "| C200 | 200 | 10M | 2.0B | concatenate; concatenate + voxelize |",
        "",
        "Standard mode runs the full C20 target. Smoke mode runs C20 with 1M points",
        "per input cloud.",
        "",
        "### Downsampling",
        "",
        "Full mode downsampling cases:",
        "",
        "| Case | Input points | Voxel sizes | Strategies |",
        "|---|---:|---|---|",
        "| D20 | 20M | 0.05, 0.15, 0.50 | `NEAREST_TO_CENTROID`, `AVERAGE`, `RANDOM_SEEDED` |",
        "| D50 | 50M | 0.05, 0.15, 0.50 | same |",
        "| D100 | 100M | 0.05, 0.15, 0.50 | same |",
        "| D200 | 200M | 0.05, 0.15, 0.50 | same |",
        "| D400 | 400M | 0.05, 0.15, 0.50 | same |",
        "",
        "Standard mode runs D20, D50, and D100 at full size. Smoke mode runs D20",
        "scaled to 2M points with the same voxel sizes and strategies.",
        "",
        "## CSV Output",
        "",
        "Each benchmark run writes a fresh CSV under `reports/benchmarks/`:",
        "",
        "```text",
        "reports/benchmarks/rfc0009-smoke.csv",
        "reports/benchmarks/rfc0009-standard.csv",
        "reports/benchmarks/rfc0009-full.csv",
        "```",
        "",
        "The CSV columns follow RFC-0009 section 3.3:",
        "",
        "| Column | Meaning |",
        "|---|---|",
        "| `case_id` | Matrix case such as `C20` or `D100`. |",
        "| `operation` | `concatenate`, `concatenate_voxelize`, or `voxel_downsample`. |",
        "| `backend` | Benchmark harness backend label. |",
        "| `input_clouds` | Number of input clouds. |",
        "| `input_points` | Total input points. |",
        "| `output_points` | Output point count after the operation. |",
        "| `voxel_size` | Voxel size when applicable. |",
        "| `strategy` | Downsampling strategy when applicable. |",
        "| `wall_time_s` | Measured wall-clock time in seconds. |",
        "| `throughput_points_s` | Input points divided by wall time. |",
        "| `peak_rss_bytes` | Process peak RSS when the platform exposes it. |",
        "| `estimated_input_bytes` | Estimated bytes for `xyz` plus RFC attributes. |",
        "| `estimated_output_bytes` | Estimated bytes for output points with the same schema. |",
        "| `device_name` | Public `PointCloud.device()` value for the result. |",
        "| `git_sha` | Git commit SHA or `GITHUB_SHA`. |",
        "",
        "Run `just benchmark-visualize` after a recorded benchmark run to render the",
        "latest JSON output.",
        "",
        "## CI Behavior",
        "",
        "GitHub Actions runs `just benchmark mode=fast compare=false` for pull requests",
        "and normal CI. A manual workflow dispatch can select `fast` or `slow` through",
        "the `benchmark_mode` input. Slow mode also requires",
        "`allow_expensive_benchmarks=true` so they are not launched accidentally on",
        "hosted runners.",
        "",
        "Standard and full modes are not intended for hosted pull-request runners. Use a",
        "dedicated machine or self-hosted runner with enough RAM, swap, disk space, and",
        "a known CPU/GPU configuration. Full C200 concatenation represents 2B input",
        "points before intermediate allocations, so memory requirements can reach",
        "hundreds of GB depending on allocator behavior and backend implementation.",
        "",
    ]
    lines.extend(render_results(rows))
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_document(read_rows(args.csv_dir)), encoding="utf-8")


if __name__ == "__main__":
    main()
