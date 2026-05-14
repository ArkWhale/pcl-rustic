"""Render benchmark charts from the unified pytest-benchmark JSON."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SUMMARY_COLUMNS = [
    "operation",
    "case_id",
    "point_count",
    "requested_point_count",
    "measured_point_count",
    "comparison_status",
    "pcl_rustic_mean_s",
    "open3d_mean_s",
    "pcl_rustic_stddev_s",
    "open3d_stddev_s",
    "pcl_rustic_vs_open3d_speedup",
    "pcl_rustic_output_points",
    "open3d_output_points",
    "not_comparable_reason",
    "git_sha",
    "pcl_rustic_version",
    "open3d_version",
    "source_json",
]


@dataclass(frozen=True)
class BenchmarkRecord:
    library: str
    operation: str
    case_id: str
    point_count: int
    requested_point_count: int
    measured_point_count: int
    mean_s: float
    stddev_s: float
    output_points: int | None
    comparison_status: str
    not_comparable_reason: str
    git_sha: str
    pcl_rustic_version: str
    open3d_version: str
    source_json: str
    benchmark_name: str

    @property
    def comparable(self) -> bool:
        return self.comparison_status == "comparable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json",
        nargs="*",
        type=Path,
        help="pytest-benchmark JSON artifacts to render.",
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=Path("reports/benchmarks"),
        help="Directory searched for last-benchmark.json when no JSON path is given.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path("reports/benchmarks/last-benchmark.html"),
        help="Interactive Plotly HTML output path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/benchmarks/last-benchmark-summary.csv"),
        help="Summary CSV output path.",
    )
    return parser.parse_args()


def read_benchmark_records(paths: Iterable[Path]) -> list[BenchmarkRecord]:
    records: list[BenchmarkRecord] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for benchmark in payload.get("benchmarks", []):
            records.append(_record_from_benchmark(path, benchmark))
    return records


def _record_from_benchmark(path: Path, benchmark: dict[str, Any]) -> BenchmarkRecord:
    params = {
        **(benchmark.get("params") or {}),
        **(benchmark.get("extra_info") or {}),
    }
    stats = benchmark.get("stats") or {}
    operation = str(params.get("operation") or benchmark.get("group") or "")
    comparison_status = _comparison_status(params)
    return BenchmarkRecord(
        library=str(params.get("library", "")),
        operation=operation,
        case_id=str(params.get("case_id", "")),
        point_count=_optional_int(params.get("point_count")) or 0,
        requested_point_count=(
            _optional_int(params.get("requested_point_count"))
            or _optional_int(params.get("point_count"))
            or 0
        ),
        measured_point_count=(
            _optional_int(params.get("measured_point_count"))
            or _optional_int(params.get("point_count"))
            or 0
        ),
        mean_s=float(stats.get("mean", 0.0)),
        stddev_s=float(stats.get("stddev", 0.0)),
        output_points=_optional_int(params.get("output_points")),
        comparison_status=comparison_status,
        not_comparable_reason=str(params.get("not_comparable_reason", "")),
        git_sha=str(params.get("git_sha", "")),
        pcl_rustic_version=str(params.get("pcl_rustic_version", "")),
        open3d_version=str(params.get("open3d_version", "")),
        source_json=str(path),
        benchmark_name=str(benchmark.get("name") or benchmark.get("fullname") or ""),
    )


def _comparison_status(params: dict[str, Any]) -> str:
    explicit = params.get("comparison_status")
    if explicit:
        return str(explicit)
    return "comparable" if bool(params.get("comparable", True)) else "not_comparable"


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    return int(value)


def build_summary_rows(records: list[BenchmarkRecord]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for key in sorted({_summary_key(record) for record in records}):
        group = [record for record in records if _summary_key(record) == key]
        rows.append(_summary_row(group))
    return rows


def _summary_key(record: BenchmarkRecord) -> tuple[str, str, int, int]:
    return (
        record.operation,
        record.case_id,
        record.requested_point_count,
        record.measured_point_count,
    )


def _summary_row(records: list[BenchmarkRecord]) -> dict[str, str]:
    by_library = {record.library: record for record in records}
    pcl = by_library.get("pcl_rustic")
    open3d = by_library.get("open3d")
    anchor = pcl or open3d or records[0]
    status_record = next(
        (record for record in records if not record.comparable), anchor
    )
    if (
        pcl is not None
        and open3d is not None
        and all(record.comparable for record in records)
    ):
        status = "comparable"
    elif pcl is not None and open3d is None and pcl.comparable:
        status = "pcl_rustic_only"
    elif open3d is not None and pcl is None and open3d.comparable:
        status = "open3d_only"
    else:
        status = status_record.comparison_status
    speedup = ""
    if (
        status == "comparable"
        and pcl is not None
        and open3d is not None
        and pcl.mean_s > 0.0
    ):
        speedup = f"{open3d.mean_s / pcl.mean_s:.6f}"

    return {
        "operation": anchor.operation,
        "case_id": anchor.case_id,
        "point_count": str(anchor.point_count),
        "requested_point_count": str(anchor.requested_point_count),
        "measured_point_count": str(anchor.measured_point_count),
        "comparison_status": status,
        "pcl_rustic_mean_s": _format_seconds(pcl.mean_s if pcl else None),
        "open3d_mean_s": _format_seconds(open3d.mean_s if open3d else None),
        "pcl_rustic_stddev_s": _format_seconds(pcl.stddev_s if pcl else None),
        "open3d_stddev_s": _format_seconds(open3d.stddev_s if open3d else None),
        "pcl_rustic_vs_open3d_speedup": speedup,
        "pcl_rustic_output_points": _format_optional_int(
            pcl.output_points if pcl else None
        ),
        "open3d_output_points": _format_optional_int(
            open3d.output_points if open3d else None
        ),
        "not_comparable_reason": status_record.not_comparable_reason,
        "git_sha": anchor.git_sha,
        "pcl_rustic_version": anchor.pcl_rustic_version,
        "open3d_version": anchor.open3d_version,
        "source_json": anchor.source_json,
    }


def _format_seconds(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


def _format_optional_int(value: int | None) -> str:
    return "" if value is None else str(value)


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {column: row.get(column, "") for column in SUMMARY_COLUMNS} for row in rows
        )


def write_plotly_html(path: Path, rows: list[dict[str, str]]) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "Plotly is required for chart rendering. Install the benchmark dependency group."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(
            "<html><body><h1>No benchmark rows to visualize</h1></body></html>",
            encoding="utf-8",
        )
        return

    runtime_fig = build_runtime_error_bar_figure(rows, go)
    html = "\n".join(
        [
            "<html><body><h1>PCL Rustic Benchmarks</h1>",
            runtime_fig.to_html(full_html=False, include_plotlyjs="cdn"),
            "</body></html>",
        ]
    )
    path.write_text(html, encoding="utf-8")


def build_runtime_error_bar_figure(rows: list[dict[str, str]], go_module: Any):
    from plotly.colors import qualitative
    from plotly.subplots import make_subplots

    colorway = qualitative.Plotly
    colors = {"pcl_rustic": colorway[0], "open3d": colorway[1]}
    symbols = {"pcl_rustic": "circle", "open3d": "diamond"}
    point_counts = sorted({int(row["point_count"]) for row in rows})
    operations = sorted({row["operation"] for row in rows})

    fig = make_subplots(
        rows=1,
        cols=len(point_counts),
        shared_yaxes=True,
        subplot_titles=[f"{point_count:,} points" for point_count in point_counts],
        horizontal_spacing=0.04,
    )

    for col, point_count in enumerate(point_counts, start=1):
        point_rows = [row for row in rows if int(row["point_count"]) == point_count]
        for library in ("pcl_rustic", "open3d"):
            mean_column = f"{library}_mean_s"
            stddev_column = f"{library}_stddev_s"
            library_rows = [
                row for row in point_rows if row.get(mean_column) not in ("", None)
            ]
            if not library_rows:
                continue
            library_rows.sort(key=lambda row: operations.index(row["operation"]))
            fig.add_trace(
                go_module.Scatter(
                    x=[row["operation"] for row in library_rows],
                    y=[float(row[mean_column]) for row in library_rows],
                    mode="markers",
                    name=library,
                    legendgroup=library,
                    showlegend=col == 1,
                    marker={
                        "color": colors[library],
                        "symbol": symbols[library],
                        "size": 10,
                    },
                    error_y={
                        "type": "data",
                        "array": [
                            max(0.0, float(row.get(stddev_column) or 0.0))
                            for row in library_rows
                        ],
                        "visible": True,
                        "color": colors[library],
                        "thickness": 1.5,
                        "width": 3,
                    },
                    customdata=[
                        [
                            row["operation"],
                            row["case_id"],
                            row["git_sha"],
                            row["source_json"],
                        ]
                        for row in library_rows
                    ],
                    hovertemplate=(
                        "operation=%{customdata[0]}<br>"
                        f"library={library}<br>"
                        f"point_count={point_count}<br>"
                        "mean_s=%{y:.6f}<br>"
                        "case_id=%{customdata[1]}<br>"
                        "git_sha=%{customdata[2]}<br>"
                        "source_json=%{customdata[3]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(
            title_text="operation",
            categoryorder="array",
            categoryarray=operations,
            tickangle=-45,
            row=1,
            col=col,
        )

    fig.update_layout(
        title="Runtime by operation and point-count scale",
        yaxis_title="mean_s",
        legend_title="library",
        colorway=colorway,
    )
    fig.update_yaxes(type="log")
    return fig


def discover_json_paths(json_dir: Path) -> list[Path]:
    path = json_dir / "last-benchmark.json"
    return [path] if path.exists() else []


def main() -> None:
    args = parse_args()
    paths = args.json or discover_json_paths(args.json_dir)
    records = read_benchmark_records(paths)
    rows = build_summary_rows(records)
    write_summary_csv(args.summary_output, rows)
    write_plotly_html(args.html_output, rows)


if __name__ == "__main__":
    main()
