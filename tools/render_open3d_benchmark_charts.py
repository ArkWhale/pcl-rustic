"""Render RFC-0015 Open3D comparison charts from pytest-benchmark JSON."""

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
    "comparison_status",
    "pcl_rustic_mean_s",
    "open3d_mean_s",
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
    mean_s: float
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
        help="Directory searched for open3d-comparison-*.json when no JSON path is given.",
    )
    parser.add_argument(
        "--html-output",
        type=Path,
        default=Path("reports/benchmarks/open3d-comparison.html"),
        help="Interactive Plotly HTML output path.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("reports/benchmarks/open3d-comparison-summary.csv"),
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
        mean_s=float(stats.get("mean", 0.0)),
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


def _summary_key(record: BenchmarkRecord) -> tuple[str, str, int]:
    return (record.operation, record.case_id, record.point_count)


def _summary_row(records: list[BenchmarkRecord]) -> dict[str, str]:
    by_library = {record.library: record for record in records}
    pcl = by_library.get("pcl_rustic")
    open3d = by_library.get("open3d")
    anchor = pcl or open3d or records[0]
    status_record = next(
        (record for record in records if not record.comparable), anchor
    )
    status = (
        "comparable"
        if pcl is not None
        and open3d is not None
        and all(record.comparable for record in records)
        else status_record.comparison_status
    )
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
        "comparison_status": status,
        "pcl_rustic_mean_s": _format_seconds(pcl.mean_s if pcl else None),
        "open3d_mean_s": _format_seconds(open3d.mean_s if open3d else None),
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
        import plotly.express as px
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise SystemExit(
            "Plotly is required for chart rendering. Install the benchmark dependency group."
        ) from exc

    comparable_rows = [row for row in rows if row["comparison_status"] == "comparable"]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not comparable_rows:
        path.write_text(
            "<html><body><h1>No comparable benchmark rows</h1></body></html>",
            encoding="utf-8",
        )
        return

    runtime_data = []
    speedup_data = []
    for row in comparable_rows:
        for library, column in (
            ("pcl_rustic", "pcl_rustic_mean_s"),
            ("open3d", "open3d_mean_s"),
        ):
            runtime_data.append(
                {
                    "library": library,
                    "operation": row["operation"],
                    "point_count": int(row["point_count"]),
                    "mean_s": float(row[column]),
                    "case_id": row["case_id"],
                    "git_sha": row["git_sha"],
                    "source_json": row["source_json"],
                }
            )
        speedup_data.append(
            {
                "operation": row["operation"],
                "point_count": int(row["point_count"]),
                "speedup": float(row["pcl_rustic_vs_open3d_speedup"]),
                "case_id": row["case_id"],
                "git_sha": row["git_sha"],
                "source_json": row["source_json"],
            }
        )

    runtime_fig = px.line(
        runtime_data,
        x="point_count",
        y="mean_s",
        color="library",
        facet_col="operation",
        markers=True,
        hover_data=["case_id", "git_sha", "source_json"],
        log_x=True,
        log_y=True,
        title="Runtime vs point count",
    )
    speedup_fig = px.line(
        speedup_data,
        x="point_count",
        y="speedup",
        color="operation",
        markers=True,
        hover_data=["case_id", "git_sha", "source_json"],
        log_x=True,
        title="pcl-rustic / Open3D speedup ratio",
    )

    # make_subplots import validates that Plotly's graphing stack is available.
    _ = make_subplots
    html = "\n".join(
        [
            "<html><body><h1>Open3D Comparison Benchmarks</h1>",
            runtime_fig.to_html(full_html=False, include_plotlyjs="cdn"),
            speedup_fig.to_html(full_html=False, include_plotlyjs=False),
            "</body></html>",
        ]
    )
    path.write_text(html, encoding="utf-8")


def discover_json_paths(json_dir: Path) -> list[Path]:
    return sorted(json_dir.glob("open3d-comparison-*.json"))


def main() -> None:
    args = parse_args()
    paths = args.json or discover_json_paths(args.json_dir)
    records = read_benchmark_records(paths)
    rows = build_summary_rows(records)
    write_summary_csv(args.summary_output, rows)
    write_plotly_html(args.html_output, rows)


if __name__ == "__main__":
    main()
