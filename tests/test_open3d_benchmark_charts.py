from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.render_open3d_benchmark_charts import (
    build_summary_rows,
    read_benchmark_records,
    write_summary_csv,
)


def _write_pytest_benchmark_json(path: Path) -> None:
    payload = {
        "machine_info": {"platform": "Linux", "processor": "test-cpu"},
        "commit_info": {"id": "abc123"},
        "benchmarks": [
            {
                "name": "pcl_rustic__voxel_downsample__smoke__N1000",
                "fullname": "tests/test_open3d_benchmark.py::test_pcl",
                "group": "voxel_downsample",
                "params": {
                    "library": "pcl_rustic",
                    "operation": "voxel_downsample",
                    "case_id": "smoke-1000",
                    "point_count": 1000,
                    "output_points": 120,
                    "comparable": True,
                    "git_sha": "abc123",
                    "open3d_version": "0.19.0",
                    "pcl_rustic_version": "0.1.0",
                },
                "stats": {"mean": 0.01, "min": 0.009, "max": 0.012},
            },
            {
                "name": "open3d__voxel_downsample__smoke__N1000",
                "fullname": "tests/test_open3d_benchmark.py::test_open3d",
                "group": "voxel_downsample",
                "params": {
                    "library": "open3d",
                    "operation": "voxel_downsample",
                    "case_id": "smoke-1000",
                    "point_count": 1000,
                    "output_points": 118,
                    "comparable": True,
                    "git_sha": "abc123",
                    "open3d_version": "0.19.0",
                    "pcl_rustic_version": "0.1.0",
                },
                "stats": {"mean": 0.025, "min": 0.023, "max": 0.03},
            },
            {
                "name": "pcl_rustic__typed_concat__smoke__N1000",
                "fullname": "tests/test_open3d_benchmark.py::test_pcl_only",
                "group": "typed_concat",
                "params": {
                    "library": "pcl_rustic",
                    "operation": "typed_concat",
                    "case_id": "smoke-1000",
                    "point_count": 1000,
                    "comparable": False,
                    "comparison_status": "pcl_rustic_only",
                    "not_comparable_reason": "Open3D has no typed strict concat.",
                    "git_sha": "abc123",
                },
                "stats": {"mean": 0.02},
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_reads_pytest_benchmark_records(tmp_path: Path) -> None:
    json_path = tmp_path / "open3d-comparison-smoke.json"
    _write_pytest_benchmark_json(json_path)

    records = read_benchmark_records([json_path])

    assert [record.library for record in records] == [
        "pcl_rustic",
        "open3d",
        "pcl_rustic",
    ]
    assert records[0].operation == "voxel_downsample"
    assert records[0].point_count == 1000
    assert records[0].mean_s == 0.01
    assert records[0].source_json == str(json_path)
    assert records[2].comparison_status == "pcl_rustic_only"


def test_summary_rows_include_speedup_and_exclusions(tmp_path: Path) -> None:
    json_path = tmp_path / "open3d-comparison-smoke.json"
    _write_pytest_benchmark_json(json_path)
    records = read_benchmark_records([json_path])

    rows = build_summary_rows(records)

    comparable = next(row for row in rows if row["operation"] == "voxel_downsample")
    assert comparable["comparison_status"] == "comparable"
    assert comparable["pcl_rustic_mean_s"] == "0.010000"
    assert comparable["open3d_mean_s"] == "0.025000"
    assert comparable["pcl_rustic_vs_open3d_speedup"] == "2.500000"

    exclusion = next(row for row in rows if row["operation"] == "typed_concat")
    assert exclusion["comparison_status"] == "pcl_rustic_only"
    assert exclusion["not_comparable_reason"] == "Open3D has no typed strict concat."


def test_writes_summary_csv(tmp_path: Path) -> None:
    json_path = tmp_path / "open3d-comparison-smoke.json"
    csv_path = tmp_path / "summary.csv"
    _write_pytest_benchmark_json(json_path)

    write_summary_csv(csv_path, build_summary_rows(read_benchmark_records([json_path])))

    text = csv_path.read_text(encoding="utf-8")
    assert "pcl_rustic_vs_open3d_speedup" in text
    assert "voxel_downsample" in text
    assert "typed_concat" in text
