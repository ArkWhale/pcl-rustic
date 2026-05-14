from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.check_open3d_speed_budget import (
    BudgetConfig,
    evaluate_speed_budget,
    evaluate_speed_budget_from_csv,
)


def _row(
    operation: str,
    *,
    speedup: float,
    requested_point_count: int = 1_000_000,
    measured_point_count: int = 1_000_000,
    case_id: str = "standard-1000000",
    comparison_status: str = "comparable",
) -> dict[str, str]:
    return {
        "operation": operation,
        "case_id": case_id,
        "point_count": str(requested_point_count),
        "requested_point_count": str(requested_point_count),
        "measured_point_count": str(measured_point_count),
        "comparison_status": comparison_status,
        "pcl_rustic_mean_s": "1.000000",
        "open3d_mean_s": f"{speedup:.6f}",
        "pcl_rustic_vs_open3d_speedup": f"{speedup:.6f}",
    }


def test_speed_budget_passes_no_baseline_when_1m_rows_clear_budget() -> None:
    rows = [
        _row("transform", speedup=2.0),
        _row("remove_radius_outlier", speedup=0.9),
        _row(
            "registration_point_to_point",
            speedup=0.2,
            requested_point_count=1_000_000,
            measured_point_count=10_000,
        ),
    ]

    report = evaluate_speed_budget(
        rows,
        BudgetConfig(
            mode="standard",
            min_1m_speedup=0.80,
            min_1m_geomean=1.25,
            no_baseline=True,
        ),
    )

    assert report.passed is True
    assert report.baseline_enforced is False
    assert report.geomean_speedup == "1.341641"
    assert "regression protection starts" in report.notes[0]


def test_speed_budget_fails_when_comparable_measured_1m_row_is_too_slow() -> None:
    report = evaluate_speed_budget(
        [_row("voxel_downsample", speedup=0.79)],
        BudgetConfig(
            mode="standard",
            min_1m_speedup=0.80,
            min_1m_geomean=0.0,
            no_baseline=True,
        ),
    )

    assert report.passed is False
    assert report.failures[0].operation == "voxel_downsample"
    assert report.failures[0].reason == "below_min_1m_speedup"


def test_speed_budget_fails_when_1m_geomean_is_too_low() -> None:
    report = evaluate_speed_budget(
        [_row("voxel_downsample", speedup=0.9), _row("estimate_normals", speedup=1.0)],
        BudgetConfig(
            mode="standard",
            min_1m_speedup=0.80,
            min_1m_geomean=1.25,
            no_baseline=True,
        ),
    )

    assert report.passed is False
    assert report.geomean_speedup == "0.948683"
    assert report.failures[-1].reason == "below_min_1m_geomean"


def test_speed_budget_fails_when_current_row_regresses_from_baseline() -> None:
    rows = [_row("transform", speedup=1.70)]
    baseline_rows = [_row("transform", speedup=2.00)]

    report = evaluate_speed_budget(
        rows,
        BudgetConfig(
            mode="standard",
            min_1m_speedup=0.80,
            min_1m_geomean=0.0,
            max_regression_ratio=0.10,
        ),
        baseline_rows=baseline_rows,
    )

    assert report.passed is False
    assert report.baseline_enforced is True
    assert report.failures[0].reason == "baseline_regression"


def test_speed_budget_reads_summary_csv_with_requested_and_measured_counts(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "summary.csv"
    rows = [_row("transform", speedup=2.0)]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = evaluate_speed_budget_from_csv(
        csv_path,
        BudgetConfig(
            mode="standard",
            min_1m_speedup=0.80,
            min_1m_geomean=1.25,
            no_baseline=True,
        ),
    )

    assert report.rows[0].requested_point_count == 1_000_000
    assert report.rows[0].measured_point_count == 1_000_000
