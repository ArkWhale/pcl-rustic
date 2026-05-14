"""Check Open3D comparison summaries against the RFC-0021 speed budget."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class BudgetConfig:
    mode: str
    min_1m_speedup: float
    min_1m_geomean: float
    baseline: Path | None = None
    max_regression_ratio: float = 0.10
    no_baseline: bool = False


@dataclass(frozen=True)
class BudgetRow:
    operation: str
    case_id: str
    requested_point_count: int
    measured_point_count: int
    comparison_status: str
    pcl_rustic_mean_s: str
    open3d_mean_s: str
    speedup: float | None

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (
            self.operation,
            self.case_id,
            self.requested_point_count,
            self.measured_point_count,
        )

    @property
    def comparable(self) -> bool:
        return self.comparison_status == "comparable"


@dataclass(frozen=True)
class BudgetFailure:
    operation: str
    case_id: str
    requested_point_count: int
    measured_point_count: int
    speedup: str
    baseline_speedup: str
    reason: str


@dataclass(frozen=True)
class BudgetReport:
    passed: bool
    rows: list[BudgetRow]
    failures: list[BudgetFailure]
    geomean_speedup: str
    baseline_enforced: bool
    notes: list[str] = field(default_factory=list)


def read_summary_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def evaluate_speed_budget(
    rows: Iterable[dict[str, str]],
    config: BudgetConfig,
    *,
    baseline_rows: Iterable[dict[str, str]] | None = None,
) -> BudgetReport:
    current = [_budget_row(row) for row in rows]
    current = [row for row in current if _in_mode(row, config.mode)]
    baseline = (
        [_budget_row(row) for row in baseline_rows]
        if baseline_rows is not None
        else None
    )
    baseline_by_key = {row.key: row for row in baseline or []}
    baseline_enforced = bool(baseline_by_key)
    notes: list[str] = []
    failures: list[BudgetFailure] = []

    if config.no_baseline:
        notes.append(
            "No accepted baseline supplied; regression protection starts after "
            "the first accepted baseline is recorded."
        )
    elif not baseline_enforced:
        failures.append(
            BudgetFailure(
                operation="*",
                case_id="*",
                requested_point_count=0,
                measured_point_count=0,
                speedup="",
                baseline_speedup="",
                reason="missing_baseline",
            )
        )

    measured_1m_rows = [
        row
        for row in current
        if row.comparable
        and row.speedup is not None
        and row.measured_point_count >= 1_000_000
    ]
    if not measured_1m_rows:
        failures.append(
            BudgetFailure(
                operation="*",
                case_id=f"{config.mode}-*",
                requested_point_count=0,
                measured_point_count=1_000_000,
                speedup="",
                baseline_speedup="",
                reason="no_comparable_measured_1m_rows",
            )
        )

    for row in measured_1m_rows:
        assert row.speedup is not None
        if row.speedup < config.min_1m_speedup:
            failures.append(
                _failure(
                    row,
                    reason="below_min_1m_speedup",
                    baseline_speedup="",
                )
            )

    geomean = _geomean([row.speedup for row in measured_1m_rows])
    geomean_text = _format_optional_float(geomean)
    if geomean is not None and geomean < config.min_1m_geomean:
        failures.append(
            BudgetFailure(
                operation="*",
                case_id=f"{config.mode}-measured-1m",
                requested_point_count=0,
                measured_point_count=1_000_000,
                speedup=geomean_text,
                baseline_speedup="",
                reason="below_min_1m_geomean",
            )
        )

    if baseline_enforced:
        for row in current:
            if not row.comparable or row.speedup is None:
                continue
            baseline_row = baseline_by_key.get(row.key)
            if baseline_row is None or baseline_row.speedup is None:
                continue
            if baseline_row.speedup <= 1.0:
                continue
            regression_ratio = (
                baseline_row.speedup - row.speedup
            ) / baseline_row.speedup
            if regression_ratio > config.max_regression_ratio:
                failures.append(
                    _failure(
                        row,
                        reason="baseline_regression",
                        baseline_speedup=_format_float(baseline_row.speedup),
                    )
                )

    return BudgetReport(
        passed=not failures,
        rows=current,
        failures=failures,
        geomean_speedup=geomean_text,
        baseline_enforced=baseline_enforced,
        notes=notes,
    )


def evaluate_speed_budget_from_csv(
    path: Path,
    config: BudgetConfig,
    *,
    baseline_path: Path | None = None,
) -> BudgetReport:
    baseline_rows = read_summary_csv(baseline_path) if baseline_path else None
    return evaluate_speed_budget(
        read_summary_csv(path), config, baseline_rows=baseline_rows
    )


def _budget_row(row: dict[str, str]) -> BudgetRow:
    requested = _int_field(
        row,
        "requested_point_count",
        fallback_key="point_count",
    )
    measured = _int_field(
        row,
        "measured_point_count",
        fallback_key="point_count",
    )
    return BudgetRow(
        operation=row.get("operation", ""),
        case_id=row.get("case_id", ""),
        requested_point_count=requested,
        measured_point_count=measured,
        comparison_status=row.get("comparison_status", ""),
        pcl_rustic_mean_s=row.get("pcl_rustic_mean_s", ""),
        open3d_mean_s=row.get("open3d_mean_s", ""),
        speedup=_speedup(row),
    )


def _int_field(row: dict[str, str], key: str, *, fallback_key: str) -> int:
    value = row.get(key) or row.get(fallback_key) or "0"
    return int(value)


def _speedup(row: dict[str, str]) -> float | None:
    explicit = row.get("pcl_rustic_vs_open3d_speedup")
    if explicit:
        return float(explicit)
    pcl_mean = row.get("pcl_rustic_mean_s")
    open3d_mean = row.get("open3d_mean_s")
    if not pcl_mean or not open3d_mean:
        return None
    pcl = float(pcl_mean)
    if pcl <= 0.0:
        return None
    return float(open3d_mean) / pcl


def _in_mode(row: BudgetRow, mode: str) -> bool:
    return row.case_id.startswith(f"{mode}-")


def _failure(
    row: BudgetRow,
    *,
    reason: str,
    baseline_speedup: str,
) -> BudgetFailure:
    return BudgetFailure(
        operation=row.operation,
        case_id=row.case_id,
        requested_point_count=row.requested_point_count,
        measured_point_count=row.measured_point_count,
        speedup=_format_optional_float(row.speedup),
        baseline_speedup=baseline_speedup,
        reason=reason,
    )


def _geomean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None and value > 0.0]
    if not present:
        return None
    return math.exp(sum(math.log(value) for value in present) / len(present))


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else _format_float(value)


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--min-1m-speedup", type=float, required=True)
    parser.add_argument("--min-1m-geomean", type=float, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--max-regression-ratio", type=float, default=0.10)
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="Skip P3 regression checks for first adoption.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.baseline and args.no_baseline:
        raise SystemExit("--baseline and --no-baseline are mutually exclusive")
    config = BudgetConfig(
        mode=args.mode,
        min_1m_speedup=args.min_1m_speedup,
        min_1m_geomean=args.min_1m_geomean,
        baseline=args.baseline,
        max_regression_ratio=args.max_regression_ratio,
        no_baseline=args.no_baseline,
    )
    report = evaluate_speed_budget_from_csv(
        args.summary_csv,
        config,
        baseline_path=args.baseline,
    )
    _print_report(report)
    raise SystemExit(0 if report.passed else 1)


def _print_report(report: BudgetReport) -> None:
    print(f"measured_1m_geomean_speedup={report.geomean_speedup or 'n/a'}")
    print(f"baseline_enforced={str(report.baseline_enforced).lower()}")
    for note in report.notes:
        print(f"note={note}")
    if not report.failures:
        print("status=pass")
        return
    print("status=fail")
    print(
        "operation,case_id,requested_point_count,measured_point_count,"
        "speedup,baseline_speedup,reason"
    )
    for failure in report.failures:
        print(
            ",".join(
                [
                    failure.operation,
                    failure.case_id,
                    str(failure.requested_point_count),
                    str(failure.measured_point_count),
                    failure.speedup,
                    failure.baseline_speedup,
                    failure.reason,
                ]
            )
        )


if __name__ == "__main__":
    main()
