"""RFC-0009 benchmark suite harness.

The standard and full matrices intentionally allocate very large point clouds.
They are selected only by explicit benchmark mode and remain marked ``slow`` so
normal test runs do not execute them by accident.
"""

from __future__ import annotations

import csv
import os
import platform
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from pcl_rustic import DownsampleStrategy, PointCloud, has_wgpu_device

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]


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

FULL_CONCAT_COUNTS = (20, 40, 80, 120, 160, 200)
FULL_POINTS_PER_CLOUD = 10_000_000
SMOKE_POINTS_PER_CLOUD = 1_000_000
SMOKE_CONCAT_VOXEL_SIZE = 0.15
FULL_CONCAT_VOXEL_SIZE = 5.0

FULL_DOWNSAMPLE_POINTS = {
    "D20": 20_000_000,
    "D50": 50_000_000,
    "D100": 100_000_000,
    "D200": 200_000_000,
    "D400": 400_000_000,
}
SMOKE_DOWNSAMPLE_POINTS = {"D20": 2_000_000}
DOWNSAMPLE_VOXEL_SIZES = (0.05, 0.15, 0.50)
DOWNSAMPLE_STRATEGIES = (
    ("NEAREST_TO_CENTROID", DownsampleStrategy.NEAREST_TO_CENTROID, None),
    ("AVERAGE", DownsampleStrategy.AVERAGE, None),
    ("RANDOM_SEEDED", DownsampleStrategy.RANDOM_SEEDED, 42),
)
XYZ_BYTES_PER_POINT = 12
NON_XYZ_ATTRIBUTE_BYTES_PER_POINT = 4 + 1 + 1 + 8
ATTRIBUTE_BYTES_PER_POINT = XYZ_BYTES_PER_POINT + NON_XYZ_ATTRIBUTE_BYTES_PER_POINT
SYNTHETIC_HOST_BYTES_PER_POINT = int(
    os.environ.get("PCL_RUSTIC_BENCH_HOST_BYTES_PER_POINT", "160")
)
DEVICE_BUDGET_FRACTION = float(
    os.environ.get("PCL_RUSTIC_BENCH_DEVICE_BUDGET_FRACTION", "0.65")
)
SINGLE_ALLOCATION_FRACTION = float(
    os.environ.get("PCL_RUSTIC_BENCH_SINGLE_ALLOCATION_FRACTION", "0.12")
)
HOST_BUDGET_FRACTION = float(
    os.environ.get("PCL_RUSTIC_BENCH_HOST_BUDGET_FRACTION", "0.70")
)
CONCAT_VOXEL_OUTPUT_RATIO_ESTIMATE = float(
    os.environ.get("PCL_RUSTIC_BENCH_CONCAT_VOXEL_OUTPUT_RATIO", "0.05")
)
CONCAT_VOXEL_MAX_OUTPUT_RATIO = 0.50
_INITIALIZED_CSV_PATHS: set[Path] = set()

SKIP_COLUMNS = [
    "case_id",
    "operation",
    "input_points",
    "voxel_size",
    "strategy",
    "reason",
    "estimated_host_bytes",
    "estimated_peak_device_bytes",
    "estimated_max_single_allocation_bytes",
    "detected_device_memory_bytes",
    "detected_device",
    "git_sha",
    "run_date",
]


@dataclass(frozen=True)
class ConcatCase:
    case_id: str
    input_clouds: int
    points_per_cloud: int

    @property
    def input_points(self) -> int:
        return self.input_clouds * self.points_per_cloud


@dataclass(frozen=True)
class DownsampleCase:
    case_id: str
    input_points: int
    voxel_size: float
    strategy_name: str
    strategy: int
    seed: int | None


@dataclass(frozen=True)
class BenchmarkBudget:
    device_name: str
    device_memory_bytes: int | None
    host_memory_bytes: int | None


@dataclass(frozen=True)
class ResourceEstimate:
    host_bytes: int
    peak_device_bytes: int
    max_single_allocation_bytes: int


def concat_cases(mode: str) -> list[ConcatCase]:
    if mode == "full":
        return [
            ConcatCase(f"C{cloud_count}", cloud_count, FULL_POINTS_PER_CLOUD)
            for cloud_count in FULL_CONCAT_COUNTS
        ]
    return [
        ConcatCase(
            "C20",
            20,
            FULL_POINTS_PER_CLOUD if mode == "standard" else SMOKE_POINTS_PER_CLOUD,
        )
    ]


def downsample_cases(mode: str) -> list[DownsampleCase]:
    if mode == "full":
        point_matrix = FULL_DOWNSAMPLE_POINTS
    elif mode == "standard":
        point_matrix = {
            case_id: FULL_DOWNSAMPLE_POINTS[case_id]
            for case_id in ("D20", "D50", "D100")
        }
    else:
        point_matrix = SMOKE_DOWNSAMPLE_POINTS

    return [
        DownsampleCase(case_id, points, voxel_size, strategy_name, strategy, seed)
        for case_id, points in point_matrix.items()
        for voxel_size in DOWNSAMPLE_VOXEL_SIZES
        for strategy_name, strategy, seed in DOWNSAMPLE_STRATEGIES
    ]


def concat_voxel_size(mode: str) -> float:
    return SMOKE_CONCAT_VOXEL_SIZE if mode == "smoke" else FULL_CONCAT_VOXEL_SIZE


def gpu_required(mode: str) -> bool:
    return mode in {"standard", "full"}


def accepted_accelerator(device_name: str) -> bool:
    return any(token in device_name for token in ("Cuda", "Mps", "Metal", "Rocm"))


def accelerator_device_name() -> str:
    if not has_wgpu_device():
        return "unavailable"
    probe = PointCloud.from_numpy(
        {
            "xyz": np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            "intensity": np.array([1.0], dtype=np.float32),
            "classification": np.array([1], dtype=np.uint8),
            "return_number": np.array([1], dtype=np.uint8),
            "gps_time": np.array([0.0], dtype=np.float64),
        }
    )
    return str(probe.device())


def benchmark_budget(mode: str) -> BenchmarkBudget:
    if not gpu_required(mode):
        return BenchmarkBudget(
            device_name="not-required",
            device_memory_bytes=detect_device_memory_bytes(),
            host_memory_bytes=detect_host_memory_bytes(),
        )
    return BenchmarkBudget(
        device_name=accelerator_device_name(),
        device_memory_bytes=detect_device_memory_bytes(),
        host_memory_bytes=detect_host_memory_bytes(),
    )


def detect_device_memory_bytes() -> int | None:
    override = os.environ.get("PCL_RUSTIC_BENCH_DEVICE_MEMORY_BYTES")
    if override:
        return int(override)
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    first_line = output.strip().splitlines()[0]
    return int(first_line.strip()) * 1024 * 1024


def detect_host_memory_bytes() -> int | None:
    override = os.environ.get("PCL_RUSTIC_BENCH_HOST_MEMORY_BYTES")
    if override:
        return int(override)
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None
    return int(page_size * page_count)


def concat_estimate(case: ConcatCase) -> ResourceEstimate:
    input_points = case.input_points
    xyz_bytes = input_points * XYZ_BYTES_PER_POINT
    return ResourceEstimate(
        host_bytes=input_points * SYNTHETIC_HOST_BYTES_PER_POINT,
        peak_device_bytes=xyz_bytes * 2,
        max_single_allocation_bytes=xyz_bytes,
    )


def concat_voxel_estimate(case: ConcatCase) -> ResourceEstimate:
    input_points = case.input_points
    output_upper = max(1, int(input_points * CONCAT_VOXEL_OUTPUT_RATIO_ESTIMATE))
    input_xyz_bytes = input_points * XYZ_BYTES_PER_POINT
    output_xyz_bytes = output_upper * XYZ_BYTES_PER_POINT
    return ResourceEstimate(
        host_bytes=input_points * SYNTHETIC_HOST_BYTES_PER_POINT,
        peak_device_bytes=(input_xyz_bytes + output_xyz_bytes) * 2,
        max_single_allocation_bytes=max(input_xyz_bytes, output_xyz_bytes * 3),
    )


def downsample_estimate(case: DownsampleCase) -> ResourceEstimate:
    input_points = case.input_points
    xyz_bytes = input_points * XYZ_BYTES_PER_POINT
    return ResourceEstimate(
        host_bytes=input_points * SYNTHETIC_HOST_BYTES_PER_POINT,
        peak_device_bytes=xyz_bytes * 4,
        max_single_allocation_bytes=xyz_bytes * 6,
    )


def skip_reason(
    mode: str,
    estimate: ResourceEstimate,
    budget: BenchmarkBudget,
) -> str | None:
    if not gpu_required(mode):
        return None
    if not accepted_accelerator(budget.device_name):
        return f"standard/full benchmark requires accelerator, got {budget.device_name}"
    if budget.device_memory_bytes is None:
        return "accelerator memory is unknown"
    if estimate.peak_device_bytes > int(
        budget.device_memory_bytes * DEVICE_BUDGET_FRACTION
    ):
        return "estimated peak device bytes exceed accelerator budget"
    if estimate.max_single_allocation_bytes > int(
        budget.device_memory_bytes * SINGLE_ALLOCATION_FRACTION
    ):
        return "estimated max single allocation exceeds accelerator budget"
    if budget.host_memory_bytes is None:
        return "host memory is unknown"
    if estimate.host_bytes > int(budget.host_memory_bytes * HOST_BUDGET_FRACTION):
        return "estimated host bytes exceed host budget"
    return None


def generate_rfc0009_cloud_data(
    num_points: int,
    *,
    seed: int,
    cloud_index: int = 0,
    cluster_count: int = 8,
    extent: float = 1_000.0,
) -> dict[str, np.ndarray]:
    """Create deterministic synthetic data with the RFC-0009 typed attributes."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-extent / 2.0, extent / 2.0, size=(cluster_count, 3))
    labels = rng.integers(0, cluster_count, size=num_points, dtype=np.int16)
    noise = rng.normal(0.0, extent / 80.0, size=(num_points, 3)).astype(np.float32)
    xyz = centers[labels].astype(np.float32) + noise

    intensity = rng.uniform(0.0, 255.0, size=num_points).astype(np.float32)
    classification_codes = np.array([1, 2, 3, 4, 5, 6, 9], dtype=np.uint8)
    classification = rng.choice(classification_codes, size=num_points).astype(np.uint8)
    return_number = rng.integers(1, 6, size=num_points, dtype=np.uint8)
    gps_time = np.arange(num_points, dtype=np.float64) * 0.001 + float(
        cloud_index * num_points
    )

    return {
        "xyz": xyz,
        "intensity": intensity,
        "classification": classification,
        "return_number": return_number,
        "gps_time": gps_time,
    }


def make_point_cloud(num_points: int, *, seed: int, cloud_index: int = 0) -> PointCloud:
    return PointCloud.from_numpy(
        generate_rfc0009_cloud_data(num_points, seed=seed, cloud_index=cloud_index)
    )


def assert_rfc_attrs(pc: PointCloud) -> None:
    info = {name: dtype for name, _length, dtype in pc.attribute_info()}
    assert info["intensity"] == "float32"
    assert info["classification"] == "uint8"
    assert info["return_number"] == "uint8"
    assert info["gps_time"] == "float64"


def benchmark_csv_path(mode: str) -> Path:
    path = Path("reports") / "benchmarks" / f"rfc0009-{mode}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path not in _INITIALIZED_CSV_PATHS:
        path.unlink(missing_ok=True)
        _INITIALIZED_CSV_PATHS.add(path)
    return path


def benchmark_skip_csv_path(mode: str) -> Path:
    path = Path("reports") / "benchmarks" / f"rfc0009-{mode}-skips.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path not in _INITIALIZED_CSV_PATHS:
        path.unlink(missing_ok=True)
        _INITIALIZED_CSV_PATHS.add(path)
    return path


def write_csv_row(path: Path, row: dict[str, object]) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def write_skip_row(
    path: Path,
    *,
    case_id: str,
    operation: str,
    input_points: int,
    voxel_size: float | None,
    strategy: str | None,
    reason: str,
    estimate: ResourceEstimate,
    budget: BenchmarkBudget,
) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SKIP_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "case_id": case_id,
                "operation": operation,
                "input_points": input_points,
                "voxel_size": "" if voxel_size is None else voxel_size,
                "strategy": "" if strategy is None else strategy,
                "reason": reason,
                "estimated_host_bytes": estimate.host_bytes,
                "estimated_peak_device_bytes": estimate.peak_device_bytes,
                "estimated_max_single_allocation_bytes": (
                    estimate.max_single_allocation_bytes
                ),
                "detected_device_memory_bytes": (
                    ""
                    if budget.device_memory_bytes is None
                    else budget.device_memory_bytes
                ),
                "detected_device": budget.device_name,
                "git_sha": git_sha(),
                "run_date": datetime.now(timezone.utc).date().isoformat(),
            }
        )


def write_measured_row(mode: str, path: Path, row: dict[str, object]) -> None:
    if gpu_required(mode) and not accepted_accelerator(str(row["device_name"])):
        raise AssertionError(
            f"standard/full benchmark row must use accelerator, got {row['device_name']}"
        )
    write_csv_row(path, row)


def measure(operation: Callable[[], PointCloud]) -> tuple[PointCloud, float, int]:
    start_rss = peak_rss_bytes()
    t0 = time.perf_counter()
    result = operation()
    wall_time = time.perf_counter() - t0
    return result, wall_time, max(start_rss, peak_rss_bytes())


def peak_rss_bytes() -> int:
    try:
        import resource
    except ImportError:
        return 0

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return int(rss)
    return int(rss * 1024)


def git_sha() -> str:
    env_sha = os.environ.get("GITHUB_SHA")
    if env_sha:
        return env_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def base_row(
    *,
    case_id: str,
    operation: str,
    input_clouds: int,
    input_points: int,
    output_points: int,
    voxel_size: float | None,
    strategy: str | None,
    wall_time_s: float,
    peak_rss: int,
    device_name: str,
) -> dict[str, object]:
    return {
        "case_id": case_id,
        "operation": operation,
        "backend": "python-api",
        "input_clouds": input_clouds,
        "input_points": input_points,
        "output_points": output_points,
        "voxel_size": "" if voxel_size is None else voxel_size,
        "strategy": "" if strategy is None else strategy,
        "wall_time_s": f"{wall_time_s:.6f}",
        "throughput_points_s": (
            f"{input_points / wall_time_s:.2f}" if wall_time_s > 0.0 else ""
        ),
        "peak_rss_bytes": peak_rss,
        "estimated_input_bytes": input_points * ATTRIBUTE_BYTES_PER_POINT,
        "estimated_output_bytes": output_points * ATTRIBUTE_BYTES_PER_POINT,
        "device_name": device_name,
        "git_sha": git_sha(),
    }


def benchmark_mode(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--benchmark-mode"))


class TestRFC0009BenchmarkSuite:
    def test_concat_matrix_writes_csv(self, request: pytest.FixtureRequest):
        mode = benchmark_mode(request)
        csv_path = benchmark_csv_path(mode)
        skip_csv_path = benchmark_skip_csv_path(mode)
        budget = benchmark_budget(mode)

        for case in concat_cases(mode):
            concat_resource_estimate = concat_estimate(case)
            reason = skip_reason(mode, concat_resource_estimate, budget)
            if reason is not None:
                write_skip_row(
                    skip_csv_path,
                    case_id=case.case_id,
                    operation="concatenate",
                    input_points=case.input_points,
                    voxel_size=None,
                    strategy=None,
                    reason=reason,
                    estimate=concat_resource_estimate,
                    budget=budget,
                )
                write_skip_row(
                    skip_csv_path,
                    case_id=case.case_id,
                    operation="concatenate_voxelize",
                    input_points=case.input_points,
                    voxel_size=concat_voxel_size(mode),
                    strategy="NEAREST_TO_CENTROID",
                    reason="concatenate prerequisite skipped",
                    estimate=concat_voxel_estimate(case),
                    budget=budget,
                )
                continue

            clouds = [
                make_point_cloud(
                    case.points_per_cloud,
                    seed=10_000 + i,
                    cloud_index=i,
                )
                for i in range(case.input_clouds)
            ]
            for cloud in clouds:
                assert_rfc_attrs(cloud)

            concatenated, concat_time, concat_rss = measure(
                lambda clouds=clouds: PointCloud.concatenate(clouds, "strict")
            )
            assert concatenated.point_count() == case.input_points
            assert_rfc_attrs(concatenated)

            write_measured_row(
                mode,
                csv_path,
                base_row(
                    case_id=case.case_id,
                    operation="concatenate",
                    input_clouds=case.input_clouds,
                    input_points=case.input_points,
                    output_points=concatenated.point_count(),
                    voxel_size=None,
                    strategy=None,
                    wall_time_s=concat_time,
                    peak_rss=concat_rss,
                    device_name=concatenated.device(),
                ),
            )

            voxel_size = concat_voxel_size(mode)
            voxel_resource_estimate = concat_voxel_estimate(case)
            reason = skip_reason(mode, voxel_resource_estimate, budget)
            if reason is not None:
                write_skip_row(
                    skip_csv_path,
                    case_id=case.case_id,
                    operation="concatenate_voxelize",
                    input_points=case.input_points,
                    voxel_size=voxel_size,
                    strategy="NEAREST_TO_CENTROID",
                    reason=reason,
                    estimate=voxel_resource_estimate,
                    budget=budget,
                )
                continue

            downsampled, voxel_time, voxel_rss = measure(
                lambda pc=concatenated, voxel_size=voxel_size: pc.voxel_downsample(
                    voxel_size,
                    DownsampleStrategy.NEAREST_TO_CENTROID,
                )
            )
            assert 0 < downsampled.point_count() <= concatenated.point_count()
            if gpu_required(mode) and case.case_id == "C20":
                assert downsampled.point_count() <= int(
                    concatenated.point_count() * CONCAT_VOXEL_MAX_OUTPUT_RATIO
                )
            assert_rfc_attrs(downsampled)

            write_measured_row(
                mode,
                csv_path,
                base_row(
                    case_id=case.case_id,
                    operation="concatenate_voxelize",
                    input_clouds=case.input_clouds,
                    input_points=case.input_points,
                    output_points=downsampled.point_count(),
                    voxel_size=voxel_size,
                    strategy="NEAREST_TO_CENTROID",
                    wall_time_s=concat_time + voxel_time,
                    peak_rss=max(concat_rss, voxel_rss),
                    device_name=downsampled.device(),
                ),
            )

    def test_downsampling_matrix_writes_csv(self, request: pytest.FixtureRequest):
        mode = benchmark_mode(request)
        csv_path = benchmark_csv_path(mode)
        skip_csv_path = benchmark_skip_csv_path(mode)
        budget = benchmark_budget(mode)

        for case in downsample_cases(mode):
            resource_estimate = downsample_estimate(case)
            reason = skip_reason(mode, resource_estimate, budget)
            if reason is not None:
                write_skip_row(
                    skip_csv_path,
                    case_id=case.case_id,
                    operation="voxel_downsample",
                    input_points=case.input_points,
                    voxel_size=case.voxel_size,
                    strategy=case.strategy_name,
                    reason=reason,
                    estimate=resource_estimate,
                    budget=budget,
                )
                continue

            pc = make_point_cloud(case.input_points, seed=20_000 + case.input_points)
            assert pc.point_count() == case.input_points
            assert_rfc_attrs(pc)

            kwargs = {"seed": case.seed} if case.seed is not None else {}
            downsampled, wall_time, rss = measure(
                lambda pc=pc, case=case, kwargs=kwargs: pc.voxel_downsample(
                    case.voxel_size,
                    case.strategy,
                    **kwargs,
                )
            )
            assert 0 < downsampled.point_count() <= pc.point_count()
            assert_rfc_attrs(downsampled)

            write_measured_row(
                mode,
                csv_path,
                base_row(
                    case_id=case.case_id,
                    operation="voxel_downsample",
                    input_clouds=1,
                    input_points=case.input_points,
                    output_points=downsampled.point_count(),
                    voxel_size=case.voxel_size,
                    strategy=case.strategy_name,
                    wall_time_s=wall_time,
                    peak_rss=rss,
                    device_name=downsampled.device(),
                ),
            )
