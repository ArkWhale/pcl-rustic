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
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from pcl_rustic import DownsampleStrategy, PointCloud

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
CONCAT_VOXEL_SIZE = 0.15

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
ATTRIBUTE_BYTES_PER_POINT = 12 + 4 + 1 + 1 + 8
_INITIALIZED_CSV_PATHS: set[Path] = set()


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


def write_csv_row(path: Path, row: dict[str, object]) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        if needs_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


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

        for case in concat_cases(mode):
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

            write_csv_row(
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

            downsampled, voxel_time, voxel_rss = measure(
                lambda pc=concatenated: pc.voxel_downsample(
                    CONCAT_VOXEL_SIZE,
                    DownsampleStrategy.NEAREST_TO_CENTROID,
                )
            )
            assert 0 < downsampled.point_count() <= concatenated.point_count()
            assert_rfc_attrs(downsampled)

            write_csv_row(
                csv_path,
                base_row(
                    case_id=case.case_id,
                    operation="concatenate_voxelize",
                    input_clouds=case.input_clouds,
                    input_points=case.input_points,
                    output_points=downsampled.point_count(),
                    voxel_size=CONCAT_VOXEL_SIZE,
                    strategy="NEAREST_TO_CENTROID",
                    wall_time_s=concat_time + voxel_time,
                    peak_rss=max(concat_rss, voxel_rss),
                    device_name=downsampled.device(),
                ),
            )

    def test_downsampling_matrix_writes_csv(self, request: pytest.FixtureRequest):
        mode = benchmark_mode(request)
        csv_path = benchmark_csv_path(mode)

        for case in downsample_cases(mode):
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

            write_csv_row(
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
