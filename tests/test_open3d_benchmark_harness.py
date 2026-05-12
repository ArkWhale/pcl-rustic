from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.test_open3d_benchmark import (
    benchmark_libraries,
    comparable_operations,
    comparison_cases,
    make_metadata,
    pcl_neighbor_execution_metadata,
)


def test_comparison_modes_follow_rfc0015_scale_matrix() -> None:
    assert [case.point_count for case in comparison_cases("smoke")] == [1000, 10_000]
    assert [case.point_count for case in comparison_cases("standard")] == [
        10_000,
        100_000,
        1_000_000,
    ]
    assert [case.point_count for case in comparison_cases("full")] == [
        10_000,
        100_000,
        1_000_000,
        10_000_000,
    ]


def test_comparable_operations_include_phase_families() -> None:
    operations = comparable_operations()

    assert "construction" in operations
    assert "transform" in operations
    assert "voxel_downsample" in operations
    assert "knn_warm" in operations
    assert "radius_search_warm" in operations
    assert "estimate_normals" in operations
    assert "remove_statistical_outlier" in operations
    assert "registration_point_to_point" in operations
    assert "typed_concat" not in operations


class Config:
    def __init__(self, compare_open3d: bool) -> None:
        self.compare_open3d = compare_open3d

    def getoption(self, name: str) -> bool:
        assert name == "--benchmark-compare-open3d"
        return self.compare_open3d


def test_benchmark_libraries_are_pcl_only_by_default() -> None:
    assert benchmark_libraries(Config(compare_open3d=False)) == ("pcl_rustic",)


def test_benchmark_libraries_add_open3d_when_requested() -> None:
    assert benchmark_libraries(Config(compare_open3d=True)) == ("pcl_rustic", "open3d")


def test_metadata_contract_is_self_describing() -> None:
    case = comparison_cases("smoke")[0]

    metadata = make_metadata(
        library="pcl_rustic",
        operation="voxel_downsample",
        case=case,
        output_points=123,
        comparable=True,
        extra={"voxel_size": 0.25, "cache_policy": "none"},
    )

    assert metadata["library"] == "pcl_rustic"
    assert metadata["operation"] == "voxel_downsample"
    assert metadata["case_id"] == "smoke-1000"
    assert metadata["point_count"] == 1000
    assert metadata["output_points"] == 123
    assert metadata["comparable"] is True
    assert metadata["voxel_size"] == 0.25
    assert metadata["cache_policy"] == "none"
    assert metadata["git_sha"]
    assert metadata["pcl_rustic_version"]
    assert "python_version" in metadata
    assert "numpy_version" in metadata


def test_pcl_neighbor_execution_metadata_distinguishes_backend_and_storage() -> None:
    class Cloud:
        def device(self) -> str:
            return "Cuda(Cuda(0))"

    metadata = pcl_neighbor_execution_metadata(Cloud())

    assert metadata["execution_backend"] == "cpu_rayon"
    assert metadata["gpu_accelerated"] is False
    assert metadata["storage_device"] == "Cuda(Cuda(0))"
    assert metadata["rayon_current_num_threads"] >= 1
