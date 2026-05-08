"""RFC-0015 pytest-benchmark comparison harness.

The benchmark tests are marked slow and remain collection-safe without optional
Open3D or pytest-benchmark dependencies. Install the benchmark dependency group
and pass ``--run-slow`` to execute them.
"""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest

import pcl_rustic
from pcl_rustic import DownsampleStrategy, NormalSearch, PointCloud, registration

pytestmark = [pytest.mark.benchmark, pytest.mark.slow]

SMOKE_POINT_COUNTS = (1_000, 10_000)
STANDARD_POINT_COUNTS = (10_000, 100_000, 1_000_000)
FULL_POINT_COUNTS = (10_000, 100_000, 1_000_000, 10_000_000)
VOXEL_SIZE = 0.20
KNN_K = 16
RADIUS = 0.35
OUTLIER_NEIGHBORS = 12
REGISTRATION_POINTS_CAP = 10_000


@dataclass(frozen=True)
class ComparisonCase:
    mode: str
    point_count: int

    @property
    def case_id(self) -> str:
        return f"{self.mode}-{self.point_count}"

    @property
    def seed(self) -> int:
        return 30_000 + self.point_count


def comparison_cases(mode: str) -> list[ComparisonCase]:
    if mode == "full":
        point_counts = FULL_POINT_COUNTS
    elif mode == "standard":
        point_counts = STANDARD_POINT_COUNTS
    else:
        point_counts = SMOKE_POINT_COUNTS
    return [ComparisonCase(mode=mode, point_count=count) for count in point_counts]


def comparable_operations() -> tuple[str, ...]:
    return (
        "construction",
        "transform",
        "voxel_downsample",
        "knn_warm",
        "radius_search_warm",
        "estimate_normals",
        "remove_statistical_outlier",
        "remove_radius_outlier",
        "registration_point_to_point",
        "registration_point_to_plane",
    )


def benchmark_mode(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--benchmark-mode"))


@pytest.fixture
def benchmark_runner(request: pytest.FixtureRequest):
    try:
        return request.getfixturevalue("benchmark")
    except pytest.FixtureLookupError:
        pytest.skip("pytest-benchmark is not installed")


@pytest.fixture
def open3d_module():
    return pytest.importorskip("open3d")


def generate_xyz(case: ComparisonCase) -> np.ndarray:
    rng = np.random.default_rng(case.seed)
    centers = rng.uniform(-20.0, 20.0, size=(8, 3)).astype(np.float32)
    labels = rng.integers(0, len(centers), size=case.point_count)
    noise = rng.normal(0.0, 0.75, size=(case.point_count, 3)).astype(np.float32)
    return centers[labels] + noise


def generate_plane_xyz(case: ComparisonCase) -> np.ndarray:
    rng = np.random.default_rng(case.seed)
    xy = rng.uniform(-10.0, 10.0, size=(case.point_count, 2)).astype(np.float32)
    z = np.zeros((case.point_count, 1), dtype=np.float32)
    return np.column_stack([xy, z])


def generate_outlier_xyz(case: ComparisonCase) -> tuple[np.ndarray, np.ndarray]:
    xyz = generate_xyz(case)
    outlier_count = max(2, min(32, case.point_count // 100))
    xyz[:outlier_count] += np.array([200.0, 200.0, 200.0], dtype=np.float32)
    expected_outliers = np.zeros(case.point_count, dtype=np.bool_)
    expected_outliers[:outlier_count] = True
    return xyz, expected_outliers


def make_pcl_cloud(xyz: np.ndarray) -> PointCloud:
    return PointCloud.from_xyz(xyz)


def make_open3d_cloud(o3d: Any, xyz: np.ndarray):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(xyz.astype(np.float64, copy=False))
    return cloud


def make_metadata(
    *,
    library: str,
    operation: str,
    case: ComparisonCase,
    output_points: int | None,
    comparable: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "library": library,
        "operation": operation,
        "case_id": case.case_id,
        "point_count": case.point_count,
        "output_points": output_points,
        "comparable": comparable,
        "git_sha": git_sha(),
        "python_version": platform.python_version(),
        "pcl_rustic_version": pcl_rustic.__version__,
        "open3d_version": package_version("open3d"),
        "numpy_version": np.__version__,
        "cpu": platform.processor() or platform.machine(),
        "gpu": "",
        "os": platform.platform(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "uninstalled"


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def attach_metadata(benchmark: Any, metadata: dict[str, Any]) -> None:
    extra_info = getattr(benchmark, "extra_info", None)
    if isinstance(extra_info, dict):
        extra_info.update(metadata)


def benchmark_fresh_input(
    benchmark: Any,
    setup: Callable[[], Any],
    operation: Callable[[Any], Any],
):
    if hasattr(benchmark, "pedantic"):

        def setup_args():
            return (setup(),), {}

        return benchmark.pedantic(operation, setup=setup_args, rounds=3, iterations=1)
    return benchmark(lambda: operation(setup()))


def assert_transform_equivalent(before: np.ndarray, after: np.ndarray) -> None:
    matrix = transform_matrix()
    expected = before @ matrix[:3, :3].T + matrix[:3, 3]
    np.testing.assert_allclose(after, expected, rtol=1e-5, atol=1e-5)


def transform_matrix() -> np.ndarray:
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 3] = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    return matrix


@pytest.mark.parametrize("library", ("pcl_rustic", "open3d"))
def test_construction_benchmark(
    request: pytest.FixtureRequest,
    benchmark_runner,
    open3d_module,
    library: str,
) -> None:
    o3d = open3d_module
    for case in comparison_cases(benchmark_mode(request)):
        xyz = generate_xyz(case)
        if library == "pcl_rustic":
            result = benchmark_runner(lambda xyz=xyz: PointCloud.from_xyz(xyz))
            output_points = result.point_count()
            np.testing.assert_allclose(result.get_xyz(), xyz, rtol=1e-6, atol=1e-6)
        else:
            result = benchmark_runner(lambda xyz=xyz: make_open3d_cloud(o3d, xyz))
            output_points = len(result.points)
            np.testing.assert_allclose(
                np.asarray(result.points), xyz, rtol=1e-6, atol=1e-6
            )
        attach_metadata(
            benchmark_runner,
            make_metadata(
                library=library,
                operation="construction",
                case=case,
                output_points=output_points,
                comparable=True,
                extra={"cache_policy": "none"},
            ),
        )


@pytest.mark.parametrize("library", ("pcl_rustic", "open3d"))
def test_transform_benchmark(
    request: pytest.FixtureRequest,
    benchmark_runner,
    open3d_module,
    library: str,
) -> None:
    o3d = open3d_module
    matrix = transform_matrix()
    for case in comparison_cases(benchmark_mode(request)):
        xyz = generate_xyz(case)
        if library == "pcl_rustic":
            result = benchmark_fresh_input(
                benchmark_runner,
                lambda xyz=xyz: make_pcl_cloud(xyz),
                lambda cloud: cloud.transform(matrix),
            )
            result_xyz = result.get_xyz()
        else:

            def transform_open3d(cloud):
                cloud.transform(matrix.astype(np.float64))
                return cloud

            result = benchmark_fresh_input(
                benchmark_runner,
                lambda xyz=xyz: make_open3d_cloud(o3d, xyz),
                transform_open3d,
            )
            result_xyz = np.asarray(result.points)
        assert_transform_equivalent(xyz, result_xyz)
        attach_metadata(
            benchmark_runner,
            make_metadata(
                library=library,
                operation="transform",
                case=case,
                output_points=len(result_xyz),
                comparable=True,
                extra={"cache_policy": "none"},
            ),
        )


@pytest.mark.parametrize("library", ("pcl_rustic", "open3d"))
def test_voxel_downsample_benchmark(
    request: pytest.FixtureRequest,
    benchmark_runner,
    open3d_module,
    library: str,
) -> None:
    o3d = open3d_module
    for case in comparison_cases(benchmark_mode(request)):
        xyz = generate_xyz(case)
        if library == "pcl_rustic":
            result = benchmark_fresh_input(
                benchmark_runner,
                lambda xyz=xyz: make_pcl_cloud(xyz),
                lambda cloud: cloud.voxel_downsample(
                    VOXEL_SIZE,
                    DownsampleStrategy.NEAREST_TO_CENTROID,
                ),
            )
            output_points = result.point_count()
        else:
            result = benchmark_fresh_input(
                benchmark_runner,
                lambda xyz=xyz: make_open3d_cloud(o3d, xyz),
                lambda cloud: cloud.voxel_down_sample(VOXEL_SIZE),
            )
            output_points = len(result.points)
        assert 0 < output_points <= case.point_count
        attach_metadata(
            benchmark_runner,
            make_metadata(
                library=library,
                operation="voxel_downsample",
                case=case,
                output_points=output_points,
                comparable=True,
                extra={"voxel_size": VOXEL_SIZE, "cache_policy": "none"},
            ),
        )


@pytest.mark.parametrize("operation", ("knn_warm", "radius_search_warm"))
@pytest.mark.parametrize("library", ("pcl_rustic", "open3d"))
def test_neighbor_benchmark(
    request: pytest.FixtureRequest,
    benchmark_runner,
    open3d_module,
    library: str,
    operation: str,
) -> None:
    o3d = open3d_module
    for case in comparison_cases(benchmark_mode(request)):
        xyz = generate_xyz(case)
        query = xyz[: min(64, len(xyz))]
        if library == "pcl_rustic":
            cloud = make_pcl_cloud(xyz)
            if operation == "knn_warm":
                cloud.knn(query[:1], KNN_K)
                result = benchmark_runner(
                    lambda cloud=cloud, query=query: cloud.knn(query, KNN_K)
                )
                output_points = len(result[0])
            else:
                cloud.radius_search(query[:1], RADIUS)
                result = benchmark_runner(
                    lambda cloud=cloud, query=query: cloud.radius_search(query, RADIUS)
                )
                output_points = sum(len(indices) for indices in result)
        else:
            cloud = make_open3d_cloud(o3d, xyz)
            tree = o3d.geometry.KDTreeFlann(cloud)
            if operation == "knn_warm":

                def run_knn(tree=tree, query=query):
                    return [
                        tree.search_knn_vector_3d(point, KNN_K)[1] for point in query
                    ]

                result = benchmark_runner(run_knn)
                output_points = sum(len(indices) for indices in result)
            else:

                def run_radius(tree=tree, query=query):
                    return [
                        tree.search_radius_vector_3d(point, RADIUS)[1]
                        for point in query
                    ]

                result = benchmark_runner(run_radius)
                output_points = sum(len(indices) for indices in result)
        attach_metadata(
            benchmark_runner,
            make_metadata(
                library=library,
                operation=operation,
                case=case,
                output_points=output_points,
                comparable=True,
                extra={
                    "query_count": len(query),
                    "neighbors": KNN_K if operation == "knn_warm" else "",
                    "radius": RADIUS if operation == "radius_search_warm" else "",
                    "cache_policy": "warm",
                },
            ),
        )


@pytest.mark.parametrize("library", ("pcl_rustic", "open3d"))
def test_estimate_normals_benchmark(
    request: pytest.FixtureRequest,
    benchmark_runner,
    open3d_module,
    library: str,
) -> None:
    o3d = open3d_module
    for case in comparison_cases(benchmark_mode(request)):
        xyz = generate_plane_xyz(case)
        if library == "pcl_rustic":

            def estimate(cloud: PointCloud):
                cloud.estimate_normals(NormalSearch.knn(KNN_K))
                return cloud

            result = benchmark_fresh_input(
                benchmark_runner,
                lambda xyz=xyz: make_pcl_cloud(xyz),
                estimate,
            )
            output_points = result.point_count()
        else:

            def estimate_open3d(cloud):
                cloud.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=KNN_K)
                )
                return cloud

            result = benchmark_fresh_input(
                benchmark_runner,
                lambda xyz=xyz: make_open3d_cloud(o3d, xyz),
                estimate_open3d,
            )
            output_points = len(result.points)
        attach_metadata(
            benchmark_runner,
            make_metadata(
                library=library,
                operation="estimate_normals",
                case=case,
                output_points=output_points,
                comparable=True,
                extra={"neighbors": KNN_K, "cache_policy": "none"},
            ),
        )


@pytest.mark.parametrize(
    ("operation", "method_args"),
    (
        (
            "remove_statistical_outlier",
            {"nb_neighbors": OUTLIER_NEIGHBORS, "std_ratio": 2.0},
        ),
        ("remove_radius_outlier", {"nb_points": 2, "radius": RADIUS}),
    ),
)
@pytest.mark.parametrize("library", ("pcl_rustic", "open3d"))
def test_outlier_benchmark(
    request: pytest.FixtureRequest,
    benchmark_runner,
    open3d_module,
    library: str,
    operation: str,
    method_args: dict[str, Any],
) -> None:
    o3d = open3d_module
    for case in comparison_cases(benchmark_mode(request)):
        xyz, _expected_outliers = generate_outlier_xyz(case)
        if library == "pcl_rustic":
            if operation == "remove_statistical_outlier":
                result = benchmark_fresh_input(
                    benchmark_runner,
                    lambda xyz=xyz: make_pcl_cloud(xyz),
                    lambda cloud: cloud.remove_statistical_outlier(**method_args)[0],
                )
            else:
                result = benchmark_fresh_input(
                    benchmark_runner,
                    lambda xyz=xyz: make_pcl_cloud(xyz),
                    lambda cloud: cloud.remove_radius_outlier(**method_args)[0],
                )
            output_points = result.point_count()
        else:
            if operation == "remove_statistical_outlier":
                result = benchmark_fresh_input(
                    benchmark_runner,
                    lambda xyz=xyz: make_open3d_cloud(o3d, xyz),
                    lambda cloud: cloud.remove_statistical_outlier(**method_args)[0],
                )
            else:
                result = benchmark_fresh_input(
                    benchmark_runner,
                    lambda xyz=xyz: make_open3d_cloud(o3d, xyz),
                    lambda cloud: cloud.remove_radius_outlier(**method_args)[0],
                )
            output_points = len(result.points)
        assert 0 < output_points <= case.point_count
        attach_metadata(
            benchmark_runner,
            make_metadata(
                library=library,
                operation=operation,
                case=case,
                output_points=output_points,
                comparable=True,
                extra={**method_args, "cache_policy": "none"},
            ),
        )


@pytest.mark.parametrize(
    "operation",
    ("registration_point_to_point", "registration_point_to_plane"),
)
@pytest.mark.parametrize("library", ("pcl_rustic", "open3d"))
def test_registration_benchmark(
    request: pytest.FixtureRequest,
    benchmark_runner,
    open3d_module,
    library: str,
    operation: str,
) -> None:
    o3d = open3d_module
    criteria = registration.ICPConvergenceCriteria(max_iteration=10)
    for case in comparison_cases(benchmark_mode(request)):
        limited_case = ComparisonCase(
            mode=case.mode,
            point_count=min(case.point_count, REGISTRATION_POINTS_CAP),
        )
        source_xyz = generate_plane_xyz(limited_case)
        target_xyz = source_xyz + np.array([0.05, -0.02, 0.01], dtype=np.float32)
        init = np.eye(4, dtype=np.float32)
        if library == "pcl_rustic":
            source = make_pcl_cloud(source_xyz)
            target = make_pcl_cloud(target_xyz)
            if operation == "registration_point_to_plane":
                target.estimate_normals(NormalSearch.knn(KNN_K))
                estimation = registration.TransformationEstimation.point_to_plane()
            else:
                estimation = registration.TransformationEstimation.point_to_point()
            result = benchmark_runner(
                lambda source=source,
                target=target,
                estimation=estimation,
                init=init,
                criteria=criteria: registration.icp(
                    source,
                    target,
                    0.5,
                    init,
                    estimation,
                    criteria,
                )
            )
            output_points = len(result.correspondence_set)
        else:
            source = make_open3d_cloud(o3d, source_xyz)
            target = make_open3d_cloud(o3d, target_xyz)
            if operation == "registration_point_to_plane":
                target.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamKNN(knn=KNN_K)
                )
                estimation = (
                    o3d.pipelines.registration.TransformationEstimationPointToPlane()
                )
            else:
                estimation = (
                    o3d.pipelines.registration.TransformationEstimationPointToPoint()
                )
            criteria_o3d = o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=10
            )
            result = benchmark_runner(
                lambda source=source,
                target=target,
                estimation=estimation,
                init=init,
                criteria_o3d=criteria_o3d: (
                    o3d.pipelines.registration.registration_icp(
                        source,
                        target,
                        0.5,
                        init.astype(np.float64),
                        estimation,
                        criteria_o3d,
                    )
                )
            )
            output_points = len(result.correspondence_set)
        attach_metadata(
            benchmark_runner,
            make_metadata(
                library=library,
                operation=operation,
                case=case,
                output_points=output_points,
                comparable=True,
                extra={
                    "estimator": operation.replace("registration_", ""),
                    "cache_policy": "warm",
                },
            ),
        )
