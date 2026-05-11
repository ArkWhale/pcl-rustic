"""
pcl_rustic 点云库的 pytest 测试用例

覆盖核心功能、边界场景、异常场景
"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from pcl_rustic import (
    DownsampleStrategy,
    NormalSearch,
    PointCloud,
    has_wgpu_device,
    registration,
)


def load_example_function(script_name: str, function_name: str):
    path = Path(__file__).resolve().parents[1] / "examples" / script_name
    spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, function_name)


class TestPointCloudLifecycle:
    """点云生命周期测试"""

    def test_create_empty_point_cloud(self):
        """测试创建空点云"""
        pc = PointCloud()
        assert pc.point_count() == 0
        assert not pc.has_intensity()
        assert not pc.has_rgb()

    def test_create_from_xyz(self):
        """测试从 numpy XYZ 数组创建点云"""
        xyz = np.array(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
        )
        pc = PointCloud.from_xyz(xyz)
        assert pc.point_count() == 3
        result = pc.get_xyz()
        np.testing.assert_array_almost_equal(result, xyz)

    def test_autocast_f64_input(self):
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        pc = PointCloud.from_xyz(xyz)
        assert pc.get_xyz().dtype == np.float32
        np.testing.assert_allclose(pc.get_xyz(), xyz.astype(np.float32))

    def test_autocast_int_input(self):
        xyz = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)
        pc = PointCloud.from_xyz(xyz)
        assert pc.get_xyz().dtype == np.float32
        np.testing.assert_allclose(pc.get_xyz(), xyz.astype(np.float32))

    def test_reject_string_input(self):
        with pytest.raises(ValueError):
            PointCloud.from_xyz(np.array([["x", "y", "z"]], dtype=str))

    def test_clone_point_cloud(self):
        """测试点云克隆"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc1 = PointCloud.from_xyz(xyz)
        pc2 = pc1.clone()
        assert pc2.point_count() == 1
        np.testing.assert_array_almost_equal(pc2.get_xyz(), pc1.get_xyz())


class TestPointCloudProperties:
    """点云属性测试"""

    def test_set_intensity(self):
        """测试设置 intensity"""
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        intensity = np.array([100.0, 200.0], dtype=np.float32)
        pc.set_intensity(intensity)

        assert pc.has_intensity()
        np.testing.assert_array_almost_equal(pc.get_intensity(), intensity)

    def test_set_rgb(self):
        """测试设置 RGB"""
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        r = np.array([255.0, 0.0], dtype=np.float32)
        g = np.array([0.0, 255.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        pc.set_rgb(r, g, b)

        assert pc.has_rgb()
        result = pc.get_rgb()
        assert result is not None
        np.testing.assert_array_almost_equal(result[0], [255, 0])  # R 通道

    def test_add_custom_attribute(self):
        """测试添加自定义属性"""
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        attr_data = np.array([1.5, 2.5], dtype=np.float32)
        pc.add_attribute("confidence", attr_data)

        assert "confidence" in pc.attribute_names()
        np.testing.assert_array_almost_equal(pc.get_attribute("confidence"), attr_data)

    def test_typed_attributes_round_trip(self):
        xyz = np.arange(12, dtype=np.float32).reshape(4, 3)
        pc = PointCloud.from_xyz(xyz)
        values = {
            "confidence": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "classification": np.array([2, 2, 6, 9], dtype=np.uint8),
            "source_id": np.array([1, 2, 3, 4], dtype=np.uint16),
            "point_source_uid": np.array([10, 20, 30, 40], dtype=np.uint32),
            "wavepacket_offset": np.array(
                [0, 2**32 + 1, 2**40 + 7, 2**48 + 11], dtype=np.uint64
            ),
            "scan_angle_raw": np.array([-120, -1, 1, 120], dtype=np.int16),
            "scan_angle": np.array([-3, -1, 1, 3], dtype=np.int32),
            "global_id": np.array([100, 200, 300, 400], dtype=np.int64),
            "gps_time": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            "flag": np.array([True, False, True, False], dtype=np.bool_),
        }
        for name, data in values.items():
            pc.set_attribute(name, data)
            out = pc.get_attribute(name)
            assert out.dtype == data.dtype
            np.testing.assert_array_equal(out, data)

        selected = pc.select(np.array([True, False, True, False], dtype=np.bool_))
        np.testing.assert_array_equal(
            selected.get_attribute("wavepacket_offset"),
            values["wavepacket_offset"][[0, 2]],
        )
        np.testing.assert_array_equal(
            selected.get_attribute("scan_angle_raw"),
            values["scan_angle_raw"][[0, 2]],
        )

        concatenated = PointCloud.concatenate([selected, selected], "strict")
        assert concatenated.get_attribute("wavepacket_offset").dtype == np.uint64
        assert concatenated.get_attribute("scan_angle_raw").dtype == np.int16
        np.testing.assert_array_equal(
            concatenated.get_attribute("wavepacket_offset"),
            np.concatenate(
                [
                    values["wavepacket_offset"][[0, 2]],
                    values["wavepacket_offset"][[0, 2]],
                ]
            ),
        )

        covariance = np.arange(24, dtype=np.float32).reshape(4, 6)
        pc.set_attribute("covariance", covariance)
        out_covariance = pc.get_attribute("covariance")
        assert out_covariance.dtype == np.float32
        assert out_covariance.shape == (4, 6)
        np.testing.assert_array_equal(out_covariance, covariance)

    def test_from_numpy_alias(self):
        data = {
            "xyz": np.arange(9, dtype=np.float32).reshape(3, 3),
            "classification": np.array([1, 2, 2], dtype=np.uint8),
        }
        pc = PointCloud.from_numpy(data)
        np.testing.assert_array_equal(
            pc.get_attribute("classification"), data["classification"]
        )

    def test_add_duplicate_attribute_fails(self):
        """测试添加重复属性时失败"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        pc.add_attribute("test", np.array([1.0], dtype=np.float32))

        with pytest.raises(ValueError):
            pc.add_attribute("test", np.array([2.0], dtype=np.float32))

    def test_set_attribute_overwrites(self):
        """测试设置属性会覆盖"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        pc.add_attribute("test", np.array([1.0], dtype=np.float32))
        pc.set_attribute("test", np.array([2.0], dtype=np.float32))

        np.testing.assert_array_almost_equal(pc.get_attribute("test"), [2.0])

    def test_remove_attribute(self):
        """测试移除属性"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        pc.add_attribute("test", np.array([1.0], dtype=np.float32))
        pc.remove_attribute("test")

        assert "test" not in pc.attribute_names()

    def test_attribute_dimension_mismatch(self):
        """测试属性维度不匹配时失败"""
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        with pytest.raises(ValueError):
            pc.add_attribute(
                "test", np.array([1.0], dtype=np.float32)
            )  # 只有1个值，点云有2个点

    def test_empty_point_cloud_rejects_non_empty_attribute(self):
        pc = PointCloud()

        with pytest.raises(ValueError):
            pc.set_attribute("classification", np.array([1], dtype=np.uint8))


class TestCoordinateTransform:
    """坐标变换测试"""

    def test_3x3_transform_scaling(self):
        """测试 3x3 变换矩阵（缩放）"""
        xyz = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        # 2 倍缩放矩阵
        matrix = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]

        pc_scaled = pc.transform(matrix)
        result = pc_scaled.get_xyz()

        assert abs(result[0][0] - 2.0) < 1e-5
        assert abs(result[1][1] - 2.0) < 1e-5

    def test_3x3_transform_rotation(self):
        """测试 3x3 变换矩阵（旋转）"""
        xyz = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        # 90 度绕 Z 轴旋转
        angle = math.pi / 2
        matrix = [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]

        pc_rotated = pc.transform(matrix)
        result = pc_rotated.get_xyz()

        # 应接近 [0, 1, 0]
        assert abs(result[0][0]) < 1e-5
        assert abs(result[0][1] - 1.0) < 1e-5

    def test_rigid_transform(self):
        """测试刚体变换（旋转+平移）"""
        xyz = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        # 恒等旋转
        rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        # 平移 [1, 2, 3]
        translation = [1.0, 2.0, 3.0]

        pc_transformed = pc.rigid_transform(rotation, translation)
        result = pc_transformed.get_xyz()

        assert abs(result[0][0] - 2.0) < 1e-5
        assert abs(result[0][1] - 2.0) < 1e-5
        assert abs(result[0][2] - 3.0) < 1e-5

    def test_invalid_matrix_dimension(self):
        """测试非法矩阵维度"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        # 2x2 矩阵（不支持）
        with pytest.raises(ValueError):
            pc.transform([[1.0, 0.0], [0.0, 1.0]])

    def test_4x4_translation_preserves_fractional_precision(self):
        xyz = np.array(
            [[3.2090764, 13.60938, -6.618447], [-4.200236, 18.88349, -20.33915]],
            dtype=np.float32,
        )
        pc = PointCloud.from_xyz(xyz)
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = np.array([1.0, -2.0, 0.5], dtype=np.float32)

        result = pc.transform(matrix).get_xyz()

        np.testing.assert_allclose(result, xyz + matrix[:3, 3], rtol=1e-5, atol=1e-5)

    def test_translate_scale_rotate_wrappers(self):
        xyz = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)
        moved = pc.translate([1.0, 2.0, 3.0])
        np.testing.assert_allclose(moved.get_xyz(), [[2.0, 2.0, 3.0]], atol=1e-5)
        scaled = pc.scale(2.0, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(scaled.get_xyz(), [[2.0, 0.0, 0.0]], atol=1e-5)
        rotated = pc.rotate(
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(rotated.get_xyz(), [[0.0, 1.0, 0.0]], atol=1e-5)


class TestSelectionAndConcatenation:
    def test_feature_and_spatial_selection(self):
        xyz = np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            dtype=np.float32,
        )
        pc = PointCloud.from_xyz(xyz)
        pc.set_attribute("classification", np.array([2, 6, 2, 9], dtype=np.uint8))
        pc.set_attribute("return_number", np.array([1, 1, 2, 2], dtype=np.uint8))
        pc.set_intensity(np.array([0.1, 0.4, 0.7, 1.0], dtype=np.float32))

        assert pc.select(np.array([True, False, True, False])).point_count() == 2
        assert pc.select_where("classification", "in", [2.0, 6.0]).point_count() == 3
        assert pc.select_where("intensity", "range", [0.3, 0.8]).point_count() == 2
        assert pc.select_by_classification([2]).point_count() == 2
        assert pc.select_return_number(1).point_count() == 2
        assert pc.select_intensity_range(0.3, 0.8).point_count() == 2
        assert pc.select_elevation_range(0.5, 2.5).point_count() == 2
        assert pc.crop_aabb([0.5, 0.5, 0.5], [2.5, 2.5, 2.5]).point_count() == 2
        assert pc.aabb() == ([0.0, 0.0, 0.0], [3.0, 3.0, 3.0])
        center, extents, rotation = pc.obb()
        assert len(center) == 3
        assert len(extents) == 3
        assert np.asarray(rotation, dtype=np.float32).shape == (3, 3)
        assert pc.crop_obb(center, extents, rotation).point_count() == 4

    def test_concatenate_policies(self):
        pc1 = PointCloud.from_xyz(np.zeros((2, 3), dtype=np.float32))
        pc1.set_attribute("classification", np.array([1, 2], dtype=np.uint8))
        pc2 = PointCloud.from_xyz(np.ones((1, 3), dtype=np.float32))
        pc2.set_attribute("classification", np.array([3], dtype=np.uint8))
        strict = PointCloud.concatenate([pc1, pc2])
        assert strict.point_count() == 3
        np.testing.assert_array_equal(strict.get_attribute("classification"), [1, 2, 3])

        pc3 = PointCloud.from_xyz(np.ones((1, 3), dtype=np.float32))
        with pytest.raises(ValueError):
            PointCloud.concatenate([pc1, pc3], "strict")
        union = PointCloud.concatenate([pc1, pc3], "union")
        np.testing.assert_array_equal(union.get_attribute("classification"), [1, 2, 0])

        intersection = PointCloud.concatenate([pc1, pc3], "intersection")
        assert intersection.point_count() == 3
        assert "classification" not in intersection.attribute_names()

    def test_concatenate_edge_cases(self):
        pc = PointCloud.from_xyz(np.arange(6, dtype=np.float32).reshape(2, 3))
        pc.set_attribute("classification", np.array([1, 2], dtype=np.uint8))

        single = PointCloud.concatenate([pc], "strict")
        assert single.point_count() == pc.point_count()
        np.testing.assert_array_equal(
            single.get_attribute("classification"), pc.get_attribute("classification")
        )

        other = PointCloud.from_xyz(np.ones((1, 3), dtype=np.float32))
        other.set_attribute("classification", np.array([1.0], dtype=np.float32))
        with pytest.raises(ValueError):
            PointCloud.concatenate([pc, other], "strict")

    def test_empty_selection_preserves_attribute_schema_for_strict_concat(self):
        xyz = np.arange(9, dtype=np.float32).reshape(3, 3)
        pc = PointCloud.from_xyz(xyz)
        pc.set_attribute("classification", np.array([1, 2, 3], dtype=np.uint8))
        pc.set_attribute("gps_time", np.array([0.1, 0.2, 0.3], dtype=np.float64))

        empty = pc.select(np.array([False, False, False], dtype=np.bool_))

        assert empty.point_count() == 0
        assert empty.get_xyz().shape == (0, 3)
        assert set(empty.attribute_names()) == {"classification", "gps_time"}
        assert empty.get_attribute("classification").dtype == np.uint8
        assert empty.get_attribute("gps_time").dtype == np.float64
        assert empty.get_attribute("classification").shape == (0,)
        assert empty.get_attribute("gps_time").shape == (0,)

        concatenated = PointCloud.concatenate([empty, pc], "strict")
        assert concatenated.point_count() == pc.point_count()
        np.testing.assert_array_equal(
            concatenated.get_attribute("classification"),
            pc.get_attribute("classification"),
        )
        np.testing.assert_array_equal(
            concatenated.get_attribute("gps_time"),
            pc.get_attribute("gps_time"),
        )


class TestVoxelDownsample:
    """体素下采样测试"""

    def test_random_downsample(self):
        """测试随机采样下采样"""
        # 创建两个体素中各 2 个点
        xyz = np.array(
            [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],  # 体素1
                [1.1, 1.1, 1.1],
                [1.2, 1.2, 1.2],  # 体素2
            ],
            dtype=np.float32,
        )
        pc = PointCloud.from_xyz(xyz)

        downsampled = pc.voxel_downsample(1.0, DownsampleStrategy.RANDOM_SEEDED)

        # 下采样后应该有 2 个点（每个体素 1 个）
        assert downsampled.point_count() <= pc.point_count()
        assert downsampled.point_count() > 0

    def test_centroid_downsample(self):
        """测试重心采样下采样"""
        xyz = np.array(
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.1, 1.1, 1.1], [1.2, 1.2, 1.2]],
            dtype=np.float32,
        )
        pc = PointCloud.from_xyz(xyz)

        downsampled = pc.voxel_downsample(1.0, DownsampleStrategy.NEAREST_TO_CENTROID)

        assert downsampled.point_count() <= pc.point_count()

    def test_downsample_with_intensity(self):
        """测试下采样保留 intensity"""
        xyz = np.array(
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.1, 1.1, 1.1]], dtype=np.float32
        )
        intensity = np.array([100.0, 200.0, 300.0], dtype=np.float32)

        pc = PointCloud.from_xyz(xyz)
        pc.set_intensity(intensity)

        downsampled = pc.voxel_downsample(1.0, DownsampleStrategy.RANDOM_SEEDED)

        assert downsampled.has_intensity()
        result_intensity = downsampled.get_intensity()
        assert result_intensity is not None
        assert len(result_intensity) == downsampled.point_count()

    def test_invalid_voxel_size(self):
        """测试无效 voxel_size"""
        xyz = np.array([[0.1, 0.1, 0.1]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        with pytest.raises(ValueError):
            pc.voxel_downsample(-1.0, DownsampleStrategy.RANDOM_SEEDED)

    def test_seeded_random_is_deterministic(self):
        xyz = np.column_stack(
            [np.linspace(0, 10, 100), np.zeros(100), np.zeros(100)]
        ).astype(np.float32)
        pc = PointCloud.from_xyz(xyz)
        a = pc.voxel_downsample(1.0, DownsampleStrategy.RANDOM_SEEDED, seed=123)
        b = pc.voxel_downsample(1.0, DownsampleStrategy.RANDOM_SEEDED, seed=123)
        np.testing.assert_array_equal(a.get_xyz(), b.get_xyz())

    def test_average_downsample_modes_integer_attributes(self):
        xyz = np.array(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
        )
        pc = PointCloud.from_xyz(xyz)
        pc.set_attribute("classification", np.array([2, 2, 6], dtype=np.uint8))
        down = pc.voxel_downsample(1.0, DownsampleStrategy.AVERAGE)
        assert down.point_count() == 2
        assert down.get_attribute("classification").dtype == np.uint8


class TestDeviceResidency:
    def _cloud(self):
        xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [2.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        pc = PointCloud.from_xyz(xyz).to("cpu")
        pc.set_attribute("classification", np.array([2, 2, 6, 6], dtype=np.uint8))
        pc.set_intensity(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
        return pc

    def test_cpu_selection_concat_voxel_and_transform_preserve_device(self):
        pc = self._cloud()
        expected_device = pc.device()

        selected = pc.select(np.array([True, False, True, False], dtype=np.bool_))
        assert selected.device() == expected_device
        empty = pc.select_indices([])
        assert empty.device() == expected_device

        concatenated = PointCloud.concatenate([selected, empty], "strict")
        assert concatenated.device() == expected_device

        for strategy in (
            DownsampleStrategy.RANDOM_SEEDED,
            DownsampleStrategy.NEAREST_TO_CENTROID,
            DownsampleStrategy.AVERAGE,
        ):
            downsampled = pc.voxel_downsample(1.0, strategy, seed=7)
            assert downsampled.device() == expected_device

        transformed = pc.transform(np.eye(4, dtype=np.float32))
        assert transformed.device() == expected_device

    def test_cpu_full_pipeline_preserves_device_and_semantics(self):
        pc = self._cloud()
        expected_device = pc.device()

        selected = pc.select_by_classification([2, 6])
        downsampled = selected.voxel_downsample(
            1.0, DownsampleStrategy.NEAREST_TO_CENTROID
        )
        transformed = downsampled.translate([1.0, 0.0, 0.0])
        result = PointCloud.concatenate([transformed], "strict")

        assert result.device() == expected_device
        assert result.point_count() == downsampled.point_count()
        assert result.get_attribute("classification").dtype == np.uint8
        np.testing.assert_allclose(
            transformed.get_xyz(),
            downsampled.get_xyz() + np.array([1.0, 0.0, 0.0], dtype=np.float32),
            atol=1e-5,
        )

    def test_gpu_full_pipeline_preserves_device_when_available(self):
        if not has_wgpu_device():
            pytest.skip("WGPU adapter unavailable")

        pc = self._cloud().to("gpu")
        expected_device = pc.device()
        selected = pc.select_by_classification([2, 6])
        downsampled = selected.voxel_downsample(
            1.0, DownsampleStrategy.RANDOM_SEEDED, seed=11
        )
        transformed = downsampled.transform(np.eye(4, dtype=np.float32))
        result = PointCloud.concatenate([transformed], "strict")

        assert result.device() == expected_device
        np.testing.assert_array_equal(
            downsampled.get_xyz(),
            selected.voxel_downsample(
                1.0, DownsampleStrategy.RANDOM_SEEDED, seed=11
            ).get_xyz(),
        )


class TestNeighborsNormalsOutliersRegistration:
    def test_knn_radius_octree_normals_covariance(self):
        xyz = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        pc = PointCloud.from_xyz(xyz)
        indices, distances = pc.knn(np.array([[0.0, 0.0, 0.0]], dtype=np.float32), 2)
        assert indices.shape == (1, 2)
        assert distances.shape == (1, 2)
        assert indices[0, 0] == 0
        radius_hits = pc.radius_search(
            np.array([[0.0, 0.0, 0.0]], dtype=np.float32), 1.1
        )
        assert set(radius_hits[0].tolist()) == {0, 1, 2}

        octree = pc.octree(3)
        assert set(octree.range_search([0.0, 0.0, 0.0], 1.1)) == {0, 1, 2}
        assert octree.voxel_centers().shape[1] == 3

        pc.estimate_normals(NormalSearch.knn(3))
        normals = np.column_stack(
            [pc.get_attribute("nx"), pc.get_attribute("ny"), pc.get_attribute("nz")]
        )
        np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
        pc.estimate_covariances(3)
        cov = pc.get_attribute("covariance")
        assert cov.shape == (4, 6)
        assert cov.dtype == np.float32
        assert isinstance(pc.device(), str)
        assert pc.to("cpu").point_count() == pc.point_count()

    def test_outlier_removal(self):
        xyz = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        pc = PointCloud.from_xyz(xyz)
        filtered, mask = pc.remove_radius_outlier(nb_points=1, radius=1.5)
        assert filtered.point_count() == 3
        np.testing.assert_array_equal(mask, [True, True, True, False])

        filtered, mask = pc.remove_statistical_outlier(nb_neighbors=2, std_ratio=1.0)
        assert filtered.point_count() < pc.point_count()
        assert mask.dtype == np.bool_

    def test_outlier_empty_input_errors(self):
        pc = PointCloud()
        with pytest.raises(ValueError):
            pc.remove_radius_outlier(nb_points=1, radius=1.0)
        with pytest.raises(ValueError):
            pc.remove_statistical_outlier(nb_neighbors=2, std_ratio=1.0)

    def test_statistical_outlier_removes_injected_outliers(self):
        rng = np.random.default_rng(42)
        cluster = rng.normal(0.0, 0.05, size=(300, 3)).astype(np.float32)
        outliers = np.column_stack(
            [
                np.linspace(10.0, 200.0, 20, dtype=np.float32),
                np.zeros(20, dtype=np.float32),
                np.zeros(20, dtype=np.float32),
            ]
        )
        xyz = np.vstack([cluster, outliers])
        pc = PointCloud.from_xyz(xyz)

        filtered, mask = pc.remove_statistical_outlier(nb_neighbors=8, std_ratio=1.0)

        assert mask.sum() == filtered.point_count()
        removed_outliers = (~mask[-len(outliers) :]).sum()
        assert removed_outliers >= int(len(outliers) * 0.95)

    def test_outlier_preserves_typed_attributes(self):
        xyz = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [5.0, 5.0, 5.0]],
            dtype=np.float32,
        )
        pc = PointCloud.from_xyz(xyz)
        intensity = np.array([1.0, 2.0, 3.0, 99.0], dtype=np.float32)
        classification = np.array([2, 2, 6, 7], dtype=np.uint8)
        gps_time = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
        pc.set_intensity(intensity)
        pc.set_attribute("classification", classification)
        pc.set_attribute("gps_time", gps_time)

        filtered, mask = pc.remove_radius_outlier(nb_points=1, radius=0.15)

        assert mask.sum() == filtered.point_count()
        np.testing.assert_array_equal(filtered.get_intensity(), intensity[mask])
        np.testing.assert_array_equal(
            filtered.get_attribute("classification"), classification[mask]
        )
        np.testing.assert_array_equal(
            filtered.get_attribute("gps_time"), gps_time[mask]
        )

    def test_registration_api(self):
        xyz = np.array(
            [[i * 0.37, np.sin(i * 1.91), np.cos(i * 0.73)] for i in range(40)],
            dtype=np.float32,
        )
        source = PointCloud.from_xyz(xyz)
        target = source.translate([0.02, -0.03, 0.01])
        result = registration.icp(
            source,
            target,
            0.2,
            np.eye(4, dtype=np.float32),
            registration.TransformationEstimation.point_to_point(),
            registration.ICPConvergenceCriteria(20, 1e-7, 1e-7),
        )
        assert result.fitness == 1.0
        np.testing.assert_allclose(
            result.transformation[:3, 3], [0.02, -0.03, 0.01], atol=1e-3
        )
        score = registration.evaluate(source, target, 0.2, result.transformation)
        assert score.fitness == 1.0

    def test_point_to_plane_and_gicp_require_inputs(self):
        pc = PointCloud.from_xyz(
            np.random.default_rng(0).normal(size=(10, 3)).astype(np.float32)
        )
        with pytest.raises(ValueError):
            registration.icp(
                pc,
                pc,
                1.0,
                np.eye(4, dtype=np.float32),
                registration.TransformationEstimation.point_to_plane(),
            )
        with pytest.raises(ValueError):
            registration.icp(
                pc,
                pc,
                1.0,
                np.eye(4, dtype=np.float32),
                registration.TransformationEstimation.generalized(),
            )


class TestTableIo:
    def test_laspy_point_format_10_fixture_loads_with_standard_dtypes(self, tmp_path):
        import laspy

        path = tmp_path / "laspy_pf10.las"
        header = laspy.LasHeader(version="1.4", point_format=10)
        header.scales = np.array([0.0001, 0.0001, 0.0001])
        header.offsets = np.array([1000.0, -2000.0, 50.0])
        las = laspy.LasData(header)
        las.x = np.array([1001.25, 1004.25])
        las.y = np.array([-1997.5, -1994.5])
        las.z = np.array([53.75, 56.75])
        las.intensity = np.array([65535, 42], dtype=np.uint16)
        las.red = np.array([4096, 65535], dtype=np.uint16)
        las.green = np.array([8192, 32768], dtype=np.uint16)
        las.blue = np.array([16384, 12345], dtype=np.uint16)
        las.nir = np.array([111, 222], dtype=np.uint16)
        las.scan_angle = np.array([-120, 120], dtype=np.int16)
        las.wavepacket_offset = np.array([0, 0], dtype=np.uint64)
        las.write(path)

        pc = PointCloud.from_las(str(path))

        assert pc.get_attribute("intensity").dtype == np.uint16
        assert pc.get_attribute("scan_angle").dtype == np.int16
        assert pc.get_attribute("wavepacket_offset").dtype == np.uint64
        np.testing.assert_array_equal(pc.get_attribute("intensity"), [65535, 42])
        np.testing.assert_array_equal(pc.get_attribute("red"), [4096, 65535])
        np.testing.assert_array_equal(pc.get_attribute("nir"), [111, 222])
        np.testing.assert_array_equal(pc.get_attribute("scan_angle"), [-120, 120])

    def test_las_fixture_backed_selectors_cover_standard_attributes(self, tmp_path):
        xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 3.0],
                [0.0, 0.0, 4.0],
            ],
            dtype=np.float32,
        )
        classification = np.array([2, 6, 2, 9, 6], dtype=np.uint8)
        return_number = np.array([1, 1, 2, 2, 1], dtype=np.uint8)
        intensity = np.array([0.05, 0.25, 0.5, 0.75, 0.95], dtype=np.float32)

        fixture = PointCloud.from_xyz(xyz)
        fixture.set_attribute("classification", classification)
        fixture.set_attribute("return_number", return_number)
        fixture.set_intensity(intensity)

        path = tmp_path / "selectors.las"
        fixture.to_las(str(path))
        pc = PointCloud.from_las(str(path))

        np.testing.assert_array_equal(
            pc.select_where("classification", "in", [2, 6]).get_attribute(
                "classification"
            ),
            [2, 6, 2, 6],
        )
        np.testing.assert_array_equal(
            pc.select_by_classification([6]).get_attribute("classification"),
            [6, 6],
        )
        np.testing.assert_array_equal(
            pc.select_return_number(1).get_attribute("return_number"),
            [1, 1, 1],
        )
        assert pc.select_intensity_range(0.2, 0.8).point_count() == 3
        np.testing.assert_allclose(
            pc.select_elevation_range(1.0, 3.0).get_xyz()[:, 2],
            [1.0, 2.0, 3.0],
            atol=1e-5,
        )

    def test_rfc0003_examples_reduce_las_fixture(self, tmp_path):
        classification_pipeline = load_example_function(
            "classification_aware_downsample.py", "run_pipeline"
        )
        split_grid_pipeline = load_example_function(
            "split_grid_downsample_concat.py", "run_pipeline"
        )

        rng = np.random.default_rng(21)
        xyz = rng.uniform([0.0, 0.0, -1.0], [8.0, 8.0, 2.0], size=(2_000, 3)).astype(
            np.float32
        )
        pc = PointCloud.from_xyz(xyz)
        pc.set_attribute(
            "classification",
            rng.choice(np.array([2, 3, 5, 6, 7], dtype=np.uint8), size=len(xyz)),
        )
        pc.set_intensity(rng.random(len(xyz), dtype=np.float32))

        path = tmp_path / "rfc0003_examples.las"
        pc.to_las(str(path))
        fixture = PointCloud.from_las(str(path))

        split = split_grid_pipeline(fixture)
        classified = classification_pipeline(fixture)

        assert 0 < split.point_count() < fixture.point_count()
        assert 0 < classified.point_count() < fixture.point_count()
        assert split.get_intensity().dtype == np.float32
        assert classified.get_attribute("classification").dtype == np.uint8

    def test_las_classification_selection_and_outlier_round_trip(self, tmp_path):
        xyz = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.1, 1.0, 1.0],
                [10.0, 10.0, 10.0],
            ],
            dtype=np.float32,
        )
        classification = np.array([2, 2, 6, 6, 7], dtype=np.uint8)
        return_number = np.array([1, 1, 2, 2, 1], dtype=np.uint8)
        number_of_returns = np.array([1, 1, 2, 2, 1], dtype=np.uint8)
        gps_time = np.array([100.0, 101.0, 200.0, 201.0, 999.0], dtype=np.float64)
        rgb = np.array(
            [[10, 20, 30], [11, 21, 31], [40, 50, 60], [41, 51, 61], [200, 210, 220]],
            dtype=np.uint8,
        )

        pc = PointCloud.from_xyz(xyz)
        pc.set_attribute("classification", classification)
        pc.set_attribute("return_number", return_number)
        pc.set_attribute("number_of_returns", number_of_returns)
        pc.set_attribute("gps_time", gps_time)
        pc.set_rgb(rgb[:, 0], rgb[:, 1], rgb[:, 2])

        selected = pc.select_by_classification([2, 6])
        first_path = tmp_path / "selected.las"
        selected.to_las(str(first_path))
        reloaded = PointCloud.from_las(str(first_path))

        np.testing.assert_array_equal(
            reloaded.get_attribute("classification"), classification[:4]
        )
        np.testing.assert_array_equal(
            reloaded.get_attribute("return_number"), return_number[:4]
        )
        np.testing.assert_array_equal(
            reloaded.get_attribute("number_of_returns"), number_of_returns[:4]
        )
        np.testing.assert_allclose(reloaded.get_attribute("gps_time"), gps_time[:4])
        np.testing.assert_array_equal(reloaded.get_rgb()[0], rgb[:4, 0])

        filtered, mask = reloaded.remove_radius_outlier(nb_points=1, radius=0.2)
        second_path = tmp_path / "filtered.las"
        filtered.to_las(str(second_path))
        filtered_reloaded = PointCloud.from_las(str(second_path))

        np.testing.assert_array_equal(
            filtered_reloaded.get_attribute("classification"),
            reloaded.get_attribute("classification")[mask],
        )
        np.testing.assert_array_equal(
            filtered_reloaded.get_attribute("return_number"),
            reloaded.get_attribute("return_number")[mask],
        )
        np.testing.assert_allclose(
            filtered_reloaded.get_attribute("gps_time"),
            reloaded.get_attribute("gps_time")[mask],
        )

    def test_las_custom_attributes_survive_outlier_round_trip(self, tmp_path):
        xyz = np.array(
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [5.0, 5.0, 5.0]],
            dtype=np.float32,
        )
        confidence = np.array([0.1, 0.2, 0.3, 9.9], dtype=np.float32)
        source_id = np.array([10, 11, 12, 99], dtype=np.uint16)
        flightline = np.array([100, 101, 102, 999], dtype=np.int32)

        pc = PointCloud.from_xyz(xyz)
        pc.set_attribute("confidence", confidence)
        pc.set_attribute("source_id", source_id)
        pc.set_attribute("flightline", flightline)

        first_path = tmp_path / "custom_attrs.las"
        pc.to_las(str(first_path))
        reloaded = PointCloud.from_las(str(first_path))

        np.testing.assert_array_equal(reloaded.get_attribute("confidence"), confidence)
        np.testing.assert_array_equal(reloaded.get_attribute("source_id"), source_id)
        np.testing.assert_array_equal(reloaded.get_attribute("flightline"), flightline)

        filtered, mask = reloaded.remove_radius_outlier(nb_points=1, radius=0.15)
        second_path = tmp_path / "custom_attrs_filtered.las"
        filtered.to_las(str(second_path))
        filtered_reloaded = PointCloud.from_las(str(second_path))

        np.testing.assert_array_equal(
            filtered_reloaded.get_attribute("confidence"), confidence[mask]
        )
        np.testing.assert_array_equal(
            filtered_reloaded.get_attribute("source_id"), source_id[mask]
        )
        np.testing.assert_array_equal(
            filtered_reloaded.get_attribute("flightline"), flightline[mask]
        )

    def test_las_point_format_10_python_api_round_trip_precision(self, tmp_path):
        xyz = np.array([[1.25, 2.5, 3.75], [4.25, 5.5, 6.75]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)
        pc.set_attribute("intensity", np.array([65535, 42], dtype=np.uint16))
        pc.set_attribute("red", np.array([4096, 65535], dtype=np.uint16))
        pc.set_attribute("green", np.array([8192, 32768], dtype=np.uint16))
        pc.set_attribute("blue", np.array([16384, 12345], dtype=np.uint16))
        pc.set_attribute("nir", np.array([111, 222], dtype=np.uint16))
        pc.set_attribute("scan_angle", np.array([-120, 120], dtype=np.int16))
        pc.set_attribute(
            "wavepacket_offset",
            np.array([2**32 + 1, 9_000_000_000], dtype=np.uint64),
        )

        path = tmp_path / "pf10_precision.las"
        pc.to_las(
            str(path),
            point_format=10,
            las_version="1.4",
            drop_waveform=True,
        )
        reloaded = PointCloud.from_las(str(path))

        assert reloaded.get_attribute("intensity").dtype == np.uint16
        assert reloaded.get_attribute("scan_angle").dtype == np.int16
        assert reloaded.get_attribute("wavepacket_offset").dtype == np.uint64
        np.testing.assert_array_equal(reloaded.get_attribute("intensity"), [65535, 42])
        np.testing.assert_array_equal(reloaded.get_attribute("red"), [4096, 65535])
        np.testing.assert_array_equal(reloaded.get_attribute("nir"), [111, 222])
        np.testing.assert_array_equal(reloaded.get_attribute("scan_angle"), [-120, 120])
        np.testing.assert_array_equal(
            reloaded.get_attribute("wavepacket_offset"),
            [0, 0],
        )

    def test_las_point_format_10_rejects_waveform_metadata_without_drop(self, tmp_path):
        pc = PointCloud.from_xyz(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        pc.set_attribute("wavepacket_offset", np.array([2**32 + 1], dtype=np.uint64))

        with pytest.raises(ValueError, match="waveform payload"):
            pc.to_las(str(tmp_path / "pf10_waveform.las"), point_format=10)

    def test_las_point_format_10_policy_validation_and_partial_rgb(self, tmp_path):
        pc = PointCloud.from_xyz(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        pc.set_attribute("red", np.array([4096], dtype=np.uint16))

        path = tmp_path / "pf10_partial_rgb.las"
        pc.to_las(str(path), point_format=10)
        reloaded = PointCloud.from_las(str(path))
        np.testing.assert_array_equal(reloaded.get_attribute("red"), [4096])
        np.testing.assert_array_equal(reloaded.get_attribute("green"), [0])
        np.testing.assert_array_equal(reloaded.get_attribute("blue"), [0])

        with pytest.raises(ValueError, match="requires las_version"):
            pc.to_las(
                str(tmp_path / "bad_version.las"), point_format=10, las_version="1.2"
            )
        with pytest.raises(ValueError, match="unsupported LAS point_format"):
            pc.to_las(str(tmp_path / "unsupported_format.las"), point_format=9)

    def test_las_extra_bytes_preserve_uint64_and_int16(self, tmp_path):
        pc = PointCloud.from_xyz(
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        )
        pc.set_attribute(
            "custom_offset", np.array([2**32 + 1, 2**40 + 7], dtype=np.uint64)
        )
        pc.set_attribute("custom_scan_angle", np.array([-120, 120], dtype=np.int16))

        path = tmp_path / "extra_bytes_u64_i16.las"
        pc.to_las(str(path))
        reloaded = PointCloud.from_las(str(path))

        assert reloaded.get_attribute("custom_offset").dtype == np.uint64
        assert reloaded.get_attribute("custom_scan_angle").dtype == np.int16
        np.testing.assert_array_equal(
            reloaded.get_attribute("custom_offset"), [2**32 + 1, 2**40 + 7]
        )
        np.testing.assert_array_equal(
            reloaded.get_attribute("custom_scan_angle"), [-120, 120]
        )

    def test_csv_and_parquet_round_trip(self, tmp_path):
        xyz = np.array(
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
            dtype=np.float32,
        )
        intensity = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        rgb = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
        pc = PointCloud.from_xyz(xyz)
        pc.set_intensity(intensity)
        pc.set_rgb(rgb[:, 0], rgb[:, 1], rgb[:, 2])

        csv_path = tmp_path / "points.csv"
        parquet_path = tmp_path / "points.parquet"
        auto_path = tmp_path / "points_auto.parquet"

        pc.to_csv(str(csv_path), delimiter=ord(","))
        csv_loaded = PointCloud.from_csv(str(csv_path), delimiter=ord(","))
        np.testing.assert_allclose(csv_loaded.get_xyz(), xyz)
        np.testing.assert_allclose(csv_loaded.get_intensity(), intensity)
        np.testing.assert_array_equal(csv_loaded.get_rgb()[0], rgb[:, 0])

        pc.to_parquet(str(parquet_path))
        parquet_loaded = PointCloud.from_parquet(str(parquet_path))
        np.testing.assert_allclose(parquet_loaded.get_xyz(), xyz)
        np.testing.assert_allclose(parquet_loaded.get_intensity(), intensity)
        np.testing.assert_array_equal(parquet_loaded.get_rgb()[1], rgb[:, 1])

        pc.save_to_file(str(auto_path))
        auto_loaded = PointCloud.load_from_file(str(auto_path))
        np.testing.assert_allclose(auto_loaded.get_xyz(), xyz)

    def test_downsample_legacy_aliases_removed(self):
        assert not hasattr(DownsampleStrategy, "RANDOM")
        assert not hasattr(DownsampleStrategy, "CENTROID")


class TestMemoryAndRepr:
    """内存和表示测试"""

    def test_memory_usage(self):
        """测试内存占用计算"""
        xyz = np.random.randn(100, 3).astype(np.float32)
        pc = PointCloud.from_xyz(xyz)

        memory = pc.memory_usage()
        assert memory > 0
        # 100 个 3D 点的 f32 数据: 100 * 3 * 4 bytes = 1200 bytes
        assert memory >= 1200

    def test_repr(self):
        """测试点云表示"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        repr_str = repr(pc)
        assert "PointCloud" in repr_str
        assert "1" in repr_str  # 点数


class TestEdgeCases:
    """边界场景测试"""

    def test_single_point(self):
        """测试单点云"""
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        pc = PointCloud.from_xyz(xyz)

        assert pc.point_count() == 1
        pc.set_intensity(np.array([100.0], dtype=np.float32))
        assert pc.has_intensity()

    def test_large_point_count(self):
        """测试大规模点云（可选性能测试）"""
        n = 10000
        xyz = np.random.randn(n, 3).astype(np.float32)
        pc = PointCloud.from_xyz(xyz)

        assert pc.point_count() == n

    def test_empty_downsample(self):
        """测试空点云下采样应该抛出错误"""
        pc = PointCloud()
        with pytest.raises(ValueError):
            pc.voxel_downsample(1.0, DownsampleStrategy.RANDOM_SEEDED)

    def test_zero_points_properties(self):
        """测试空点云的属性操作"""
        pc = PointCloud()

        assert not pc.has_intensity()
        assert not pc.has_rgb()
        assert pc.attribute_names() == []


class TestIntegration:
    """集成测试"""

    def test_full_workflow(self):
        """测试完整工作流"""
        # 1. 创建点云
        xyz = np.array(
            [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.1, 1.1, 1.1], [1.2, 1.2, 1.2]],
            dtype=np.float32,
        )
        pc = PointCloud.from_xyz(xyz)

        # 2. 添加属性
        pc.set_intensity(np.array([100.0, 150.0, 200.0, 250.0], dtype=np.float32))
        pc.set_rgb(
            np.array([255.0, 0.0, 0.0, 255.0], dtype=np.float32),
            np.array([0.0, 255.0, 0.0, 255.0], dtype=np.float32),
            np.array([0.0, 0.0, 255.0, 0.0], dtype=np.float32),
        )
        pc.add_attribute("confidence", np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32))

        # 3. 变换
        rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        translation = [1.0, 2.0, 3.0]
        pc_transformed = pc.rigid_transform(rotation, translation)

        # 4. 下采样
        pc_downsampled = pc_transformed.voxel_downsample(
            1.0, DownsampleStrategy.NEAREST_TO_CENTROID
        )

        # 5. 验证结果
        assert pc_downsampled.point_count() > 0
        assert pc_downsampled.has_intensity()
        assert pc_downsampled.has_rgb()
        assert "confidence" in pc_downsampled.attribute_names()

    def test_chain_operations(self):
        """测试链式操作"""
        xyz = np.arange(30, dtype=np.float32).reshape(10, 3)
        pc = PointCloud.from_xyz(xyz)

        # 变换 -> 下采样
        matrix = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        result = pc.transform(matrix).voxel_downsample(
            2.0, DownsampleStrategy.RANDOM_SEEDED
        )

        assert result.point_count() > 0
        assert result.point_count() <= pc.point_count()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
