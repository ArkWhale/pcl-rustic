"""
pcl_rustic 点云库的 pytest 测试用例

覆盖核心功能、边界场景、异常场景
"""

import math

import numpy as np
import pytest

from pcl_rustic import DownsampleStrategy, NormalSearch, PointCloud, registration


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
