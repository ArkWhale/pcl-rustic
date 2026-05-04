#!/usr/bin/env python3
"""
PCL Rustic 使用示例

演示完整的点云处理工作流
"""

import math

import numpy as np

from pcl_rustic import DownsampleStrategy, PointCloud


def example_basic_operations():
    """基本操作示例"""
    print("\n=== 基本操作 ===")

    # 创建点云
    xyz = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    pc = PointCloud.from_xyz(np.asarray(xyz, dtype=np.float32))
    print(f"创建点云：{pc}")
    print(f"点数：{pc.point_count()}")
    print(f"XYZ坐标：{pc.get_xyz()}")


def example_properties():
    """属性操作示例"""
    print("\n=== 属性操作 ===")

    xyz = [[i, i, i] for i in range(5)]
    pc = PointCloud.from_xyz(np.asarray(xyz, dtype=np.float32))

    # 设置intensity
    intensity = np.asarray([100.0, 110.0, 120.0, 130.0, 140.0], dtype=np.float32)
    pc.set_intensity(intensity)
    print(f"设置intensity：{pc.get_intensity()}")

    # 设置RGB
    rgb = [
        [255, 0, 0],  # 红
        [0, 255, 0],  # 绿
        [0, 0, 255],  # 蓝
        [255, 255, 0],  # 黄
        [255, 0, 255],  # 紫
    ]
    rgb = np.asarray(rgb, dtype=np.uint8)
    pc.set_rgb(rgb[:, 0], rgb[:, 1], rgb[:, 2])
    print(f"设置RGB：{pc.get_rgb()}")

    # 添加自定义属性
    pc.add_attribute(
        "confidence", np.asarray([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    )
    pc.add_attribute("category", np.asarray([1, 1, 2, 2, 3], dtype=np.uint8))
    print(f"属性列表：{pc.attribute_names()}")
    print(f"confidence：{pc.get_attribute('confidence')}")


def example_transform():
    """坐标变换示例"""
    print("\n=== 坐标变换 ===")

    # 创建简单点云
    xyz = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    pc = PointCloud.from_xyz(np.asarray(xyz, dtype=np.float32))

    # 示例1：缩放变换
    print("\n缩放变换（2倍）：")
    scale_matrix = [
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ]
    pc_scaled = pc.transform(scale_matrix)
    print(f"原始：{pc.get_xyz()[0]}")
    print(f"缩放后：{pc_scaled.get_xyz()[0]}")

    # 示例2：旋转变换（绕Z轴45度）
    print("\n旋转变换（绕Z轴45度）：")
    angle = math.pi / 4  # 45度
    rotation_matrix = [
        [math.cos(angle), -math.sin(angle), 0.0],
        [math.sin(angle), math.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]
    pc_rotated = pc.transform(rotation_matrix)
    print(f"旋转后：{[round(v, 2) for v in pc_rotated.get_xyz()[0]]}")

    # 示例3：刚体变换（旋转+平移）
    print("\n刚体变换（恒等旋转+平移）：")
    identity_rotation = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    translation = [10.0, 20.0, 30.0]
    pc_rigid = pc.rigid_transform(identity_rotation, translation)
    print(f"平移后：{pc_rigid.get_xyz()[0]}")


def example_downsample():
    """下采样示例"""
    print("\n=== 体素下采样 ===")

    # 创建密集点云（一个体素内有多个点）
    xyz = [
        # 体素1：围绕(0,0,0)
        [0.1, 0.1, 0.1],
        [0.2, 0.1, 0.1],
        [0.1, 0.2, 0.1],
        [0.2, 0.2, 0.2],
        # 体素2：围绕(1,1,1)
        [1.1, 1.1, 1.1],
        [1.2, 1.1, 1.1],
        [1.1, 1.2, 1.1],
        [1.2, 1.2, 1.2],
        # 体素3：围绕(2,2,2)
        [2.1, 2.1, 2.1],
        [2.2, 2.1, 2.1],
    ]
    pc = PointCloud.from_xyz(np.asarray(xyz, dtype=np.float32))
    print(f"原始点数：{pc.point_count()}")

    # 随机采样
    pc_random = pc.voxel_downsample(1.0, DownsampleStrategy.RANDOM_SEEDED, seed=42)
    print(f"随机采样后：{pc_random.point_count()}个点")

    # 重心采样
    pc_centroid = pc.voxel_downsample(1.0, DownsampleStrategy.NEAREST_TO_CENTROID)
    print(f"重心采样后：{pc_centroid.point_count()}个点")


def example_complete_workflow():
    """完整工作流示例"""
    print("\n=== 完整工作流 ===")

    # 1. 创建点云
    xyz = [[i * 0.1, i * 0.1, i * 0.1] for i in range(100)]
    pc = PointCloud.from_xyz(np.asarray(xyz, dtype=np.float32))
    print(f"1. 创建点云：{pc.point_count()}个点")

    # 2. 添加属性
    intensity = np.asarray([i * 1.0 for i in range(100)], dtype=np.float32)
    pc.set_intensity(intensity)
    pc.add_attribute(
        "height", np.asarray([h * 0.5 for h in range(100)], dtype=np.float32)
    )
    print("2. 添加属性：intensity, height")

    # 3. 应用变换
    matrix = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    pc = pc.transform(matrix)
    print("3. 应用变换")

    # 4. 下采样
    pc_downsampled = pc.voxel_downsample(0.5, DownsampleStrategy.NEAREST_TO_CENTROID)
    print(f"4. 下采样：{pc.point_count()}个点 -> {pc_downsampled.point_count()}个点")

    # 5. 查看结果
    print("5. 结果统计：")
    print(f"   - 点数：{pc_downsampled.point_count()}")
    print(f"   - 有intensity：{pc_downsampled.has_intensity()}")
    print(f"   - 属性：{pc_downsampled.attribute_names()}")
    print(f"   - 内存占用：{pc_downsampled.memory_usage()} 字节")


def example_parquet_io():
    """Parquet 读写示例"""
    print("\n=== Parquet IO ===")

    xyz = np.asarray([[i * 0.1, i * 0.2, i * 0.3] for i in range(10)], dtype=np.float32)
    pc = PointCloud.from_xyz(xyz)
    pc.set_intensity(np.asarray([float(i) for i in range(10)], dtype=np.float32))
    rgb = np.asarray(
        [[i * 10 % 256, i * 20 % 256, i * 30 % 256] for i in range(10)],
        dtype=np.uint8,
    )
    pc.set_rgb(rgb[:, 0], rgb[:, 1], rgb[:, 2])

    path = "./tmp_points.parquet"
    pc.to_parquet(path)
    pc_loaded = PointCloud.from_parquet(path)

    print(f"写入Parquet: {path}")
    print(f"读取Parquet点数: {pc_loaded.point_count()}")


def example_memory_efficiency():
    """内存效率示例"""
    print("\n=== 内存效率 ===")

    # 只有XYZ
    pc1 = PointCloud.from_xyz(
        np.asarray([[i, i, i] for i in range(1000)], dtype=np.float32)
    )
    mem1 = pc1.memory_usage()
    print(f"只有XYZ：{mem1} 字节")

    # XYZ + intensity
    pc2 = PointCloud.from_xyz(
        np.asarray([[i, i, i] for i in range(1000)], dtype=np.float32)
    )
    pc2.set_intensity(np.asarray([float(i) for i in range(1000)], dtype=np.float32))
    mem2 = pc2.memory_usage()
    print(f"XYZ + intensity：{mem2} 字节（增加{mem2 - mem1}字节）")

    # XYZ + intensity + RGB
    pc3 = PointCloud.from_xyz(
        np.asarray([[i, i, i] for i in range(1000)], dtype=np.float32)
    )
    pc3.set_intensity(np.asarray([float(i) for i in range(1000)], dtype=np.float32))
    rgb = np.asarray([[i % 256, i % 256, i % 256] for i in range(1000)], dtype=np.uint8)
    pc3.set_rgb(rgb[:, 0], rgb[:, 1], rgb[:, 2])
    mem3 = pc3.memory_usage()
    print(f"XYZ + intensity + RGB：{mem3} 字节（增加{mem3 - mem2}字节）")


if __name__ == "__main__":
    print("PCL Rustic 使用示例")
    print("=" * 50)

    try:
        example_basic_operations()
        example_properties()
        example_transform()
        example_downsample()
        example_complete_workflow()
        example_parquet_io()
        example_memory_efficiency()

        print("\n" + "=" * 50)
        print("所有示例完成！")
    except Exception as e:
        print(f"\n错误：{e}")
        import traceback

        traceback.print_exc()
