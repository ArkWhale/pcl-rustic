# 示例代码

本页面提供完整的使用示例，展示如何在实际场景中使用 `pcl-rustic`。

## RFC 工作流示例

仓库中包含两个端到端脚本：

- `examples/split_grid_downsample_concat.py`：按 AABB 网格切片，每个格子使用不同体素大小下采样，再 `PointCloud.concatenate(..., policy="union")` 合并。
- `examples/classification_aware_downsample.py`：按 LAS classification 分组，按类别设置体素大小，可选异常点清理后合并。

## 基础示例

### 创建和可视化点云

```python
import numpy as np
from pcl_rustic import PointCloud

# 生成球形点云
n_points = 10000
theta = np.random.rand(n_points) * 2 * np.pi
phi = np.random.rand(n_points) * np.pi
r = 10.0

x = (r * np.sin(phi) * np.cos(theta)).astype(np.float32)
y = (r * np.sin(phi) * np.sin(theta)).astype(np.float32)
z = (r * np.cos(phi)).astype(np.float32)

xyz = np.column_stack([x, y, z])
pc = PointCloud.from_xyz(xyz)

# 根据高度着色
z_norm = (z - z.min()) / (z.max() - z.min())
intensity = (z_norm * 255).astype(np.float32)
pc.set_intensity(intensity)

print(f"创建了球形点云: {pc.point_count()} 个点")
```

### 多级体素下采样

```python
from pcl_rustic import DownsampleStrategy

# 原始点云
print(f"原始点数: {pc.point_count():,}")

# 逐级下采样
voxel_sizes = [0.5, 1.0, 2.0]
for voxel_size in voxel_sizes:
    pc_down = pc.voxel_downsample(
        voxel_size=voxel_size,
        strategy=DownsampleStrategy.NEAREST_TO_CENTROID
    )
    reduction = (1 - pc_down.point_count() / pc.point_count()) * 100
    print(f"体素 {voxel_size}m: {pc_down.point_count():,} 点 ({reduction:.1f}% 减少)")
```

## 文件处理

### 批量处理 LAZ 文件

```python
from pathlib import Path
from pcl_rustic import PointCloud, DownsampleStrategy

input_dir = Path("data/raw")
output_dir = Path("data/processed")
output_dir.mkdir(exist_ok=True)

# 处理目录中的所有 LAZ 文件
for laz_file in input_dir.glob("*.laz"):
    print(f"处理: {laz_file.name}")

    # 读取
    pc = PointCloud.from_las(str(laz_file))
    original_count = pc.point_count()

    # 下采样
    pc_down = pc.voxel_downsample(
        voxel_size=0.1,
        strategy=DownsampleStrategy.AVERAGE
    )

    # 保存
    output_path = output_dir / f"{laz_file.stem}_downsampled.laz"
    pc_down.to_las(str(output_path), compress=True)

    reduction = (1 - pc_down.point_count() / original_count) * 100
    print(f"  {original_count:,} → {pc_down.point_count():,} ({reduction:.1f}% 减少)")
```

### 合并多个点云

```python
def merge_point_clouds(point_clouds: list[PointCloud]) -> PointCloud:
    """合并多个点云"""
    # 收集所有坐标
    all_xyz = [pc.get_xyz() for pc in point_clouds]
    merged_xyz = np.vstack(all_xyz)

    # 创建合并后的点云
    merged_pc = PointCloud.from_xyz(merged_xyz)

    # 如果所有点云都有强度，合并强度
    if all(pc.has_intensity() for pc in point_clouds):
        all_intensity = [pc.get_intensity() for pc in point_clouds]
        merged_intensity = np.concatenate(all_intensity)
        merged_pc.set_intensity(merged_intensity)

    return merged_pc

# 使用示例
pcs = [PointCloud.from_las(f"tile_{i}.laz") for i in range(4)]
merged = merge_point_clouds(pcs)
print(f"合并后点数: {merged.point_count():,}")
```

## 坐标变换

### 点云配准

```python
def align_point_clouds(source: PointCloud, target: PointCloud) -> PointCloud:
    """简单的点云对齐（平移到相同中心）"""
    # 计算中心
    source_xyz = source.get_xyz()
    target_xyz = target.get_xyz()

    source_center = source_xyz.mean(axis=0)
    target_center = target_xyz.mean(axis=0)

    # 计算平移向量
    translation = target_center - source_center

    # 应用平移
    aligned = source.rigid_transform(np.eye(3, dtype=np.float32), translation)

    return aligned

# 使用示例
pc1 = PointCloud.from_las("scan1.laz")
pc2 = PointCloud.from_las("scan2.laz")
pc1_aligned = align_point_clouds(pc1, pc2)
```

### 坐标系转换

```python
def transform_coordinate_system(
    pc: PointCloud,
    from_system: str,
    to_system: str
) -> PointCloud:
    """坐标系转换示例（简化版）"""
    if from_system == "xyz" and to_system == "enu":
        # XYZ → ENU (东-北-上)
        rotation = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        return pc.rigid_transform(rotation, np.zeros(3, dtype=np.float32))

    elif from_system == "enu" and to_system == "xyz":
        # ENU → XYZ
        rotation = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ], dtype=np.float32)
        return pc.rigid_transform(rotation, np.zeros(3, dtype=np.float32))

    else:
        raise ValueError(f"不支持的转换: {from_system} → {to_system}")

# 使用示例
pc_xyz = PointCloud.from_las("data.laz")
pc_enu = transform_coordinate_system(pc_xyz, "xyz", "enu")
```

## 数据分析

### 点云统计信息

```python
def analyze_point_cloud(pc: PointCloud) -> dict:
    """计算点云统计信息"""
    xyz = pc.get_xyz()

    stats = {
        "点数": pc.point_count(),
        "最小边界": xyz.min(axis=0),
        "最大边界": xyz.max(axis=0),
        "中心": xyz.mean(axis=0),
        "标准差": xyz.std(axis=0),
    }

    if pc.has_intensity():
        intensity = pc.get_intensity()
        stats["强度范围"] = (intensity.min(), intensity.max())
        stats["平均强度"] = intensity.mean()

    # 计算密度
    volume = np.prod(stats["最大边界"] - stats["最小边界"])
    stats["体积"] = volume
    stats["密度 (点/m³)"] = stats["点数"] / volume

    return stats

# 使用示例
pc = PointCloud.from_las("data.laz")
stats = analyze_point_cloud(pc)

for key, value in stats.items():
    print(f"{key}: {value}")
```

### 高度分层分析

```python
def analyze_by_height(pc: PointCloud, n_layers: int = 10):
    """按高度分层分析点云"""
    xyz = pc.get_xyz()
    z = xyz[:, 2]

    z_min, z_max = z.min(), z.max()
    layer_height = (z_max - z_min) / n_layers

    print(f"高度范围: {z_min:.2f} - {z_max:.2f} m")
    print(f"层高: {layer_height:.2f} m\n")

    for i in range(n_layers):
        layer_min = z_min + i * layer_height
        layer_max = layer_min + layer_height

        mask = (z >= layer_min) & (z < layer_max)
        layer_count = mask.sum()
        percentage = (layer_count / len(z)) * 100

        print(f"层 {i+1:2d} [{layer_min:6.2f}, {layer_max:6.2f}): "
              f"{layer_count:8,} 点 ({percentage:5.2f}%)")

# 使用示例
pc = PointCloud.from_las("forest.laz")
analyze_by_height(pc, n_layers=10)
```

## 性能测试

### 体素下采样性能测试

```python
import time
from pcl_rustic import DownsampleStrategy

def benchmark_downsample(pc: PointCloud, voxel_sizes: list[float]):
    """测试不同体素大小的下采样性能"""
    print(f"输入点云: {pc.point_count():,} 点\n")

    for voxel_size in voxel_sizes:
        start_time = time.time()

        pc_down = pc.voxel_downsample(
            voxel_size=voxel_size,
            strategy=DownsampleStrategy.NEAREST_TO_CENTROID
        )

        elapsed = time.time() - start_time
        reduction = (1 - pc_down.point_count() / pc.point_count()) * 100
        throughput = pc.point_count() / elapsed / 1e6

        print(f"体素大小 {voxel_size:4.2f}m:")
        print(f"  输出点数: {pc_down.point_count():,}")
        print(f"  减少率: {reduction:.1f}%")
        print(f"  耗时: {elapsed:.2f}s")
        print(f"  吞吐量: {throughput:.2f}M 点/秒\n")

# 生成大点云测试
xyz = np.random.randn(10_000_000, 3).astype(np.float32) * 100
pc = PointCloud.from_xyz(xyz)

benchmark_downsample(pc, voxel_sizes=[0.05, 0.10, 0.15, 0.20])
```

## 实用工具

### 点云裁剪

```python
def crop_point_cloud(
    pc: PointCloud,
    min_bound: np.ndarray,
    max_bound: np.ndarray
) -> PointCloud:
    """裁剪点云到指定边界框"""
    xyz = pc.get_xyz()

    # 创建掩码
    mask = np.all((xyz >= min_bound) & (xyz <= max_bound), axis=1)

    # 裁剪坐标
    xyz_cropped = xyz[mask]
    pc_cropped = PointCloud.from_xyz(xyz_cropped)

    # 裁剪属性
    if pc.has_intensity():
        intensity = pc.get_intensity()[mask]
        pc_cropped.set_intensity(intensity)

    if pc.has_rgb():
        r, g, b = pc.rgb()
        pc_cropped.set_rgb(r[mask], g[mask], b[mask])

    return pc_cropped

# 使用示例
pc = PointCloud.from_las("large_area.laz")
min_bound = np.array([0, 0, 0], dtype=np.float32)
max_bound = np.array([100, 100, 50], dtype=np.float32)
pc_cropped = crop_point_cloud(pc, min_bound, max_bound)

print(f"裁剪: {pc.point_count():,} → {pc_cropped.point_count():,}")
```

## 下一步

- [API 文档](../api/overview.md) - 完整的 API 参考
- [性能优化](../performance/optimization.md) - 性能优化技巧
- [开发指南](../development/setup.md) - 参与项目开发
