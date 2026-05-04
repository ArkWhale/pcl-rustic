# 基本使用

本指南介绍 `pcl-rustic` 的核心功能和常见用法。

## 创建点云

### 从 NumPy 数组

最常用的方式是从 NumPy 数组创建点云：

```python
import numpy as np
from pcl_rustic import PointCloud

# 创建随机 XYZ 数据（必须是 float32）
xyz = np.random.randn(10000, 3).astype(np.float32) * 100

# 创建点云
pc = PointCloud.from_xyz(xyz)
print(f"点数: {pc.point_count()}")
```

!!! warning "数据类型要求"
    所有数组必须是 `dtype=float32`。如果数据是其他类型，使用 `.astype(np.float32)` 转换。

### 添加属性

点云支持多种属性：

```python
# 强度值 [0, 255]
intensity = np.random.rand(10000).astype(np.float32) * 255
pc.set_intensity(intensity)

# RGB 颜色（3 个独立通道）
r = np.random.rand(10000).astype(np.float32) * 255
g = np.random.rand(10000).astype(np.float32) * 255
b = np.random.rand(10000).astype(np.float32) * 255
pc.set_rgb(r, g, b)

# 自定义维度
dimension1 = np.random.randn(10000).astype(np.float32)
pc.set_dimension_1(dimension1)
```

## 读取属性

使用对应的 getter 方法获取属性数组：

```python
# 获取坐标
xyz = pc.get_xyz()  # shape: (N, 3)

# 获取单个属性
    intensity = pc.get_intensity()  # shape: (N,)
    r, g, b = pc.get_rgb()  # 3 个 shape: (N,) 的数组

# 检查属性是否存在
if pc.has_intensity():
    print("点云包含强度信息")
```

## 体素下采样

体素下采样是最常用的点云降采样方法：

```python
from pcl_rustic import DownsampleStrategy

# 选择最接近体素质心的原始点
pc_down = pc.voxel_downsample(
    voxel_size=0.15,
    strategy=DownsampleStrategy.NEAREST_TO_CENTROID
)

# 输出体素平均点
pc_down = pc.voxel_downsample(
    voxel_size=0.15,
    strategy=DownsampleStrategy.AVERAGE
)

# 使用随机采样
pc_down = pc.voxel_downsample(
    voxel_size=0.15,
    strategy=DownsampleStrategy.RANDOM_SEEDED
)

print(f"原始: {pc.point_count()}, 下采样: {pc_down.point_count()}")
```

### 降采样策略

| 策略 | 描述 | 性能 | 适用场景 |
|------|------|------|----------|
| `NEAREST_TO_CENTROID` | 选择最接近体素质心的原始点 | ⭐⭐⭐ | 通用，保持原始点属性 |
| `AVERAGE` | 输出体素内平均点，整数属性取众数 | ⭐⭐ | 需要 Open3D 风格平均语义 |
| `RANDOM_SEEDED` | 按 seed 随机选择一个点 | ⭐⭐⭐⭐⭐ | 可复现快速预览 |

## 坐标变换

### 平移

```python
# 平移向量 (x, y, z)
translation = np.array([10.0, 20.0, 30.0], dtype=np.float32)
pc_translated = pc.rigid_transform(np.eye(3, dtype=np.float32), translation)
```

### 旋转

```python
# 3x3 旋转矩阵
rotation = np.array([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
], dtype=np.float32)

pc_rotated = pc.rigid_transform(rotation, np.zeros(3, dtype=np.float32))
```

### 仿射变换

```python
# 4x4 变换矩阵
transform = np.eye(4, dtype=np.float32)
transform[:3, :3] = rotation  # 旋转部分
transform[:3, 3] = translation  # 平移部分

pc_transformed = pc.transform(transform)
```

## 文件 I/O

### 读取点云

```python
from pcl_rustic import PointCloud

# 读取 LAZ 文件
pc = PointCloud.from_las("data/sample.laz")

# 读取 LAS 文件
pc = PointCloud.from_las("data/sample.las")

print(f"读取了 {pc.point_count()} 个点")
```

### 写入点云

```python
from pcl_rustic import PointCloud

# 写入 LAZ（压缩）
pc.to_las("output/result.laz", compress=True)

# 写入 LAS（未压缩）
pc.to_las("output/result.las", compress=False)
```

## 点云信息

```python
# 点数
count = pc.point_count()

# 边界框
xyz = pc.get_xyz()
min_bound = xyz.min(axis=0)
max_bound = xyz.max(axis=0)
center = (min_bound + max_bound) / 2

print(f"点数: {count:,}")
print(f"范围: {min_bound} - {max_bound}")
print(f"中心: {center}")
```

## 性能提示

!!! tip "优化建议"
    1. **数据类型**: 始终使用 `float32` 而不是 `float64`
    2. **批量操作**: 一次性设置所有属性，避免多次调用
    3. **体素大小**: 选择合适的体素大小，过小会导致处理时间增加
    4. **内存管理**: 对于大点云，及时删除不需要的中间结果

```python
# ✅ 推荐：批量设置
xyz = np.random.randn(1000000, 3).astype(np.float32)
pc = PointCloud.from_xyz(xyz)
pc.set_intensity(intensity)
pc.set_rgb(r, g, b)

# ❌ 不推荐：多次小规模操作
for i in range(1000):
    xyz_chunk = np.random.randn(1000, 3).astype(np.float32)
    pc_chunk = PointCloud.from_xyz(xyz_chunk)
```

## 下一步

- [更多示例](examples.md) - 查看完整的应用示例
- [API 文档](../api/overview.md) - 深入了解所有 API
- [性能基准](../performance/benchmarks.md) - 了解性能表现
