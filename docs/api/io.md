# 文件 I/O

点云文件读写功能，支持 LAZ/LAS、CSV 和 Parquet 格式。

## API 列表

### LAZ/LAS 格式

- `PointCloud.from_las(path: str) -> PointCloud` - 从 LAS/LAZ 文件读取点云
- `PointCloud.to_las(path: str, compress: bool = False, *, point_format: int | None = None, las_version: str = "1.4", drop_waveform: bool = False) -> None` - 将点云写入 LAS/LAZ 文件

### CSV 格式

- `PointCloud.from_csv(path: str, delimiter: int = ord(','), x: str | None = None, y: str | None = None, z: str | None = None, intensity: str | None = None, ...) -> PointCloud`
- `PointCloud.to_csv(path: str, delimiter: int = ord(','), x: str | None = None, y: str | None = None, z: str | None = None, intensity: str | None = None, ...) -> None`

### Parquet 格式

- `PointCloud.from_parquet(path: str, x: str | None = None, y: str | None = None, z: str | None = None, intensity: str | None = None, ...) -> PointCloud`
- `PointCloud.to_parquet(path: str, x: str | None = None, y: str | None = None, z: str | None = None, intensity: str | None = None, ...) -> None`

### 通用接口

- `PointCloud.load_from_file(path: str, x: str | None = None, ...) -> PointCloud` - 自动检测格式并读取
- `PointCloud.save_to_file(path: str, x: str | None = None, ...) -> None` - 根据文件扩展名自动选择格式

## 使用示例

### 基本读写

```python
from pcl_rustic import PointCloud

# 读取 LAZ 文件
pc = PointCloud.from_las("input.laz")
print(f"读取了 {pc.point_count():,} 个点")

# 处理点云
pc_down = pc.voxel_downsample(0.15)

# 保存为 LAZ（压缩）
pc_down.to_las("output.laz", compress=True)

# 保存为 LAS（未压缩）
pc_down.to_las("output.las", compress=False)
```

### LAS 1.4 Point Format 10

Use `point_format=10` to explicitly write LAS 1.4 point format 10. XYZ is the
only required input; all non-XYZ LAS dimensions are optional and missing values
are written as LAS defaults.

```python
from pcl_rustic import PointCloud
import numpy as np

pc = PointCloud.from_xyz(np.array([[1.25, 2.5, 3.75]], dtype=np.float32))
pc.set_attribute("intensity", np.array([65535], dtype=np.uint16))
pc.set_attribute("red", np.array([4096], dtype=np.uint16))
pc.set_attribute("green", np.array([8192], dtype=np.uint16))
pc.set_attribute("blue", np.array([16384], dtype=np.uint16))
pc.set_attribute("nir", np.array([1234], dtype=np.uint16))
pc.set_attribute("scan_angle", np.array([-120], dtype=np.int16))

pc.to_las("pf10.las", point_format=10, las_version="1.4")
```

Point format 10 standard attributes use these preferred dtypes: `uint16` for
`intensity`, `red`, `green`, `blue`, and `nir`; `int16` for raw LAS 1.4
`scan_angle`; `uint64` for `wavepacket_offset`; `uint32` for
`wavepacket_size`; `uint8` for packed return/classification fields; `float64`
for `gps_time`; and `float32` for waveform location/vector fields. Legacy
`float32` intensity and `uint8` RGB are still accepted on write.

Non-default waveform metadata is rejected unless `drop_waveform=True`, because
waveform descriptor and payload preservation is not implemented yet. Imported
unmodified LAS coordinates preserve raw integer XYZ records and scale/offset
metadata through row-preserving operations such as selection; coordinate
mutations such as translate, rotate, scale, and transform drop that metadata.

### CSV 读写

```python
from pcl_rustic import PointCloud

# 从 CSV 读取
pc = PointCloud.from_csv(
    "input.csv",
    delimiter=ord(","),
    x="x",
    y="y",
    z="z",
    intensity="intensity",
)

# 写入 CSV
pc.to_csv(
    "output.csv",
    delimiter=ord(","),
    x="x",
    y="y",
    z="z",
    intensity="intensity",
)
```

### Parquet 读写

```python
from pcl_rustic import PointCloud

# 从 Parquet 读取
pc = PointCloud.from_parquet(
    "input.parquet",
    x="x",
    y="y",
    z="z",
    intensity="intensity",
)

# 写入 Parquet
pc.to_parquet(
    "output.parquet",
    x="x",
    y="y",
    z="z",
    intensity="intensity",
)
```

### 自动格式检测

```python
from pcl_rustic import PointCloud

# 根据文件扩展名自动检测格式
pc = PointCloud.load_from_file(
    "data.laz",
    x="x",
    y="y",
    z="z",
)

# 自动根据扩展名选择格式
pc.save_to_file(
    "output.parquet",
    x="x",
    y="y",
    z="z",
)
```

### 文件格式对比

```python
# LAS：未压缩，文件较大
pc.to_las("output.las", compress=False)

# LAZ：压缩格式，文件较小
pc.to_las("output.laz", compress=True)

# CSV：文本格式，易于处理，文件最大
pc.to_csv("output.csv", delimiter=ord(","), x="x", y="y", z="z")

# Parquet：列式存储，高效处理，适合大规模数据
pc.to_parquet("output.parquet", x="x", y="y", z="z")
```

## 文件格式选择指南

| 格式 | 压缩 | 文件大小 | 读写速度 | 适用场景 |
|-----|------|---------|---------|---------|
| LAS | ❌ | 很大 | 快 | 行业标准，兼容性好 |
| LAZ | ✅ | 小 | 中等 | 存储、传输 |
| CSV | ❌ | 很大 | 慢 | 数据交换、人工检查 |
| Parquet | ✅ | 小 | 快 | 大规模数据、分析 |

## 属性处理

### 保存标准属性

```python
from pcl_rustic import PointCloud
import numpy as np

# 创建点云
xyz = np.random.randn(1000, 3).astype(np.float32)
pc = PointCloud.from_xyz(xyz)

# 添加强度
intensity = np.random.randint(0, 65535, 1000, dtype=np.uint16)
pc.set_intensity(intensity.astype(np.float32) / 65535)

# 保存到 Parquet（写出 XYZ、intensity 和 RGB 标准属性）
pc.to_parquet("output.parquet", x="x", y="y", z="z", intensity="intensity")
```

CSV/Parquet helpers currently write XYZ plus the standard `intensity` and RGB
attributes. Full custom-attribute table export is tracked separately from the
RFC-0002 typed-attribute storage work.

## 相关链接

- [PointCloud](pointcloud.md) - 点云核心类
- [下采样](downsample.md) - 点云下采样
- [变换](transform.md) - 坐标变换
