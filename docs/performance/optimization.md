# 性能优化指南

本指南提供深入的性能优化技巧和最佳实践。

## 数据类型优化

### 使用 float32

`pcl-rustic` 只支持 `float32`，这是经过权衡的设计选择：

**优势**：
- 内存占用减半（相比 float64）
- 缓存命中率更高
- SIMD 指令利用率更好
- 点云处理通常不需要 float64 精度

```python
# ✅ 推荐：直接使用 float32
xyz = np.random.randn(1000000, 3).astype(np.float32)
pc = PointCloud.from_xyz(xyz)

# ❌ 避免：使用 float64 后转换
xyz = np.random.randn(1000000, 3)  # float64
xyz = xyz.astype(np.float32)  # 额外的转换开销
```

### NumPy 数组布局

使用 C-连续数组以获得最佳性能：

```python
# ✅ 好：C-连续数组
xyz = np.ascontiguousarray(xyz)
pc = PointCloud.from_xyz(xyz)

# ❌ 差：非连续数组
xyz = xyz[::2]  # 步长切片导致非连续
pc = PointCloud.from_xyz(xyz)  # 需要内部复制
```

检查数组是否连续：

```python
if not xyz.flags['C_CONTIGUOUS']:
    xyz = np.ascontiguousarray(xyz)
```

## 体素下采样优化

### 选择合适的体素大小

体素大小应该根据点云密度和应用需求选择：

```python
def calculate_optimal_voxel_size(pc: PointCloud, target_ratio: float = 0.5):
    """计算最优体素大小以达到目标减少率"""
    xyz = pc.get_xyz()
    
    # 计算边界框
    min_bound = xyz.min(axis=0)
    max_bound = xyz.max(axis=0)
    extent = max_bound - min_bound
    
    # 估算体积和密度
    volume = np.prod(extent)
    density = pc.point_count() / volume
    
    # 计算目标体素大小
    target_points = pc.point_count() * target_ratio
    target_voxel_volume = volume / target_points
    voxel_size = target_voxel_volume ** (1/3)
    
    return voxel_size

# 使用
pc = read_laz("data.laz")
voxel_size = calculate_optimal_voxel_size(pc, target_ratio=0.3)
pc_down = pc.voxel_downsample(voxel_size)
```

### 策略选择

根据应用场景选择降采样策略：

| 场景 | 策略 | 原因 |
|------|------|------|
| 可视化 | `RANDOM_SEEDED` | 速度快且可复现 |
| 配准/重建 | `NEAREST_TO_CENTROID` | 保留真实点坐标和属性 |
| 聚合分析 | `AVERAGE` | 输出体素均值，整数属性取众数 |

```python
# 快速预览
pc_preview = pc.voxel_downsample(0.2, DownsampleStrategy.RANDOM_SEEDED)

# 精确处理
pc_precise = pc.voxel_downsample(0.1, DownsampleStrategy.NEAREST_TO_CENTROID)
```

## 内存管理

### 分块处理大文件

对于超大点云（>1 亿点），考虑分块处理：

```python
def process_large_point_cloud(file_path: str, chunk_size: int = 10_000_000):
    """分块处理大点云"""
    # 伪代码：实际需要支持流式读取的格式
    results = []
    
    for chunk_xyz in read_chunks(file_path, chunk_size):
        pc_chunk = PointCloud.from_xyz(chunk_xyz)
        pc_down = pc_chunk.voxel_downsample(0.15)
        results.append(pc_down.get_xyz())
        
        # 显式释放内存
        del pc_chunk
    
    # 合并结果
    merged_xyz = np.vstack(results)
    return PointCloud.from_xyz(merged_xyz)
```

### 及时释放中间结果

```python
# ✅ 好：及时释放
pc = read_laz("large.laz")
pc_down = pc.voxel_downsample(0.15)
del pc  # 释放原始点云
write_laz(pc_down, "output.laz")

# ❌ 差：保留多个副本
pc = read_laz("large.laz")
pc1 = pc.voxel_downsample(0.10)
pc2 = pc.voxel_downsample(0.15)
pc3 = pc.voxel_downsample(0.20)
# 内存占用是单个副本的 4 倍
```

### 监控内存使用

```python
import psutil
import os

def get_memory_usage():
    """获取当前进程内存使用（MB）"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

# 使用
mem_before = get_memory_usage()
pc = read_laz("large.laz")
mem_after = get_memory_usage()
print(f"加载点云使用: {mem_after - mem_before:.1f} MB")
```

## 并行处理

### 使用 multiprocessing

对于多个独立的点云文件，使用多进程并行处理：

```python
from multiprocessing import Pool
from pathlib import Path

def process_single_file(file_path: str) -> tuple:
    """处理单个文件"""
    pc = read_laz(file_path)
    pc_down = pc.voxel_downsample(0.15)
    
    output_path = file_path.replace(".laz", "_downsampled.laz")
    write_laz(pc_down, output_path)
    
    return (file_path, pc.point_count(), pc_down.point_count())

def batch_process(input_dir: str, n_workers: int = 4):
    """批量并行处理"""
    files = list(Path(input_dir).glob("*.laz"))
    
    with Pool(n_workers) as pool:
        results = pool.map(process_single_file, [str(f) for f in files])
    
    # 汇总结果
    for file_path, original, downsampled in results:
        reduction = (1 - downsampled / original) * 100
        print(f"{Path(file_path).name}: {original:,} → {downsampled:,} ({reduction:.1f}%)")

# 使用
batch_process("data/raw", n_workers=8)
```

### 避免过度并行

```python
import os

# ✅ 好：根据 CPU 核心数选择
n_workers = min(os.cpu_count(), len(files))

# ❌ 差：过多进程导致资源竞争
n_workers = 32  # 在 8 核 CPU 上
```

## I/O 优化

### 批量读取

```python
# ✅ 好：批量读取
files = list(Path("data").glob("*.laz"))
point_clouds = [read_laz(str(f)) for f in files]

# ❌ 差：多次小规模读取
for i in range(100):
    pc = read_laz(f"data/tile_{i}.laz")
    # 处理...
```

### 使用 LAZ 而非 LAS

LAZ 文件通常比 LAS 小 7-10 倍，读取速度相似：

```python
# ✅ 推荐：使用压缩格式
write_laz(pc, "output.laz")  # ~100 MB

# ❌ 不推荐：未压缩
write_las(pc, "output.las")  # ~700 MB
```

## 算法优化

### 空间索引

对于多次查询，考虑构建空间索引：

```python
from scipy.spatial import cKDTree

# 构建 KD 树
xyz = pc.get_xyz()
tree = cKDTree(xyz)

# 半径搜索
indices = tree.query_ball_point([0, 0, 0], r=10.0)
nearby_points = xyz[indices]

# K 近邻搜索
distances, indices = tree.query([0, 0, 0], k=100)
```

### 预计算常用属性

```python
class CachedPointCloud:
    """带缓存的点云类"""
    def __init__(self, pc: PointCloud):
        self.pc = pc
        self._xyz = None
        self._center = None
        self._bounds = None
    
    @property
    def xyz(self):
        if self._xyz is None:
            self._xyz = self.pc.get_xyz()
        return self._xyz
    
    @property
    def center(self):
        if self._center is None:
            self._center = self.xyz.mean(axis=0)
        return self._center
    
    @property
    def bounds(self):
        if self._bounds is None:
            self._bounds = (self.xyz.min(axis=0), self.xyz.max(axis=0))
        return self._bounds
```

## 性能分析

### 使用 cProfile

```python
import cProfile
import pstats

def profile_function():
    pc = read_laz("data.laz")
    pc_down = pc.voxel_downsample(0.15)
    write_laz(pc_down, "output.laz")

# 运行分析
cProfile.run('profile_function()', 'profile_stats')

# 查看结果
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### 使用 line_profiler

```python
from line_profiler import LineProfiler

@profile
def process_point_cloud(file_path: str):
    pc = read_laz(file_path)  # 行级分析
    pc_down = pc.voxel_downsample(0.15)
    write_laz(pc_down, "output.laz")
    return pc_down

# 运行
lp = LineProfiler()
lp.add_function(process_point_cloud)
lp.run('process_point_cloud("data.laz")')
lp.print_stats()
```

### 使用 memory_profiler

```python
from memory_profiler import profile

@profile
def memory_intensive_operation():
    pc = read_laz("large.laz")
    pc_down = pc.voxel_downsample(0.15)
    return pc_down

# 运行
memory_intensive_operation()
```

## 常见性能陷阱

### 1. 重复转换

```python
# ❌ 差：重复转换
for i in range(100):
    xyz_i = pc.get_xyz()  # 每次都分配新数组
    # ...

# ✅ 好：缓存结果
xyz = pc.get_xyz()
for i in range(100):
    # 使用缓存的 xyz
    # ...
```

### 2. 不必要的拷贝

```python
# ❌ 差：创建副本
xyz_copy = xyz.copy()
pc = PointCloud.from_xyz(xyz_copy)

# ✅ 好：直接使用
pc = PointCloud.from_xyz(xyz)
```

### 3. 过小的体素

```python
# ❌ 差：体素过小导致输出点数过多
pc_down = pc.voxel_downsample(0.001)  # 几乎没有减少

# ✅ 好：选择合理的体素大小
pc_down = pc.voxel_downsample(0.15)  # 30% 减少
```

### 4. 忽略数据类型

```python
# ❌ 差：使用 float64
xyz = np.random.randn(1000000, 3)  # float64
pc = PointCloud.from_xyz(xyz.astype(np.float32))  # 额外转换

# ✅ 好：直接使用 float32
xyz = np.random.randn(1000000, 3).astype(np.float32)
pc = PointCloud.from_xyz(xyz)
```

## 性能检查清单

在部署到生产环境前，检查以下项目：

- [ ] 使用 `float32` 而非 `float64`
- [ ] 数组是 C-连续的
- [ ] 选择了合适的体素大小
- [ ] 选择了合适的降采样策略
- [ ] 及时释放大对象
- [ ] 使用 LAZ 而非 LAS
- [ ] 考虑了并行处理
- [ ] 缓存了常用属性
- [ ] 进行了性能分析
- [ ] 监控了内存使用

## 性能目标

根据我们的基准测试，以下是合理的性能目标：

| 操作 | 目标性能 |
|------|---------|
| 体素下采样 (10M) | < 10 秒 |
| 体素下采样 (50M) | < 50 秒 |
| 读取 LAZ (1GB) | < 30 秒 |
| 写入 LAZ (1GB) | < 40 秒 |
| 坐标变换 (10M) | < 1 秒 |

如果你的性能低于这些目标，请参考本指南进行优化。

## 下一步

- [基准测试](benchmarks.md) - 查看详细的性能数据
- [API 文档](../api/overview.md) - 了解 API 细节
- [开发指南](../development/contributing.md) - 参与性能优化
