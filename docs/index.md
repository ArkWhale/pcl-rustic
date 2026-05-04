# PCL Rustic

**高性能 Python 点云处理库**

基于 Rust + PyO3 的高性能 Python 点云处理库，使用 [Burn](https://github.com/tracel-ai/burn) 张量框架实现批量运算，支持 CPU/GPU 加速。

## ✨ 核心特性

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } __高性能批量运算__

    ---

    基于 Burn 张量框架，支持 CPU/GPU 加速，10M 点云体素下采样仅需 ~7 秒

-   :material-link-variant:{ .lg .middle } __NumPy 互通__

    ---

    与 NumPy 数组无缝转换，支持多种 dtype，保留 typed attributes

-   :material-file-multiple:{ .lg .middle } __多格式 I/O__

    ---

    支持 LAZ/LAS/Parquet/CSV 格式读写

-   :material-shield-check:{ .lg .middle } __类型安全__

    ---

    完整的类型注解和 `.pyi` 存根文件，支持 IDE 自动补全

</div>

## 🚀 快速开始

### 安装

=== "uv"

    ```bash
    uv pip install pcl-rustic
    ```

=== "pip"

    ```bash
    pip install pcl-rustic
    ```

=== "从源码"

    ```bash
    git clone https://github.com/ArkWhale/pcl-rustic.git
    cd pcl-rustic
    uv build
    ```

### 第一个示例

```python
import numpy as np
from pcl_rustic import PointCloud, DownsampleStrategy

# 使用 NumPy 数组创建点云
xyz = np.random.randn(10000, 3).astype(np.float32) * 100
pc = PointCloud.from_xyz(xyz)

# 添加属性
intensity = np.random.rand(10000).astype(np.float32) * 255
pc.set_intensity(intensity)

# 体素下采样
pc_downsampled = pc.voxel_downsample(
    voxel_size=0.15,
    strategy=DownsampleStrategy.NEAREST_TO_CENTROID
)

print(f"原始点数: {pc.point_count():,}")
print(f"下采样后: {pc_downsampled.point_count():,}")
```

## 📊 性能表现

基准测试结果（MacBook M1）：

| 输入点数 | Voxel | 输出点数 | 减少率 | 耗时 | 吞吐量 |
|---------|-------|---------|-------|-----|--------|
| 10M | 0.06 | 8.8M | 11.6% | 7.70s | 1.3M/s |
| 10M | 0.15 | 7.9M | 21.3% | 7.13s | 1.4M/s |
| 10M | 0.20 | 7.0M | 29.5% | 6.45s | 1.5M/s |

查看 [性能基准测试](performance/benchmarks.md) 了解更多详情。

## 📖 文档导航

<div class="grid cards" markdown>

-   [快速开始](getting-started/installation.md)

    安装指南和基本使用示例

-   [API 文档](api/overview.md)

    完整的 API 参考文档

-   [性能](performance/benchmarks.md)

    基准测试结果和优化建议

-   [开发](development/setup.md)

    开发环境设置和贡献指南

</div>

## 🤝 社区

- [GitHub 仓库](https://github.com/ArkWhale/pcl-rustic)
- [问题反馈](https://github.com/ArkWhale/pcl-rustic/issues)
- [Pull Requests](https://github.com/ArkWhale/pcl-rustic/pulls)

## 📄 许可证

本项目采用 MIT 许可证 - 查看项目根目录中的 LICENSE 文件了解详情。
