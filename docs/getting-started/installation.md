# 安装指南

## 系统要求

- **Python**: 3.9 或更高版本
- **Rust**: 1.70+ (仅从源码安装时需要)
- **操作系统**: Linux, macOS, Windows

## 使用 pip 安装

推荐使用 [uv](https://github.com/astral-sh/uv) 作为快速的包管理器：

```bash
uv pip install pcl-rustic
```

或使用标准的 pip：

```bash
pip install pcl-rustic
```

## 从源码安装

### 1. 安装 Rust 工具链

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. 克隆仓库

```bash
git clone https://github.com/ArkWhale/pcl-rustic.git
cd pcl-rustic
```

### 3. 使用 justfile 构建

如果安装了 [just](https://github.com/casey/just):

```bash
# 安装依赖和预提交钩子
just install

# 开发模式构建
just dev

# 生产模式构建
just build

# 构建 wheel
just wheel
```

### 4. 或使用 maturin 手动构建

```bash
# 安装 maturin
pip install maturin

# 开发模式（可编辑安装）
maturin develop --release

# 构建 wheel
maturin build --release
```

## 验证安装

```python
import pcl_rustic

# 检查版本
print(pcl_rustic.__version__)

# 测试基本功能
import numpy as np
from pcl_rustic import PointCloud

xyz = np.random.randn(100, 3).astype(np.float32)
pc = PointCloud.from_xyz(xyz)
print(f"创建了包含 {pc.point_count()} 个点的点云")
```

## 可选依赖

### 开发工具

```bash
pip install pcl-rustic[dev]
```

包含：

- `pytest` - 测试框架
- `loguru` - 日志记录
- `ruff` - 代码格式化和检查
- `pre-commit` - 预提交钩子

### 文档工具

```bash
pip install pcl-rustic[docs]
```

包含：

- `mkdocs-material` - 文档主题
- `mkdocs-git-revision-date-localized-plugin` - Git 修订日期插件

## 故障排除

### macOS: 找不到 libtorch

如果遇到 libtorch 相关错误，确保安装了 libtorch：

```bash
# 使用 Homebrew
brew install libtorch
```

或设置 `LIBTORCH` 环境变量：

```bash
export LIBTORCH=/path/to/libtorch
```

### Windows: MSVC 工具链问题

确保安装了 Visual Studio Build Tools:

1. 下载 [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. 选择 "Desktop development with C++" 工作负载
3. 重启终端后重试

### Linux: GPU 支持

使用 CUDA 后端需要安装 NVIDIA 驱动和 CUDA 工具包：

```bash
# Ubuntu/Debian
sudo apt install nvidia-cuda-toolkit

# Fedora
sudo dnf install cuda
```

## 下一步

- [基本使用](basic-usage.md) - 学习基本 API
- [示例代码](examples.md) - 查看更多示例
- [API 文档](../api/overview.md) - 完整 API 参考
