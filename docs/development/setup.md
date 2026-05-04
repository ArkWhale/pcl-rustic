# 开发环境设置

本页面介绍如何设置开发环境以贡献代码到 pcl-rustic。

## 系统要求

- **Python**: 3.9 或更高版本
- **Rust**: 1.70+
- **Just**: 命令运行器（可选但推荐）
- **操作系统**: Linux, macOS, Windows

## 安装工具

### 1. 安装 Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### 2. 安装 Just (推荐)

=== "macOS"

    ```bash
    brew install just
    ```

=== "Linux"

    ```bash
    cargo install just
    ```

=== "Windows"

    ```powershell
    cargo install just
    ```

### 3. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 克隆仓库

```bash
git clone https://github.com/ArkWhale/pcl-rustic.git
cd pcl-rustic
```

## 安装依赖

使用 `just` 命令一键设置：

```bash
just install
```

这将自动：
- 安装所有Python依赖（dev + docs）
- 设置pre-commit hooks

或手动安装：

```bash
# 创建虚拟环境并安装依赖
uv sync --all-groups

# 安装pre-commit hooks
pre-commit install
```

## 构建项目

### 开发模式构建

```bash
just dev
```

或：

```bash
maturin develop
```

### 发布模式构建

```bash
just build
```

或：

```bash
maturin develop --release
```

## 运行测试

```bash
# 快速测试
just test

# 所有测试
just test-slow

# 基准测试
just benchmark

# Rust测试
just test-rust
```

## 代码质量检查

### 格式化代码

```bash
just fmt
```

这将运行：
- `cargo fmt` (Rust)
- `ruff format` (Python)

### Linting

```bash
just lint
```

这将运行：
- `cargo clippy` (Rust)
- `ruff check` (Python)

### Pre-commit检查

```bash
just pre-commit
```

## 文档开发

### 本地预览

```bash
just docs-serve
```

然后访问 http://127.0.0.1:8000

### 构建文档

```bash
just docs-build
```

静态文件将生成到 `site/` 目录。

## 常用开发流程

### 添加新功能

```bash
# 1. 创建新分支
git checkout -b feature/my-feature

# 2. 开发代码
# 编辑 src/ 中的文件

# 3. 构建测试
just dev
just test

# 4. 格式化和lint
just fmt
just lint

# 5. 提交
git add .
git commit -m "Add my feature"

# 6. 推送
git push origin feature/my-feature
```

### 修复Bug

```bash
# 1. 创建bug分支
git checkout -b fix/bug-description

# 2. 修复代码

# 3. 添加测试
# 编辑 tests/ 中的文件

# 4. 验证修复
just test

# 5. 提交和推送
git commit -am "Fix bug: description"
git push origin fix/bug-description
```

## IDE设置

### VS Code

推荐安装以下扩展：

- rust-analyzer
- Python
- Ruff
- Even Better TOML

推荐设置（`.vscode/settings.json`）：

```json
{
  "rust-analyzer.cargo.features": ["pyo3/extension-module"],
  "python.linting.enabled": true,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "ruff",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": true,
      "source.organizeImports": true
    }
  },
  "[rust]": {
    "editor.formatOnSave": true
  }
}
```

### PyCharm

1. 安装 Rust 插件
2. 配置 Python 解释器指向 `.venv/bin/python`
3. 启用 Ruff linter

## 故障排除

### 编译错误

**问题**: `error: failed to compile pcl-rustic`

**解决**:
```bash
# 更新Rust工具链
rustup update stable

# 清理并重新构建
cargo clean
just build
```

### 导入错误

**问题**: `ModuleNotFoundError: No module named 'pcl_rustic._core'`

**解决**:
```bash
# 重新构建扩展
just dev
```

### Pre-commit失败

**问题**: Pre-commit hooks失败

**解决**:
```bash
# 运行格式化和lint
just fmt
just lint

# 再次尝试提交
git commit
```

## 下一步

- [贡献指南](contributing.md) - 了解如何贡献代码
- [API文档](../api/overview.md) - 了解API设计
