# 贡献指南

感谢您对 pcl-rustic 的贡献兴趣！本指南将帮助您了解如何参与项目开发。

## 行为准则

- 尊重所有贡献者
- 保持建设性的讨论
- 专注于技术问题
- 欢迎新手贡献者

## 如何贡献

### 报告Bug

在[GitHub Issues](https://github.com/ArkWhale/pcl-rustic/issues)中创建bug报告，请包含：

- **描述**: 清晰简洁的问题描述
- **复现步骤**: 详细的复现步骤
- **期望行为**: 应该发生什么
- **实际行为**: 实际发生了什么
- **环境信息**: OS、Python版本、pcl-rustic版本
- **代码示例**: 最小可复现示例

**Bug报告模板**:

```markdown
**描述**
简短描述bug

**复现步骤**
1. ...
2. ...
3. ...

**期望行为**
应该...

**实际行为**
实际...

**环境**
- OS: macOS 14.0
- Python: 3.11
- pcl-rustic: 0.1.0

**代码示例**
\```python
# 最小可复现示例
\```
```

### 提议新功能

在[GitHub Discussions](https://github.com/ArkWhale/pcl-rustic/discussions)中讨论新功能：

- 描述功能的用途
- 解释为什么需要这个功能
- 提供使用示例
- 讨论可能的实现方式

### 贡献代码

1. **Fork仓库**

   点击GitHub页面右上角的"Fork"按钮

2. **克隆你的fork**

   ```bash
   git clone https://github.com/ArkWhale/pcl-rustic.git
   cd pcl-rustic
   ```

3. **创建特性分支**

   ```bash
   git checkout -b feature/amazing-feature
   ```

4. **设置开发环境**

   ```bash
   just install
   ```

5. **进行更改**

   编辑代码，确保：
   - 代码符合项目规范
   - 添加了单元测试
   - 测试通过
   - 文档已更新

6. **运行测试**

   ```bash
   just test
   just fmt
   just lint
   ```

7. **提交更改**

   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

   提交消息应遵循[Conventional Commits](https://www.conventionalcommits.org/)规范：
   - `feat:` 新功能
   - `fix:` Bug修复
   - `docs:` 文档更新
   - `style:` 代码格式（不影响功能）
   - `refactor:` 重构
   - `perf:` 性能优化
   - `test:` 测试相关
   - `chore:` 构建/工具相关

8. **推送到GitHub**

   ```bash
   git push origin feature/amazing-feature
   ```

9. **创建Pull Request**

   在GitHub上创建PR，描述清楚：
   - 解决了什么问题
   - 如何解决的
   - 相关Issue编号
   - 测试情况

## 代码规范

### Rust代码

- 遵循Rust标准风格（`cargo fmt`）
- 通过所有clippy检查（`cargo clippy`）
- 添加文档注释（`///`）
- 为公共API编写测试

**示例**:

```rust
/// 从numpy XYZ数组创建点云
///
/// # Arguments
///
/// * `xyz` - 形状为[N, 3]的2D numpy数组（dtype=float32）
///
/// # Returns
///
/// `PyResult<Self>` - 成功返回点云对象，失败返回错误
///
/// # Errors
///
/// 当输入不是float32的2D数组时返回错误
#[staticmethod]
fn from_xyz(xyz: &Bound<'_, PyAny>) -> PyResult<Self> {
    // ...
}
```

### Python代码

- 遵循Ruff风格（`ruff format`）
- 通过Ruff检查（`ruff check`）
- 使用类型注解
- 编写Google风格的docstring

**示例**:

```python
def process_point_cloud(pc: PointCloud, voxel_size: float) -> PointCloud:
    """处理点云进行体素下采样。

    Args:
        pc: 输入点云
        voxel_size: 体素大小（米）

    Returns:
        下采样后的点云

    Raises:
        ValueError: 当voxel_size <= 0时

    Examples:
        ```python
        pc_down = process_point_cloud(pc, 0.15)
        ```
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    return pc.voxel_downsample(voxel_size)
```

### 测试

- 为新功能添加测试
- 测试覆盖主要代码路径
- 使用pytest编写Python测试
- 使用`#[cfg(test)]`编写Rust测试

**Python测试示例**:

```python
def test_voxel_downsample():
    """测试体素下采样功能"""
    xyz = np.random.randn(10000, 3).astype(np.float32)
    pc = PointCloud.from_xyz(xyz)

    pc_down = pc.voxel_downsample(0.15)

    assert pc_down.point_count() < pc.point_count()
    assert pc_down.point_count() > 0
```

**Rust测试示例**:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0, 2.0, 3.0];
        let tensor = tensor1_from_slice(&data);
        assert_eq!(tensor.dims(), [3]);
    }
}
```

## 文档

### 更新文档

当您添加新功能或修改API时，请更新文档：

1. **API文档**: 在代码中添加docstring
2. **使用指南**: 在`docs/`中添加markdown文件
3. **示例**: 在`docs/getting-started/examples.md`中添加示例

### 本地预览文档

```bash
just docs-serve
```

访问 http://127.0.0.1:8000 查看效果。

## Pull Request流程

1. **PR标题**: 使用清晰的标题
2. **描述**: 详细说明更改内容
3. **关联Issue**: 使用`Fixes #123`链接相关Issue
4. **检查列表**: 确保所有检查项都完成

**PR模板**:

```markdown
## 更改内容

简要描述此PR的更改

## 相关Issue

Fixes #123

## 类型

- [ ] Bug修复
- [ ] 新功能
- [ ] 性能改进
- [ ] 文档更新
- [ ] 代码重构

## 测试

- [ ] 添加了新测试
- [ ] 所有测试通过
- [ ] 手动测试通过

## 检查列表

- [ ] 代码符合项目规范
- [ ] 通过`just fmt`格式化
- [ ] 通过`just lint`检查
- [ ] 添加/更新了文档
- [ ] 更新了CHANGELOG.md（如适用）
```

## Review流程

1. **自动检查**: CI会自动运行测试和检查
2. **代码Review**: 维护者会Review您的代码
3. **修改**: 根据反馈进行修改
4. **合并**: Review通过后会合并到main分支

## 开发技巧

### 快速迭代

```bash
# 监视文件变化，自动重新构建
cargo watch -x 'run --example my_example'
```

### 调试

```bash
# 启用详细日志
RUST_LOG=debug just test

# 使用Python调试器
python -m pdb tests/test_xxx.py
```

### 性能分析

```bash
# 使用cargo flamegraph
cargo install flamegraph
cargo flamegraph --example benchmark
```

## 获得帮助

如果您遇到问题或有疑问：

1. 查看[文档](https://ArkWhale.github.io/pcl-rustic)
2. 搜索[现有Issues](https://github.com/ArkWhale/pcl-rustic/issues)
3. 在[Discussions](https://github.com/ArkWhale/pcl-rustic/discussions)中提问
4. 联系维护者：liuzhen19@xiaomi.com

## 致谢

感谢所有贡献者的努力！您的贡献让pcl-rustic变得更好。

## 相关资源

- [开发环境设置](setup.md)
- [API文档](../api/overview.md)
- [Rust Book](https://doc.rust-lang.org/book/)
- [PyO3指南](https://pyo3.rs/)
