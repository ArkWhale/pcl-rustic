"""
PCL Rustic pytest 配置和 fixtures

提供测试数据生成器和共享 fixtures
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pytest
from loguru import logger

# 配置 loguru 用于测试输出
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO",
    colorize=True,
)


def pytest_configure(config):
    """注册自定义标记"""
    config.addinivalue_line("markers", "slow: 标记为慢速测试（100M+ 点）")
    config.addinivalue_line("markers", "benchmark: RFC-0009 benchmark suite tests")
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("reports/benchmarks").mkdir(parents=True, exist_ok=True)


def pytest_addoption(parser):
    """添加命令行选项"""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="运行慢速测试（100M+ 点云）",
    )
    parser.addoption(
        "--benchmark-mode",
        action="store",
        default="smoke",
        choices=("smoke", "standard", "full"),
        help="Benchmark suite scope: smoke, standard, or full (default: smoke)",
    )


def pytest_collection_modifyitems(config, items):
    """跳过慢速测试（除非指定 --run-slow）"""
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(reason="需要 --run-slow 选项来运行")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def small_xyz() -> np.ndarray:
    """小型 XYZ 数据（10 点）"""
    return np.array(
        [[float(i), float(i), float(i) * 0.5] for i in range(10)],
        dtype=np.float32,
    )


@pytest.fixture
def medium_xyz() -> np.ndarray:
    """中型 XYZ 数据（1000 点）"""
    return np.random.randn(1000, 3).astype(np.float32)


@pytest.fixture
def large_xyz() -> np.ndarray:
    """大型 XYZ 数据（100000 点）"""
    return np.random.randn(100000, 3).astype(np.float32) * 100


@pytest.fixture
def gaussian_point_cloud_10m() -> dict[str, np.ndarray]:
    """10M 点高斯分布点云"""
    return _generate_gaussian_point_cloud(10_000_000)


@pytest.fixture
def gaussian_point_cloud_1m() -> dict[str, np.ndarray]:
    """1M 点高斯分布点云（用于快速测试）"""
    return _generate_gaussian_point_cloud(1_000_000)


def _generate_gaussian_point_cloud(
    num_points: int,
    x_range: Tuple[float, float] = (-100, 250),
    y_range: Tuple[float, float] = (-100, 250),
    z_range: Tuple[float, float] = (-3, 7),
    seed: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    生成高斯分布的点云数据

    6 个维度: x, y, z, intensity, d1, d2

    Args:
        num_points: 点数
        x_range: X 坐标范围
        y_range: Y 坐标范围
        z_range: Z 坐标范围
        seed: 随机种子

    Returns:
        包含 xyz, intensity, d1, d2 的字典（全部为 float32）
    """
    rng = np.random.default_rng(seed)

    # 随机中心点
    center_x = rng.uniform(x_range[0] + 50, x_range[1] - 50)
    center_y = rng.uniform(y_range[0] + 50, y_range[1] - 50)
    center_z = rng.uniform(z_range[0] + 1, z_range[1] - 1)

    # 随机 sigma（标准差）
    sigma_xy = rng.uniform(30, 80)
    sigma_z = rng.uniform(1, 3)

    # 生成高斯分布坐标
    x = rng.normal(center_x, sigma_xy, num_points).astype(np.float32)
    y = rng.normal(center_y, sigma_xy, num_points).astype(np.float32)
    z = rng.normal(center_z, sigma_z, num_points).astype(np.float32)

    # 裁剪到范围内
    x = np.clip(x, x_range[0], x_range[1])
    y = np.clip(y, y_range[0], y_range[1])
    z = np.clip(z, z_range[0], z_range[1])

    # 合并为 [N, 3] 数组
    xyz = np.column_stack([x, y, z]).astype(np.float32)

    # 生成 intensity（距离衰减）
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2)
    intensity = (1.0 / (1.0 + dist / sigma_xy) * 255).astype(np.float32)

    # 生成自定义属性 d1, d2
    d1 = rng.normal(0, 1, num_points).astype(np.float32)
    d2 = rng.normal(0, 1, num_points).astype(np.float32)

    return {
        "xyz": xyz,
        "intensity": intensity,
        "d1": d1,
        "d2": d2,
        "metadata": {
            "center": (center_x, center_y, center_z),
            "sigma_xy": sigma_xy,
            "sigma_z": sigma_z,
        },
    }
