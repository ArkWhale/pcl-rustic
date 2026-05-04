import numpy as np

from pcl_rustic import DownsampleStrategy, PointCloud


def build_fixture() -> PointCloud:
    rng = np.random.default_rng(7)
    xyz = rng.uniform([0.0, 0.0, -1.0], [20.0, 20.0, 3.0], size=(10_000, 3)).astype(
        np.float32
    )
    pc = PointCloud.from_xyz(xyz)
    pc.set_intensity(rng.random(len(xyz), dtype=np.float32))
    return pc


def main() -> None:
    pc = build_fixture()
    min_bound, max_bound = pc.aabb()
    min_bound = np.asarray(min_bound, dtype=np.float32)
    max_bound = np.asarray(max_bound, dtype=np.float32)

    parts = []
    grid_x = 4
    grid_y = 4
    step = (max_bound - min_bound) / np.array([grid_x, grid_y, 1], dtype=np.float32)

    for ix in range(grid_x):
        for iy in range(grid_y):
            lo = min_bound + np.array([ix * step[0], iy * step[1], 0], dtype=np.float32)
            hi = min_bound + np.array(
                [(ix + 1) * step[0], (iy + 1) * step[1], max_bound[2] - min_bound[2]],
                dtype=np.float32,
            )
            cell = pc.crop_aabb(lo, hi)
            if cell.point_count() == 0:
                continue
            voxel = 0.08 + 0.02 * (ix + iy)
            parts.append(
                cell.voxel_downsample(voxel, DownsampleStrategy.NEAREST_TO_CENTROID)
            )

    merged = PointCloud.concatenate(parts, policy="union")
    print(pc.point_count(), merged.point_count())


if __name__ == "__main__":
    main()

