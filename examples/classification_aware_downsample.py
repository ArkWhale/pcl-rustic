import numpy as np

from pcl_rustic import DownsampleStrategy, PointCloud


def build_fixture() -> PointCloud:
    rng = np.random.default_rng(11)
    xyz = rng.normal(size=(10_000, 3)).astype(np.float32)
    xyz[:, :2] *= 20.0
    xyz[:, 2] *= 2.0
    classification = rng.choice(
        np.array([2, 3, 5, 6, 7], dtype=np.uint8),
        size=len(xyz),
        p=[0.35, 0.2, 0.2, 0.2, 0.05],
    )
    pc = PointCloud.from_xyz(xyz)
    pc.set_attribute("classification", classification)
    pc.set_intensity(rng.random(len(xyz), dtype=np.float32))
    return pc


def run_pipeline(pc: PointCloud | None = None) -> PointCloud:
    pc = build_fixture() if pc is None else pc
    class_voxels = {
        2: 0.5,  # ground
        3: 0.25,  # low vegetation
        5: 0.2,  # high vegetation
        6: 0.1,  # buildings
    }

    parts = []
    for code, voxel_size in class_voxels.items():
        subset = pc.select_by_classification([code])
        if subset.point_count() == 0:
            continue
        clean, _ = subset.remove_statistical_outlier(nb_neighbors=8, std_ratio=2.5)
        clean, _ = clean.remove_radius_outlier(
            nb_points=1, radius=max(voxel_size * 4, 0.5)
        )
        if clean.point_count() == 0:
            continue
        parts.append(clean.voxel_downsample(voxel_size, DownsampleStrategy.AVERAGE))

    if not parts:
        raise ValueError(
            "classification-aware pipeline produced no non-empty partitions"
        )
    return PointCloud.concatenate(parts, policy="union")


def main() -> None:
    pc = build_fixture()
    merged = run_pipeline(pc)
    print(pc.point_count(), merged.point_count())


if __name__ == "__main__":
    main()
