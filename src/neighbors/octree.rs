use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::{PointCloudError, Result};
use std::collections::HashMap;

#[derive(Clone)]
pub struct Octree {
    xyz: Vec<[f32; 3]>,
    min: [f32; 3],
    max: [f32; 3],
    max_depth: u8,
    cells: HashMap<[u32; 3], Vec<u64>>,
}

impl Octree {
    pub fn build(pc: &HighPerformancePointCloud, max_depth: u8) -> Result<Self> {
        if pc.is_empty() {
            return Err(PointCloudError::InvalidParameter(
                "cannot build octree for an empty point cloud".to_string(),
            ));
        }
        if max_depth > 21 {
            return Err(PointCloudError::InvalidParameter(
                "max_depth must be <= 21".to_string(),
            ));
        }

        let xyz = pc.get_xyz_vec();
        let (min, max) = pc.aabb();
        let mut cells: HashMap<[u32; 3], Vec<u64>> = HashMap::new();
        for (idx, point) in xyz.iter().enumerate() {
            cells
                .entry(cell_for_point(point, min, max, max_depth))
                .or_default()
                .push(idx as u64);
        }

        Ok(Self {
            xyz,
            min,
            max,
            max_depth,
            cells,
        })
    }

    pub fn range_search(&self, center: &[f32; 3], radius: f32) -> Result<Vec<u64>> {
        if radius < 0.0 || !radius.is_finite() {
            return Err(PointCloudError::InvalidParameter(
                "radius must be finite and non-negative".to_string(),
            ));
        }
        let r2 = radius * radius;
        Ok(self
            .xyz
            .iter()
            .enumerate()
            .filter(|(_, p)| squared_distance(p, center) <= r2)
            .map(|(idx, _)| idx as u64)
            .collect())
    }

    pub fn voxel_centers(&self) -> Vec<[f32; 3]> {
        let scale = (1u32 << self.max_depth) as f32;
        self.cells
            .keys()
            .map(|cell| {
                let mut out = [0.0; 3];
                for axis in 0..3 {
                    let width = (self.max[axis] - self.min[axis]).max(f32::EPSILON);
                    out[axis] = self.min[axis] + ((cell[axis] as f32 + 0.5) / scale) * width;
                }
                out
            })
            .collect()
    }
}

impl HighPerformancePointCloud {
    pub fn octree(&self, max_depth: u8) -> Result<Octree> {
        Octree::build(self, max_depth)
    }
}

fn cell_for_point(point: &[f32; 3], min: [f32; 3], max: [f32; 3], max_depth: u8) -> [u32; 3] {
    let resolution = 1u32 << max_depth;
    let mut cell = [0; 3];
    for axis in 0..3 {
        let width = (max[axis] - min[axis]).max(f32::EPSILON);
        let normalized = ((point[axis] - min[axis]) / width).clamp(0.0, 1.0);
        cell[axis] = (normalized * (resolution - 1) as f32).floor() as u32;
    }
    cell
}

fn squared_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn range_search_matches_bruteforce_case() {
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ])
        .unwrap();
        let octree = pc.octree(4).unwrap();
        assert_eq!(
            octree.range_search(&[0.0, 0.0, 0.0], 1.1).unwrap(),
            vec![0, 1]
        );
        assert!(!octree.voxel_centers().is_empty());
    }

    #[test]
    fn random_10k_range_search_matches_bruteforce_oracle() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let xyz: Vec<[f32; 3]> = (0..10_000)
            .map(|_| {
                [
                    rng.gen_range(-500.0..500.0),
                    rng.gen_range(-500.0..500.0),
                    rng.gen_range(-500.0..500.0),
                ]
            })
            .collect();
        let pc = HighPerformancePointCloud::from_xyz_vec(xyz.clone()).unwrap();
        let octree = pc.octree(8).unwrap();

        for _ in 0..64 {
            let center = [
                rng.gen_range(-500.0..500.0),
                rng.gen_range(-500.0..500.0),
                rng.gen_range(-500.0..500.0),
            ];
            let radius = 75.0;
            let mut actual = octree.range_search(&center, radius).unwrap();
            let r2 = radius * radius;
            let mut expected: Vec<u64> = xyz
                .iter()
                .enumerate()
                .filter(|(_, point)| squared_distance(point, &center) <= r2)
                .map(|(idx, _)| idx as u64)
                .collect();
            actual.sort_unstable();
            expected.sort_unstable();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn empty_cloud_returns_error() {
        let empty = HighPerformancePointCloud::new();
        match empty.octree(4) {
            Ok(_) => panic!("empty cloud unexpectedly built an octree"),
            Err(err) => assert!(err.to_string().contains("empty")),
        }
    }
}
