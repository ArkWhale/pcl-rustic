use crate::point_cloud::attribute_value::AttributeValue;
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::{PointCloudError, Result};
use kiddo::{float::kdtree::KdTree as FloatKdTree, KdTree, SquaredEuclidean};
use nalgebra::{Matrix3, SymmetricEigen, Vector3};

const NORMAL_PROJECTED_BUCKET_SIZE: usize = 4096;
type ProjectedKdTree2d = FloatKdTree<f32, u64, 2, NORMAL_PROJECTED_BUCKET_SIZE, u32>;

#[derive(Clone, Debug)]
pub enum NormalSearch {
    Knn(usize),
    Radius(f32),
    Hybrid(f32, usize),
}

impl HighPerformancePointCloud {
    pub fn estimate_normals(&mut self, search: NormalSearch) -> Result<()> {
        let xyz = self.get_xyz_vec();
        if xyz.is_empty() {
            return Err(PointCloudError::InvalidParameter(
                "cannot estimate normals for an empty point cloud".to_string(),
            ));
        }
        validate_search(&search)?;

        let mut nx = Vec::with_capacity(xyz.len());
        let mut ny = Vec::with_capacity(xyz.len());
        let mut nz = Vec::with_capacity(xyz.len());
        let index = NormalNeighborIndex::build(&xyz)?;

        for point_index in 0..xyz.len() {
            let indices = index.neighbor_indices(&xyz, point_index, &search)?;
            let normal = estimate_normal_for_indices(&xyz, &indices);
            nx.push(normal[0]);
            ny.push(normal[1]);
            nz.push(normal[2]);
        }

        self.attributes_mut()
            .insert("nx".to_string(), AttributeValue::F32(nx));
        self.attributes_mut()
            .insert("ny".to_string(), AttributeValue::F32(ny));
        self.attributes_mut()
            .insert("nz".to_string(), AttributeValue::F32(nz));
        Ok(())
    }

    pub fn estimate_covariances(&mut self, knn: usize) -> Result<()> {
        if knn < 3 {
            return Err(PointCloudError::InvalidParameter(
                "knn must be >= 3 for covariance estimation".to_string(),
            ));
        }
        let xyz = self.get_xyz_vec();
        if xyz.is_empty() {
            return Err(PointCloudError::InvalidParameter(
                "cannot estimate covariances for an empty point cloud".to_string(),
            ));
        }
        let neighborhoods = neighborhoods(&xyz, NormalSearch::Knn(knn))?;
        let covariances = neighborhoods
            .iter()
            .map(|indices| pack_covariance(covariance_for_indices(&xyz, indices)))
            .collect();
        self.attributes_mut()
            .insert("covariance".to_string(), AttributeValue::F32x6(covariances));
        Ok(())
    }
}

fn neighborhoods(xyz: &[[f32; 3]], search: NormalSearch) -> Result<Vec<Vec<usize>>> {
    validate_search(&search)?;
    let index = NormalNeighborIndex::build(xyz)?;
    (0..xyz.len())
        .map(|point_index| index.neighbor_indices(xyz, point_index, &search))
        .collect()
}

#[cfg(test)]
#[derive(Debug, PartialEq, Eq)]
enum NormalNeighborIndexKind {
    Kd3d,
    Projected2d,
    BruteForce,
}

enum NormalNeighborIndex {
    Kd3d(KdTree<f32, 3>),
    Projected2d {
        tree: ProjectedKdTree2d,
        axes: [usize; 2],
    },
    BruteForce,
}

impl NormalNeighborIndex {
    fn build(xyz: &[[f32; 3]]) -> Result<Self> {
        if xyz.is_empty() {
            return Err(PointCloudError::InvalidParameter(
                "cannot build normal neighbor index for an empty point cloud".to_string(),
            ));
        }
        if xyz.iter().flatten().any(|v| !v.is_finite()) {
            return Err(PointCloudError::InvalidParameter(
                "cannot build normal neighbor index with non-finite coordinates".to_string(),
            ));
        }

        let constant_axes = constant_axes(xyz);
        if constant_axes.len() == 1 {
            let axes = projected_axes(constant_axes[0]);
            if has_axis_bucket_over_limit(xyz, &axes, NORMAL_PROJECTED_BUCKET_SIZE) {
                return Ok(Self::BruteForce);
            }
            let mut tree: ProjectedKdTree2d = ProjectedKdTree2d::new();
            for (idx, point) in xyz.iter().enumerate() {
                tree.add(&project(point, axes), idx as u64);
            }
            Ok(Self::Projected2d { tree, axes })
        } else if constant_axes.is_empty() {
            if has_axis_bucket_over_limit(xyz, &[0, 1, 2], 32) {
                return Ok(Self::BruteForce);
            }
            let mut tree: KdTree<f32, 3> = KdTree::new();
            for (idx, point) in xyz.iter().enumerate() {
                tree.add(point, idx as u64);
            }
            Ok(Self::Kd3d(tree))
        } else {
            Ok(Self::BruteForce)
        }
    }

    #[cfg(test)]
    fn kind(&self) -> NormalNeighborIndexKind {
        match self {
            Self::Kd3d(_) => NormalNeighborIndexKind::Kd3d,
            Self::Projected2d { .. } => NormalNeighborIndexKind::Projected2d,
            Self::BruteForce => NormalNeighborIndexKind::BruteForce,
        }
    }

    fn neighbor_indices(
        &self,
        xyz: &[[f32; 3]],
        point_index: usize,
        search: &NormalSearch,
    ) -> Result<Vec<usize>> {
        match *search {
            NormalSearch::Knn(k) => Ok(self
                .knn_hits(xyz, point_index, k.saturating_add(1).min(xyz.len()))
                .into_iter()
                .map(|(idx, _)| idx)
                .filter(|&idx| idx != point_index)
                .take(k)
                .collect()),
            NormalSearch::Radius(radius) => Ok(self
                .radius_hits(xyz, point_index, radius)
                .into_iter()
                .map(|(idx, _)| idx)
                .filter(|&idx| idx != point_index)
                .collect()),
            NormalSearch::Hybrid(radius, k) => {
                let mut hits = self.radius_hits(xyz, point_index, radius);
                hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                Ok(hits
                    .into_iter()
                    .map(|(idx, _)| idx)
                    .filter(|&idx| idx != point_index)
                    .take(k)
                    .collect())
            }
        }
    }

    fn knn_hits(&self, xyz: &[[f32; 3]], point_index: usize, k: usize) -> Vec<(usize, f32)> {
        match self {
            Self::Kd3d(tree) => tree
                .nearest_n::<SquaredEuclidean>(&xyz[point_index], k)
                .into_iter()
                .map(|hit| (hit.item as usize, hit.distance))
                .collect(),
            Self::Projected2d { tree, axes } => tree
                .nearest_n::<SquaredEuclidean>(&project(&xyz[point_index], *axes), k)
                .into_iter()
                .map(|hit| (hit.item as usize, hit.distance))
                .collect(),
            Self::BruteForce => brute_force_knn_hits(xyz, &xyz[point_index], k),
        }
    }

    fn radius_hits(&self, xyz: &[[f32; 3]], point_index: usize, radius: f32) -> Vec<(usize, f32)> {
        let radius_sq = radius * radius;
        match self {
            Self::Kd3d(tree) => tree
                .within::<SquaredEuclidean>(&xyz[point_index], radius_sq)
                .into_iter()
                .map(|hit| (hit.item as usize, hit.distance))
                .collect(),
            Self::Projected2d { tree, axes } => tree
                .within::<SquaredEuclidean>(&project(&xyz[point_index], *axes), radius_sq)
                .into_iter()
                .map(|hit| (hit.item as usize, hit.distance))
                .collect(),
            Self::BruteForce => brute_force_radius_hits(xyz, &xyz[point_index], radius_sq),
        }
    }
}

fn validate_search(search: &NormalSearch) -> Result<()> {
    match *search {
        NormalSearch::Knn(k) => validate_knn(k, "knn"),
        NormalSearch::Radius(radius) => validate_radius(radius),
        NormalSearch::Hybrid(radius, k) => {
            validate_radius(radius)?;
            validate_knn(k, "hybrid knn")
        }
    }
}

fn validate_knn(k: usize, label: &str) -> Result<()> {
    if k < 3 {
        return Err(PointCloudError::InvalidParameter(format!(
            "{label} must be >= 3 for normal estimation"
        )));
    }
    Ok(())
}

fn validate_radius(radius: f32) -> Result<()> {
    if radius < 0.0 || !radius.is_finite() {
        return Err(PointCloudError::InvalidParameter(
            "radius must be finite and non-negative".to_string(),
        ));
    }
    Ok(())
}

fn constant_axes(xyz: &[[f32; 3]]) -> Vec<usize> {
    (0..3)
        .filter(|&axis| xyz.iter().all(|point| point[axis] == xyz[0][axis]))
        .collect()
}

fn has_axis_bucket_over_limit(xyz: &[[f32; 3]], axes: &[usize], limit: usize) -> bool {
    for &axis in axes {
        let mut values: Vec<u32> = xyz.iter().map(|p| p[axis].to_bits()).collect();
        values.sort_unstable();
        let mut run = 1usize;
        for pair in values.windows(2) {
            if pair[0] == pair[1] {
                run += 1;
                if run > limit {
                    return true;
                }
            } else {
                run = 1;
            }
        }
    }
    false
}

fn projected_axes(constant_axis: usize) -> [usize; 2] {
    match constant_axis {
        0 => [1, 2],
        1 => [0, 2],
        _ => [0, 1],
    }
}

fn project(point: &[f32; 3], axes: [usize; 2]) -> [f32; 2] {
    [point[axes[0]], point[axes[1]]]
}

fn brute_force_knn_hits(xyz: &[[f32; 3]], query: &[f32; 3], k: usize) -> Vec<(usize, f32)> {
    let mut hits: Vec<(usize, f32)> = xyz
        .iter()
        .enumerate()
        .map(|(idx, point)| (idx, squared_distance(point, query)))
        .collect();
    hits.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(k);
    hits
}

fn brute_force_radius_hits(
    xyz: &[[f32; 3]],
    query: &[f32; 3],
    radius_sq: f32,
) -> Vec<(usize, f32)> {
    xyz.iter()
        .enumerate()
        .filter_map(|(idx, point)| {
            let distance = squared_distance(point, query);
            (distance <= radius_sq).then_some((idx, distance))
        })
        .collect()
}

fn squared_distance(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn estimate_normal_for_indices(xyz: &[[f32; 3]], indices: &[usize]) -> [f32; 3] {
    if indices.len() < 3 {
        return [f32::NAN; 3];
    }
    let cov = covariance_for_indices(xyz, indices);
    let eig = SymmetricEigen::new(cov);
    let min_idx = (0..3)
        .min_by(|&a, &b| {
            eig.eigenvalues[a]
                .partial_cmp(&eig.eigenvalues[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let v = eig.eigenvectors.column(min_idx).normalize();
    [v[0], v[1], v[2]]
}

pub(crate) fn covariance_for_indices(xyz: &[[f32; 3]], indices: &[usize]) -> Matrix3<f32> {
    if indices.is_empty() {
        return Matrix3::identity();
    }
    let n = indices.len() as f32;
    let mut mean = Vector3::zeros();
    for &idx in indices {
        mean += Vector3::new(xyz[idx][0], xyz[idx][1], xyz[idx][2]);
    }
    mean /= n;

    let mut cov = Matrix3::zeros();
    for &idx in indices {
        let p = Vector3::new(xyz[idx][0], xyz[idx][1], xyz[idx][2]) - mean;
        cov += p * p.transpose();
    }
    cov / n.max(1.0)
}

fn pack_covariance(cov: Matrix3<f32>) -> [f32; 6] {
    [
        cov[(0, 0)],
        cov[(0, 1)],
        cov[(0, 2)],
        cov[(1, 1)],
        cov[(1, 2)],
        cov[(2, 2)],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_normals_are_unit_and_axis_aligned() {
        let mut xyz = Vec::new();
        for x in 0..100 {
            for y in 0..100 {
                xyz.push([x as f32, y as f32, 0.0]);
            }
        }
        let mut pc = HighPerformancePointCloud::from_xyz_vec(xyz).unwrap();
        pc.estimate_normals(NormalSearch::Knn(12)).unwrap();
        let nx = pc.get_attribute("nx").unwrap().as_f32().unwrap();
        let ny = pc.get_attribute("ny").unwrap().as_f32().unwrap();
        let nz = pc.get_attribute("nz").unwrap().as_f32().unwrap();
        let aligned = nx
            .iter()
            .zip(ny)
            .zip(nz)
            .filter(|((&x, &y), &z)| (x * x + y * y + z * z - 1.0).abs() < 1e-5 && z.abs() > 0.99)
            .count();
        assert!(aligned >= 9_900);
    }

    #[test]
    fn covariances_are_stored_as_packed_rows() {
        let mut pc = HighPerformancePointCloud::from_xyz_vec(vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        .unwrap();
        pc.estimate_covariances(3).unwrap();
        assert_eq!(pc.get_attribute("covariance").unwrap().len(), 4);
    }

    #[test]
    fn flat_plane_normals_use_projected_knn_backend() {
        let xyz: Vec<[f32; 3]> = (0..128)
            .map(|i| {
                let x = (i % 16) as f32;
                let y = (i / 16) as f32;
                [x, y, 0.0]
            })
            .collect();

        assert_eq!(
            NormalNeighborIndex::build(&xyz).unwrap().kind(),
            NormalNeighborIndexKind::Projected2d
        );
    }
}
