use crate::point_cloud::attribute_value::AttributeValue;
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::{PointCloudError, Result};
use nalgebra::{Matrix3, SymmetricEigen, Vector3};

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

        let neighborhoods = neighborhoods(self, &xyz, search)?;
        let mut nx = Vec::with_capacity(xyz.len());
        let mut ny = Vec::with_capacity(xyz.len());
        let mut nz = Vec::with_capacity(xyz.len());

        for indices in neighborhoods {
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
        let neighborhoods = neighborhoods(self, &xyz, NormalSearch::Knn(knn))?;
        let covariances = neighborhoods
            .iter()
            .map(|indices| pack_covariance(covariance_for_indices(&xyz, indices)))
            .collect();
        self.attributes_mut()
            .insert("covariance".to_string(), AttributeValue::F32x6(covariances));
        Ok(())
    }
}

fn neighborhoods(
    pc: &HighPerformancePointCloud,
    xyz: &[[f32; 3]],
    search: NormalSearch,
) -> Result<Vec<Vec<usize>>> {
    match search {
        NormalSearch::Knn(k) => {
            if k < 3 {
                return Err(PointCloudError::InvalidParameter(
                    "knn must be >= 3 for normal estimation".to_string(),
                ));
            }
            let hits = pc.kdtree()?.knn(xyz, (k + 1).min(xyz.len()))?;
            Ok(hits
                .into_iter()
                .enumerate()
                .map(|(i, row)| {
                    row.into_iter()
                        .map(|hit| hit.index as usize)
                        .filter(|&idx| idx != i)
                        .take(k)
                        .collect()
                })
                .collect())
        }
        NormalSearch::Radius(radius) => {
            let hits = pc.kdtree()?.radius_search(xyz, radius)?;
            Ok(hits
                .into_iter()
                .enumerate()
                .map(|(i, row)| {
                    row.into_iter()
                        .map(|hit| hit.index as usize)
                        .filter(|&idx| idx != i)
                        .collect()
                })
                .collect())
        }
        NormalSearch::Hybrid(radius, k) => {
            if k < 3 {
                return Err(PointCloudError::InvalidParameter(
                    "hybrid knn must be >= 3 for normal estimation".to_string(),
                ));
            }
            let hits = pc.kdtree()?.radius_search(xyz, radius)?;
            Ok(hits
                .into_iter()
                .enumerate()
                .map(|(i, mut row)| {
                    row.sort_by(|a, b| {
                        a.distance
                            .partial_cmp(&b.distance)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    row.into_iter()
                        .map(|hit| hit.index as usize)
                        .filter(|&idx| idx != i)
                        .take(k)
                        .collect()
                })
                .collect())
        }
    }
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
        for x in 0..20 {
            for y in 0..20 {
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
            .filter(|((&x, &y), &z)| (x * x + y * y + z * z - 1.0).abs() < 1e-4 && z.abs() > 0.99)
            .count();
        assert!(aligned > 390);
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
}
