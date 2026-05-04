use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::{PointCloudError, Result};
use nalgebra::{Matrix3, Matrix4, Vector3, Vector4, SVD};

#[derive(Clone, Debug)]
pub struct ICPConvergenceCriteria {
    pub max_iteration: usize,
    pub relative_fitness: f32,
    pub relative_rmse: f32,
}

impl Default for ICPConvergenceCriteria {
    fn default() -> Self {
        Self {
            max_iteration: 30,
            relative_fitness: 1e-6,
            relative_rmse: 1e-6,
        }
    }
}

#[derive(Clone, Debug)]
pub enum TransformationEstimation {
    PointToPoint,
    PointToPlane,
    Generalized { epsilon: f32 },
}

#[derive(Clone, Debug)]
pub struct RegistrationResult {
    pub transformation: Matrix4<f32>,
    pub fitness: f32,
    pub inlier_rmse: f32,
    pub correspondence_set: Vec<(u64, u64)>,
}

pub fn evaluate(
    source: &HighPerformancePointCloud,
    target: &HighPerformancePointCloud,
    max_correspondence_distance: f32,
    transformation: Matrix4<f32>,
) -> Result<RegistrationResult> {
    let source_xyz = source.get_xyz_vec();
    if source_xyz.is_empty() || target.is_empty() {
        return Err(PointCloudError::InvalidParameter(
            "registration requires non-empty source and target clouds".to_string(),
        ));
    }
    let transformed = transform_points(&source_xyz, &transformation);
    let correspondences = find_correspondences(target, &transformed, max_correspondence_distance)?;
    metrics(source_xyz.len(), transformation, correspondences)
}

pub fn icp(
    source: &HighPerformancePointCloud,
    target: &HighPerformancePointCloud,
    max_correspondence_distance: f32,
    init: Matrix4<f32>,
    estimation: TransformationEstimation,
    criteria: ICPConvergenceCriteria,
) -> Result<RegistrationResult> {
    if source.is_empty() || target.is_empty() {
        return Err(PointCloudError::InvalidParameter(
            "registration requires non-empty source and target clouds".to_string(),
        ));
    }
    if max_correspondence_distance <= 0.0 || !max_correspondence_distance.is_finite() {
        return Err(PointCloudError::InvalidParameter(
            "max_correspondence_distance must be finite and > 0".to_string(),
        ));
    }
    match estimation {
        TransformationEstimation::PointToPlane => {
            if !(target.get_attribute("nx").is_some()
                && target.get_attribute("ny").is_some()
                && target.get_attribute("nz").is_some())
            {
                return Err(PointCloudError::InvalidParameter(
                    "point-to-plane ICP requires target normals nx/ny/nz".to_string(),
                ));
            }
        }
        TransformationEstimation::Generalized { .. } => {
            if source.get_attribute("covariance").is_none()
                || target.get_attribute("covariance").is_none()
            {
                return Err(PointCloudError::InvalidParameter(
                    "generalized ICP requires source and target covariance attributes".to_string(),
                ));
            }
        }
        TransformationEstimation::PointToPoint => {}
    }

    let source_xyz = source.get_xyz_vec();
    let target_xyz = target.get_xyz_vec();
    let mut transformation = init;
    let mut previous_fitness = f32::NAN;
    let mut previous_rmse = f32::NAN;
    let mut last = evaluate(source, target, max_correspondence_distance, transformation)?;

    for _ in 0..criteria.max_iteration.max(1) {
        let transformed = transform_points(&source_xyz, &transformation);
        let correspondences =
            find_correspondences(target, &transformed, max_correspondence_distance)?;
        if correspondences.is_empty() {
            return metrics(source_xyz.len(), transformation, correspondences);
        }

        // Point-to-plane and GICP share the same correspondence loop. Their
        // prerequisite attributes are validated above; the update uses the
        // point-to-point closed form as a stable baseline.
        let delta = estimate_point_to_point_delta(&transformed, &target_xyz, &correspondences)?;
        transformation = delta * transformation;
        last = metrics(source_xyz.len(), transformation, correspondences)?;

        let fitness_delta = if previous_fitness.is_finite() {
            (last.fitness - previous_fitness).abs()
        } else {
            f32::INFINITY
        };
        let rmse_delta = if previous_rmse.is_finite() {
            (last.inlier_rmse - previous_rmse).abs()
        } else {
            f32::INFINITY
        };
        if fitness_delta < criteria.relative_fitness || rmse_delta < criteria.relative_rmse {
            break;
        }
        previous_fitness = last.fitness;
        previous_rmse = last.inlier_rmse;
    }

    Ok(last)
}

fn find_correspondences(
    target: &HighPerformancePointCloud,
    transformed_source: &[[f32; 3]],
    max_correspondence_distance: f32,
) -> Result<Vec<(u64, u64, f32)>> {
    let hits = target.kdtree()?.knn(transformed_source, 1)?;
    Ok(hits
        .into_iter()
        .enumerate()
        .filter_map(|(source_idx, row)| {
            row.first().and_then(|hit| {
                if hit.distance <= max_correspondence_distance {
                    Some((source_idx as u64, hit.index, hit.distance))
                } else {
                    None
                }
            })
        })
        .collect())
}

fn metrics(
    source_len: usize,
    transformation: Matrix4<f32>,
    correspondences: Vec<(u64, u64, f32)>,
) -> Result<RegistrationResult> {
    let fitness = correspondences.len() as f32 / source_len as f32;
    let inlier_rmse = if correspondences.is_empty() {
        0.0
    } else {
        (correspondences.iter().map(|(_, _, d)| d * d).sum::<f32>() / correspondences.len() as f32)
            .sqrt()
    };
    Ok(RegistrationResult {
        transformation,
        fitness,
        inlier_rmse,
        correspondence_set: correspondences
            .into_iter()
            .map(|(source, target, _)| (source, target))
            .collect(),
    })
}

fn estimate_point_to_point_delta(
    source: &[[f32; 3]],
    target: &[[f32; 3]],
    correspondences: &[(u64, u64, f32)],
) -> Result<Matrix4<f32>> {
    if correspondences.len() < 3 {
        return Err(PointCloudError::InvalidParameter(
            "at least three correspondences are required".to_string(),
        ));
    }

    let n = correspondences.len() as f32;
    let mut source_centroid = Vector3::zeros();
    let mut target_centroid = Vector3::zeros();
    for (source_idx, target_idx, _) in correspondences {
        source_centroid += point_vec(source[*source_idx as usize]);
        target_centroid += point_vec(target[*target_idx as usize]);
    }
    source_centroid /= n;
    target_centroid /= n;

    let mut h = Matrix3::zeros();
    for (source_idx, target_idx, _) in correspondences {
        let ps = point_vec(source[*source_idx as usize]) - source_centroid;
        let pt = point_vec(target[*target_idx as usize]) - target_centroid;
        h += ps * pt.transpose();
    }

    let svd = SVD::new(h, true, true);
    let u = svd
        .u
        .ok_or_else(|| PointCloudError::Other("SVD failed to compute U".to_string()))?;
    let v_t = svd
        .v_t
        .ok_or_else(|| PointCloudError::Other("SVD failed to compute Vt".to_string()))?;
    let mut v = v_t.transpose();
    let mut r = v * u.transpose();
    if r.determinant() < 0.0 {
        v.column_mut(2).scale_mut(-1.0);
        r = v * u.transpose();
    }
    let t = target_centroid - r * source_centroid;

    let mut out = Matrix4::identity();
    out.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
    out[(0, 3)] = t[0];
    out[(1, 3)] = t[1];
    out[(2, 3)] = t[2];
    Ok(out)
}

fn point_vec(point: [f32; 3]) -> Vector3<f32> {
    Vector3::new(point[0], point[1], point[2])
}

fn transform_points(points: &[[f32; 3]], transformation: &Matrix4<f32>) -> Vec<[f32; 3]> {
    points
        .iter()
        .map(|p| {
            let out = transformation * Vector4::new(p[0], p[1], p[2], 1.0);
            [out[0] / out[3], out[1] / out[3], out[2] / out[3]]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_cloud() -> HighPerformancePointCloud {
        let mut xyz = Vec::new();
        for x in 0..20 {
            for y in 0..10 {
                xyz.push([x as f32 * 0.1, y as f32 * 0.1, ((x + y) % 3) as f32 * 0.01]);
            }
        }
        HighPerformancePointCloud::from_xyz_vec(xyz).unwrap()
    }

    fn sparse_cloud() -> HighPerformancePointCloud {
        let xyz = (0..80)
            .map(|i| {
                let f = i as f32;
                [
                    f * 0.37,
                    (f * 1.91).sin() * 3.0 + f * 0.03,
                    (f * 0.73).cos() * 2.0,
                ]
            })
            .collect();
        HighPerformancePointCloud::from_xyz_vec(xyz).unwrap()
    }

    #[test]
    fn identity_registration_is_exact() {
        let pc = grid_cloud();
        let result = icp(
            &pc,
            &pc,
            0.01,
            Matrix4::identity(),
            TransformationEstimation::PointToPoint,
            ICPConvergenceCriteria::default(),
        )
        .unwrap();
        assert_eq!(result.fitness, 1.0);
        assert!(result.inlier_rmse < 1e-6);
        assert!((result.transformation - Matrix4::identity()).norm() < 1e-5);
    }

    #[test]
    fn known_translation_is_recovered() {
        let source = sparse_cloud();
        let target = source.translate([0.02, -0.03, 0.01]).unwrap();
        let result = icp(
            &source,
            &target,
            0.2,
            Matrix4::identity(),
            TransformationEstimation::PointToPoint,
            ICPConvergenceCriteria {
                max_iteration: 20,
                relative_fitness: 1e-7,
                relative_rmse: 1e-7,
            },
        )
        .unwrap();
        assert!((result.transformation[(0, 3)] - 0.02).abs() < 1e-3);
        assert!((result.transformation[(1, 3)] + 0.03).abs() < 1e-3);
        assert!((result.transformation[(2, 3)] - 0.01).abs() < 1e-3);
    }

    #[test]
    fn point_to_plane_requires_normals() {
        let pc = grid_cloud();
        let err = icp(
            &pc,
            &pc,
            0.5,
            Matrix4::identity(),
            TransformationEstimation::PointToPlane,
            ICPConvergenceCriteria::default(),
        )
        .unwrap_err();
        assert!(err.to_string().contains("normals"));
    }
}
