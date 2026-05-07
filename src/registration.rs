use crate::point_cloud::{attribute_value::AttributeValue, core::HighPerformancePointCloud};
use crate::utils::error::{PointCloudError, Result};
use nalgebra::{Matrix3, Matrix4, SMatrix, SVector, Vector3, Vector4, SVD};

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
    match &estimation {
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

        let delta = match &estimation {
            TransformationEstimation::PointToPoint => {
                estimate_point_to_point_delta(&transformed, &target_xyz, &correspondences)?
            }
            TransformationEstimation::PointToPlane => {
                estimate_point_to_plane_delta(&transformed, target, &correspondences)?
            }
            TransformationEstimation::Generalized { epsilon } => estimate_generalized_delta(
                &transformed,
                source,
                target,
                &correspondences,
                *epsilon,
            )?,
        };
        transformation = delta * transformation;
        let updated = transform_points(&source_xyz, &transformation);
        let updated_correspondences =
            find_correspondences(target, &updated, max_correspondence_distance)?;
        last = metrics(source_xyz.len(), transformation, updated_correspondences)?;

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

fn estimate_point_to_plane_delta(
    transformed_source: &[[f32; 3]],
    target: &HighPerformancePointCloud,
    correspondences: &[(u64, u64, f32)],
) -> Result<Matrix4<f32>> {
    let target_xyz = target.get_xyz_vec();
    let nx = target
        .get_attribute("nx")
        .and_then(|attr| attr.as_f32())
        .ok_or_else(|| {
            PointCloudError::InvalidParameter("point-to-plane ICP requires target nx".to_string())
        })?;
    let ny = target
        .get_attribute("ny")
        .and_then(|attr| attr.as_f32())
        .ok_or_else(|| {
            PointCloudError::InvalidParameter("point-to-plane ICP requires target ny".to_string())
        })?;
    let nz = target
        .get_attribute("nz")
        .and_then(|attr| attr.as_f32())
        .ok_or_else(|| {
            PointCloudError::InvalidParameter("point-to-plane ICP requires target nz".to_string())
        })?;

    let mut normal_matrix = SMatrix::<f32, 6, 6>::zeros();
    let mut rhs = SVector::<f32, 6>::zeros();
    for (source_idx, target_idx, _) in correspondences {
        let source_point = point_vec(transformed_source[*source_idx as usize]);
        let target_point = point_vec(target_xyz[*target_idx as usize]);
        let normal = Vector3::new(
            nx[*target_idx as usize],
            ny[*target_idx as usize],
            nz[*target_idx as usize],
        );
        if normal.norm_squared() <= f32::EPSILON {
            continue;
        }
        let cross = source_point.cross(&normal);
        let jacobian = SVector::<f32, 6>::new(
            cross[0], cross[1], cross[2], normal[0], normal[1], normal[2],
        );
        let residual = normal.dot(&(target_point - source_point));
        normal_matrix += jacobian * jacobian.transpose();
        rhs += jacobian * residual;
    }

    let svd = SVD::new(normal_matrix, true, true);
    let Ok(delta) = svd.solve(&rhs, 1e-6) else {
        return Err(PointCloudError::InvalidParameter(
            "point-to-plane normal equations are singular".to_string(),
        ));
    };
    Ok(delta_vector_to_transform(delta))
}

fn estimate_generalized_delta(
    transformed_source: &[[f32; 3]],
    source: &HighPerformancePointCloud,
    target: &HighPerformancePointCloud,
    correspondences: &[(u64, u64, f32)],
    epsilon: f32,
) -> Result<Matrix4<f32>> {
    let target_xyz = target.get_xyz_vec();
    let source_covariances = source
        .get_attribute("covariance")
        .and_then(|attr| attr.as_f32x6())
        .ok_or_else(|| {
            PointCloudError::InvalidParameter(
                "generalized ICP requires source covariance attributes".to_string(),
            )
        })?;
    let target_covariances = target
        .get_attribute("covariance")
        .and_then(|attr| attr.as_f32x6())
        .ok_or_else(|| {
            PointCloudError::InvalidParameter(
                "generalized ICP requires target covariance attributes".to_string(),
            )
        })?;

    let mut normal_matrix = SMatrix::<f32, 6, 6>::zeros();
    let mut rhs = SVector::<f32, 6>::zeros();
    for (source_idx, target_idx, _) in correspondences {
        let source_point = point_vec(transformed_source[*source_idx as usize]);
        let target_point = point_vec(target_xyz[*target_idx as usize]);
        let residual = target_point - source_point;
        let covariance = packed_covariance(source_covariances[*source_idx as usize])
            + packed_covariance(target_covariances[*target_idx as usize])
            + Matrix3::identity() * epsilon.max(1e-9);
        let information = covariance
            .try_inverse()
            .unwrap_or_else(|| Matrix3::identity() / epsilon.max(1e-6));
        let skew = skew_matrix(source_point);
        let mut jacobian = SMatrix::<f32, 3, 6>::zeros();
        jacobian.fixed_view_mut::<3, 3>(0, 0).copy_from(&(-skew));
        jacobian
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&Matrix3::identity());

        normal_matrix += jacobian.transpose() * information * jacobian;
        rhs += jacobian.transpose() * information * residual;
    }

    let svd = SVD::new(normal_matrix, true, true);
    let Ok(delta) = svd.solve(&rhs, 1e-6) else {
        return Err(PointCloudError::InvalidParameter(
            "GICP normal equations are singular".to_string(),
        ));
    };
    Ok(delta_vector_to_transform(delta))
}

fn packed_covariance(row: [f32; 6]) -> Matrix3<f32> {
    Matrix3::new(
        row[0], row[1], row[2], row[1], row[3], row[4], row[2], row[4], row[5],
    )
}

fn skew_matrix(v: Vector3<f32>) -> Matrix3<f32> {
    Matrix3::new(0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0)
}

fn delta_vector_to_transform(delta: SVector<f32, 6>) -> Matrix4<f32> {
    let rotation = small_angle_rotation(Vector3::new(delta[0], delta[1], delta[2]));
    let mut out = Matrix4::identity();
    out.fixed_view_mut::<3, 3>(0, 0).copy_from(&rotation);
    out[(0, 3)] = delta[3];
    out[(1, 3)] = delta[4];
    out[(2, 3)] = delta[5];
    out
}

fn small_angle_rotation(omega: Vector3<f32>) -> Matrix3<f32> {
    let theta = omega.norm();
    if theta <= 1e-8 {
        return Matrix3::identity();
    }
    let axis = omega / theta;
    let k = Matrix3::new(
        0.0, -axis[2], axis[1], axis[2], 0.0, -axis[0], -axis[1], axis[0], 0.0,
    );
    Matrix3::identity() + k * theta.sin() + (k * k) * (1.0 - theta.cos())
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
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

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
    fn known_rotation_10k_is_recovered() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let xyz: Vec<[f32; 3]> = (0..10_000)
            .map(|_| {
                [
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                    rng.gen_range(-1.0..1.0),
                ]
            })
            .collect();
        let target = HighPerformancePointCloud::from_xyz_vec(xyz).unwrap();
        let angle = 0.03f32;
        let c = angle.cos();
        let s = angle.sin();
        let rotation = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]];
        let translation = [0.02, -0.015, 0.01];
        let source = target.rigid_transform(&rotation, translation).unwrap();

        let mut t_gt = Matrix4::identity();
        t_gt.fixed_view_mut::<3, 3>(0, 0).copy_from(&Matrix3::new(
            rotation[0][0],
            rotation[0][1],
            rotation[0][2],
            rotation[1][0],
            rotation[1][1],
            rotation[1][2],
            rotation[2][0],
            rotation[2][1],
            rotation[2][2],
        ));
        t_gt[(0, 3)] = translation[0];
        t_gt[(1, 3)] = translation[1];
        t_gt[(2, 3)] = translation[2];

        let result = icp(
            &source,
            &target,
            0.15,
            Matrix4::identity(),
            TransformationEstimation::PointToPoint,
            ICPConvergenceCriteria {
                max_iteration: 30,
                relative_fitness: 1e-7,
                relative_rmse: 1e-7,
            },
        )
        .unwrap();

        assert!(result.fitness > 0.99);
        assert!((result.transformation * t_gt - Matrix4::identity()).norm() < 1e-3);
    }

    #[test]
    fn single_iteration_metrics_match_returned_transform() {
        let source = sparse_cloud();
        let target = source.translate([0.02, -0.03, 0.01]).unwrap();
        let result = icp(
            &source,
            &target,
            0.2,
            Matrix4::identity(),
            TransformationEstimation::PointToPoint,
            ICPConvergenceCriteria {
                max_iteration: 1,
                relative_fitness: 0.0,
                relative_rmse: 0.0,
            },
        )
        .unwrap();

        assert_eq!(result.fitness, 1.0);
        assert!(result.inlier_rmse < 1e-4);
        let score = evaluate(&source, &target, 0.2, result.transformation).unwrap();
        assert!((result.inlier_rmse - score.inlier_rmse).abs() < 1e-6);
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

    #[test]
    fn point_to_plane_recovers_single_normal_translation() {
        let source = HighPerformancePointCloud::from_xyz_vec(vec![[0.0, 0.0, 1.0]]).unwrap();
        let mut target = HighPerformancePointCloud::from_xyz_vec(vec![[0.0, 0.0, 0.0]]).unwrap();
        target
            .set_attribute("nx".to_string(), AttributeValue::F32(vec![0.0]))
            .unwrap();
        target
            .set_attribute("ny".to_string(), AttributeValue::F32(vec![0.0]))
            .unwrap();
        target
            .set_attribute("nz".to_string(), AttributeValue::F32(vec![1.0]))
            .unwrap();

        let result = icp(
            &source,
            &target,
            2.0,
            Matrix4::identity(),
            TransformationEstimation::PointToPlane,
            ICPConvergenceCriteria {
                max_iteration: 10,
                relative_fitness: 1e-7,
                relative_rmse: 1e-7,
            },
        )
        .unwrap();

        assert_eq!(result.fitness, 1.0);
        assert!(result.inlier_rmse < 1e-4);
        assert!((result.transformation[(2, 3)] + 1.0).abs() < 1e-4);
    }

    #[test]
    fn generalized_icp_recovers_covariance_weighted_translation() {
        let source = HighPerformancePointCloud::from_xyz_vec(vec![[0.0, 0.0, 1.0]]).unwrap();
        let mut target = HighPerformancePointCloud::from_xyz_vec(vec![[0.0, 0.0, 0.0]]).unwrap();
        let covariance = AttributeValue::F32x6(vec![[1.0, 0.0, 0.0, 1.0, 0.0, 0.001]]);
        let mut source = source;
        source
            .set_attribute("covariance".to_string(), covariance.clone())
            .unwrap();
        target
            .set_attribute("covariance".to_string(), covariance)
            .unwrap();

        let result = icp(
            &source,
            &target,
            2.0,
            Matrix4::identity(),
            TransformationEstimation::Generalized { epsilon: 1e-3 },
            ICPConvergenceCriteria {
                max_iteration: 10,
                relative_fitness: 1e-7,
                relative_rmse: 1e-7,
            },
        )
        .unwrap();

        assert_eq!(result.fitness, 1.0);
        assert!(result.inlier_rmse < 1e-4);
        assert!((result.transformation[(2, 3)] + 1.0).abs() < 1e-4);
    }
}
