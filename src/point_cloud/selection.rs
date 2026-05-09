use crate::point_cloud::attribute_value::AttributeValue;
use crate::point_cloud::core::{HighPerformancePointCloud, LasCoordinateMetadata};
use crate::utils::error::{PointCloudError, Result};
use crate::utils::tensor;
use nalgebra::{Matrix3, SymmetricEigen, Vector3};
use std::collections::{HashMap, HashSet};

use crate::ConcatPolicy;

impl HighPerformancePointCloud {
    // === Feature-based selection ===

    pub fn select_where(
        &self,
        name: &str,
        op: &str,
        values: &[f64],
        inclusive: bool,
    ) -> Result<Self> {
        let attr = self.get_attribute(name).ok_or_else(|| {
            PointCloudError::InvalidParameter(format!("point cloud has no '{}' attribute", name))
        })?;
        if values.is_empty() {
            return Err(PointCloudError::InvalidParameter(
                "values must not be empty".to_string(),
            ));
        }
        let mask = match attr {
            AttributeValue::F32(data) => {
                numeric_mask(data.iter().map(|&v| v as f64), op, values, inclusive)?
            }
            AttributeValue::F64(data) => numeric_mask(data.iter().copied(), op, values, inclusive)?,
            AttributeValue::U8(data) => {
                numeric_mask(data.iter().map(|&v| v as f64), op, values, inclusive)?
            }
            AttributeValue::U16(data) => {
                numeric_mask(data.iter().map(|&v| v as f64), op, values, inclusive)?
            }
            AttributeValue::U32(data) => {
                numeric_mask(data.iter().map(|&v| v as f64), op, values, inclusive)?
            }
            AttributeValue::U64(data) => {
                numeric_mask(data.iter().map(|&v| v as f64), op, values, inclusive)?
            }
            AttributeValue::I16(data) => {
                numeric_mask(data.iter().map(|&v| v as f64), op, values, inclusive)?
            }
            AttributeValue::I32(data) => {
                numeric_mask(data.iter().map(|&v| v as f64), op, values, inclusive)?
            }
            AttributeValue::I64(data) => {
                numeric_mask(data.iter().map(|&v| v as f64), op, values, inclusive)?
            }
            AttributeValue::Bool(data) => bool_mask(data, op, values)?,
            AttributeValue::F32x6(_) => {
                return Err(PointCloudError::InvalidParameter(
                    "select_where does not support float32[6] attributes".to_string(),
                ))
            }
        };
        self.select_mask(&mask)
    }

    pub fn select_by_classification(&self, codes: &[u8]) -> Result<Self> {
        let attr = self.get_attribute("classification").ok_or_else(|| {
            PointCloudError::InvalidParameter(
                "point cloud has no 'classification' attribute".to_string(),
            )
        })?;
        let class_data = attr.as_u8().ok_or_else(|| {
            PointCloudError::InvalidParameter("classification attribute is not u8".to_string())
        })?;
        let code_set: HashSet<u8> = codes.iter().copied().collect();
        let mask: Vec<bool> = class_data.iter().map(|c| code_set.contains(c)).collect();
        self.select_mask(&mask)
    }

    pub fn select_intensity_range(&self, lo: f32, hi: f32) -> Result<Self> {
        let attr = self.get_attribute("intensity").ok_or_else(|| {
            PointCloudError::InvalidParameter(
                "point cloud has no 'intensity' attribute".to_string(),
            )
        })?;
        let intensity_data = attr.as_f32().ok_or_else(|| {
            PointCloudError::InvalidParameter("intensity attribute is not f32".to_string())
        })?;
        let mask: Vec<bool> = intensity_data.iter().map(|&v| v >= lo && v <= hi).collect();
        self.select_mask(&mask)
    }

    pub fn select_return_number(&self, n: u8) -> Result<Self> {
        let attr = self.get_attribute("return_number").ok_or_else(|| {
            PointCloudError::InvalidParameter(
                "point cloud has no 'return_number' attribute".to_string(),
            )
        })?;
        let data = attr.as_u8().ok_or_else(|| {
            PointCloudError::InvalidParameter("return_number attribute is not u8".to_string())
        })?;
        let mask: Vec<bool> = data.iter().map(|&v| v == n).collect();
        self.select_mask(&mask)
    }

    pub fn select_elevation_range(&self, lo: f32, hi: f32) -> Result<Self> {
        let xyz = self.get_xyz_vec();
        let mask: Vec<bool> = xyz.iter().map(|p| p[2] >= lo && p[2] <= hi).collect();
        self.select_mask(&mask)
    }

    // === Spatial selection ===

    pub fn crop_aabb(&self, min: [f32; 3], max: [f32; 3]) -> Result<Self> {
        let xyz = self.get_xyz_vec();
        let mask: Vec<bool> = xyz
            .iter()
            .map(|p| {
                p[0] >= min[0]
                    && p[0] <= max[0]
                    && p[1] >= min[1]
                    && p[1] <= max[1]
                    && p[2] >= min[2]
                    && p[2] <= max[2]
            })
            .collect();
        self.select_mask(&mask)
    }

    pub fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        let xyz = self.get_xyz_vec();
        if xyz.is_empty() {
            return ([0.0; 3], [0.0; 3]);
        }
        let mut min = xyz[0];
        let mut max = xyz[0];
        for p in &xyz[1..] {
            for i in 0..3 {
                if p[i] < min[i] {
                    min[i] = p[i];
                }
                if p[i] > max[i] {
                    max[i] = p[i];
                }
            }
        }
        (min, max)
    }

    pub fn crop_obb(
        &self,
        center: [f32; 3],
        extents: [f32; 3],
        rotation: [[f32; 3]; 3],
    ) -> Result<Self> {
        if extents.iter().any(|&v| v < 0.0 || !v.is_finite()) {
            return Err(PointCloudError::InvalidParameter(
                "extents must be finite and non-negative".to_string(),
            ));
        }
        let axes = Matrix3::from_columns(&[
            Vector3::new(rotation[0][0], rotation[1][0], rotation[2][0]),
            Vector3::new(rotation[0][1], rotation[1][1], rotation[2][1]),
            Vector3::new(rotation[0][2], rotation[1][2], rotation[2][2]),
        ]);
        let center = Vector3::new(center[0], center[1], center[2]);
        let half = Vector3::new(extents[0] * 0.5, extents[1] * 0.5, extents[2] * 0.5);
        let mask: Vec<bool> = self
            .get_xyz_vec()
            .iter()
            .map(|p| {
                let local = axes.transpose() * (Vector3::new(p[0], p[1], p[2]) - center);
                local[0].abs() <= half[0] && local[1].abs() <= half[1] && local[2].abs() <= half[2]
            })
            .collect();
        self.select_mask(&mask)
    }

    pub fn obb(&self) -> ([f32; 3], [f32; 3], [[f32; 3]; 3]) {
        let xyz = self.get_xyz_vec();
        if xyz.is_empty() {
            return ([0.0; 3], [0.0; 3], identity_rotation());
        }
        if xyz.len() < 3 {
            let (min, max) = self.aabb();
            let center = [
                (min[0] + max[0]) * 0.5,
                (min[1] + max[1]) * 0.5,
                (min[2] + max[2]) * 0.5,
            ];
            let extents = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
            return (center, extents, identity_rotation());
        }

        let centroid = xyz.iter().fold(Vector3::zeros(), |acc, p| {
            acc + Vector3::new(p[0], p[1], p[2])
        }) / xyz.len() as f32;
        let mut cov = Matrix3::zeros();
        for p in &xyz {
            let d = Vector3::new(p[0], p[1], p[2]) - centroid;
            cov += d * d.transpose();
        }
        cov /= xyz.len() as f32;

        let eigen = SymmetricEigen::new(cov);
        let mut axes = eigen.eigenvectors;
        if axes.determinant() < 0.0 {
            axes.column_mut(2).scale_mut(-1.0);
        }

        let mut min = [f32::INFINITY; 3];
        let mut max = [f32::NEG_INFINITY; 3];
        for p in &xyz {
            let local = axes.transpose() * (Vector3::new(p[0], p[1], p[2]) - centroid);
            for axis in 0..3 {
                min[axis] = min[axis].min(local[axis]);
                max[axis] = max[axis].max(local[axis]);
            }
        }
        let local_center = Vector3::new(
            (min[0] + max[0]) * 0.5,
            (min[1] + max[1]) * 0.5,
            (min[2] + max[2]) * 0.5,
        );
        let world_center = centroid + axes * local_center;
        let extents = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];

        (
            [world_center[0], world_center[1], world_center[2]],
            extents,
            [
                [axes[(0, 0)], axes[(0, 1)], axes[(0, 2)]],
                [axes[(1, 0)], axes[(1, 1)], axes[(1, 2)]],
                [axes[(2, 0)], axes[(2, 1)], axes[(2, 2)]],
            ],
        )
    }

    // === Concatenation ===

    pub fn concatenate(clouds: &[&Self], policy: ConcatPolicy) -> Result<Self> {
        if clouds.is_empty() {
            return Ok(Self::new());
        }
        if clouds.len() == 1 {
            return Ok(clouds[0].clone());
        }

        let result_device = clouds
            .iter()
            .find(|cloud| !cloud.is_empty())
            .unwrap_or(&clouds[0])
            .xyz_device();
        for (i, cloud) in clouds.iter().enumerate() {
            if !cloud.is_empty() && cloud.xyz_device() != result_device {
                return Err(PointCloudError::InvalidParameter(format!(
                    "cannot concatenate non-empty cloud {} on device {:?} with result device {:?}",
                    i,
                    cloud.xyz_device(),
                    result_device
                )));
            }
        }

        // Collect all XYZ
        let total_points: usize = clouds.iter().map(|c| c.point_count()).sum();
        let mut all_xyz: Vec<f32> = Vec::with_capacity(total_points * 3);
        for cloud in clouds {
            all_xyz.extend(cloud.get_xyz_flat());
        }
        let xyz_tensor = if total_points == 0 {
            None
        } else {
            Some(tensor::tensor2_from_slice_on_device(
                &all_xyz,
                total_points,
                3,
                &result_device,
            )?)
        };

        // Determine which attributes to include
        let attr_sets: Vec<HashSet<String>> = clouds
            .iter()
            .map(|c| c.attribute_names().into_iter().collect())
            .collect();

        let target_attrs: HashSet<String> = match &policy {
            ConcatPolicy::Strict => {
                let first = &attr_sets[0];
                for (i, set) in attr_sets.iter().enumerate().skip(1) {
                    if set != first {
                        return Err(PointCloudError::InvalidParameter(format!(
                            "ConcatPolicy::Strict: cloud {} has different attributes than cloud 0",
                            i
                        )));
                    }
                }
                first.clone()
            }
            ConcatPolicy::Union => {
                let mut all: HashSet<String> = HashSet::new();
                for set in &attr_sets {
                    all.extend(set.iter().cloned());
                }
                all
            }
            ConcatPolicy::Intersection => {
                let mut common = attr_sets[0].clone();
                for set in attr_sets.iter().skip(1) {
                    common.retain(|k| set.contains(k));
                }
                common
            }
        };

        // Also verify dtypes match in Strict mode
        if matches!(policy, ConcatPolicy::Strict) {
            for attr_name in &target_attrs {
                let first_dtype = clouds[0].get_attribute(attr_name).unwrap().dtype();
                for (i, cloud) in clouds.iter().enumerate().skip(1) {
                    if let Some(attr) = cloud.get_attribute(attr_name) {
                        if attr.dtype() != first_dtype {
                            return Err(PointCloudError::InvalidParameter(format!(
                                "ConcatPolicy::Strict: attribute '{}' has dtype {} in cloud 0 but {} in cloud {}",
                                attr_name, first_dtype, attr.dtype(), i
                            )));
                        }
                    }
                }
            }
        }

        // Concatenate attributes
        let mut new_attrs: HashMap<String, AttributeValue> = HashMap::new();
        for attr_name in &target_attrs {
            // Determine the dtype from the first cloud that has this attribute
            let dtype = clouds
                .iter()
                .find_map(|c| c.get_attribute(attr_name).map(|a| a.dtype()))
                .unwrap();

            let parts: Vec<AttributeValue> = clouds
                .iter()
                .map(|c| {
                    c.get_attribute(attr_name)
                        .cloned()
                        .unwrap_or_else(|| AttributeValue::zeros(&dtype, c.point_count()))
                })
                .collect();

            let refs: Vec<&AttributeValue> = parts.iter().collect();
            let concatenated = AttributeValue::concatenate(&refs)?;
            new_attrs.insert(attr_name.clone(), concatenated);
        }

        let mut result = match xyz_tensor {
            Some(xyz_tensor) => Self::from_tensor_xyz(xyz_tensor)?,
            None => Self::empty_on_device(result_device),
        };
        *result.attributes_mut() = new_attrs;
        if let Some(metadata) = concatenate_las_coordinate_metadata(clouds) {
            result.set_las_coordinate_metadata(metadata)?;
        }
        Ok(result)
    }
}

fn concatenate_las_coordinate_metadata(
    clouds: &[&HighPerformancePointCloud],
) -> Option<LasCoordinateMetadata> {
    let mut non_empty = clouds.iter().copied().filter(|cloud| !cloud.is_empty());
    let first = non_empty.next()?;
    let first_metadata = first.las_coordinate_metadata()?;
    let mut raw_x = first_metadata.raw_x.clone();
    let mut raw_y = first_metadata.raw_y.clone();
    let mut raw_z = first_metadata.raw_z.clone();

    for cloud in non_empty {
        let metadata = cloud.las_coordinate_metadata()?;
        if metadata.scale != first_metadata.scale || metadata.offset != first_metadata.offset {
            return None;
        }
        raw_x.extend_from_slice(&metadata.raw_x);
        raw_y.extend_from_slice(&metadata.raw_y);
        raw_z.extend_from_slice(&metadata.raw_z);
    }

    Some(LasCoordinateMetadata {
        raw_x,
        raw_y,
        raw_z,
        scale: first_metadata.scale,
        offset: first_metadata.offset,
    })
}

fn numeric_mask<I>(data: I, op: &str, values: &[f64], inclusive: bool) -> Result<Vec<bool>>
where
    I: Iterator<Item = f64>,
{
    match op {
        "eq" => Ok(data.map(|v| v == values[0]).collect()),
        "in" => Ok(data.map(|v| values.contains(&v)).collect()),
        "range" => {
            if values.len() < 2 {
                return Err(PointCloudError::InvalidParameter(
                    "range requires [lo, hi] values".to_string(),
                ));
            }
            let lo = values[0];
            let hi = values[1];
            Ok(data
                .map(|v| {
                    if inclusive {
                        v >= lo && v <= hi
                    } else {
                        v > lo && v < hi
                    }
                })
                .collect())
        }
        "gt" => Ok(data.map(|v| v > values[0]).collect()),
        "ge" => Ok(data.map(|v| v >= values[0]).collect()),
        "lt" => Ok(data.map(|v| v < values[0]).collect()),
        "le" => Ok(data.map(|v| v <= values[0]).collect()),
        _ => Err(PointCloudError::InvalidParameter(
            "op must be one of 'eq', 'in', 'range', 'gt', 'ge', 'lt', or 'le'".to_string(),
        )),
    }
}

fn bool_mask(data: &[bool], op: &str, values: &[f64]) -> Result<Vec<bool>> {
    match op {
        "eq" => {
            let expected = values[0] != 0.0;
            Ok(data.iter().map(|&v| v == expected).collect())
        }
        "in" => {
            let accepts_true = values.iter().any(|&v| v != 0.0);
            let accepts_false = values.contains(&0.0);
            Ok(data
                .iter()
                .map(|&v| if v { accepts_true } else { accepts_false })
                .collect())
        }
        _ => Err(PointCloudError::InvalidParameter(
            "bool select_where only supports 'eq' and 'in'".to_string(),
        )),
    }
}

fn identity_rotation() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tensor;

    #[test]
    fn selection_and_concat_preserve_cpu_device() {
        let mut pc =
            HighPerformancePointCloud::from_xyz_vec(vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
                .unwrap()
                .to_device(tensor::cpu_device());
        pc.set_attribute("classification".to_string(), AttributeValue::U8(vec![2, 6]))
            .unwrap();
        let device = pc.xyz_device();

        let selected = pc.select_indices(&[0]).unwrap();
        assert_eq!(selected.xyz_device(), device);
        let empty = pc.select_indices(&[]).unwrap();
        assert_eq!(empty.xyz_device(), device);
        let concatenated =
            HighPerformancePointCloud::concatenate(&[&selected, &empty], ConcatPolicy::Strict)
                .unwrap();
        assert_eq!(concatenated.xyz_device(), device);
    }

    #[test]
    fn concatenate_rejects_mixed_non_empty_devices_when_gpu_available() {
        if !tensor::has_wgpu_device() {
            return;
        }
        let cpu = HighPerformancePointCloud::from_xyz_vec(vec![[0.0, 0.0, 0.0]])
            .unwrap()
            .to_device(tensor::cpu_device());
        let gpu = HighPerformancePointCloud::from_xyz_vec(vec![[1.0, 0.0, 0.0]])
            .unwrap()
            .to_device(tensor::gpu_device().unwrap());

        let err = match HighPerformancePointCloud::concatenate(&[&cpu, &gpu], ConcatPolicy::Strict)
        {
            Ok(_) => panic!("mixed-device concatenation should fail"),
            Err(err) => err,
        };
        assert!(err
            .to_string()
            .contains("cannot concatenate non-empty cloud"));
    }
}
