use crate::point_cloud::attribute_value::AttributeValue;
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::{PointCloudError, Result};
use crate::utils::tensor;
use std::collections::{HashMap, HashSet};

use crate::ConcatPolicy;

impl HighPerformancePointCloud {
    // === Feature-based selection ===

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

    // === Concatenation ===

    pub fn concatenate(clouds: &[&Self], policy: ConcatPolicy) -> Result<Self> {
        if clouds.is_empty() {
            return Ok(Self::new());
        }
        if clouds.len() == 1 {
            return Ok(clouds[0].clone());
        }

        // Collect all XYZ
        let total_points: usize = clouds.iter().map(|c| c.point_count()).sum();
        let mut all_xyz: Vec<f32> = Vec::with_capacity(total_points * 3);
        for cloud in clouds {
            all_xyz.extend(cloud.get_xyz_flat());
        }
        let xyz_tensor = tensor::tensor2_from_slice(&all_xyz, total_points, 3)?;

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

        let mut result = Self::from_tensor_xyz(xyz_tensor)?;
        *result.attributes_mut() = new_attrs;
        Ok(result)
    }
}
