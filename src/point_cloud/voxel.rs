use crate::point_cloud::attribute_value::AttributeValue;
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::{PointCloudError, Result};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;

#[derive(Clone, Debug)]
pub enum DownsampleStrategy {
    RandomSeeded { seed: u64 },
    NearestToCentroid,
    Average,
}

impl HighPerformancePointCloud {
    pub fn voxel_downsample(
        &self,
        voxel_size: f32,
        strategy: &DownsampleStrategy,
    ) -> Result<Self> {
        if self.is_empty() {
            return Err(PointCloudError::InvalidParameter(
                "cannot downsample an empty point cloud".to_string(),
            ));
        }
        if voxel_size <= 0.0 {
            return Err(PointCloudError::InvalidParameter(
                "voxel_size must be > 0".to_string(),
            ));
        }

        let xyz = self.get_xyz_vec();
        let voxel_groups = group_points_by_voxel(&xyz, voxel_size);

        match strategy {
            DownsampleStrategy::RandomSeeded { seed } => {
                self.downsample_random_seeded(&voxel_groups, *seed)
            }
            DownsampleStrategy::NearestToCentroid => {
                self.downsample_nearest_to_centroid(&xyz, &voxel_groups)
            }
            DownsampleStrategy::Average => self.downsample_average(&xyz, &voxel_groups),
        }
    }

    fn downsample_random_seeded(
        &self,
        voxel_groups: &HashMap<[i32; 3], Vec<usize>>,
        seed: u64,
    ) -> Result<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut selected_indices: Vec<usize> = Vec::with_capacity(voxel_groups.len());

        // Sort voxel keys for deterministic iteration order
        let mut keys: Vec<&[i32; 3]> = voxel_groups.keys().collect();
        keys.sort();

        for key in keys {
            let indices = &voxel_groups[key];
            if !indices.is_empty() {
                let pick = indices[rng.gen_range(0..indices.len())];
                selected_indices.push(pick);
            }
        }
        selected_indices.sort_unstable();
        self.select_indices(&selected_indices)
    }

    fn downsample_nearest_to_centroid(
        &self,
        xyz: &[[f32; 3]],
        voxel_groups: &HashMap<[i32; 3], Vec<usize>>,
    ) -> Result<Self> {
        let mut selected_indices: Vec<usize> = Vec::with_capacity(voxel_groups.len());

        for indices in voxel_groups.values() {
            if indices.is_empty() {
                continue;
            }
            let centroid = compute_centroid(xyz, indices);
            let closest = indices
                .iter()
                .min_by(|&&a, &&b| {
                    let da = dist_sq(&xyz[a], &centroid);
                    let db = dist_sq(&xyz[b], &centroid);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap();
            selected_indices.push(closest);
        }
        selected_indices.sort_unstable();
        self.select_indices(&selected_indices)
    }

    fn downsample_average(
        &self,
        xyz: &[[f32; 3]],
        voxel_groups: &HashMap<[i32; 3], Vec<usize>>,
    ) -> Result<Self> {
        let n_voxels = voxel_groups.len();
        let mut new_xyz: Vec<[f32; 3]> = Vec::with_capacity(n_voxels);
        let mut new_attrs: HashMap<String, AttributeValue> = HashMap::new();

        // Initialize attribute accumulators
        for (name, attr) in self.attributes() {
            let empty = match attr.dtype() {
                crate::point_cloud::attribute_value::AttrDType::F32 => {
                    AttributeValue::F32(Vec::with_capacity(n_voxels))
                }
                crate::point_cloud::attribute_value::AttrDType::F64 => {
                    AttributeValue::F64(Vec::with_capacity(n_voxels))
                }
                crate::point_cloud::attribute_value::AttrDType::U8 => {
                    AttributeValue::U8(Vec::with_capacity(n_voxels))
                }
                crate::point_cloud::attribute_value::AttrDType::U16 => {
                    AttributeValue::U16(Vec::with_capacity(n_voxels))
                }
                crate::point_cloud::attribute_value::AttrDType::U32 => {
                    AttributeValue::U32(Vec::with_capacity(n_voxels))
                }
                crate::point_cloud::attribute_value::AttrDType::I32 => {
                    AttributeValue::I32(Vec::with_capacity(n_voxels))
                }
                crate::point_cloud::attribute_value::AttrDType::I64 => {
                    AttributeValue::I64(Vec::with_capacity(n_voxels))
                }
                crate::point_cloud::attribute_value::AttrDType::Bool => {
                    AttributeValue::Bool(Vec::with_capacity(n_voxels))
                }
            };
            new_attrs.insert(name.clone(), empty);
        }

        for indices in voxel_groups.values() {
            if indices.is_empty() {
                continue;
            }
            // Average XYZ
            new_xyz.push(compute_centroid(xyz, indices));

            // Average/mode attributes
            for (name, attr) in self.attributes() {
                let out = new_attrs.get_mut(name).unwrap();
                average_attribute_into(out, attr, indices);
            }
        }

        let mut result = Self::from_xyz_vec(new_xyz)?;
        result.attributes_mut().extend(new_attrs);
        Ok(result)
    }
}

fn group_points_by_voxel(xyz: &[[f32; 3]], voxel_size: f32) -> HashMap<[i32; 3], Vec<usize>> {
    let inv = 1.0 / voxel_size;
    let mut groups: HashMap<[i32; 3], Vec<usize>> = HashMap::new();
    for (idx, point) in xyz.iter().enumerate() {
        let vx = (point[0] * inv).floor() as i32;
        let vy = (point[1] * inv).floor() as i32;
        let vz = (point[2] * inv).floor() as i32;
        groups.entry([vx, vy, vz]).or_default().push(idx);
    }
    groups
}

fn compute_centroid(xyz: &[[f32; 3]], indices: &[usize]) -> [f32; 3] {
    let n = indices.len() as f32;
    let mut c = [0.0f32; 3];
    for &i in indices {
        c[0] += xyz[i][0];
        c[1] += xyz[i][1];
        c[2] += xyz[i][2];
    }
    c[0] /= n;
    c[1] /= n;
    c[2] /= n;
    c
}

fn dist_sq(a: &[f32; 3], b: &[f32; 3]) -> f32 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

fn average_attribute_into(out: &mut AttributeValue, src: &AttributeValue, indices: &[usize]) {
    match (out, src) {
        (AttributeValue::F32(out_vec), AttributeValue::F32(src_vec)) => {
            let mean: f32 =
                indices.iter().map(|&i| src_vec[i]).sum::<f32>() / indices.len() as f32;
            out_vec.push(mean);
        }
        (AttributeValue::F64(out_vec), AttributeValue::F64(src_vec)) => {
            let mean: f64 =
                indices.iter().map(|&i| src_vec[i]).sum::<f64>() / indices.len() as f64;
            out_vec.push(mean);
        }
        // For integer types, use mode (most common value)
        (AttributeValue::U8(out_vec), AttributeValue::U8(src_vec)) => {
            out_vec.push(mode_u8(src_vec, indices));
        }
        (AttributeValue::U16(out_vec), AttributeValue::U16(src_vec)) => {
            out_vec.push(mode_u16(src_vec, indices));
        }
        (AttributeValue::U32(out_vec), AttributeValue::U32(src_vec)) => {
            out_vec.push(mode_u32(src_vec, indices));
        }
        (AttributeValue::I32(out_vec), AttributeValue::I32(src_vec)) => {
            out_vec.push(mode_i32(src_vec, indices));
        }
        (AttributeValue::I64(out_vec), AttributeValue::I64(src_vec)) => {
            out_vec.push(mode_i64(src_vec, indices));
        }
        (AttributeValue::Bool(out_vec), AttributeValue::Bool(src_vec)) => {
            // Majority vote
            let trues = indices.iter().filter(|&&i| src_vec[i]).count();
            out_vec.push(trues > indices.len() / 2);
        }
        _ => {}
    }
}

fn mode_u8(data: &[u8], indices: &[usize]) -> u8 {
    let mut counts: HashMap<u8, usize> = HashMap::new();
    for &i in indices {
        *counts.entry(data[i]).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(v, _)| v)
        .unwrap_or(0)
}

fn mode_u16(data: &[u16], indices: &[usize]) -> u16 {
    let mut counts: HashMap<u16, usize> = HashMap::new();
    for &i in indices {
        *counts.entry(data[i]).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(v, _)| v)
        .unwrap_or(0)
}

fn mode_u32(data: &[u32], indices: &[usize]) -> u32 {
    let mut counts: HashMap<u32, usize> = HashMap::new();
    for &i in indices {
        *counts.entry(data[i]).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(v, _)| v)
        .unwrap_or(0)
}

fn mode_i32(data: &[i32], indices: &[usize]) -> i32 {
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &i in indices {
        *counts.entry(data[i]).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(v, _)| v)
        .unwrap_or(0)
}

fn mode_i64(data: &[i64], indices: &[usize]) -> i64 {
    let mut counts: HashMap<i64, usize> = HashMap::new();
    for &i in indices {
        *counts.entry(data[i]).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(v, _)| v)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_downsample_nearest_to_centroid() {
        let xyz = vec![[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [1.1, 1.1, 1.1], [1.2, 1.2, 1.2]];
        let pc = HighPerformancePointCloud::from_xyz_vec(xyz).unwrap();
        let result = pc
            .voxel_downsample(1.0, &DownsampleStrategy::NearestToCentroid)
            .unwrap();
        assert_eq!(result.point_count(), 2);
    }

    #[test]
    fn test_voxel_downsample_random_seeded_deterministic() {
        let xyz: Vec<[f32; 3]> = (0..100).map(|i| [i as f32 * 0.1, 0.0, 0.0]).collect();
        let pc = HighPerformancePointCloud::from_xyz_vec(xyz).unwrap();

        let r1 = pc
            .voxel_downsample(0.5, &DownsampleStrategy::RandomSeeded { seed: 42 })
            .unwrap();
        let r2 = pc
            .voxel_downsample(0.5, &DownsampleStrategy::RandomSeeded { seed: 42 })
            .unwrap();

        assert_eq!(r1.point_count(), r2.point_count());
        let xyz1 = r1.get_xyz_vec();
        let xyz2 = r2.get_xyz_vec();
        for (a, b) in xyz1.iter().zip(xyz2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_voxel_downsample_average() {
        let xyz = vec![[0.0, 0.0, 0.0], [0.2, 0.2, 0.2], [1.0, 1.0, 1.0]];
        let pc = HighPerformancePointCloud::from_xyz_vec(xyz).unwrap();
        let result = pc
            .voxel_downsample(1.0, &DownsampleStrategy::Average)
            .unwrap();
        assert!(result.point_count() <= 2);
    }
}
