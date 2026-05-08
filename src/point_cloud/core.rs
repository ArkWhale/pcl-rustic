use crate::neighbors::kdtree::KdTreeIndex;
use crate::point_cloud::attribute_value::AttributeValue;
use crate::utils::error::{PointCloudError, Result};
use crate::utils::tensor;
use crate::utils::tensor::Tensor2;
use once_cell::sync::OnceCell;
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LasCoordinateMetadata {
    pub raw_x: Vec<i32>,
    pub raw_y: Vec<i32>,
    pub raw_z: Vec<i32>,
    pub scale: [f64; 3],
    pub offset: [f64; 3],
}

pub struct HighPerformancePointCloud {
    xyz: Tensor2,
    xyz_device: tensor::BackendDevice,
    attributes: HashMap<String, AttributeValue>,
    las_coordinates: Option<LasCoordinateMetadata>,
    kdtree_cache: OnceCell<KdTreeIndex>,
}

impl Clone for HighPerformancePointCloud {
    fn clone(&self) -> Self {
        Self {
            xyz: self.xyz.clone(),
            xyz_device: self.xyz_device.clone(),
            attributes: self.attributes.clone(),
            las_coordinates: self.las_coordinates.clone(),
            kdtree_cache: OnceCell::new(),
        }
    }
}

impl HighPerformancePointCloud {
    pub fn new() -> Self {
        Self {
            xyz: tensor::empty_xyz(),
            xyz_device: tensor::cpu_device(),
            attributes: HashMap::new(),
            las_coordinates: None,
            kdtree_cache: OnceCell::new(),
        }
    }

    pub fn from_tensor_xyz(xyz: Tensor2) -> Result<Self> {
        let device = xyz.device();
        let cols = tensor::tensor2_cols(&xyz);
        if cols != 3 {
            return Err(PointCloudError::TensorShapeError(format!(
                "XYZ tensor must have 3 columns, got {}",
                cols
            )));
        }
        Ok(Self {
            xyz,
            xyz_device: device,
            attributes: HashMap::new(),
            las_coordinates: None,
            kdtree_cache: OnceCell::new(),
        })
    }

    pub fn empty_on_device(device: tensor::BackendDevice) -> Self {
        Self {
            xyz: tensor::empty_xyz(),
            xyz_device: device,
            attributes: HashMap::new(),
            las_coordinates: None,
            kdtree_cache: OnceCell::new(),
        }
    }

    pub fn from_xyz_vec(xyz: Vec<[f32; 3]>) -> Result<Self> {
        let device = tensor::default_device();
        Self::from_xyz_vec_on_device(xyz, &device)
    }

    pub fn from_xyz_vec_on_device(
        xyz: Vec<[f32; 3]>,
        device: &tensor::BackendDevice,
    ) -> Result<Self> {
        if xyz.is_empty() {
            return Err(PointCloudError::TensorShapeError(
                "XYZ data is empty".to_string(),
            ));
        }
        let flat: Vec<f32> = xyz.iter().flat_map(|p| p.iter().copied()).collect();
        let t = tensor::tensor2_from_slice_on_device(&flat, xyz.len(), 3, device)?;
        Ok(Self {
            xyz: t,
            xyz_device: device.clone(),
            attributes: HashMap::new(),
            las_coordinates: None,
            kdtree_cache: OnceCell::new(),
        })
    }

    pub fn point_count(&self) -> usize {
        tensor::tensor2_rows(&self.xyz)
    }

    pub fn xyz_ref(&self) -> &Tensor2 {
        &self.xyz
    }

    #[cfg(test)]
    pub fn xyz_mut(&mut self) -> &mut Tensor2 {
        self.kdtree_cache = OnceCell::new();
        self.las_coordinates = None;
        &mut self.xyz
    }

    pub fn set_xyz(&mut self, xyz: Tensor2) {
        self.kdtree_cache = OnceCell::new();
        self.las_coordinates = None;
        self.xyz_device = xyz.device();
        self.xyz = xyz;
    }

    pub fn get_xyz_vec(&self) -> Vec<[f32; 3]> {
        if self.point_count() == 0 {
            return Vec::new();
        }
        let data = tensor::tensor2_to_vec(&self.xyz);
        data.into_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect()
    }

    pub fn get_xyz_flat(&self) -> Vec<f32> {
        if self.point_count() == 0 {
            return Vec::new();
        }
        let data = self.xyz.to_data();
        data.to_vec::<f32>()
            .expect("Failed to convert XYZ tensor to Vec<f32>")
    }

    pub fn kdtree(&self) -> Result<&KdTreeIndex> {
        self.kdtree_cache
            .get_or_try_init(|| KdTreeIndex::build(self))
    }

    pub fn to_device(&self, device: tensor::BackendDevice) -> Self {
        let xyz = if self.point_count() == 0 {
            tensor::empty_xyz()
        } else {
            self.xyz.clone().to_device(&device)
        };
        Self {
            xyz,
            xyz_device: device,
            attributes: self.attributes.clone(),
            las_coordinates: self.las_coordinates.clone(),
            kdtree_cache: OnceCell::new(),
        }
    }

    pub fn device_name(&self) -> String {
        format!("{:?}", self.xyz_device)
    }

    pub fn xyz_device(&self) -> tensor::BackendDevice {
        self.xyz_device.clone()
    }

    // === Attribute access ===

    pub fn attributes(&self) -> &HashMap<String, AttributeValue> {
        &self.attributes
    }

    pub fn attributes_mut(&mut self) -> &mut HashMap<String, AttributeValue> {
        &mut self.attributes
    }

    pub(crate) fn las_coordinate_metadata(&self) -> Option<&LasCoordinateMetadata> {
        self.las_coordinates.as_ref()
    }

    pub(crate) fn set_las_coordinate_metadata(
        &mut self,
        metadata: LasCoordinateMetadata,
    ) -> Result<()> {
        let point_count = self.point_count();
        if metadata.raw_x.len() != point_count
            || metadata.raw_y.len() != point_count
            || metadata.raw_z.len() != point_count
        {
            return Err(PointCloudError::DimensionMismatch {
                expected: point_count,
                actual: metadata
                    .raw_x
                    .len()
                    .max(metadata.raw_y.len())
                    .max(metadata.raw_z.len()),
            });
        }
        self.las_coordinates = Some(metadata);
        Ok(())
    }

    pub fn get_attribute(&self, name: &str) -> Option<&AttributeValue> {
        self.attributes.get(name)
    }

    pub fn set_attribute(&mut self, name: String, value: AttributeValue) -> Result<()> {
        if value.len() != self.point_count() {
            return Err(PointCloudError::DimensionMismatch {
                expected: self.point_count(),
                actual: value.len(),
            });
        }
        self.attributes.insert(name, value);
        Ok(())
    }

    pub fn remove_attribute(&mut self, name: &str) -> Result<()> {
        if self.attributes.remove(name).is_none() {
            return Err(PointCloudError::InvalidParameter(format!(
                "attribute '{}' does not exist",
                name
            )));
        }
        Ok(())
    }

    pub fn attribute_names(&self) -> Vec<String> {
        self.attributes.keys().cloned().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.point_count() == 0
    }

    // === Convenience accessors for standard attributes ===

    pub fn has_intensity(&self) -> bool {
        self.attributes.contains_key("intensity")
    }

    pub fn has_rgb(&self) -> bool {
        self.attributes.contains_key("red")
            && self.attributes.contains_key("green")
            && self.attributes.contains_key("blue")
    }

    // === Memory ===

    pub fn memory_usage(&self) -> usize {
        let xyz_bytes = self.point_count() * 3 * 4;
        let attr_bytes: usize = self.attributes.values().map(|v| v.memory_usage()).sum();
        xyz_bytes + attr_bytes
    }

    // === Selection primitives (M2) ===

    pub fn select_indices(&self, indices: &[usize]) -> Result<Self> {
        let device = self.xyz_device();
        if indices.is_empty() {
            let mut result = Self::empty_on_device(device);
            for (name, attr) in &self.attributes {
                result
                    .attributes
                    .insert(name.clone(), attr.gather(indices)?);
            }
            if let Some(metadata) = self.gather_las_coordinate_metadata(indices) {
                result.las_coordinates = Some(metadata);
            }
            return Ok(result);
        }
        let n = self.point_count();
        for &idx in indices {
            if idx >= n {
                return Err(PointCloudError::InvalidParameter(format!(
                    "index {} out of bounds for point cloud of size {}",
                    idx, n
                )));
            }
        }

        let xyz_vec = self.get_xyz_vec();
        let new_xyz: Vec<[f32; 3]> = indices.iter().map(|&i| xyz_vec[i]).collect();

        let mut result = Self::from_xyz_vec_on_device(new_xyz, &device)?;
        for (name, attr) in &self.attributes {
            result
                .attributes
                .insert(name.clone(), attr.gather(indices)?);
        }
        if let Some(metadata) = self.gather_las_coordinate_metadata(indices) {
            result.las_coordinates = Some(metadata);
        }
        Ok(result)
    }

    fn gather_las_coordinate_metadata(&self, indices: &[usize]) -> Option<LasCoordinateMetadata> {
        let metadata = self.las_coordinates.as_ref()?;
        Some(LasCoordinateMetadata {
            raw_x: indices.iter().map(|&i| metadata.raw_x[i]).collect(),
            raw_y: indices.iter().map(|&i| metadata.raw_y[i]).collect(),
            raw_z: indices.iter().map(|&i| metadata.raw_z[i]).collect(),
            scale: metadata.scale,
            offset: metadata.offset,
        })
    }

    pub fn select_mask(&self, mask: &[bool]) -> Result<Self> {
        if mask.len() != self.point_count() {
            return Err(PointCloudError::DimensionMismatch {
                expected: self.point_count(),
                actual: mask.len(),
            });
        }
        let indices: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter(|(_, &m)| m)
            .map(|(i, _)| i)
            .collect();
        self.select_indices(&indices)
    }
}

impl Default for HighPerformancePointCloud {
    fn default() -> Self {
        Self::new()
    }
}
