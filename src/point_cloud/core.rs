use crate::neighbors::kdtree::KdTreeIndex;
use crate::point_cloud::attribute_value::AttributeValue;
use crate::utils::error::{PointCloudError, Result};
use crate::utils::tensor;
use crate::utils::tensor::Tensor2;
use once_cell::sync::OnceCell;
use std::collections::HashMap;

pub struct HighPerformancePointCloud {
    xyz: Tensor2,
    attributes: HashMap<String, AttributeValue>,
    kdtree_cache: OnceCell<KdTreeIndex>,
}

impl Clone for HighPerformancePointCloud {
    fn clone(&self) -> Self {
        Self {
            xyz: self.xyz.clone(),
            attributes: self.attributes.clone(),
            kdtree_cache: OnceCell::new(),
        }
    }
}

impl HighPerformancePointCloud {
    pub fn new() -> Self {
        Self {
            xyz: tensor::empty_xyz(),
            attributes: HashMap::new(),
            kdtree_cache: OnceCell::new(),
        }
    }

    pub fn from_tensor_xyz(xyz: Tensor2) -> Result<Self> {
        let cols = tensor::tensor2_cols(&xyz);
        if cols != 3 {
            return Err(PointCloudError::TensorShapeError(format!(
                "XYZ tensor must have 3 columns, got {}",
                cols
            )));
        }
        Ok(Self {
            xyz,
            attributes: HashMap::new(),
            kdtree_cache: OnceCell::new(),
        })
    }

    pub fn from_xyz_vec(xyz: Vec<[f32; 3]>) -> Result<Self> {
        if xyz.is_empty() {
            return Err(PointCloudError::TensorShapeError(
                "XYZ data is empty".to_string(),
            ));
        }
        let flat: Vec<f32> = xyz.iter().flat_map(|p| p.iter().copied()).collect();
        let t = tensor::tensor2_from_slice(&flat, xyz.len(), 3)?;
        Ok(Self {
            xyz: t,
            attributes: HashMap::new(),
            kdtree_cache: OnceCell::new(),
        })
    }

    pub fn from_xyz(xyz: Vec<Vec<f32>>) -> Result<Self> {
        if xyz.is_empty() {
            return Err(PointCloudError::TensorShapeError(
                "XYZ data is empty".to_string(),
            ));
        }
        if !xyz.iter().all(|row| row.len() == 3) {
            return Err(PointCloudError::TensorShapeError(
                "XYZ must have shape [N, 3]".to_string(),
            ));
        }
        let t = tensor::xyz_to_tensor(xyz)?;
        Ok(Self {
            xyz: t,
            attributes: HashMap::new(),
            kdtree_cache: OnceCell::new(),
        })
    }

    pub fn point_count(&self) -> usize {
        tensor::tensor2_rows(&self.xyz)
    }

    pub fn xyz_ref(&self) -> &Tensor2 {
        &self.xyz
    }

    pub fn xyz_mut(&mut self) -> &mut Tensor2 {
        self.kdtree_cache = OnceCell::new();
        &mut self.xyz
    }

    pub fn get_xyz_vec(&self) -> Vec<[f32; 3]> {
        let data = tensor::tensor2_to_vec(&self.xyz);
        data.into_iter()
            .map(|row| [row[0], row[1], row[2]])
            .collect()
    }

    pub fn get_xyz_flat(&self) -> Vec<f32> {
        let data = self.xyz.to_data();
        data.to_vec::<f32>()
            .expect("Failed to convert XYZ tensor to Vec<f32>")
    }

    pub fn kdtree(&self) -> Result<&KdTreeIndex> {
        self.kdtree_cache
            .get_or_try_init(|| KdTreeIndex::build(self))
    }

    pub fn to_device(&self, device: tensor::BackendDevice) -> Self {
        Self {
            xyz: self.xyz.clone().to_device(&device),
            attributes: self.attributes.clone(),
            kdtree_cache: OnceCell::new(),
        }
    }

    pub fn device_name(&self) -> String {
        format!("{:?}", self.xyz.device())
    }

    // === Attribute access ===

    pub fn attributes(&self) -> &HashMap<String, AttributeValue> {
        &self.attributes
    }

    pub fn attributes_mut(&mut self) -> &mut HashMap<String, AttributeValue> {
        &mut self.attributes
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

    pub fn has_normals(&self) -> bool {
        self.attributes.contains_key("nx")
            && self.attributes.contains_key("ny")
            && self.attributes.contains_key("nz")
    }

    pub fn get_intensity_f32(&self) -> Option<&Vec<f32>> {
        self.attributes.get("intensity").and_then(|v| v.as_f32())
    }

    pub fn get_classification_u8(&self) -> Option<&Vec<u8>> {
        self.attributes
            .get("classification")
            .and_then(|v| v.as_u8())
    }

    // === Memory ===

    pub fn memory_usage(&self) -> usize {
        let xyz_bytes = self.point_count() * 3 * 4;
        let attr_bytes: usize = self.attributes.values().map(|v| v.memory_usage()).sum();
        xyz_bytes + attr_bytes
    }

    // === Selection primitives (M2) ===

    pub fn select_indices(&self, indices: &[usize]) -> Result<Self> {
        if indices.is_empty() {
            let mut result = Self::new();
            for (name, attr) in &self.attributes {
                result
                    .attributes
                    .insert(name.clone(), attr.gather(indices)?);
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

        let mut result = Self::from_xyz_vec(new_xyz)?;
        for (name, attr) in &self.attributes {
            result
                .attributes
                .insert(name.clone(), attr.gather(indices)?);
        }
        Ok(result)
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
