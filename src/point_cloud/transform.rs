use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::Result;
use crate::utils::tensor;
use crate::utils::tensor::Backend;
use burn::tensor::{Tensor, TensorData};

impl HighPerformancePointCloud {
    pub fn transform(&self, matrix: &[[f32; 4]; 4]) -> Result<Self> {
        if self.point_count() == 0 {
            return Ok(self.clone());
        }
        let device = self.xyz_device();
        let flat: Vec<f32> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
        let mat_tensor = tensor::tensor2_from_slice_on_device(&flat, 4, 4, &device)?;
        let mat_t = mat_tensor.transpose();

        let xyz = self.xyz_ref().clone();
        let n = self.point_count();

        // Build homogeneous coordinates [N, 4]
        let ones_data = TensorData::from(vec![1.0f32; n].as_slice());
        let ones = Tensor::<Backend, 1>::from_data(ones_data, &device).reshape([n, 1]);
        let homo = Tensor::cat(vec![xyz, ones], 1); // [N, 4]
        let transformed = homo.matmul(mat_t); // [N, 4]

        // Extract XYZ and divide by w
        let new_xyz = transformed.clone().slice([0..n, 0..3]);
        let w = transformed.slice([0..n, 3..4]); // [N, 1]
        let new_xyz = new_xyz / w;

        let mut result = self.clone();
        result.set_xyz(new_xyz);
        Ok(result)
    }

    pub fn transform_3x3(&self, matrix: &[[f32; 3]; 3]) -> Result<Self> {
        if self.point_count() == 0 {
            return Ok(self.clone());
        }
        let device = self.xyz_device();
        let flat: Vec<f32> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
        let mat_tensor = tensor::tensor2_from_slice_on_device(&flat, 3, 3, &device)?;
        let mat_t = mat_tensor.transpose();

        let xyz = self.xyz_ref().clone();
        let new_xyz = xyz.matmul(mat_t);

        let mut result = self.clone();
        result.set_xyz(new_xyz);
        Ok(result)
    }

    pub fn translate(&self, t: [f32; 3]) -> Result<Self> {
        if self.point_count() == 0 {
            return Ok(self.clone());
        }
        let device = self.xyz_device();
        let translation_data = TensorData::from(t.as_slice());
        let translation_tensor =
            Tensor::<Backend, 1>::from_data(translation_data, &device).reshape([1, 3]);

        let mut result = self.clone();
        let new_xyz = self.xyz_ref().clone() + translation_tensor;
        result.set_xyz(new_xyz);
        Ok(result)
    }

    pub fn scale(&self, s: f32, center: Option<[f32; 3]>) -> Result<Self> {
        if self.point_count() == 0 {
            return Ok(self.clone());
        }
        let device = self.xyz_device();
        let center = center.unwrap_or_else(|| self.compute_centroid_xyz());
        let center_data = TensorData::from(center.as_slice());
        let center_tensor = Tensor::<Backend, 1>::from_data(center_data, &device).reshape([1, 3]);

        let xyz = self.xyz_ref().clone();
        let centered = xyz - center_tensor.clone();
        let scaled = centered * s;
        let new_xyz = scaled + center_tensor;

        let mut result = self.clone();
        result.set_xyz(new_xyz);
        Ok(result)
    }

    pub fn rotate(&self, r: &[[f32; 3]; 3], center: Option<[f32; 3]>) -> Result<Self> {
        if self.point_count() == 0 {
            return Ok(self.clone());
        }
        let device = self.xyz_device();
        let center = center.unwrap_or_else(|| self.compute_centroid_xyz());
        let center_data = TensorData::from(center.as_slice());
        let center_tensor = Tensor::<Backend, 1>::from_data(center_data, &device).reshape([1, 3]);

        let flat: Vec<f32> = r.iter().flat_map(|row| row.iter().copied()).collect();
        let rot_tensor = tensor::tensor2_from_slice_on_device(&flat, 3, 3, &device)?;
        let rot_t = rot_tensor.transpose();

        let xyz = self.xyz_ref().clone();
        let centered = xyz - center_tensor.clone();
        let rotated = centered.matmul(rot_t);
        let new_xyz = rotated + center_tensor;

        let mut result = self.clone();
        result.set_xyz(new_xyz);
        Ok(result)
    }

    pub fn rigid_transform(&self, rotation: &[[f32; 3]; 3], translation: [f32; 3]) -> Result<Self> {
        if self.point_count() == 0 {
            return Ok(self.clone());
        }
        let device = self.xyz_device();
        let flat: Vec<f32> = rotation
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect();
        let rot_tensor = tensor::tensor2_from_slice_on_device(&flat, 3, 3, &device)?;
        let rot_t = rot_tensor.transpose();

        let translation_data = TensorData::from(translation.as_slice());
        let translation_tensor =
            Tensor::<Backend, 1>::from_data(translation_data, &device).reshape([1, 3]);

        let xyz = self.xyz_ref().clone();
        let rotated = xyz.matmul(rot_t);
        let new_xyz = rotated + translation_tensor;

        let mut result = self.clone();
        result.set_xyz(new_xyz);
        Ok(result)
    }

    fn compute_centroid_xyz(&self) -> [f32; 3] {
        let xyz = self.get_xyz_vec();
        if xyz.is_empty() {
            return [0.0; 3];
        }
        let n = xyz.len() as f32;
        let mut c = [0.0f32; 3];
        for p in &xyz {
            c[0] += p[0];
            c[1] += p[1];
            c[2] += p[2];
        }
        c[0] /= n;
        c[1] /= n;
        c[2] /= n;
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate() {
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![[1.0, 0.0, 0.0]]).unwrap();
        let result = pc.translate([1.0, 2.0, 3.0]).unwrap();
        let xyz = result.get_xyz_vec();
        assert!((xyz[0][0] - 2.0).abs() < 1e-5);
        assert!((xyz[0][1] - 2.0).abs() < 1e-5);
        assert!((xyz[0][2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_scale() {
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
            .unwrap();
        let result = pc.scale(2.0, Some([0.0, 0.0, 0.0])).unwrap();
        let xyz = result.get_xyz_vec();
        assert!((xyz[0][0] - 2.0).abs() < 1e-5);
        assert!((xyz[1][0] + 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_rigid_transform() {
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![[1.0, 0.0, 0.0]]).unwrap();
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let result = pc.rigid_transform(&identity, [1.0, 2.0, 3.0]).unwrap();
        let xyz = result.get_xyz_vec();
        assert!((xyz[0][0] - 2.0).abs() < 1e-5);
        assert!((xyz[0][1] - 2.0).abs() < 1e-5);
        assert!((xyz[0][2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn transforms_preserve_source_device() {
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![[1.0, 0.0, 0.0]])
            .unwrap()
            .to_device(crate::utils::tensor::cpu_device());
        let device = pc.xyz_device();
        let identity3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let identity4 = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];

        assert_eq!(pc.transform(&identity4).unwrap().xyz_device(), device);
        assert_eq!(pc.transform_3x3(&identity3).unwrap().xyz_device(), device);
        assert_eq!(pc.translate([1.0, 0.0, 0.0]).unwrap().xyz_device(), device);
        assert_eq!(
            pc.scale(2.0, Some([0.0, 0.0, 0.0])).unwrap().xyz_device(),
            device
        );
        assert_eq!(
            pc.rotate(&identity3, Some([0.0, 0.0, 0.0]))
                .unwrap()
                .xyz_device(),
            device
        );
        assert_eq!(
            pc.rigid_transform(&identity3, [0.0, 0.0, 0.0])
                .unwrap()
                .xyz_device(),
            device
        );
    }
}
