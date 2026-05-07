use crate::utils::error::{PointCloudError, Result};
use burn::backend::ndarray::NdArrayDevice;
use burn::backend::wgpu::WgpuDevice;
/// burn张量工具函数：类型转换、维度检查等
use burn::backend::{NdArray, Router, Wgpu};
use burn::prelude::DeviceOps;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Tensor, TensorData};

// Router backend: automatically selects GPU (Wgpu) or CPU (NdArray) at runtime
pub type Backend = Router<(Wgpu, NdArray)>;
pub type BackendDevice = <Backend as BackendTrait>::Device;

/// Default device with automatic GPU->CPU fallback
///
/// 工作原理：
/// 1. Router backend 包含两个后端：Wgpu (GPU) 和 NdArray (CPU)
/// 2. MultiDevice::B1 对应第一个后端 (Wgpu)，B2 对应第二个 (NdArray)
/// 3. 运行时尝试初始化 GPU，失败则自动降级到 CPU
pub fn default_device() -> BackendDevice {
    use burn::backend::router::duo::MultiDevice;

    // 尝试创建 GPU 设备，如果失败则降级到 CPU
    // let result = panic::catch_unwind(|| WgpuDevice::default());
    let wgpu_cnt = WgpuDevice::device_count_total();
    match wgpu_cnt {
        0 => {
            log::warn!("GPU not available, falling back to CPU backend");
            MultiDevice::B2(NdArrayDevice::Cpu)
        }
        _ => {
            log::info!("Using GPU backend (WGPU)");
            MultiDevice::B1(WgpuDevice::default())
        }
    }
}

pub fn has_wgpu_device() -> bool {
    WgpuDevice::device_count_total() > 0
}

/// Get a GPU device if available
#[allow(dead_code)]
pub fn gpu_device() -> BackendDevice {
    use burn::backend::router::duo::MultiDevice;
    MultiDevice::B1(WgpuDevice::default())
}

/// Get a CPU device
#[allow(dead_code)]
pub fn cpu_device() -> BackendDevice {
    use burn::backend::router::duo::MultiDevice;
    MultiDevice::B2(NdArrayDevice::Cpu)
}

pub fn empty_xyz() -> Tensor2 {
    Tensor::<Backend, 2>::zeros([0, 3], &cpu_device())
}

pub type Tensor1 = Tensor<Backend, 1>;
pub type Tensor2 = Tensor<Backend, 2>;
// ============ 从 slice 创建 Tensor（避免 Vec 复制）============

/// 从 &[f32] 创建 Tensor1
pub fn tensor1_from_slice(data: &[f32]) -> Tensor1 {
    let tensor_data = TensorData::from(data);
    Tensor::<Backend, 1>::from_data(tensor_data, &default_device())
}

/// 从 flat &[f32] 创建 Tensor2，形状为 [rows, cols]
pub fn tensor2_from_slice(data: &[f32], rows: usize, cols: usize) -> Result<Tensor2> {
    let device = default_device();
    tensor2_from_slice_on_device(data, rows, cols, &device)
}

pub fn tensor2_from_slice_on_device(
    data: &[f32],
    rows: usize,
    cols: usize,
    device: &BackendDevice,
) -> Result<Tensor2> {
    if data.len() != rows * cols {
        return Err(PointCloudError::TensorShapeError(format!(
            "数据长度{}与形状[{},{}]不匹配",
            data.len(),
            rows,
            cols
        )));
    }
    if data.is_empty() {
        return Ok(Tensor::<Backend, 2>::zeros([rows, cols], device));
    }
    let tensor_data = TensorData::from(data);
    let tensor = Tensor::<Backend, 1>::from_data(tensor_data, device).reshape([rows, cols]);
    Ok(tensor)
}

// /// 从 flat &[f32] 创建 XYZ Tensor2，形状为 [N, 3]
// pub fn xyz_from_slice(data: &[f32]) -> Result<Tensor2> {
//     if !data.len().is_multiple_of(3) {
//         return Err(PointCloudError::TensorShapeError(
//             "XYZ数据长度必须是3的倍数".to_string(),
//         ));
//     }
//     if data.is_empty() {
//         return Err(PointCloudError::TensorShapeError("XYZ数据为空".to_string()));
//     }
//     let rows = data.len() / 3;
//     tensor2_from_slice(data, rows, 3)
// }

/// 检查XYZ张量维度
pub fn validate_xyz_shape(xyz: &[Vec<f32>]) -> Result<()> {
    if xyz.is_empty() {
        return Err(PointCloudError::TensorShapeError("XYZ数据为空".to_string()));
    }

    if !xyz.iter().all(|row| row.len() == 3) {
        return Err(PointCloudError::TensorShapeError(
            "XYZ必须为[M,3]的形状".to_string(),
        ));
    }

    Ok(())
}

/// 检查RGB通道维度
pub fn validate_rgb_channel_shape(channel: &[u8], point_count: usize) -> Result<()> {
    if channel.len() != point_count {
        return Err(PointCloudError::DimensionMismatch {
            expected: point_count,
            actual: channel.len(),
        });
    }
    Ok(())
}

/// RGB通道 Vec<u8> -> Tensor1 (内部存储为f32)
pub fn rgb_channel_to_tensor(channel: Vec<u8>) -> Tensor1 {
    let f32_data: Vec<f32> = channel.into_iter().map(|v| v as f32).collect();
    let tensor_data = TensorData::from(f32_data.as_slice());
    Tensor::<Backend, 1>::from_data(tensor_data, &default_device())
}

/// Tensor1 -> Vec<u8> (从f32转换回u8)
pub fn tensor1_to_u8_vec(tensor: &Tensor1) -> Vec<u8> {
    let data: TensorData = tensor.to_data();
    let f32_vec: Vec<f32> = data
        .to_vec::<f32>()
        .expect("Failed to convert tensor data to Vec<f32>");
    f32_vec
        .into_iter()
        .map(|v| v.clamp(0.0, 255.0) as u8)
        .collect()
}

/// 检查intensity张量维度
pub fn validate_intensity_shape(intensity: &[f32], point_count: usize) -> Result<()> {
    if intensity.len() != point_count {
        return Err(PointCloudError::DimensionMismatch {
            expected: point_count,
            actual: intensity.len(),
        });
    }
    Ok(())
}

/// 检查属性张量维度
pub fn validate_attribute_shape(attribute: &[f32], point_count: usize) -> Result<()> {
    if attribute.len() != point_count {
        return Err(PointCloudError::DimensionMismatch {
            expected: point_count,
            actual: attribute.len(),
        });
    }
    Ok(())
}

/// 检查矩阵维度（3x3或4x4）
pub fn validate_matrix_shape(matrix: &[Vec<f32>]) -> Result<(usize, usize)> {
    if matrix.is_empty() {
        return Err(PointCloudError::MatrixError("矩阵为空".to_string()));
    }

    let rows = matrix.len();
    let cols = matrix[0].len();

    // 检查所有行的列数一致
    if !matrix.iter().all(|row| row.len() == cols) {
        return Err(PointCloudError::MatrixError("矩阵行列数不一致".to_string()));
    }

    // 只允许3x3或4x4矩阵
    if (rows == 3 && cols == 3) || (rows == 4 && cols == 4) {
        Ok((rows, cols))
    } else {
        Err(PointCloudError::MatrixError(format!(
            "仅支持3x3或4x4矩阵，实际{}x{}",
            rows, cols
        )))
    }
}

/// Vec<Vec<f32>> -> Tensor<[M,N]>
pub fn vec2_to_tensor(data: Vec<Vec<f32>>) -> Result<Tensor2> {
    let device = default_device();
    vec2_to_tensor_on_device(data, &device)
}

pub fn vec2_to_tensor_on_device(data: Vec<Vec<f32>>, device: &BackendDevice) -> Result<Tensor2> {
    if data.is_empty() {
        return Err(PointCloudError::TensorShapeError("数据为空".to_string()));
    }

    let rows = data.len();
    let cols = data[0].len();
    if !data.iter().all(|row| row.len() == cols) {
        return Err(PointCloudError::TensorShapeError(
            "数据行列数不一致".to_string(),
        ));
    }

    let flat: Vec<f32> = data.into_iter().flatten().collect();
    let tensor_data = TensorData::from(flat.as_slice());
    let tensor = Tensor::<Backend, 1>::from_data(tensor_data, device).reshape([rows, cols]);
    Ok(tensor)
}

/// XYZ Vec -> Tensor<[M,3]>
pub fn xyz_to_tensor(xyz: Vec<Vec<f32>>) -> Result<Tensor2> {
    validate_xyz_shape(&xyz)?;
    vec2_to_tensor(xyz)
}

pub fn intensity_to_tensor(intensity: Vec<f32>) -> Tensor1 {
    let tensor_data = TensorData::from(intensity.as_slice());
    Tensor::<Backend, 1>::from_data(tensor_data, &default_device())
}

pub fn tensor1_len(tensor: &Tensor1) -> usize {
    let shape = tensor.shape();
    shape.dims[0]
}

pub fn tensor2_rows(tensor: &Tensor2) -> usize {
    let shape = tensor.shape();
    shape.dims[0]
}

pub fn tensor2_cols(tensor: &Tensor2) -> usize {
    let shape = tensor.shape();
    shape.dims[1]
}

pub fn tensor1_to_vec(tensor: &Tensor1) -> Vec<f32> {
    let data: TensorData = tensor.to_data();
    data.to_vec::<f32>()
        .expect("Failed to convert tensor data to Vec<f32>")
}

pub fn tensor2_to_vec(tensor: &Tensor2) -> Vec<Vec<f32>> {
    let data: TensorData = tensor.to_data();
    let shape = &data.shape;
    let rows = shape[0];
    let cols = shape[1];
    let flat: Vec<f32> = data
        .to_vec::<f32>()
        .expect("Failed to convert tensor data to Vec<f32>");
    flat.chunks(cols)
        .take(rows)
        .map(|chunk| chunk.to_vec())
        .collect()
}

pub fn matrix_to_tensor(matrix: Vec<Vec<f32>>) -> Result<Tensor2> {
    let (rows, cols) = validate_matrix_shape(&matrix)?;
    let tensor = vec2_to_tensor(matrix)?;
    let actual_rows = tensor2_rows(&tensor);
    let actual_cols = tensor2_cols(&tensor);
    if actual_rows != rows || actual_cols != cols {
        return Err(PointCloudError::MatrixError("矩阵维度不一致".to_string()));
    }
    Ok(tensor)
}
