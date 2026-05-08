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

pub type Tensor2 = Tensor<Backend, 2>;

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

pub fn tensor2_rows(tensor: &Tensor2) -> usize {
    let shape = tensor.shape();
    shape.dims[0]
}

pub fn tensor2_cols(tensor: &Tensor2) -> usize {
    let shape = tensor.shape();
    shape.dims[1]
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
