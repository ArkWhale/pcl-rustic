use crate::utils::error::{PointCloudError, Result};
use burn::backend::wgpu::WgpuDevice;
use burn::tensor::backend::{Backend as BackendTrait, BackendTypes};
use burn::tensor::{Tensor, TensorData};
use burn::{Dispatch, DispatchDevice};
use std::sync::OnceLock;

// burn张量工具函数：类型转换、维度检查等
// Dispatch backend: dynamically selects the feature-enabled Burn backend device.
pub type Backend = Dispatch;
pub type BackendDevice = <Backend as BackendTypes>::Device;
pub type Tensor1 = Tensor<Backend, 1>;
pub type Tensor2 = Tensor<Backend, 2>;
static GPU_DEVICE: OnceLock<Option<BackendDevice>> = OnceLock::new();

/// Default device with GPU-first selection.
///
/// 工作原理：
/// 1. Dispatch backend 包含 Cargo feature 启用的后端设备
/// 2. 默认选择 WGPU/Vulkan GPU
/// 3. 只有没有 GPU 设备时才回退到 CPU
pub fn default_device() -> BackendDevice {
    if has_wgpu_device() {
        log::info!("Using GPU backend (Dispatch/Vulkan)");
        return gpu_device().expect("WGPU device disappeared after availability check");
    }
    log::warn!("GPU not available, falling back to CPU backend");
    cpu_device()
}

pub fn has_wgpu_device() -> bool {
    detect_gpu_device().is_some()
}

/// Get a GPU device if available
#[allow(dead_code)]
pub fn gpu_device() -> Result<BackendDevice> {
    detect_gpu_device().ok_or_else(|| {
        PointCloudError::InvalidParameter(
            "GPU device requested but no WGPU device is available".to_string(),
        )
    })
}

/// Get a CPU device
#[allow(dead_code)]
pub fn cpu_device() -> BackendDevice {
    DispatchDevice::Cpu(Default::default())
}

fn detect_gpu_device() -> Option<BackendDevice> {
    GPU_DEVICE.get_or_init(probe_gpu_device).clone()
}

fn probe_gpu_device() -> Option<BackendDevice> {
    for candidate in [
        WgpuDevice::DiscreteGpu(0),
        WgpuDevice::IntegratedGpu(0),
        WgpuDevice::VirtualGpu(0),
    ] {
        let device = DispatchDevice::Vulkan(candidate);
        if gpu_smoke_check(&device) {
            return Some(device);
        }
    }
    None
}

fn gpu_smoke_check(device: &BackendDevice) -> bool {
    std::panic::catch_unwind(|| {
        let tensor = Tensor::<Backend, 2>::zeros([1, 1], device);
        drop(tensor);
        <Backend as BackendTrait>::sync(device).is_ok()
    })
    .unwrap_or(false)
}

pub fn empty_xyz() -> Tensor2 {
    Tensor::<Backend, 2>::zeros([0, 3], &cpu_device())
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

pub fn tensor1_from_slice_on_device(data: &[f32], device: &BackendDevice) -> Tensor1 {
    let tensor_data = TensorData::from(data);
    Tensor::<Backend, 1>::from_data(tensor_data, device)
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
    shape.dims::<2>()[0]
}

pub fn tensor2_cols(tensor: &Tensor2) -> usize {
    let shape = tensor.shape();
    shape.dims::<2>()[1]
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_uses_dispatch_cpu_variant() {
        assert!(format!("{:?}", cpu_device()).starts_with("Cpu("));
    }

    #[test]
    fn tensor1_helper_uses_adapter_backend() {
        let device = cpu_device();
        let tensor = tensor1_from_slice_on_device(&[1.0, 2.0, 3.0], &device);

        assert_eq!(tensor.shape().dims::<1>(), [3]);
    }
}
