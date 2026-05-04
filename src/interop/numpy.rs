use crate::point_cloud::attribute_value::AttributeValue;
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::Result;
use crate::utils::tensor;
use numpy::ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyDictMethods};

impl HighPerformancePointCloud {
    pub fn to_numpy<'py>(&self, py: Python<'py>) -> Result<Py<PyAny>> {
        let dict = PyDict::new(py);

        // XYZ
        let xyz_flat = self.get_xyz_flat();
        let n = self.point_count();
        let xyz_nd = Array2::from_shape_vec((n, 3), xyz_flat)
            .map_err(|e| format!("XYZ shape error: {}", e))?;
        let xyz_np = IntoPyArray::into_pyarray(xyz_nd, py);
        dict.set_item("xyz", xyz_np)
            .map_err(|e: PyErr| e.to_string())?;

        // Attributes with correct dtypes
        for (name, attr) in self.attributes() {
            let np_obj = attribute_to_numpy(py, attr)?;
            dict.set_item(name.as_str(), np_obj)
                .map_err(|e: PyErr| e.to_string())?;
        }

        Ok(dict.into())
    }

    pub fn from_xyz_array(xyz_obj: &Bound<'_, pyo3::PyAny>) -> Result<Self> {
        let xyz = read_xyz_from_pyany(xyz_obj)?;
        Self::from_tensor_xyz(xyz)
    }

    pub fn from_numpy(_py: Python, data: &Bound<'_, PyDict>) -> Result<Self> {
        let xyz_obj = data
            .get_item("xyz")
            .map_err(|_| "failed to get xyz".to_string())?
            .ok_or("xyz field missing".to_string())?;

        let xyz = read_xyz_from_pyany(&xyz_obj)?;
        let mut result = Self::from_tensor_xyz(xyz)?;
        let n = result.point_count();

        // Read all other keys as typed attributes
        for item in data.iter() {
            let key: String = item
                .0
                .extract()
                .map_err(|_| "dict key must be a string".to_string())?;
            if key == "xyz" {
                continue;
            }
            let value = &item.1;
            let attr = read_attribute_from_pyany(value)?;
            if attr.len() == n {
                result.attributes_mut().insert(key, attr);
            }
        }

        Ok(result)
    }

    pub fn set_intensity_from_array(&mut self, arr: &Bound<'_, pyo3::PyAny>) -> Result<()> {
        let attr = read_attribute_from_pyany(arr)?;
        if attr.len() != self.point_count() {
            return Err(crate::utils::error::PointCloudError::DimensionMismatch {
                expected: self.point_count(),
                actual: attr.len(),
            });
        }
        // Intensity is stored as f32
        let f32_val = match attr {
            AttributeValue::F32(v) => AttributeValue::F32(v),
            other => AttributeValue::F32(other.to_f32_vec()),
        };
        self.attributes_mut().insert("intensity".to_string(), f32_val);
        Ok(())
    }

    pub fn set_rgb_from_arrays(
        &mut self,
        r_arr: &Bound<'_, pyo3::PyAny>,
        g_arr: &Bound<'_, pyo3::PyAny>,
        b_arr: &Bound<'_, pyo3::PyAny>,
    ) -> Result<()> {
        let r = read_attribute_from_pyany(r_arr)?;
        let g = read_attribute_from_pyany(g_arr)?;
        let b = read_attribute_from_pyany(b_arr)?;

        let n = self.point_count();
        if r.len() != n || g.len() != n || b.len() != n {
            return Err(crate::utils::error::PointCloudError::DimensionMismatch {
                expected: n,
                actual: r.len(),
            });
        }

        // Convert to u8 for storage
        let r_u8 = to_u8_attr(r);
        let g_u8 = to_u8_attr(g);
        let b_u8 = to_u8_attr(b);

        self.attributes_mut().insert("red".to_string(), r_u8);
        self.attributes_mut().insert("green".to_string(), g_u8);
        self.attributes_mut().insert("blue".to_string(), b_u8);
        Ok(())
    }
}

fn to_u8_attr(attr: AttributeValue) -> AttributeValue {
    match attr {
        AttributeValue::U8(_) => attr,
        AttributeValue::F32(v) => {
            AttributeValue::U8(v.iter().map(|&x| x.clamp(0.0, 255.0) as u8).collect())
        }
        AttributeValue::F64(v) => {
            AttributeValue::U8(v.iter().map(|&x| x.clamp(0.0, 255.0) as u8).collect())
        }
        AttributeValue::U16(v) => AttributeValue::U8(v.iter().map(|&x| (x >> 8) as u8).collect()),
        AttributeValue::I32(v) => {
            AttributeValue::U8(v.iter().map(|&x| x.clamp(0, 255) as u8).collect())
        }
        _ => {
            let f = attr.to_f32_vec();
            AttributeValue::U8(f.iter().map(|&x| x.clamp(0.0, 255.0) as u8).collect())
        }
    }
}

// === NumPy reading helpers ===

use crate::utils::tensor::Tensor2;

fn read_xyz_from_pyany(obj: &Bound<'_, pyo3::PyAny>) -> Result<Tensor2> {
    // Try f32 first
    if let Ok(arr) = obj.cast::<PyArray2<f32>>() {
        let shape = arr.shape();
        if shape[1] != 3 {
            return Err(format!("XYZ must have shape [N,3], got [{},{}]", shape[0], shape[1]).into());
        }
        let readonly = arr.readonly();
        let slice = readonly
            .as_slice()
            .map_err(|_| "cannot read xyz data, array may not be contiguous")?;
        return tensor::tensor2_from_slice(slice, shape[0], shape[1]);
    }

    // Try f64 (autocast to f32)
    if let Ok(arr) = obj.cast::<PyArray2<f64>>() {
        let shape = arr.shape();
        if shape[1] != 3 {
            return Err(format!("XYZ must have shape [N,3], got [{},{}]", shape[0], shape[1]).into());
        }
        log::debug!("Autocasting f64 XYZ input to f32");
        let readonly = arr.readonly();
        let slice = readonly
            .as_slice()
            .map_err(|_| "cannot read xyz data")?;
        let f32_data: Vec<f32> = slice.iter().map(|&x| x as f32).collect();
        return tensor::tensor2_from_slice(&f32_data, shape[0], shape[1]);
    }

    // Try i32 (autocast to f32)
    if let Ok(arr) = obj.cast::<PyArray2<i32>>() {
        let shape = arr.shape();
        if shape[1] != 3 {
            return Err(format!("XYZ must have shape [N,3], got [{},{}]", shape[0], shape[1]).into());
        }
        log::debug!("Autocasting i32 XYZ input to f32");
        let readonly = arr.readonly();
        let slice = readonly
            .as_slice()
            .map_err(|_| "cannot read xyz data")?;
        let f32_data: Vec<f32> = slice.iter().map(|&x| x as f32).collect();
        return tensor::tensor2_from_slice(&f32_data, shape[0], shape[1]);
    }

    // Try i64 (autocast to f32)
    if let Ok(arr) = obj.cast::<PyArray2<i64>>() {
        let shape = arr.shape();
        if shape[1] != 3 {
            return Err(format!("XYZ must have shape [N,3], got [{},{}]", shape[0], shape[1]).into());
        }
        log::debug!("Autocasting i64 XYZ input to f32");
        let readonly = arr.readonly();
        let slice = readonly
            .as_slice()
            .map_err(|_| "cannot read xyz data")?;
        let f32_data: Vec<f32> = slice.iter().map(|&x| x as f32).collect();
        return tensor::tensor2_from_slice(&f32_data, shape[0], shape[1]);
    }

    Err("xyz must be a 2D numpy array with dtype float32, float64, int32, or int64".into())
}

pub fn read_attribute_from_pyany(obj: &Bound<'_, pyo3::PyAny>) -> Result<AttributeValue> {
    // Try f32
    if let Ok(arr) = obj.cast::<PyArray1<f32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice().map_err(|_| "cannot read array data")?;
        return Ok(AttributeValue::F32(slice.to_vec()));
    }
    // Try f64
    if let Ok(arr) = obj.cast::<PyArray1<f64>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice().map_err(|_| "cannot read array data")?;
        return Ok(AttributeValue::F64(slice.to_vec()));
    }
    // Try u8
    if let Ok(arr) = obj.cast::<PyArray1<u8>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice().map_err(|_| "cannot read array data")?;
        return Ok(AttributeValue::U8(slice.to_vec()));
    }
    // Try u16
    if let Ok(arr) = obj.cast::<PyArray1<u16>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice().map_err(|_| "cannot read array data")?;
        return Ok(AttributeValue::U16(slice.to_vec()));
    }
    // Try u32
    if let Ok(arr) = obj.cast::<PyArray1<u32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice().map_err(|_| "cannot read array data")?;
        return Ok(AttributeValue::U32(slice.to_vec()));
    }
    // Try i32
    if let Ok(arr) = obj.cast::<PyArray1<i32>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice().map_err(|_| "cannot read array data")?;
        return Ok(AttributeValue::I32(slice.to_vec()));
    }
    // Try i64
    if let Ok(arr) = obj.cast::<PyArray1<i64>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice().map_err(|_| "cannot read array data")?;
        return Ok(AttributeValue::I64(slice.to_vec()));
    }
    // Try bool
    if let Ok(arr) = obj.cast::<PyArray1<bool>>() {
        let readonly = arr.readonly();
        let slice = readonly.as_slice().map_err(|_| "cannot read array data")?;
        return Ok(AttributeValue::Bool(slice.to_vec()));
    }

    Err("attribute must be a 1D numpy array with a numeric or bool dtype".into())
}

pub fn attribute_to_numpy(py: Python<'_>, attr: &AttributeValue) -> Result<Py<PyAny>> {
    Ok(match attr {
        AttributeValue::F32(v) => {
            let nd = Array1::from_vec(v.clone());
            IntoPyArray::into_pyarray(nd, py).into()
        }
        AttributeValue::F64(v) => {
            let nd = Array1::from_vec(v.clone());
            IntoPyArray::into_pyarray(nd, py).into()
        }
        AttributeValue::U8(v) => {
            let nd = Array1::from_vec(v.clone());
            IntoPyArray::into_pyarray(nd, py).into()
        }
        AttributeValue::U16(v) => {
            let nd = Array1::from_vec(v.clone());
            IntoPyArray::into_pyarray(nd, py).into()
        }
        AttributeValue::U32(v) => {
            let nd = Array1::from_vec(v.clone());
            IntoPyArray::into_pyarray(nd, py).into()
        }
        AttributeValue::I32(v) => {
            let nd = Array1::from_vec(v.clone());
            IntoPyArray::into_pyarray(nd, py).into()
        }
        AttributeValue::I64(v) => {
            let nd = Array1::from_vec(v.clone());
            IntoPyArray::into_pyarray(nd, py).into()
        }
        AttributeValue::Bool(v) => {
            let nd = Array1::from_vec(v.clone());
            IntoPyArray::into_pyarray(nd, py).into()
        }
    })
}
