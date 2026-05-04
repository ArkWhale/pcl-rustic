#![recursion_limit = "256"]
mod interop;
mod io;
mod point_cloud;
mod utils;

use interop::numpy::{attribute_to_numpy, read_attribute_from_pyany};
use point_cloud::core::HighPerformancePointCloud;
use point_cloud::voxel::DownsampleStrategy;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyDownsampleStrategy>()?;
    Ok(())
}

#[pyclass(name = "PointCloud")]
pub struct PyPointCloud {
    inner: HighPerformancePointCloud,
}

#[pymethods]
impl PyPointCloud {
    #[new]
    fn new() -> Self {
        PyPointCloud {
            inner: HighPerformancePointCloud::new(),
        }
    }

    #[staticmethod]
    fn from_xyz(xyz: &Bound<'_, pyo3::PyAny>) -> PyResult<Self> {
        let inner = HighPerformancePointCloud::from_xyz_array(xyz).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    #[staticmethod]
    fn from_xyz_intensity(
        xyz: &Bound<'_, pyo3::PyAny>,
        intensity: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<Self> {
        let mut inner = HighPerformancePointCloud::from_xyz_array(xyz).map_err(PyErr::from)?;
        inner
            .set_intensity_from_array(intensity)
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    #[staticmethod]
    fn from_xyz_rgb(
        xyz: &Bound<'_, pyo3::PyAny>,
        r: &Bound<'_, pyo3::PyAny>,
        g: &Bound<'_, pyo3::PyAny>,
        b: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<Self> {
        let mut inner = HighPerformancePointCloud::from_xyz_array(xyz).map_err(PyErr::from)?;
        inner.set_rgb_from_arrays(r, g, b).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    #[staticmethod]
    fn from_dict(py: Python, data: &Bound<'_, PyDict>) -> PyResult<Self> {
        let inner = HighPerformancePointCloud::from_numpy(py, data).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    fn point_count(&self) -> usize {
        self.inner.point_count()
    }

    fn get_xyz(&self, py: Python) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Array2;
        use numpy::IntoPyArray;

        let xyz_flat = self.inner.get_xyz_flat();
        let n = self.inner.point_count();
        let xyz_nd = Array2::from_shape_vec((n, 3), xyz_flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("shape error: {}", e)))?;
        let xyz_np = IntoPyArray::into_pyarray(xyz_nd, py);
        Ok(xyz_np.into())
    }

    fn has_intensity(&self) -> bool {
        self.inner.has_intensity()
    }

    fn has_rgb(&self) -> bool {
        self.inner.has_rgb()
    }

    fn get_intensity(&self, py: Python) -> PyResult<Option<Py<PyAny>>> {
        match self.inner.get_attribute("intensity") {
            Some(attr) => Ok(Some(attribute_to_numpy(py, attr).map_err(PyErr::from)?)),
            None => Ok(None),
        }
    }

    fn get_rgb(&self, py: Python) -> PyResult<Option<(Py<PyAny>, Py<PyAny>, Py<PyAny>)>> {
        let r = self.inner.get_attribute("red");
        let g = self.inner.get_attribute("green");
        let b = self.inner.get_attribute("blue");

        if let (Some(r_attr), Some(g_attr), Some(b_attr)) = (r, g, b) {
            let r_np = attribute_to_numpy(py, r_attr).map_err(PyErr::from)?;
            let g_np = attribute_to_numpy(py, g_attr).map_err(PyErr::from)?;
            let b_np = attribute_to_numpy(py, b_attr).map_err(PyErr::from)?;
            Ok(Some((r_np, g_np, b_np)))
        } else {
            Ok(None)
        }
    }

    fn set_intensity(&mut self, intensity: &Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        self.inner
            .set_intensity_from_array(intensity)
            .map_err(PyErr::from)?;
        Ok(())
    }

    fn set_rgb(
        &mut self,
        r: &Bound<'_, pyo3::PyAny>,
        g: &Bound<'_, pyo3::PyAny>,
        b: &Bound<'_, pyo3::PyAny>,
    ) -> PyResult<()> {
        self.inner
            .set_rgb_from_arrays(r, g, b)
            .map_err(PyErr::from)?;
        Ok(())
    }

    fn set_attribute(&mut self, name: String, data: &Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        let attr = read_attribute_from_pyany(data).map_err(PyErr::from)?;
        self.inner.set_attribute(name, attr).map_err(PyErr::from)?;
        Ok(())
    }

    fn add_attribute(&mut self, name: String, data: &Bound<'_, pyo3::PyAny>) -> PyResult<()> {
        if self.inner.attributes().contains_key(&name) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "attribute '{}' already exists",
                name
            )));
        }
        let attr = read_attribute_from_pyany(data).map_err(PyErr::from)?;
        self.inner.set_attribute(name, attr).map_err(PyErr::from)?;
        Ok(())
    }

    fn get_attribute(&self, py: Python, name: &str) -> PyResult<Option<Py<PyAny>>> {
        match self.inner.get_attribute(name) {
            Some(attr) => Ok(Some(attribute_to_numpy(py, attr).map_err(PyErr::from)?)),
            None => Ok(None),
        }
    }

    fn attribute_names(&self) -> Vec<String> {
        self.inner.attribute_names()
    }

    fn remove_attribute(&mut self, name: &str) -> PyResult<()> {
        self.inner.remove_attribute(name).map_err(PyErr::from)?;
        Ok(())
    }

    fn clear_attributes(&mut self) {
        self.inner.attributes_mut().clear();
    }

    fn has_attributes(&self, names: Vec<String>) -> bool {
        names.iter().all(|n| self.inner.attributes().contains_key(n))
    }

    fn attribute_info(&self) -> Vec<(String, usize, String)> {
        self.inner
            .attributes()
            .iter()
            .map(|(name, attr)| (name.clone(), attr.len(), attr.dtype().to_string()))
            .collect()
    }

    fn remove_intensity(&mut self) {
        self.inner.attributes_mut().remove("intensity");
    }

    fn remove_rgb(&mut self) {
        self.inner.attributes_mut().remove("red");
        self.inner.attributes_mut().remove("green");
        self.inner.attributes_mut().remove("blue");
    }

    #[staticmethod]
    fn delete_file(path: &str) -> PyResult<()> {
        HighPerformancePointCloud::delete_file(path).map_err(PyErr::from)?;
        Ok(())
    }

    #[pyo3(signature = (matrix))]
    fn transform(&self, matrix: Vec<Vec<f32>>) -> PyResult<Self> {
        if matrix.len() == 4 && matrix.iter().all(|r| r.len() == 4) {
            let m: [[f32; 4]; 4] = [
                [matrix[0][0], matrix[0][1], matrix[0][2], matrix[0][3]],
                [matrix[1][0], matrix[1][1], matrix[1][2], matrix[1][3]],
                [matrix[2][0], matrix[2][1], matrix[2][2], matrix[2][3]],
                [matrix[3][0], matrix[3][1], matrix[3][2], matrix[3][3]],
            ];
            let result = self.inner.transform(&m).map_err(PyErr::from)?;
            Ok(PyPointCloud { inner: result })
        } else if matrix.len() == 3 && matrix.iter().all(|r| r.len() == 3) {
            let m: [[f32; 3]; 3] = [
                [matrix[0][0], matrix[0][1], matrix[0][2]],
                [matrix[1][0], matrix[1][1], matrix[1][2]],
                [matrix[2][0], matrix[2][1], matrix[2][2]],
            ];
            let result = self.inner.transform_3x3(&m).map_err(PyErr::from)?;
            Ok(PyPointCloud { inner: result })
        } else {
            Err(pyo3::exceptions::PyValueError::new_err(
                "matrix must be 3x3 or 4x4",
            ))
        }
    }

    fn rigid_transform(&self, rotation: Vec<Vec<f32>>, translation: Vec<f32>) -> PyResult<Self> {
        if rotation.len() != 3 || !rotation.iter().all(|r| r.len() == 3) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rotation must be 3x3",
            ));
        }
        if translation.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "translation must be length 3",
            ));
        }
        let rot: [[f32; 3]; 3] = [
            [rotation[0][0], rotation[0][1], rotation[0][2]],
            [rotation[1][0], rotation[1][1], rotation[1][2]],
            [rotation[2][0], rotation[2][1], rotation[2][2]],
        ];
        let t: [f32; 3] = [translation[0], translation[1], translation[2]];
        let result = self.inner.rigid_transform(&rot, t).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    fn translate(&self, t: Vec<f32>) -> PyResult<Self> {
        if t.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "translation must be length 3",
            ));
        }
        let result = self
            .inner
            .translate([t[0], t[1], t[2]])
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    #[pyo3(signature = (s, center = None))]
    fn scale(&self, s: f32, center: Option<Vec<f32>>) -> PyResult<Self> {
        let c = center.map(|v| {
            if v.len() == 3 {
                [v[0], v[1], v[2]]
            } else {
                [0.0, 0.0, 0.0]
            }
        });
        let result = self.inner.scale(s, c).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    #[pyo3(signature = (rotation, center = None))]
    fn rotate(&self, rotation: Vec<Vec<f32>>, center: Option<Vec<f32>>) -> PyResult<Self> {
        if rotation.len() != 3 || !rotation.iter().all(|r| r.len() == 3) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rotation must be 3x3",
            ));
        }
        let rot: [[f32; 3]; 3] = [
            [rotation[0][0], rotation[0][1], rotation[0][2]],
            [rotation[1][0], rotation[1][1], rotation[1][2]],
            [rotation[2][0], rotation[2][1], rotation[2][2]],
        ];
        let c = center.map(|v| {
            if v.len() == 3 {
                [v[0], v[1], v[2]]
            } else {
                [0.0, 0.0, 0.0]
            }
        });
        let result = self.inner.rotate(&rot, c).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    #[pyo3(signature = (voxel_size, strategy = 1, seed = None))]
    fn voxel_downsample(
        &self,
        voxel_size: f32,
        strategy: i32,
        seed: Option<u64>,
    ) -> PyResult<Self> {
        let strat = match strategy {
            0 => DownsampleStrategy::RandomSeeded {
                seed: seed.unwrap_or(42),
            },
            1 => DownsampleStrategy::NearestToCentroid,
            2 => DownsampleStrategy::Average,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "strategy must be 0 (RANDOM_SEEDED), 1 (NEAREST_TO_CENTROID), or 2 (AVERAGE)",
                ))
            }
        };
        let result = self
            .inner
            .voxel_downsample(voxel_size, &strat)
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    // === Selection (M2) ===

    fn select(&self, mask: &Bound<'_, pyo3::PyAny>) -> PyResult<Self> {
        use numpy::{PyArray1, PyArrayMethods};
        let arr = mask.cast::<PyArray1<bool>>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("mask must be a boolean numpy array")
        })?;
        let readonly = arr.readonly();
        let slice = readonly.as_slice().map_err(|_| {
            pyo3::exceptions::PyValueError::new_err("cannot read mask data")
        })?;
        let result = self.inner.select_mask(slice).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    fn select_indices(&self, indices: Vec<usize>) -> PyResult<Self> {
        let result = self
            .inner
            .select_indices(&indices)
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    fn select_by_classification(&self, codes: Vec<u8>) -> PyResult<Self> {
        let result = self
            .inner
            .select_by_classification(&codes)
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    fn select_intensity_range(&self, lo: f32, hi: f32) -> PyResult<Self> {
        let result = self
            .inner
            .select_intensity_range(lo, hi)
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    // === Concatenation ===

    #[staticmethod]
    #[pyo3(signature = (clouds, policy = "strict"))]
    fn concatenate(clouds: Vec<PyRef<'_, PyPointCloud>>, policy: &str) -> PyResult<Self> {
        use crate::point_cloud::core::HighPerformancePointCloud;
        let inners: Vec<&HighPerformancePointCloud> = clouds.iter().map(|c| &c.inner).collect();
        let concat_policy = match policy {
            "strict" => ConcatPolicy::Strict,
            "union" => ConcatPolicy::Union,
            "intersection" => ConcatPolicy::Intersection,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "policy must be 'strict', 'union', or 'intersection'",
                ))
            }
        };
        let result =
            HighPerformancePointCloud::concatenate(&inners, concat_policy).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    // === I/O ===

    #[staticmethod]
    fn from_las(path: &str) -> PyResult<Self> {
        let inner = HighPerformancePointCloud::from_las_laz(path).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    #[pyo3(signature = (path, compress = false))]
    fn to_las(&self, path: &str, compress: bool) -> PyResult<()> {
        self.inner.to_las(path, compress).map_err(PyErr::from)?;
        Ok(())
    }

    fn memory_usage(&self) -> usize {
        self.inner.memory_usage()
    }

    fn to_dict(&self, py: Python) -> PyResult<Py<PyAny>> {
        self.inner.to_numpy(py).map_err(PyErr::from)
    }

    fn clone(&self) -> Self {
        PyPointCloud {
            inner: self.inner.clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PointCloud(points={}, intensity={}, rgb={}, attributes={})",
            self.inner.point_count(),
            if self.inner.has_intensity() {
                "Yes"
            } else {
                "No"
            },
            if self.inner.has_rgb() { "Yes" } else { "No" },
            self.inner.attribute_names().len()
        )
    }
}

// === ConcatPolicy ===

#[derive(Clone, Debug)]
pub enum ConcatPolicy {
    Strict,
    Union,
    Intersection,
}

// === DownsampleStrategy Python class ===

#[pyclass(name = "DownsampleStrategy")]
pub struct PyDownsampleStrategy;

#[pymethods]
impl PyDownsampleStrategy {
    #[classattr]
    #[allow(non_snake_case)]
    fn RANDOM_SEEDED() -> i32 {
        0
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn NEAREST_TO_CENTROID() -> i32 {
        1
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn AVERAGE() -> i32 {
        2
    }

    // Keep old names as aliases for backwards compat during transition
    #[classattr]
    #[allow(non_snake_case)]
    fn RANDOM() -> i32 {
        0
    }

    #[classattr]
    #[allow(non_snake_case)]
    fn CENTROID() -> i32 {
        1
    }
}
