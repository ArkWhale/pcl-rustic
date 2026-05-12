#![recursion_limit = "256"]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
mod interop;
mod io;
mod neighbors;
mod point_cloud;
mod registration;
mod utils;

use interop::numpy::{attribute_to_numpy, read_attribute_from_pyany};
use io::table::TableColumnNames;
use neighbors::normals::NormalSearch;
use neighbors::octree::Octree;
use point_cloud::core::HighPerformancePointCloud;
use point_cloud::voxel::DownsampleStrategy;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use registration::{ICPConvergenceCriteria, RegistrationResult, TransformationEstimation};

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPointCloud>()?;
    m.add_class::<PyDownsampleStrategy>()?;
    m.add_class::<PyNormalSearch>()?;
    m.add_class::<PyOctree>()?;
    m.add_function(wrap_pyfunction!(py_has_wgpu_device, m)?)?;
    m.add_function(wrap_pyfunction!(py_rayon_current_num_threads, m)?)?;
    let reg = PyModule::new(m.py(), "registration")?;
    reg.add_class::<PyICPConvergenceCriteria>()?;
    reg.add_class::<PyTransformationEstimation>()?;
    reg.add_class::<PyRegistrationResult>()?;
    reg.add_function(wrap_pyfunction!(py_icp, &reg)?)?;
    reg.add_function(wrap_pyfunction!(py_evaluate, &reg)?)?;
    m.add_submodule(&reg)?;
    Ok(())
}

#[pyfunction(name = "has_wgpu_device")]
fn py_has_wgpu_device() -> bool {
    crate::utils::tensor::has_wgpu_device()
}

#[pyfunction(name = "rayon_current_num_threads")]
fn py_rayon_current_num_threads() -> usize {
    rayon::current_num_threads()
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

    #[staticmethod]
    fn from_numpy(py: Python, data: &Bound<'_, PyDict>) -> PyResult<Self> {
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
        names
            .iter()
            .all(|n| self.inner.attributes().contains_key(n))
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
        let strat =
            match strategy {
                0 => DownsampleStrategy::RandomSeeded {
                    seed: seed.unwrap_or(42),
                },
                1 => DownsampleStrategy::NearestToCentroid,
                2 => DownsampleStrategy::Average,
                _ => return Err(pyo3::exceptions::PyValueError::new_err(
                    "strategy must be 0 (RANDOM_SEEDED), 1 (NEAREST_TO_CENTROID), or 2 (AVERAGE)",
                )),
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
        let slice = readonly
            .as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("cannot read mask data"))?;
        let result = self.inner.select_mask(slice).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    fn select_indices(&self, indices: Vec<usize>) -> PyResult<Self> {
        let result = self.inner.select_indices(&indices).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    fn select_by_classification(&self, codes: Vec<u8>) -> PyResult<Self> {
        let result = self
            .inner
            .select_by_classification(&codes)
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    #[pyo3(signature = (name, op, values, inclusive = true))]
    fn select_where(
        &self,
        name: &str,
        op: &str,
        values: Vec<f64>,
        inclusive: bool,
    ) -> PyResult<Self> {
        let result = self
            .inner
            .select_where(name, op, &values, inclusive)
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

    fn select_return_number(&self, n: u8) -> PyResult<Self> {
        let result = self.inner.select_return_number(n).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    fn select_elevation_range(&self, lo: f32, hi: f32) -> PyResult<Self> {
        let result = self
            .inner
            .select_elevation_range(lo, hi)
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    fn crop_aabb(&self, min: Vec<f32>, max: Vec<f32>) -> PyResult<Self> {
        if min.len() != 3 || max.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "min and max must both have length 3",
            ));
        }
        let result = self
            .inner
            .crop_aabb([min[0], min[1], min[2]], [max[0], max[1], max[2]])
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    fn aabb(&self) -> ([f32; 3], [f32; 3]) {
        self.inner.aabb()
    }

    fn crop_obb(
        &self,
        center: Vec<f32>,
        extents: Vec<f32>,
        rotation: Vec<Vec<f32>>,
    ) -> PyResult<Self> {
        if center.len() != 3 || extents.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "center and extents must both have length 3",
            ));
        }
        if rotation.len() != 3 || !rotation.iter().all(|row| row.len() == 3) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "rotation must be 3x3",
            ));
        }
        let result = self
            .inner
            .crop_obb(
                [center[0], center[1], center[2]],
                [extents[0], extents[1], extents[2]],
                [
                    [rotation[0][0], rotation[0][1], rotation[0][2]],
                    [rotation[1][0], rotation[1][1], rotation[1][2]],
                    [rotation[2][0], rotation[2][1], rotation[2][2]],
                ],
            )
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner: result })
    }

    fn obb(&self) -> ([f32; 3], [f32; 3], [[f32; 3]; 3]) {
        self.inner.obb()
    }

    // === Neighbors, normals, outliers ===

    fn knn(
        &self,
        py: Python,
        query: &Bound<'_, pyo3::PyAny>,
        k: usize,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
        use numpy::ndarray::Array2;
        use numpy::IntoPyArray;

        let query = read_query_points(query)?;
        let hits = self.inner.knn(&query, k).map_err(PyErr::from)?;
        let q = hits.len();
        let width = hits.first().map(|row| row.len()).unwrap_or(0);
        let mut indices = Vec::with_capacity(q * width);
        let mut distances = Vec::with_capacity(q * width);
        for row in hits {
            for hit in row {
                indices.push(hit.index as i64);
                distances.push(hit.distance);
            }
        }
        let idx_np = Array2::from_shape_vec((q, width), indices)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        let dist_np = Array2::from_shape_vec((q, width), distances)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .into_pyarray(py);
        Ok((idx_np.into_any().unbind(), dist_np.into_any().unbind()))
    }

    fn radius_search(
        &self,
        py: Python,
        query: &Bound<'_, pyo3::PyAny>,
        radius: f32,
    ) -> PyResult<Vec<Py<PyAny>>> {
        use numpy::ndarray::Array1;
        use numpy::IntoPyArray;

        let query = read_query_points(query)?;
        let hits = self
            .inner
            .radius_search(&query, radius)
            .map_err(PyErr::from)?;
        let mut out = Vec::with_capacity(hits.len());
        for row in hits {
            let indices: Vec<i64> = row.into_iter().map(|hit| hit.index as i64).collect();
            out.push(
                Array1::from_vec(indices)
                    .into_pyarray(py)
                    .into_any()
                    .unbind(),
            );
        }
        Ok(out)
    }

    fn estimate_normals(&mut self, search: PyRef<'_, PyNormalSearch>) -> PyResult<()> {
        self.inner
            .estimate_normals(search.inner.clone())
            .map_err(PyErr::from)
    }

    fn estimate_covariances(&mut self, knn: usize) -> PyResult<()> {
        self.inner.estimate_covariances(knn).map_err(PyErr::from)
    }

    fn remove_statistical_outlier(
        &self,
        py: Python,
        nb_neighbors: usize,
        std_ratio: f32,
    ) -> PyResult<(Self, Py<PyAny>)> {
        use numpy::ndarray::Array1;
        use numpy::IntoPyArray;

        let (filtered, mask) = self
            .inner
            .remove_statistical_outlier(nb_neighbors, std_ratio)
            .map_err(PyErr::from)?;
        let mask_np = Array1::from_vec(mask).into_pyarray(py).into_any().unbind();
        Ok((PyPointCloud { inner: filtered }, mask_np))
    }

    fn remove_radius_outlier(
        &self,
        py: Python,
        nb_points: usize,
        radius: f32,
    ) -> PyResult<(Self, Py<PyAny>)> {
        use numpy::ndarray::Array1;
        use numpy::IntoPyArray;

        let (filtered, mask) = self
            .inner
            .remove_radius_outlier(nb_points, radius)
            .map_err(PyErr::from)?;
        let mask_np = Array1::from_vec(mask).into_pyarray(py).into_any().unbind();
        Ok((PyPointCloud { inner: filtered }, mask_np))
    }

    fn octree(&self, max_depth: u8) -> PyResult<PyOctree> {
        Ok(PyOctree {
            inner: self.inner.octree(max_depth).map_err(PyErr::from)?,
        })
    }

    #[pyo3(signature = (device))]
    fn to(&self, device: &str) -> PyResult<Self> {
        let device = match device {
            "cpu" => crate::utils::tensor::cpu_device(),
            "gpu" => crate::utils::tensor::gpu_device().map_err(PyErr::from)?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "device must be 'cpu' or 'gpu'",
                ))
            }
        };
        Ok(PyPointCloud {
            inner: self.inner.to_device(device),
        })
    }

    fn device(&self) -> String {
        self.inner.device_name()
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

    #[staticmethod]
    #[pyo3(signature = (path, delimiter = 44, x = None, y = None, z = None, intensity = None, rgb_r = None, rgb_g = None, rgb_b = None))]
    fn from_csv(
        path: &str,
        delimiter: u8,
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> PyResult<Self> {
        let columns = table_columns(x, y, z, intensity, rgb_r, rgb_g, rgb_b);
        let inner = HighPerformancePointCloud::from_table_csv(path, delimiter, columns)
            .map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (path, x = None, y = None, z = None, intensity = None, rgb_r = None, rgb_g = None, rgb_b = None))]
    fn from_parquet(
        path: &str,
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> PyResult<Self> {
        let columns = table_columns(x, y, z, intensity, rgb_r, rgb_g, rgb_b);
        let inner =
            HighPerformancePointCloud::from_table_parquet(path, columns).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    #[staticmethod]
    #[pyo3(signature = (path, x = None, y = None, z = None, intensity = None, rgb_r = None, rgb_g = None, rgb_b = None))]
    fn load_from_file(
        path: &str,
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> PyResult<Self> {
        let columns = table_columns(x, y, z, intensity, rgb_r, rgb_g, rgb_b);
        let inner =
            HighPerformancePointCloud::load_from_file(path, Some(columns)).map_err(PyErr::from)?;
        Ok(PyPointCloud { inner })
    }

    #[pyo3(signature = (path, compress = false, *, point_format = None, las_version = "1.4", drop_waveform = false))]
    fn to_las(
        &self,
        path: &str,
        compress: bool,
        point_format: Option<u8>,
        las_version: &str,
        drop_waveform: bool,
    ) -> PyResult<()> {
        self.inner
            .to_las_with_options(
                path,
                compress,
                point_format,
                Some(las_version),
                drop_waveform,
            )
            .map_err(PyErr::from)?;
        Ok(())
    }

    #[pyo3(signature = (path, delimiter = 44, x = None, y = None, z = None, intensity = None, rgb_r = None, rgb_g = None, rgb_b = None))]
    fn to_csv(
        &self,
        path: &str,
        delimiter: u8,
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> PyResult<()> {
        let columns = table_columns(x, y, z, intensity, rgb_r, rgb_g, rgb_b);
        self.inner
            .to_table_csv(path, delimiter, columns)
            .map_err(PyErr::from)?;
        Ok(())
    }

    #[pyo3(signature = (path, x = None, y = None, z = None, intensity = None, rgb_r = None, rgb_g = None, rgb_b = None))]
    fn to_parquet(
        &self,
        path: &str,
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> PyResult<()> {
        let columns = table_columns(x, y, z, intensity, rgb_r, rgb_g, rgb_b);
        self.inner
            .to_table_parquet(path, columns)
            .map_err(PyErr::from)?;
        Ok(())
    }

    #[pyo3(signature = (path, x = None, y = None, z = None, intensity = None, rgb_r = None, rgb_g = None, rgb_b = None))]
    fn save_to_file(
        &self,
        path: &str,
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> PyResult<()> {
        let columns = table_columns(x, y, z, intensity, rgb_r, rgb_g, rgb_b);
        self.inner
            .save_to_file(path, Some(columns))
            .map_err(PyErr::from)?;
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
}

// === NormalSearch Python class ===

#[pyclass(name = "NormalSearch")]
pub struct PyNormalSearch {
    inner: NormalSearch,
}

#[pymethods]
impl PyNormalSearch {
    #[staticmethod]
    fn knn(k: usize) -> Self {
        Self {
            inner: NormalSearch::Knn(k),
        }
    }

    #[staticmethod]
    fn radius(radius: f32) -> Self {
        Self {
            inner: NormalSearch::Radius(radius),
        }
    }

    #[staticmethod]
    fn hybrid(radius: f32, k: usize) -> Self {
        Self {
            inner: NormalSearch::Hybrid(radius, k),
        }
    }
}

// === Octree Python class ===

#[pyclass(name = "Octree")]
pub struct PyOctree {
    inner: Octree,
}

#[pymethods]
impl PyOctree {
    fn range_search(&self, center: Vec<f32>, radius: f32) -> PyResult<Vec<u64>> {
        if center.len() != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "center must have length 3",
            ));
        }
        self.inner
            .range_search(&[center[0], center[1], center[2]], radius)
            .map_err(PyErr::from)
    }

    fn voxel_centers(&self, py: Python) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Array2;
        use numpy::IntoPyArray;

        let centers = self.inner.voxel_centers();
        let flat: Vec<f32> = centers.iter().flat_map(|p| p.iter().copied()).collect();
        let arr = Array2::from_shape_vec((centers.len(), 3), flat)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py).into_any().unbind())
    }
}

// === Registration Python classes and functions ===

#[pyclass(name = "ICPConvergenceCriteria")]
#[derive(Clone)]
pub struct PyICPConvergenceCriteria {
    inner: ICPConvergenceCriteria,
}

#[pymethods]
impl PyICPConvergenceCriteria {
    #[new]
    #[pyo3(signature = (max_iteration = 30, relative_fitness = 1e-6, relative_rmse = 1e-6))]
    fn new(max_iteration: usize, relative_fitness: f32, relative_rmse: f32) -> Self {
        Self {
            inner: ICPConvergenceCriteria {
                max_iteration,
                relative_fitness,
                relative_rmse,
            },
        }
    }
}

#[pyclass(name = "TransformationEstimation")]
#[derive(Clone)]
pub struct PyTransformationEstimation {
    inner: TransformationEstimation,
}

#[pymethods]
impl PyTransformationEstimation {
    #[staticmethod]
    fn point_to_point() -> Self {
        Self {
            inner: TransformationEstimation::PointToPoint,
        }
    }

    #[staticmethod]
    fn point_to_plane() -> Self {
        Self {
            inner: TransformationEstimation::PointToPlane,
        }
    }

    #[staticmethod]
    #[pyo3(signature = (epsilon = 1e-3))]
    fn generalized(epsilon: f32) -> Self {
        Self {
            inner: TransformationEstimation::Generalized { epsilon },
        }
    }
}

#[pyclass(name = "RegistrationResult")]
#[derive(Clone)]
pub struct PyRegistrationResult {
    inner: RegistrationResult,
}

#[pymethods]
impl PyRegistrationResult {
    #[getter]
    fn transformation(&self, py: Python) -> PyResult<Py<PyAny>> {
        use numpy::ndarray::Array2;
        use numpy::IntoPyArray;

        let mut row_major = Vec::with_capacity(16);
        for row in 0..4 {
            for col in 0..4 {
                row_major.push(self.inner.transformation[(row, col)]);
            }
        }
        let arr = Array2::from_shape_vec((4, 4), row_major)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(arr.into_pyarray(py).into_any().unbind())
    }

    #[getter]
    fn fitness(&self) -> f32 {
        self.inner.fitness
    }

    #[getter]
    fn inlier_rmse(&self) -> f32 {
        self.inner.inlier_rmse
    }

    #[getter]
    fn correspondence_set(&self) -> Vec<(u64, u64)> {
        self.inner.correspondence_set.clone()
    }
}

#[pyfunction(name = "icp")]
#[pyo3(signature = (source, target, max_correspondence_distance, init, estimation = None, criteria = None))]
fn py_icp(
    source: PyRef<'_, PyPointCloud>,
    target: PyRef<'_, PyPointCloud>,
    max_correspondence_distance: f32,
    init: &Bound<'_, pyo3::PyAny>,
    estimation: Option<PyRef<'_, PyTransformationEstimation>>,
    criteria: Option<PyRef<'_, PyICPConvergenceCriteria>>,
) -> PyResult<PyRegistrationResult> {
    let init = read_matrix4(init)?;
    let estimation = estimation
        .as_ref()
        .map(|e| e.inner.clone())
        .unwrap_or(TransformationEstimation::PointToPoint);
    let criteria = criteria
        .as_ref()
        .map(|c| c.inner.clone())
        .unwrap_or_default();
    let inner = registration::icp(
        &source.inner,
        &target.inner,
        max_correspondence_distance,
        init,
        estimation,
        criteria,
    )
    .map_err(PyErr::from)?;
    Ok(PyRegistrationResult { inner })
}

#[pyfunction(name = "evaluate")]
fn py_evaluate(
    source: PyRef<'_, PyPointCloud>,
    target: PyRef<'_, PyPointCloud>,
    max_correspondence_distance: f32,
    transformation: &Bound<'_, pyo3::PyAny>,
) -> PyResult<PyRegistrationResult> {
    let transformation = read_matrix4(transformation)?;
    let inner = registration::evaluate(
        &source.inner,
        &target.inner,
        max_correspondence_distance,
        transformation,
    )
    .map_err(PyErr::from)?;
    Ok(PyRegistrationResult { inner })
}

fn read_query_points(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<Vec<[f32; 3]>> {
    use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};

    if let Ok(arr) = obj.cast::<PyArray2<f32>>() {
        let shape = arr.shape();
        if shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "query must have shape [N,3]",
            ));
        }
        let readonly = arr.readonly();
        let slice = readonly
            .as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("query must be contiguous"))?;
        return Ok(slice
            .chunks_exact(3)
            .map(|row| [row[0], row[1], row[2]])
            .collect());
    }

    if let Ok(arr) = obj.cast::<PyArray2<f64>>() {
        let shape = arr.shape();
        if shape[1] != 3 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "query must have shape [N,3]",
            ));
        }
        let readonly = arr.readonly();
        let slice = readonly
            .as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("query must be contiguous"))?;
        return Ok(slice
            .chunks_exact(3)
            .map(|row| [row[0] as f32, row[1] as f32, row[2] as f32])
            .collect());
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "query must be a float32 or float64 numpy array",
    ))
}

fn read_matrix4(obj: &Bound<'_, pyo3::PyAny>) -> PyResult<nalgebra::Matrix4<f32>> {
    use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};

    if let Ok(arr) = obj.cast::<PyArray2<f32>>() {
        let shape = arr.shape();
        if shape != [4, 4] {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "matrix must have shape [4,4]",
            ));
        }
        let readonly = arr.readonly();
        let slice = readonly
            .as_slice()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("matrix must be contiguous"))?;
        return Ok(nalgebra::Matrix4::from_row_slice(slice));
    }

    let rows: Vec<Vec<f32>> = obj.extract()?;
    if rows.len() != 4 || !rows.iter().all(|row| row.len() == 4) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "matrix must have shape [4,4]",
        ));
    }
    Ok(nalgebra::Matrix4::from_row_slice(
        &rows.into_iter().flatten().collect::<Vec<_>>(),
    ))
}

fn table_columns(
    x: Option<String>,
    y: Option<String>,
    z: Option<String>,
    intensity: Option<String>,
    rgb_r: Option<String>,
    rgb_g: Option<String>,
    rgb_b: Option<String>,
) -> TableColumnNames {
    TableColumnNames::resolve(x, y, z, intensity, rgb_r, rgb_g, rgb_b)
}
