use crate::point_cloud::attribute_value::AttributeValue;
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::{PointCloudError, Result};
use las::point::Format;
use las::{Builder, Color, Point, Reader, Writer};
use std::fs;

impl HighPerformancePointCloud {
    pub fn from_las_laz(path: &str) -> Result<Self> {
        let mut reader =
            Reader::from_path(path).map_err(|e| format!("cannot read LAS file: {}", e))?;

        let mut xyz_vec: Vec<[f32; 3]> = Vec::new();
        let mut intensity_vec: Vec<f32> = Vec::new();
        let mut classification_vec: Vec<u8> = Vec::new();
        let mut return_number_vec: Vec<u8> = Vec::new();
        let mut number_of_returns_vec: Vec<u8> = Vec::new();
        let mut gps_time_vec: Vec<f64> = Vec::new();
        let mut rgb_r_vec: Vec<u8> = Vec::new();
        let mut rgb_g_vec: Vec<u8> = Vec::new();
        let mut rgb_b_vec: Vec<u8> = Vec::new();

        let has_color = reader.header().point_format().has_color;
        let has_gps_time = reader.header().point_format().has_gps_time;

        for point_result in reader.points() {
            let point = point_result.map_err(|_| "failed to read LAS point".to_string())?;

            xyz_vec.push([point.x as f32, point.y as f32, point.z as f32]);
            intensity_vec.push(point.intensity as f32 / 65535.0);
            classification_vec.push(point.classification.into());
            return_number_vec.push(point.return_number);
            number_of_returns_vec.push(point.number_of_returns);

            if has_gps_time {
                gps_time_vec.push(point.gps_time.unwrap_or(0.0));
            }

            if has_color {
                if let Some(color) = point.color {
                    rgb_r_vec.push((color.red >> 8) as u8);
                    rgb_g_vec.push((color.green >> 8) as u8);
                    rgb_b_vec.push((color.blue >> 8) as u8);
                } else {
                    rgb_r_vec.push(0);
                    rgb_g_vec.push(0);
                    rgb_b_vec.push(0);
                }
            }
        }

        if xyz_vec.is_empty() {
            return Err(PointCloudError::TensorShapeError(
                "LAS file contains no points".to_string(),
            ));
        }

        let mut result = Self::from_xyz_vec(xyz_vec)?;
        let n = result.point_count();

        // Store all LAS attributes as typed attributes
        if intensity_vec.len() == n {
            result
                .attributes_mut()
                .insert("intensity".to_string(), AttributeValue::F32(intensity_vec));
        }
        if classification_vec.len() == n {
            result.attributes_mut().insert(
                "classification".to_string(),
                AttributeValue::U8(classification_vec),
            );
        }
        if return_number_vec.len() == n {
            result.attributes_mut().insert(
                "return_number".to_string(),
                AttributeValue::U8(return_number_vec),
            );
        }
        if number_of_returns_vec.len() == n {
            result.attributes_mut().insert(
                "number_of_returns".to_string(),
                AttributeValue::U8(number_of_returns_vec),
            );
        }
        if gps_time_vec.len() == n {
            result
                .attributes_mut()
                .insert("gps_time".to_string(), AttributeValue::F64(gps_time_vec));
        }
        if rgb_r_vec.len() == n {
            result
                .attributes_mut()
                .insert("red".to_string(), AttributeValue::U8(rgb_r_vec));
            result
                .attributes_mut()
                .insert("green".to_string(), AttributeValue::U8(rgb_g_vec));
            result
                .attributes_mut()
                .insert("blue".to_string(), AttributeValue::U8(rgb_b_vec));
        }

        Ok(result)
    }

    pub fn to_las(&self, path: &str, compress: bool) -> Result<()> {
        if self.point_count() == 0 {
            return Err("point cloud is empty".into());
        }

        if let Some(parent) = std::path::Path::new(path).parent() {
            if !parent.exists() {
                fs::create_dir_all(parent).map_err(PointCloudError::IoError)?;
            }
        }

        let has_rgb = self.has_rgb();
        let has_gps_time = self.attributes().contains_key("gps_time");
        let format_id = match (has_rgb, has_gps_time) {
            (true, true) => 3,
            (true, false) => 2,
            (false, true) => 1,
            (false, false) => 0,
        };

        let mut builder = Builder::from((1, 4));
        let mut format = Format::new(format_id).map_err(|e| e.to_string())?;
        format.is_compressed = compress || path.to_lowercase().ends_with(".laz");
        builder.point_format = format;
        let header = builder.into_header().map_err(|e| e.to_string())?;

        let mut writer = Writer::from_path(path, header).map_err(|e| e.to_string())?;

        let xyz = self.get_xyz_vec();
        let intensity = self.get_attribute("intensity");
        let classification = self.get_attribute("classification");
        let return_number = self.get_attribute("return_number");
        let number_of_returns = self.get_attribute("number_of_returns");
        let gps_time = self.get_attribute("gps_time");
        let red = self.get_attribute("red");
        let green = self.get_attribute("green");
        let blue = self.get_attribute("blue");

        for (idx, point_xyz) in xyz.iter().enumerate() {
            let mut point = Point {
                x: point_xyz[0] as f64,
                y: point_xyz[1] as f64,
                z: point_xyz[2] as f64,
                ..Default::default()
            };

            if let Some(attr) = intensity {
                if let Some(v) = attr.as_f32() {
                    point.intensity = (v[idx] * 65535.0).clamp(0.0, 65535.0) as u16;
                }
            }

            if let Some(attr) = classification {
                if let Some(v) = attr.as_u8() {
                    point.classification =
                        las::point::Classification::new(v[idx]).unwrap_or_default();
                }
            }

            if let Some(attr) = return_number {
                if let Some(v) = attr.as_u8() {
                    point.return_number = v[idx];
                }
            }

            if let Some(attr) = number_of_returns {
                if let Some(v) = attr.as_u8() {
                    point.number_of_returns = v[idx];
                }
            }

            if let Some(attr) = gps_time {
                if let Some(v) = attr.as_f64() {
                    point.gps_time = Some(v[idx]);
                }
            }

            if let (Some(r_attr), Some(g_attr), Some(b_attr)) = (red, green, blue) {
                if let (Some(r_vec), Some(g_vec), Some(b_vec)) =
                    (r_attr.as_u8(), g_attr.as_u8(), b_attr.as_u8())
                {
                    point.color = Some(Color {
                        red: (r_vec[idx] as u16) << 8,
                        green: (g_vec[idx] as u16) << 8,
                        blue: (b_vec[idx] as u16) << 8,
                    });
                }
            }

            writer.write_point(point).map_err(|e| e.to_string())?;
        }

        writer.close().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn delete_file(path: &str) -> Result<()> {
        if fs::metadata(path).is_err() {
            return Err(format!("file not found: {}", path).into());
        }
        fs::remove_file(path).map_err(PointCloudError::IoError)?;
        Ok(())
    }
}
