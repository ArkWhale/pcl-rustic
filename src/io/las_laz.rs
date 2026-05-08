use crate::point_cloud::attribute_value::AttributeValue;
use crate::point_cloud::core::{HighPerformancePointCloud, LasCoordinateMetadata};
use crate::utils::error::{PointCloudError, Result};
use las::point::{Format, ScanDirection};
use las::raw::point::Waveform;
use las::{raw, Builder, Color, Point, Reader, Version, Vlr, Writer};
use std::fs;
use std::io::{Read, Seek, SeekFrom};

const EXTRA_BYTES_USER_ID: &str = "pcl-rustic";
const EXTRA_BYTES_RECORD_ID: u16 = 1;
const LAS_STANDARD_EXTRA_BYTES_USER_ID: &str = "LASF_Spec";
const LAS_STANDARD_EXTRA_BYTES_RECORD_ID: u16 = 4;
const LAS14_SCAN_ANGLE_SCALE: f32 = 0.006;
const STANDARD_LAS_ATTRIBUTES: &[&str] = &[
    "intensity",
    "return_number",
    "number_of_returns",
    "synthetic",
    "key_point",
    "withheld",
    "overlap",
    "scanner_channel",
    "scan_direction_flag",
    "edge_of_flight_line",
    "classification",
    "user_data",
    "scan_angle",
    "point_source_id",
    "gps_time",
    "red",
    "green",
    "blue",
    "nir",
    "wavepacket_index",
    "wavepacket_offset",
    "wavepacket_size",
    "return_point_wave_location",
    "x_t",
    "y_t",
    "z_t",
];

#[derive(Clone, Debug)]
struct ExtraByteField {
    name: String,
    dtype: String,
    offset: usize,
    size: usize,
}

impl HighPerformancePointCloud {
    pub fn from_las_laz(path: &str) -> Result<Self> {
        let mut reader =
            Reader::from_path(path).map_err(|e| format!("cannot read LAS file: {}", e))?;

        let mut xyz_vec: Vec<[f32; 3]> = Vec::new();
        let point_format_id = reader.header().point_format().to_u8().unwrap_or_default();
        let is_point_format_10 = point_format_id == 10;

        let mut intensity_f32_vec: Vec<f32> = Vec::new();
        let mut intensity_u16_vec: Vec<u16> = Vec::new();
        let mut classification_vec: Vec<u8> = Vec::new();
        let mut return_number_vec: Vec<u8> = Vec::new();
        let mut number_of_returns_vec: Vec<u8> = Vec::new();
        let mut synthetic_vec: Vec<bool> = Vec::new();
        let mut key_point_vec: Vec<bool> = Vec::new();
        let mut withheld_vec: Vec<bool> = Vec::new();
        let mut overlap_vec: Vec<bool> = Vec::new();
        let mut scanner_channel_vec: Vec<u8> = Vec::new();
        let mut scan_direction_flag_vec: Vec<bool> = Vec::new();
        let mut edge_of_flight_line_vec: Vec<bool> = Vec::new();
        let mut user_data_vec: Vec<u8> = Vec::new();
        let mut scan_angle_vec: Vec<i16> = Vec::new();
        let mut point_source_id_vec: Vec<u16> = Vec::new();
        let mut gps_time_vec: Vec<f64> = Vec::new();
        let mut rgb_r_vec: Vec<u8> = Vec::new();
        let mut rgb_g_vec: Vec<u8> = Vec::new();
        let mut rgb_b_vec: Vec<u8> = Vec::new();
        let mut rgb_r_u16_vec: Vec<u16> = Vec::new();
        let mut rgb_g_u16_vec: Vec<u16> = Vec::new();
        let mut rgb_b_u16_vec: Vec<u16> = Vec::new();
        let mut nir_vec: Vec<u16> = Vec::new();
        let mut wavepacket_index_vec: Vec<u8> = Vec::new();
        let mut wavepacket_offset_vec: Vec<u64> = Vec::new();
        let mut wavepacket_size_vec: Vec<u32> = Vec::new();
        let mut return_point_wave_location_vec: Vec<f32> = Vec::new();
        let mut x_t_vec: Vec<f32> = Vec::new();
        let mut y_t_vec: Vec<f32> = Vec::new();
        let mut z_t_vec: Vec<f32> = Vec::new();

        let has_color = reader.header().point_format().has_color;
        let has_gps_time = reader.header().point_format().has_gps_time;
        let has_nir = reader.header().point_format().has_nir;
        let has_waveform = reader.header().point_format().has_waveform;
        let extra_fields = extra_byte_fields(reader.header().vlrs())?;
        let mut extra_columns: Vec<Vec<u8>> = extra_fields.iter().map(|_| Vec::new()).collect();

        for point_result in reader.points() {
            let point = point_result.map_err(|_| "failed to read LAS point".to_string())?;

            xyz_vec.push([point.x as f32, point.y as f32, point.z as f32]);
            if is_point_format_10 {
                intensity_u16_vec.push(point.intensity);
            } else {
                intensity_f32_vec.push(point.intensity as f32 / 65535.0);
            }
            classification_vec.push(point.classification.into());
            return_number_vec.push(point.return_number);
            number_of_returns_vec.push(point.number_of_returns);
            if is_point_format_10 {
                synthetic_vec.push(point.is_synthetic);
                key_point_vec.push(point.is_key_point);
                withheld_vec.push(point.is_withheld);
                overlap_vec.push(point.is_overlap);
                scanner_channel_vec.push(point.scanner_channel);
                scan_direction_flag_vec.push(point.scan_direction == ScanDirection::LeftToRight);
                edge_of_flight_line_vec.push(point.is_edge_of_flight_line);
                user_data_vec.push(point.user_data);
                scan_angle_vec.push((point.scan_angle / LAS14_SCAN_ANGLE_SCALE).round() as i16);
                point_source_id_vec.push(point.point_source_id);
            }

            if has_gps_time {
                gps_time_vec.push(point.gps_time.unwrap_or(0.0));
            }

            if has_color {
                if let Some(color) = point.color {
                    if is_point_format_10 {
                        rgb_r_u16_vec.push(color.red);
                        rgb_g_u16_vec.push(color.green);
                        rgb_b_u16_vec.push(color.blue);
                    } else {
                        rgb_r_vec.push((color.red >> 8) as u8);
                        rgb_g_vec.push((color.green >> 8) as u8);
                        rgb_b_vec.push((color.blue >> 8) as u8);
                    }
                } else {
                    if is_point_format_10 {
                        rgb_r_u16_vec.push(0);
                        rgb_g_u16_vec.push(0);
                        rgb_b_u16_vec.push(0);
                    } else {
                        rgb_r_vec.push(0);
                        rgb_g_vec.push(0);
                        rgb_b_vec.push(0);
                    }
                }
            }

            if has_nir {
                nir_vec.push(point.nir.unwrap_or(0));
            }
            if has_waveform {
                let waveform = point.waveform.unwrap_or_default();
                wavepacket_index_vec.push(waveform.wave_packet_descriptor_index);
                wavepacket_offset_vec.push(waveform.byte_offset_to_waveform_data);
                wavepacket_size_vec.push(waveform.waveform_packet_size_in_bytes);
                return_point_wave_location_vec.push(waveform.return_point_waveform_location);
                x_t_vec.push(waveform.x_t);
                y_t_vec.push(waveform.y_t);
                z_t_vec.push(waveform.z_t);
            }

            for (field_idx, field) in extra_fields.iter().enumerate() {
                let end = field.offset + field.size;
                if point.extra_bytes.len() >= end {
                    extra_columns[field_idx]
                        .extend_from_slice(&point.extra_bytes[field.offset..end]);
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
        if intensity_u16_vec.len() == n {
            result.attributes_mut().insert(
                "intensity".to_string(),
                AttributeValue::U16(intensity_u16_vec),
            );
        } else if intensity_f32_vec.len() == n {
            result.attributes_mut().insert(
                "intensity".to_string(),
                AttributeValue::F32(intensity_f32_vec),
            );
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
        insert_bool_attr(&mut result, "synthetic", synthetic_vec, n);
        insert_bool_attr(&mut result, "key_point", key_point_vec, n);
        insert_bool_attr(&mut result, "withheld", withheld_vec, n);
        insert_bool_attr(&mut result, "overlap", overlap_vec, n);
        insert_u8_attr(&mut result, "scanner_channel", scanner_channel_vec, n);
        insert_bool_attr(
            &mut result,
            "scan_direction_flag",
            scan_direction_flag_vec,
            n,
        );
        insert_bool_attr(
            &mut result,
            "edge_of_flight_line",
            edge_of_flight_line_vec,
            n,
        );
        insert_u8_attr(&mut result, "user_data", user_data_vec, n);
        if scan_angle_vec.len() == n {
            result.attributes_mut().insert(
                "scan_angle".to_string(),
                AttributeValue::I16(scan_angle_vec),
            );
        }
        if point_source_id_vec.len() == n {
            result.attributes_mut().insert(
                "point_source_id".to_string(),
                AttributeValue::U16(point_source_id_vec),
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
        if rgb_r_u16_vec.len() == n {
            result
                .attributes_mut()
                .insert("red".to_string(), AttributeValue::U16(rgb_r_u16_vec));
            result
                .attributes_mut()
                .insert("green".to_string(), AttributeValue::U16(rgb_g_u16_vec));
            result
                .attributes_mut()
                .insert("blue".to_string(), AttributeValue::U16(rgb_b_u16_vec));
        }
        if nir_vec.len() == n {
            result
                .attributes_mut()
                .insert("nir".to_string(), AttributeValue::U16(nir_vec));
        }
        insert_u8_attr(&mut result, "wavepacket_index", wavepacket_index_vec, n);
        if wavepacket_offset_vec.len() == n {
            result.attributes_mut().insert(
                "wavepacket_offset".to_string(),
                AttributeValue::U64(wavepacket_offset_vec),
            );
        }
        if wavepacket_size_vec.len() == n {
            result.attributes_mut().insert(
                "wavepacket_size".to_string(),
                AttributeValue::U32(wavepacket_size_vec),
            );
        }
        insert_f32_attr(
            &mut result,
            "return_point_wave_location",
            return_point_wave_location_vec,
            n,
        );
        insert_f32_attr(&mut result, "x_t", x_t_vec, n);
        insert_f32_attr(&mut result, "y_t", y_t_vec, n);
        insert_f32_attr(&mut result, "z_t", z_t_vec, n);
        for (field, bytes) in extra_fields.iter().zip(extra_columns.iter()) {
            if let Some(attr) = decode_extra_attribute(&field.dtype, bytes, n)? {
                result.attributes_mut().insert(field.name.clone(), attr);
            }
        }
        if let Some(metadata) = read_las_coordinate_metadata(path, n)? {
            result.set_las_coordinate_metadata(metadata)?;
        }
        if is_point_format_10 {
            patch_point_format_10_raw_attributes(path, &mut result)?;
        }

        Ok(result)
    }

    pub fn to_las(&self, path: &str, compress: bool) -> Result<()> {
        self.to_las_with_options(path, compress, None, None, false)
    }

    pub fn to_las_with_options(
        &self,
        path: &str,
        compress: bool,
        point_format: Option<u8>,
        las_version: Option<&str>,
        drop_waveform: bool,
    ) -> Result<()> {
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
        let format_id = match point_format {
            Some(10) => {
                if las_version.unwrap_or("1.4") != "1.4" {
                    return Err(PointCloudError::InvalidParameter(
                        "point_format=10 requires las_version='1.4'".to_string(),
                    ));
                }
                10
            }
            Some(0..=3) => point_format.unwrap(),
            Some(id) => {
                return Err(PointCloudError::InvalidParameter(format!(
                    "unsupported LAS point_format {id}"
                )))
            }
            None => match (has_rgb, has_gps_time) {
                (true, true) => 3,
                (true, false) => 2,
                (false, true) => 1,
                (false, false) => 0,
            },
        };
        let is_point_format_10 = format_id == 10;

        let extra_fields = collect_extra_byte_fields(self.attributes())?;

        let version = if point_format.is_none() {
            Version::new(1, 4)
        } else {
            parse_las_version(las_version.unwrap_or("1.4"))?
        };
        let mut builder = Builder::from(version);
        let mut format = Format::new(format_id).map_err(|e| e.to_string())?;
        format.is_compressed = compress || path.to_lowercase().ends_with(".laz");
        format.extra_bytes = extra_fields.iter().map(|field| field.size as u16).sum();
        builder.point_format = format;
        if let Some(metadata) = self.las_coordinate_metadata() {
            builder.transforms = las::Vector {
                x: las::Transform {
                    scale: metadata.scale[0],
                    offset: metadata.offset[0],
                },
                y: las::Transform {
                    scale: metadata.scale[1],
                    offset: metadata.offset[1],
                },
                z: las::Transform {
                    scale: metadata.scale[2],
                    offset: metadata.offset[2],
                },
            };
        }
        if !extra_fields.is_empty() {
            builder.vlrs.push(extra_bytes_vlr(&extra_fields));
        }
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
        let nir = self.get_attribute("nir");

        for (idx, point_xyz) in xyz.iter().enumerate() {
            let xyz = self
                .las_coordinate_metadata()
                .map(|metadata| {
                    [
                        metadata.offset[0] + f64::from(metadata.raw_x[idx]) * metadata.scale[0],
                        metadata.offset[1] + f64::from(metadata.raw_y[idx]) * metadata.scale[1],
                        metadata.offset[2] + f64::from(metadata.raw_z[idx]) * metadata.scale[2],
                    ]
                })
                .unwrap_or([
                    point_xyz[0] as f64,
                    point_xyz[1] as f64,
                    point_xyz[2] as f64,
                ]);
            let mut point = Point {
                x: xyz[0],
                y: xyz[1],
                z: xyz[2],
                ..Default::default()
            };

            if let Some(attr) = intensity {
                point.intensity = intensity_to_u16(attr, idx)?;
            }

            if let Some(attr) = classification {
                if let Some(v) = attr.as_u8() {
                    point.classification =
                        las::point::Classification::new(v[idx]).unwrap_or_default();
                }
            }

            if let Some(attr) = return_number {
                if let Some(v) = attr.as_u8() {
                    if v[idx] > 15 {
                        return Err(PointCloudError::InvalidParameter(
                            "return_number must be in 0..=15".to_string(),
                        ));
                    }
                    point.return_number = v[idx];
                }
            }

            if let Some(attr) = number_of_returns {
                if let Some(v) = attr.as_u8() {
                    if v[idx] > 15 {
                        return Err(PointCloudError::InvalidParameter(
                            "number_of_returns must be in 0..=15".to_string(),
                        ));
                    }
                    point.number_of_returns = v[idx];
                }
            }

            point.is_synthetic = bool_at(self.get_attribute("synthetic"), idx)?;
            point.is_key_point = bool_at(self.get_attribute("key_point"), idx)?;
            point.is_withheld = bool_at(self.get_attribute("withheld"), idx)?;
            point.is_overlap = bool_at(self.get_attribute("overlap"), idx)?;
            point.scanner_channel = u8_at(self.get_attribute("scanner_channel"), idx, 0)?;
            if point.scanner_channel > 3 {
                return Err(PointCloudError::InvalidParameter(
                    "scanner_channel must be in 0..=3".to_string(),
                ));
            }
            point.scan_direction = if bool_at(self.get_attribute("scan_direction_flag"), idx)? {
                ScanDirection::LeftToRight
            } else {
                ScanDirection::RightToLeft
            };
            point.is_edge_of_flight_line = bool_at(self.get_attribute("edge_of_flight_line"), idx)?;
            point.user_data = u8_at(self.get_attribute("user_data"), idx, 0)?;
            point.point_source_id = u16_at(self.get_attribute("point_source_id"), idx, 0)?;
            if let Some(attr) = self.get_attribute("scan_angle") {
                if let Some(v) = attr.as_i16() {
                    point.scan_angle = v[idx] as f32 * LAS14_SCAN_ANGLE_SCALE;
                } else if let Some(v) = attr.as_f32() {
                    point.scan_angle = v[idx];
                }
            }

            if let Some(attr) = gps_time {
                if let Some(v) = attr.as_f64() {
                    point.gps_time = Some(v[idx]);
                }
            } else if is_point_format_10 {
                point.gps_time = Some(0.0);
            }

            let needs_color =
                is_point_format_10 || red.is_some() || green.is_some() || blue.is_some();
            if needs_color {
                point.color = Some(Color {
                    red: color_channel_to_u16(red, idx)?,
                    green: color_channel_to_u16(green, idx)?,
                    blue: color_channel_to_u16(blue, idx)?,
                });
            }

            if is_point_format_10 {
                point.nir = Some(u16_at(nir, idx, 0)?);
                point.waveform = Some(waveform_for_point(self.attributes(), idx, drop_waveform)?);
            }

            if !extra_fields.is_empty() {
                point.extra_bytes = encode_extra_bytes(self.attributes(), &extra_fields, idx)?;
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

fn insert_f32_attr(
    cloud: &mut HighPerformancePointCloud,
    name: &str,
    values: Vec<f32>,
    point_count: usize,
) {
    if values.len() == point_count {
        cloud
            .attributes_mut()
            .insert(name.to_string(), AttributeValue::F32(values));
    }
}

fn insert_u8_attr(
    cloud: &mut HighPerformancePointCloud,
    name: &str,
    values: Vec<u8>,
    point_count: usize,
) {
    if values.len() == point_count {
        cloud
            .attributes_mut()
            .insert(name.to_string(), AttributeValue::U8(values));
    }
}

fn insert_bool_attr(
    cloud: &mut HighPerformancePointCloud,
    name: &str,
    values: Vec<bool>,
    point_count: usize,
) {
    if values.len() == point_count {
        cloud
            .attributes_mut()
            .insert(name.to_string(), AttributeValue::Bool(values));
    }
}

fn parse_las_version(version: &str) -> Result<Version> {
    match version {
        "1.0" => Ok(Version::new(1, 0)),
        "1.1" => Ok(Version::new(1, 1)),
        "1.2" => Ok(Version::new(1, 2)),
        "1.3" => Ok(Version::new(1, 3)),
        "1.4" => Ok(Version::new(1, 4)),
        _ => Err(PointCloudError::InvalidParameter(format!(
            "unsupported LAS version '{version}'"
        ))),
    }
}

fn intensity_to_u16(attr: &AttributeValue, idx: usize) -> Result<u16> {
    match attr {
        AttributeValue::U16(v) => Ok(v[idx]),
        AttributeValue::F32(v) => Ok((v[idx] * 65535.0).clamp(0.0, 65535.0) as u16),
        AttributeValue::U8(v) => Ok((v[idx] as u16) << 8),
        _ => Err(PointCloudError::InvalidParameter(
            "intensity must be uint16, float32, or uint8 for LAS export".to_string(),
        )),
    }
}

fn color_channel_to_u16(attr: Option<&AttributeValue>, idx: usize) -> Result<u16> {
    let Some(attr) = attr else {
        return Ok(0);
    };
    match attr {
        AttributeValue::U16(v) => Ok(v[idx]),
        AttributeValue::U8(v) => Ok((v[idx] as u16) << 8),
        _ => Err(PointCloudError::InvalidParameter(
            "LAS color channels must be uint16 or uint8".to_string(),
        )),
    }
}

fn bool_at(attr: Option<&AttributeValue>, idx: usize) -> Result<bool> {
    let Some(attr) = attr else {
        return Ok(false);
    };
    attr.as_bool().map(|v| v[idx]).ok_or_else(|| {
        PointCloudError::InvalidParameter("LAS boolean attributes must be bool".to_string())
    })
}

fn u8_at(attr: Option<&AttributeValue>, idx: usize, default: u8) -> Result<u8> {
    let Some(attr) = attr else {
        return Ok(default);
    };
    attr.as_u8().map(|v| v[idx]).ok_or_else(|| {
        PointCloudError::InvalidParameter("LAS uint8 attributes must be uint8".to_string())
    })
}

fn u16_at(attr: Option<&AttributeValue>, idx: usize, default: u16) -> Result<u16> {
    let Some(attr) = attr else {
        return Ok(default);
    };
    attr.as_u16().map(|v| v[idx]).ok_or_else(|| {
        PointCloudError::InvalidParameter("LAS uint16 attributes must be uint16".to_string())
    })
}

fn waveform_for_point(
    attributes: &std::collections::HashMap<String, AttributeValue>,
    idx: usize,
    drop_waveform: bool,
) -> Result<Waveform> {
    if drop_waveform {
        return Ok(Waveform::default());
    }

    let wavepacket_index = u8_at(attributes.get("wavepacket_index"), idx, 0)?;
    let wavepacket_offset = u64_at(attributes.get("wavepacket_offset"), idx, 0)?;
    let wavepacket_size = u32_at(attributes.get("wavepacket_size"), idx, 0)?;
    let return_point_wave_location = f32_at(
        attributes.get("return_point_wave_location"),
        idx,
        0.0,
        "return_point_wave_location",
    )?;
    let x_t = f32_at(attributes.get("x_t"), idx, 0.0, "x_t")?;
    let y_t = f32_at(attributes.get("y_t"), idx, 0.0, "y_t")?;
    let z_t = f32_at(attributes.get("z_t"), idx, 0.0, "z_t")?;

    if wavepacket_index != 0
        || wavepacket_offset != 0
        || wavepacket_size != 0
        || return_point_wave_location != 0.0
        || x_t != 0.0
        || y_t != 0.0
        || z_t != 0.0
    {
        return Err(PointCloudError::InvalidParameter(
            "point format 10 waveform payload preservation is not implemented; use drop_waveform=True to write no-waveform defaults".to_string(),
        ));
    }

    Ok(Waveform::default())
}

fn u32_at(attr: Option<&AttributeValue>, idx: usize, default: u32) -> Result<u32> {
    let Some(attr) = attr else {
        return Ok(default);
    };
    attr.as_u32().map(|v| v[idx]).ok_or_else(|| {
        PointCloudError::InvalidParameter("LAS uint32 attributes must be uint32".to_string())
    })
}

fn u64_at(attr: Option<&AttributeValue>, idx: usize, default: u64) -> Result<u64> {
    let Some(attr) = attr else {
        return Ok(default);
    };
    attr.as_u64().map(|v| v[idx]).ok_or_else(|| {
        PointCloudError::InvalidParameter("LAS uint64 attributes must be uint64".to_string())
    })
}

fn f32_at(attr: Option<&AttributeValue>, idx: usize, default: f32, name: &str) -> Result<f32> {
    let Some(attr) = attr else {
        return Ok(default);
    };
    attr.as_f32().map(|v| v[idx]).ok_or_else(|| {
        PointCloudError::InvalidParameter(format!("LAS attribute '{name}' must be float32"))
    })
}

fn read_las_coordinate_metadata(
    path: &str,
    point_count: usize,
) -> Result<Option<LasCoordinateMetadata>> {
    let mut file = fs::File::open(path).map_err(PointCloudError::IoError)?;
    let raw_header = raw::Header::read_from(&mut file).map_err(|e| {
        PointCloudError::InvalidParameter(format!("failed to read LAS header: {e}"))
    })?;
    if raw_header.point_data_record_format & 0x80 != 0 {
        return Ok(None);
    }
    let record_len = raw_header.point_data_record_length as usize;
    if record_len < 12 {
        return Ok(None);
    }
    file.seek(SeekFrom::Start(u64::from(raw_header.offset_to_point_data)))
        .map_err(PointCloudError::IoError)?;

    let mut raw_x = Vec::with_capacity(point_count);
    let mut raw_y = Vec::with_capacity(point_count);
    let mut raw_z = Vec::with_capacity(point_count);
    let mut record = vec![0u8; record_len];
    for _ in 0..point_count {
        file.read_exact(&mut record)
            .map_err(PointCloudError::IoError)?;
        raw_x.push(i32::from_le_bytes(record[0..4].try_into().unwrap()));
        raw_y.push(i32::from_le_bytes(record[4..8].try_into().unwrap()));
        raw_z.push(i32::from_le_bytes(record[8..12].try_into().unwrap()));
    }

    Ok(Some(LasCoordinateMetadata {
        raw_x,
        raw_y,
        raw_z,
        scale: [
            raw_header.x_scale_factor,
            raw_header.y_scale_factor,
            raw_header.z_scale_factor,
        ],
        offset: [
            raw_header.x_offset,
            raw_header.y_offset,
            raw_header.z_offset,
        ],
    }))
}

fn patch_point_format_10_raw_attributes(
    path: &str,
    cloud: &mut HighPerformancePointCloud,
) -> Result<()> {
    let mut file = fs::File::open(path).map_err(PointCloudError::IoError)?;
    let raw_header = raw::Header::read_from(&mut file).map_err(|e| {
        PointCloudError::InvalidParameter(format!("failed to read LAS header: {e}"))
    })?;
    if raw_header.point_data_record_format & 0x3f != 10 {
        return Ok(());
    }
    if raw_header.point_data_record_format & 0x80 != 0 {
        return Ok(());
    }
    let record_len = raw_header.point_data_record_length as usize;
    if record_len < 67 {
        return Err(PointCloudError::InvalidParameter(
            "point format 10 record length is smaller than 67 bytes".to_string(),
        ));
    }

    let point_count = cloud.point_count();
    file.seek(SeekFrom::Start(u64::from(raw_header.offset_to_point_data)))
        .map_err(PointCloudError::IoError)?;

    let mut intensity = Vec::with_capacity(point_count);
    let mut return_number = Vec::with_capacity(point_count);
    let mut number_of_returns = Vec::with_capacity(point_count);
    let mut synthetic = Vec::with_capacity(point_count);
    let mut key_point = Vec::with_capacity(point_count);
    let mut withheld = Vec::with_capacity(point_count);
    let mut overlap = Vec::with_capacity(point_count);
    let mut scanner_channel = Vec::with_capacity(point_count);
    let mut scan_direction_flag = Vec::with_capacity(point_count);
    let mut edge_of_flight_line = Vec::with_capacity(point_count);
    let mut classification = Vec::with_capacity(point_count);
    let mut user_data = Vec::with_capacity(point_count);
    let mut scan_angle = Vec::with_capacity(point_count);
    let mut point_source_id = Vec::with_capacity(point_count);
    let mut gps_time = Vec::with_capacity(point_count);
    let mut red = Vec::with_capacity(point_count);
    let mut green = Vec::with_capacity(point_count);
    let mut blue = Vec::with_capacity(point_count);
    let mut nir = Vec::with_capacity(point_count);
    let mut wavepacket_index = Vec::with_capacity(point_count);
    let mut wavepacket_offset = Vec::with_capacity(point_count);
    let mut wavepacket_size = Vec::with_capacity(point_count);
    let mut return_point_wave_location = Vec::with_capacity(point_count);
    let mut x_t = Vec::with_capacity(point_count);
    let mut y_t = Vec::with_capacity(point_count);
    let mut z_t = Vec::with_capacity(point_count);
    let mut record = vec![0u8; record_len];

    for _ in 0..point_count {
        file.read_exact(&mut record)
            .map_err(PointCloudError::IoError)?;
        intensity.push(u16::from_le_bytes([record[12], record[13]]));
        return_number.push(record[14] & 15);
        number_of_returns.push((record[14] >> 4) & 15);
        synthetic.push(record[15] & 1 == 1);
        key_point.push(record[15] & 2 == 2);
        withheld.push(record[15] & 4 == 4);
        overlap.push(record[15] & 8 == 8);
        scanner_channel.push((record[15] >> 4) & 3);
        scan_direction_flag.push(record[15] & 64 == 64);
        edge_of_flight_line.push(record[15] & 128 == 128);
        classification.push(record[16]);
        user_data.push(record[17]);
        scan_angle.push(i16::from_le_bytes([record[18], record[19]]));
        point_source_id.push(u16::from_le_bytes([record[20], record[21]]));
        gps_time.push(f64::from_le_bytes(record[22..30].try_into().unwrap()));
        red.push(u16::from_le_bytes([record[30], record[31]]));
        green.push(u16::from_le_bytes([record[32], record[33]]));
        blue.push(u16::from_le_bytes([record[34], record[35]]));
        nir.push(u16::from_le_bytes([record[36], record[37]]));
        wavepacket_index.push(record[38]);
        wavepacket_offset.push(u64::from_le_bytes(record[39..47].try_into().unwrap()));
        wavepacket_size.push(u32::from_le_bytes(record[47..51].try_into().unwrap()));
        return_point_wave_location.push(f32::from_le_bytes(record[51..55].try_into().unwrap()));
        x_t.push(f32::from_le_bytes(record[55..59].try_into().unwrap()));
        y_t.push(f32::from_le_bytes(record[59..63].try_into().unwrap()));
        z_t.push(f32::from_le_bytes(record[63..67].try_into().unwrap()));
    }

    let attrs = cloud.attributes_mut();
    attrs.insert("intensity".to_string(), AttributeValue::U16(intensity));
    attrs.insert(
        "return_number".to_string(),
        AttributeValue::U8(return_number),
    );
    attrs.insert(
        "number_of_returns".to_string(),
        AttributeValue::U8(number_of_returns),
    );
    attrs.insert("synthetic".to_string(), AttributeValue::Bool(synthetic));
    attrs.insert("key_point".to_string(), AttributeValue::Bool(key_point));
    attrs.insert("withheld".to_string(), AttributeValue::Bool(withheld));
    attrs.insert("overlap".to_string(), AttributeValue::Bool(overlap));
    attrs.insert(
        "scanner_channel".to_string(),
        AttributeValue::U8(scanner_channel),
    );
    attrs.insert(
        "scan_direction_flag".to_string(),
        AttributeValue::Bool(scan_direction_flag),
    );
    attrs.insert(
        "edge_of_flight_line".to_string(),
        AttributeValue::Bool(edge_of_flight_line),
    );
    attrs.insert(
        "classification".to_string(),
        AttributeValue::U8(classification),
    );
    attrs.insert("user_data".to_string(), AttributeValue::U8(user_data));
    attrs.insert("scan_angle".to_string(), AttributeValue::I16(scan_angle));
    attrs.insert(
        "point_source_id".to_string(),
        AttributeValue::U16(point_source_id),
    );
    attrs.insert("gps_time".to_string(), AttributeValue::F64(gps_time));
    attrs.insert("red".to_string(), AttributeValue::U16(red));
    attrs.insert("green".to_string(), AttributeValue::U16(green));
    attrs.insert("blue".to_string(), AttributeValue::U16(blue));
    attrs.insert("nir".to_string(), AttributeValue::U16(nir));
    attrs.insert(
        "wavepacket_index".to_string(),
        AttributeValue::U8(wavepacket_index),
    );
    attrs.insert(
        "wavepacket_offset".to_string(),
        AttributeValue::U64(wavepacket_offset),
    );
    attrs.insert(
        "wavepacket_size".to_string(),
        AttributeValue::U32(wavepacket_size),
    );
    attrs.insert(
        "return_point_wave_location".to_string(),
        AttributeValue::F32(return_point_wave_location),
    );
    attrs.insert("x_t".to_string(), AttributeValue::F32(x_t));
    attrs.insert("y_t".to_string(), AttributeValue::F32(y_t));
    attrs.insert("z_t".to_string(), AttributeValue::F32(z_t));
    Ok(())
}

fn collect_extra_byte_fields(
    attributes: &std::collections::HashMap<String, AttributeValue>,
) -> Result<Vec<ExtraByteField>> {
    let mut fields = Vec::new();
    let mut offset = 0usize;
    let mut names: Vec<&String> = attributes.keys().collect();
    names.sort();
    for name in names {
        if STANDARD_LAS_ATTRIBUTES.contains(&name.as_str()) {
            continue;
        }
        let attr = &attributes[name];
        let dtype = match attr {
            AttributeValue::F32(_) => "float32",
            AttributeValue::F64(_) => "float64",
            AttributeValue::U8(_) => "uint8",
            AttributeValue::U16(_) => "uint16",
            AttributeValue::U32(_) => "uint32",
            AttributeValue::U64(_) => "uint64",
            AttributeValue::I16(_) => "int16",
            AttributeValue::I32(_) => "int32",
            AttributeValue::I64(_) => "int64",
            AttributeValue::Bool(_) => "bool",
            AttributeValue::F32x6(_) => continue,
        };
        let size = attr.bytes_per_element();
        fields.push(ExtraByteField {
            name: name.clone(),
            dtype: dtype.to_string(),
            offset,
            size,
        });
        offset += size;
    }
    if offset > u16::MAX as usize {
        return Err(PointCloudError::InvalidParameter(
            "LAS ExtraBytes payload exceeds u16::MAX bytes per point".to_string(),
        ));
    }
    Ok(fields)
}

fn extra_bytes_vlr(fields: &[ExtraByteField]) -> Vlr {
    let lines = fields
        .iter()
        .map(|field| format!("{},{},{}", field.name, field.dtype, field.size))
        .collect::<Vec<_>>()
        .join("\n");
    Vlr {
        user_id: EXTRA_BYTES_USER_ID.to_string(),
        record_id: EXTRA_BYTES_RECORD_ID,
        description: "pcl-rustic ExtraBytes schema".to_string(),
        data: lines.into_bytes(),
    }
}

fn extra_byte_fields(vlrs: &[Vlr]) -> Result<Vec<ExtraByteField>> {
    if let Some(vlr) = vlrs
        .iter()
        .find(|vlr| vlr.user_id == EXTRA_BYTES_USER_ID && vlr.record_id == EXTRA_BYTES_RECORD_ID)
    {
        let schema = std::str::from_utf8(&vlr.data).map_err(|e| {
            PointCloudError::InvalidParameter(format!("invalid ExtraBytes schema: {e}"))
        })?;
        let mut fields = Vec::new();
        let mut offset = 0usize;
        for line in schema.lines().filter(|line| !line.trim().is_empty()) {
            let mut parts = line.split(',');
            let name = parts.next().unwrap_or_default();
            let dtype = parts.next().unwrap_or_default();
            let recorded_size = parts.next().unwrap_or_default();
            let size = extra_dtype_size(dtype).ok_or_else(|| {
                PointCloudError::InvalidParameter(format!("unsupported ExtraBytes dtype '{dtype}'"))
            })?;
            if parts.next().is_some()
                || name.is_empty()
                || recorded_size.parse::<usize>() != Ok(size)
            {
                return Err(PointCloudError::InvalidParameter(format!(
                    "invalid ExtraBytes schema line '{line}'"
                )));
            }
            push_extra_field(&mut fields, name, dtype, offset, size);
            offset += size;
        }
        return Ok(fields);
    }

    let Some(vlr) = vlrs.iter().find(|vlr| {
        vlr.user_id == LAS_STANDARD_EXTRA_BYTES_USER_ID
            && vlr.record_id == LAS_STANDARD_EXTRA_BYTES_RECORD_ID
    }) else {
        return Ok(Vec::new());
    };
    if vlr.data.len() % 192 != 0 {
        return Err(PointCloudError::InvalidParameter(
            "invalid LAS Extra Bytes VLR length".to_string(),
        ));
    }
    let mut fields = Vec::new();
    let mut offset = 0usize;
    for record in vlr.data.chunks_exact(192) {
        let data_type = record[2];
        let Some((dtype, size)) = standard_extra_dtype(data_type) else {
            return Err(PointCloudError::InvalidParameter(format!(
                "unsupported LAS Extra Bytes data type {data_type}"
            )));
        };
        let name_bytes = &record[4..36];
        let name_end = name_bytes
            .iter()
            .position(|&byte| byte == 0)
            .unwrap_or(name_bytes.len());
        let name = std::str::from_utf8(&name_bytes[..name_end])
            .map_err(|e| {
                PointCloudError::InvalidParameter(format!("invalid Extra Bytes name: {e}"))
            })?
            .trim();
        if !name.is_empty() {
            push_extra_field(&mut fields, name, dtype, offset, size);
        }
        offset += size;
    }
    Ok(fields)
}

fn push_extra_field(
    fields: &mut Vec<ExtraByteField>,
    name: &str,
    dtype: &str,
    offset: usize,
    size: usize,
) {
    let mut resolved = if STANDARD_LAS_ATTRIBUTES.contains(&name) {
        format!("extra_{name}")
    } else {
        name.to_string()
    };
    while fields.iter().any(|field| field.name == resolved) {
        resolved = format!("extra_{resolved}");
    }
    fields.push(ExtraByteField {
        name: resolved,
        dtype: dtype.to_string(),
        offset,
        size,
    });
}

fn standard_extra_dtype(data_type: u8) -> Option<(&'static str, usize)> {
    match data_type {
        1 => Some(("uint8", 1)),
        3 => Some(("uint16", 2)),
        4 => Some(("int16", 2)),
        5 => Some(("uint32", 4)),
        6 => Some(("int32", 4)),
        7 => Some(("uint64", 8)),
        8 => Some(("int64", 8)),
        9 => Some(("float32", 4)),
        10 => Some(("float64", 8)),
        _ => None,
    }
}

fn extra_dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "float32" | "uint32" | "int32" => Some(4),
        "float64" | "uint64" | "int64" => Some(8),
        "uint8" | "bool" => Some(1),
        "uint16" | "int16" => Some(2),
        _ => None,
    }
}

fn encode_extra_bytes(
    attributes: &std::collections::HashMap<String, AttributeValue>,
    fields: &[ExtraByteField],
    idx: usize,
) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(fields.iter().map(|field| field.size).sum());
    for field in fields {
        let attr = attributes.get(&field.name).ok_or_else(|| {
            PointCloudError::InvalidParameter(format!(
                "missing ExtraBytes attribute '{}'",
                field.name
            ))
        })?;
        match (field.dtype.as_str(), attr) {
            ("float32", AttributeValue::F32(v)) => out.extend_from_slice(&v[idx].to_le_bytes()),
            ("float64", AttributeValue::F64(v)) => out.extend_from_slice(&v[idx].to_le_bytes()),
            ("uint8", AttributeValue::U8(v)) => out.push(v[idx]),
            ("uint16", AttributeValue::U16(v)) => out.extend_from_slice(&v[idx].to_le_bytes()),
            ("uint32", AttributeValue::U32(v)) => out.extend_from_slice(&v[idx].to_le_bytes()),
            ("uint64", AttributeValue::U64(v)) => out.extend_from_slice(&v[idx].to_le_bytes()),
            ("int16", AttributeValue::I16(v)) => out.extend_from_slice(&v[idx].to_le_bytes()),
            ("int32", AttributeValue::I32(v)) => out.extend_from_slice(&v[idx].to_le_bytes()),
            ("int64", AttributeValue::I64(v)) => out.extend_from_slice(&v[idx].to_le_bytes()),
            ("bool", AttributeValue::Bool(v)) => out.push(u8::from(v[idx])),
            _ => {
                return Err(PointCloudError::InvalidParameter(format!(
                    "ExtraBytes dtype mismatch for '{}'",
                    field.name
                )))
            }
        }
    }
    Ok(out)
}

fn decode_extra_attribute(dtype: &str, bytes: &[u8], len: usize) -> Result<Option<AttributeValue>> {
    if len == 0 {
        return Ok(None);
    }
    match dtype {
        "float32" => Ok(Some(AttributeValue::F32(
            bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        "float64" => Ok(Some(AttributeValue::F64(
            bytes
                .chunks_exact(8)
                .map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        "uint8" => Ok(Some(AttributeValue::U8(bytes.to_vec()))),
        "uint16" => Ok(Some(AttributeValue::U16(
            bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        "uint32" => Ok(Some(AttributeValue::U32(
            bytes
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        "uint64" => Ok(Some(AttributeValue::U64(
            bytes
                .chunks_exact(8)
                .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        "int16" => Ok(Some(AttributeValue::I16(
            bytes
                .chunks_exact(2)
                .map(|chunk| i16::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        "int32" => Ok(Some(AttributeValue::I32(
            bytes
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        "int64" => Ok(Some(AttributeValue::I64(
            bytes
                .chunks_exact(8)
                .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
                .collect(),
        ))),
        "bool" => Ok(Some(AttributeValue::Bool(
            bytes.iter().map(|value| *value != 0).collect(),
        ))),
        _ => Err(PointCloudError::InvalidParameter(format!(
            "unsupported ExtraBytes dtype '{dtype}'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use las::raw::point::Waveform;
    use las::Version;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT_PATH_ID: AtomicUsize = AtomicUsize::new(0);

    fn temp_las_path(label: &str) -> PathBuf {
        let id = NEXT_PATH_ID.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "pcl_rustic_{label}_{}_{}.las",
            std::process::id(),
            id
        ))
    }

    #[test]
    fn point_format_10_xyz_only_export_materializes_default_option_groups() {
        let path = temp_las_path("pf10_xyz_only");
        let pc = HighPerformancePointCloud::from_xyz_vec(vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            .unwrap();

        pc.to_las_with_options(path.to_str().unwrap(), false, Some(10), Some("1.4"), false)
            .unwrap();

        let mut reader = Reader::from_path(&path).unwrap();
        assert_eq!(reader.header().version(), Version::new(1, 4));
        assert_eq!(reader.header().point_format().to_u8().unwrap(), 10);
        for point in reader.points() {
            let point = point.unwrap();
            assert_eq!(point.gps_time, Some(0.0));
            assert_eq!(point.color, Some(Color::default()));
            assert_eq!(point.waveform, Some(Waveform::default()));
        }
        let reloaded = HighPerformancePointCloud::from_las_laz(path.to_str().unwrap()).unwrap();
        assert_eq!(
            reloaded.get_attribute("nir").unwrap().as_u16().unwrap(),
            &vec![0, 0]
        );
        assert_eq!(
            reloaded
                .get_attribute("wavepacket_offset")
                .unwrap()
                .as_u64()
                .unwrap(),
            &vec![0, 0]
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn point_format_10_round_trips_standard_attribute_precision() {
        let path = temp_las_path("pf10_precision");
        let mut pc =
            HighPerformancePointCloud::from_xyz_vec(vec![[1.25, 2.5, 3.75], [4.25, 5.5, 6.75]])
                .unwrap();
        pc.set_attribute(
            "intensity".to_string(),
            AttributeValue::U16(vec![65535, 42]),
        )
        .unwrap();
        pc.set_attribute("red".to_string(), AttributeValue::U16(vec![4096, 65535]))
            .unwrap();
        pc.set_attribute("green".to_string(), AttributeValue::U16(vec![8192, 32768]))
            .unwrap();
        pc.set_attribute("blue".to_string(), AttributeValue::U16(vec![16384, 12345]))
            .unwrap();
        pc.set_attribute("nir".to_string(), AttributeValue::U16(vec![111, 222]))
            .unwrap();
        pc.set_attribute(
            "scan_angle".to_string(),
            AttributeValue::I16(vec![-120, 120]),
        )
        .unwrap();
        pc.set_attribute("return_number".to_string(), AttributeValue::U8(vec![1, 2]))
            .unwrap();
        pc.set_attribute(
            "number_of_returns".to_string(),
            AttributeValue::U8(vec![3, 4]),
        )
        .unwrap();
        pc.set_attribute(
            "synthetic".to_string(),
            AttributeValue::Bool(vec![true, false]),
        )
        .unwrap();
        pc.set_attribute(
            "scan_direction_flag".to_string(),
            AttributeValue::Bool(vec![true, false]),
        )
        .unwrap();
        pc.set_attribute(
            "scanner_channel".to_string(),
            AttributeValue::U8(vec![1, 3]),
        )
        .unwrap();
        pc.set_attribute("classification".to_string(), AttributeValue::U8(vec![2, 6]))
            .unwrap();
        pc.set_attribute("user_data".to_string(), AttributeValue::U8(vec![9, 10]))
            .unwrap();
        pc.set_attribute(
            "point_source_id".to_string(),
            AttributeValue::U16(vec![100, 101]),
        )
        .unwrap();
        pc.set_attribute(
            "wavepacket_index".to_string(),
            AttributeValue::U8(vec![0, 0]),
        )
        .unwrap();
        pc.set_attribute(
            "wavepacket_size".to_string(),
            AttributeValue::U32(vec![0, 0]),
        )
        .unwrap();

        pc.to_las_with_options(path.to_str().unwrap(), false, Some(10), Some("1.4"), false)
            .unwrap();

        let reloaded = HighPerformancePointCloud::from_las_laz(path.to_str().unwrap()).unwrap();
        assert_eq!(
            reloaded
                .get_attribute("intensity")
                .unwrap()
                .as_u16()
                .unwrap(),
            &vec![65535, 42]
        );
        assert_eq!(
            reloaded.get_attribute("red").unwrap().as_u16().unwrap(),
            &vec![4096, 65535]
        );
        assert_eq!(
            reloaded.get_attribute("nir").unwrap().as_u16().unwrap(),
            &vec![111, 222]
        );
        assert_eq!(
            reloaded
                .get_attribute("scan_angle")
                .unwrap()
                .as_i16()
                .unwrap(),
            &vec![-120, 120]
        );
        assert_eq!(
            reloaded
                .get_attribute("return_number")
                .unwrap()
                .as_u8()
                .unwrap(),
            &vec![1, 2]
        );
        assert_eq!(
            reloaded
                .get_attribute("number_of_returns")
                .unwrap()
                .as_u8()
                .unwrap(),
            &vec![3, 4]
        );
        assert_eq!(
            reloaded
                .get_attribute("scanner_channel")
                .unwrap()
                .as_u8()
                .unwrap(),
            &vec![1, 3]
        );
        assert_eq!(
            reloaded
                .get_attribute("classification")
                .unwrap()
                .as_u8()
                .unwrap(),
            &vec![2, 6]
        );
        assert_eq!(
            reloaded
                .get_attribute("user_data")
                .unwrap()
                .as_u8()
                .unwrap(),
            &vec![9, 10]
        );
        assert_eq!(
            reloaded
                .get_attribute("point_source_id")
                .unwrap()
                .as_u16()
                .unwrap(),
            &vec![100, 101]
        );
        assert_eq!(
            reloaded
                .get_attribute("synthetic")
                .unwrap()
                .as_bool()
                .unwrap(),
            &vec![true, false]
        );
        assert_eq!(
            reloaded
                .get_attribute("scan_direction_flag")
                .unwrap()
                .as_bool()
                .unwrap(),
            &vec![true, false]
        );
        assert_eq!(
            reloaded
                .get_attribute("wavepacket_offset")
                .unwrap()
                .as_u64()
                .unwrap(),
            &vec![0, 0]
        );
        let _ = fs::remove_file(path);
    }

    #[test]
    fn point_format_10_rejects_non_default_waveform_without_explicit_drop() {
        let path = temp_las_path("pf10_waveform_reject");
        let mut pc = HighPerformancePointCloud::from_xyz_vec(vec![[1.0, 2.0, 3.0]]).unwrap();
        pc.set_attribute("wavepacket_index".to_string(), AttributeValue::U8(vec![1]))
            .unwrap();

        let err = pc
            .to_las_with_options(path.to_str().unwrap(), false, Some(10), Some("1.4"), false)
            .unwrap_err();
        assert!(err.to_string().contains("waveform payload"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn point_format_10_rejects_out_of_range_packed_fields() {
        let path = temp_las_path("pf10_range_reject");
        let mut pc = HighPerformancePointCloud::from_xyz_vec(vec![[1.0, 2.0, 3.0]]).unwrap();
        pc.set_attribute("return_number".to_string(), AttributeValue::U8(vec![16]))
            .unwrap();
        let err = pc
            .to_las_with_options(path.to_str().unwrap(), false, Some(10), Some("1.4"), false)
            .unwrap_err();
        assert!(err.to_string().contains("return_number"));

        pc.set_attribute("return_number".to_string(), AttributeValue::U8(vec![1]))
            .unwrap();
        pc.set_attribute("scanner_channel".to_string(), AttributeValue::U8(vec![4]))
            .unwrap();
        let err = pc
            .to_las_with_options(path.to_str().unwrap(), false, Some(10), Some("1.4"), false)
            .unwrap_err();
        assert!(err.to_string().contains("scanner_channel"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn point_format_10_preserves_raw_coordinate_records_on_unchanged_round_trip() {
        let input_path = temp_las_path("pf10_raw_xyz_input");
        let output_path = temp_las_path("pf10_raw_xyz_output");
        write_pf10_raw_coordinate_fixture(&input_path, &[[123_456, -5, 42], [123_457, -4, 43]]);

        let pc = HighPerformancePointCloud::from_las_laz(input_path.to_str().unwrap()).unwrap();
        pc.to_las_with_options(
            output_path.to_str().unwrap(),
            false,
            Some(10),
            Some("1.4"),
            false,
        )
        .unwrap();

        assert_eq!(
            read_raw_xyz_records(&input_path),
            read_raw_xyz_records(&output_path)
        );
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn point_format_10_gathers_raw_coordinate_records_through_selection() {
        let input_path = temp_las_path("pf10_raw_xyz_select_input");
        let output_path = temp_las_path("pf10_raw_xyz_select_output");
        let raw = [[123_456, -5, 42], [123_457, -4, 43], [123_458, -3, 44]];
        write_pf10_raw_coordinate_fixture(&input_path, &raw);

        let pc = HighPerformancePointCloud::from_las_laz(input_path.to_str().unwrap()).unwrap();
        let selected = pc.select_indices(&[2, 0]).unwrap();
        selected
            .to_las_with_options(
                output_path.to_str().unwrap(),
                false,
                Some(10),
                Some("1.4"),
                false,
            )
            .unwrap();

        assert_eq!(read_raw_xyz_records(&output_path), vec![raw[2], raw[0]]);
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn point_format_10_preserves_compatible_raw_coordinate_records_through_concat() {
        let input_path = temp_las_path("pf10_raw_xyz_concat_input");
        let output_path = temp_las_path("pf10_raw_xyz_concat_output");
        let raw = [[123_456, -5, 42], [123_457, -4, 43], [123_458, -3, 44]];
        write_pf10_raw_coordinate_fixture(&input_path, &raw);

        let pc = HighPerformancePointCloud::from_las_laz(input_path.to_str().unwrap()).unwrap();
        let a = pc.select_indices(&[0, 2]).unwrap();
        let b = pc.select_indices(&[1]).unwrap();
        let concatenated =
            HighPerformancePointCloud::concatenate(&[&a, &b], crate::ConcatPolicy::Strict).unwrap();
        concatenated
            .to_las_with_options(
                output_path.to_str().unwrap(),
                false,
                Some(10),
                Some("1.4"),
                false,
            )
            .unwrap();

        assert_eq!(
            read_raw_xyz_records(&output_path),
            vec![raw[0], raw[2], raw[1]]
        );
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn point_format_10_drops_raw_coordinate_records_after_translation() {
        let input_path = temp_las_path("pf10_raw_xyz_translate_input");
        let output_path = temp_las_path("pf10_raw_xyz_translate_output");
        let raw = [[123_456, -5, 42], [123_457, -4, 43]];
        write_pf10_raw_coordinate_fixture(&input_path, &raw);

        let pc = HighPerformancePointCloud::from_las_laz(input_path.to_str().unwrap()).unwrap();
        let translated = pc.translate([1.0, 0.0, 0.0]).unwrap();
        translated
            .to_las_with_options(
                output_path.to_str().unwrap(),
                false,
                Some(10),
                Some("1.4"),
                false,
            )
            .unwrap();

        assert_ne!(read_raw_xyz_records(&output_path), raw);
        let _ = fs::remove_file(input_path);
        let _ = fs::remove_file(output_path);
    }

    #[test]
    fn point_format_10_reads_standard_extra_bytes_with_collision_prefix() {
        let path = temp_las_path("pf10_standard_extra_bytes");
        let mut builder = Builder::from((1, 4));
        let mut format = Format::new(10).unwrap();
        format.extra_bytes = 18;
        builder.point_format = format;
        builder.vlrs.push(standard_extra_bytes_vlr(&[
            ("custom_u64", 7, 8),
            ("custom_i16", 4, 2),
            ("intensity", 7, 8),
        ]));
        let header = builder.into_header().unwrap();
        let mut writer = Writer::from_path(&path, header).unwrap();
        let mut extra_bytes = Vec::new();
        extra_bytes.extend_from_slice(&(u64::from(u32::MAX) + 9).to_le_bytes());
        extra_bytes.extend_from_slice(&(-123i16).to_le_bytes());
        extra_bytes.extend_from_slice(&77u64.to_le_bytes());
        writer
            .write_point(Point {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                gps_time: Some(0.0),
                color: Some(Color::default()),
                nir: Some(0),
                waveform: Some(Waveform::default()),
                extra_bytes,
                ..Default::default()
            })
            .unwrap();
        writer.close().unwrap();

        let pc = HighPerformancePointCloud::from_las_laz(path.to_str().unwrap()).unwrap();
        assert_eq!(
            pc.get_attribute("custom_u64").unwrap().as_u64().unwrap(),
            &vec![u64::from(u32::MAX) + 9]
        );
        assert_eq!(
            pc.get_attribute("custom_i16").unwrap().as_i16().unwrap(),
            &vec![-123]
        );
        assert_eq!(
            pc.get_attribute("extra_intensity")
                .unwrap()
                .as_u64()
                .unwrap(),
            &vec![77]
        );
        let _ = fs::remove_file(path);
    }

    fn standard_extra_bytes_vlr(fields: &[(&str, u8, usize)]) -> Vlr {
        let mut data = Vec::new();
        for (name, data_type, _size) in fields {
            let mut record = vec![0u8; 192];
            record[2] = *data_type;
            let name_bytes = name.as_bytes();
            let name_len = name_bytes.len().min(32);
            record[4..4 + name_len].copy_from_slice(&name_bytes[..name_len]);
            data.extend_from_slice(&record);
        }
        Vlr {
            user_id: "LASF_Spec".to_string(),
            record_id: 4,
            description: "Extra Bytes Record".to_string(),
            data,
        }
    }

    fn write_pf10_raw_coordinate_fixture(path: &std::path::Path, raw_xyz: &[[i32; 3]]) {
        let mut builder = Builder::from((1, 4));
        builder.point_format = Format::new(10).unwrap();
        builder.transforms = las::Vector {
            x: las::Transform {
                scale: 0.0001,
                offset: 1000.0,
            },
            y: las::Transform {
                scale: 0.0001,
                offset: -2000.0,
            },
            z: las::Transform {
                scale: 0.0001,
                offset: 50.0,
            },
        };
        let header = builder.into_header().unwrap();
        let mut writer = Writer::from_path(path, header).unwrap();
        for raw in raw_xyz {
            writer
                .write_point(Point {
                    x: 1000.0 + f64::from(raw[0]) * 0.0001,
                    y: -2000.0 + f64::from(raw[1]) * 0.0001,
                    z: 50.0 + f64::from(raw[2]) * 0.0001,
                    gps_time: Some(0.0),
                    color: Some(Color::default()),
                    nir: Some(0),
                    waveform: Some(Waveform::default()),
                    ..Default::default()
                })
                .unwrap();
        }
        writer.close().unwrap();
    }

    fn read_raw_xyz_records(path: &std::path::Path) -> Vec<[i32; 3]> {
        let mut file = fs::File::open(path).unwrap();
        let header = raw::Header::read_from(&mut file).unwrap();
        file.seek(SeekFrom::Start(u64::from(header.offset_to_point_data)))
            .unwrap();
        let mut record = vec![0u8; header.point_data_record_length as usize];
        let point_count = header.number_of_point_records as usize;
        (0..point_count)
            .map(|_| {
                file.read_exact(&mut record).unwrap();
                [
                    i32::from_le_bytes(record[0..4].try_into().unwrap()),
                    i32::from_le_bytes(record[4..8].try_into().unwrap()),
                    i32::from_le_bytes(record[8..12].try_into().unwrap()),
                ]
            })
            .collect()
    }
}
