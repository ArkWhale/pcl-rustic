use crate::point_cloud::attribute_value::AttributeValue;
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::{PointCloudError, Result};
use las::point::Format;
use las::{Builder, Color, Point, Reader, Vlr, Writer};
use std::fs;

const EXTRA_BYTES_USER_ID: &str = "pcl-rustic";
const EXTRA_BYTES_RECORD_ID: u16 = 1;
const STANDARD_LAS_ATTRIBUTES: &[&str] = &[
    "intensity",
    "classification",
    "return_number",
    "number_of_returns",
    "gps_time",
    "red",
    "green",
    "blue",
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
        let extra_fields = extra_byte_fields(reader.header().vlrs())?;
        let mut extra_columns: Vec<Vec<u8>> = extra_fields.iter().map(|_| Vec::new()).collect();

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
        for (field, bytes) in extra_fields.iter().zip(extra_columns.iter()) {
            if let Some(attr) = decode_extra_attribute(&field.dtype, bytes, n)? {
                result.attributes_mut().insert(field.name.clone(), attr);
            }
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

        let extra_fields = collect_extra_byte_fields(self.attributes())?;

        let mut builder = Builder::from((1, 4));
        let mut format = Format::new(format_id).map_err(|e| e.to_string())?;
        format.is_compressed = compress || path.to_lowercase().ends_with(".laz");
        format.extra_bytes = extra_fields.iter().map(|field| field.size as u16).sum();
        builder.point_format = format;
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
    let Some(vlr) = vlrs
        .iter()
        .find(|vlr| vlr.user_id == EXTRA_BYTES_USER_ID && vlr.record_id == EXTRA_BYTES_RECORD_ID)
    else {
        return Ok(Vec::new());
    };
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
        if parts.next().is_some() || name.is_empty() || recorded_size.parse::<usize>() != Ok(size) {
            return Err(PointCloudError::InvalidParameter(format!(
                "invalid ExtraBytes schema line '{line}'"
            )));
        }
        fields.push(ExtraByteField {
            name: name.to_string(),
            dtype: dtype.to_string(),
            offset,
            size,
        });
        offset += size;
    }
    Ok(fields)
}

fn extra_dtype_size(dtype: &str) -> Option<usize> {
    match dtype {
        "float32" | "uint32" | "int32" => Some(4),
        "float64" | "int64" => Some(8),
        "uint8" | "bool" => Some(1),
        "uint16" => Some(2),
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
