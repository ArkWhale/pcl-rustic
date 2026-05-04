use crate::point_cloud::attribute_value::AttributeValue;
use crate::point_cloud::core::HighPerformancePointCloud;
use crate::utils::error::{PointCloudError, Result};
use polars::prelude::*;
use std::fs::File;

#[derive(Clone, Debug)]
pub struct TableColumnNames {
    pub x: String,
    pub y: String,
    pub z: String,
    pub intensity: Option<String>,
    pub rgb_r: Option<String>,
    pub rgb_g: Option<String>,
    pub rgb_b: Option<String>,
}

impl Default for TableColumnNames {
    fn default() -> Self {
        Self {
            x: "x".to_string(),
            y: "y".to_string(),
            z: "z".to_string(),
            intensity: Some("intensity".to_string()),
            rgb_r: Some("r".to_string()),
            rgb_g: Some("g".to_string()),
            rgb_b: Some("b".to_string()),
        }
    }
}

impl TableColumnNames {
    pub fn resolve(
        x: Option<String>,
        y: Option<String>,
        z: Option<String>,
        intensity: Option<String>,
        rgb_r: Option<String>,
        rgb_g: Option<String>,
        rgb_b: Option<String>,
    ) -> Self {
        let mut cols = TableColumnNames::default();
        if let Some(v) = x {
            cols.x = v;
        }
        if let Some(v) = y {
            cols.y = v;
        }
        if let Some(v) = z {
            cols.z = v;
        }
        if intensity.is_some() {
            cols.intensity = intensity;
        }
        if rgb_r.is_some() {
            cols.rgb_r = rgb_r;
        }
        if rgb_g.is_some() {
            cols.rgb_g = rgb_g;
        }
        if rgb_b.is_some() {
            cols.rgb_b = rgb_b;
        }
        cols
    }
}

impl HighPerformancePointCloud {
    pub fn from_table_csv(path: &str, delimiter: u8, columns: TableColumnNames) -> Result<Self> {
        let df = CsvReadOptions::default()
            .with_has_header(true)
            .map_parse_options(|opts| opts.with_separator(delimiter))
            .try_into_reader_with_file_path(Some(path.into()))
            .map_err(|e| PointCloudError::ParseError(e.to_string()))?
            .finish()
            .map_err(|e| PointCloudError::ParseError(e.to_string()))?;
        from_dataframe(df, columns)
    }

    pub fn from_table_parquet(path: &str, columns: TableColumnNames) -> Result<Self> {
        let file = File::open(path).map_err(PointCloudError::IoError)?;
        let df = ParquetReader::new(file)
            .finish()
            .map_err(|e| PointCloudError::ParseError(e.to_string()))?;
        from_dataframe(df, columns)
    }

    pub fn to_table_csv(&self, path: &str, delimiter: u8, columns: TableColumnNames) -> Result<()> {
        let mut df = to_dataframe(self, columns)?;
        let file = File::create(path).map_err(PointCloudError::IoError)?;
        CsvWriter::new(file)
            .include_header(true)
            .with_separator(delimiter)
            .finish(&mut df)
            .map_err(|e| PointCloudError::ParseError(e.to_string()))?;
        Ok(())
    }

    pub fn to_table_parquet(&self, path: &str, columns: TableColumnNames) -> Result<()> {
        let mut df = to_dataframe(self, columns)?;
        let file = File::create(path).map_err(PointCloudError::IoError)?;
        ParquetWriter::new(file)
            .finish(&mut df)
            .map_err(|e| PointCloudError::ParseError(e.to_string()))?;
        Ok(())
    }
}

fn from_dataframe(df: DataFrame, columns: TableColumnNames) -> Result<HighPerformancePointCloud> {
    let x = get_f32_col(&df, &columns.x)?;
    let y = get_f32_col(&df, &columns.y)?;
    let z = get_f32_col(&df, &columns.z)?;

    if x.len() != y.len() || x.len() != z.len() {
        return Err(PointCloudError::DimensionMismatch {
            expected: x.len(),
            actual: y.len().max(z.len()),
        });
    }

    let xyz: Vec<[f32; 3]> = x
        .iter()
        .zip(y.iter())
        .zip(z.iter())
        .map(|((&xi, &yi), &zi)| [xi, yi, zi])
        .collect();

    let mut pc = HighPerformancePointCloud::from_xyz_vec(xyz)?;

    if let Some(intensity_name) = &columns.intensity {
        if df.column(intensity_name).is_ok() {
            let intensity = get_f32_col(&df, intensity_name)?;
            if intensity.len() == pc.point_count() {
                pc.set_attribute("intensity".to_string(), AttributeValue::F32(intensity))?;
            }
        }
    }

    if let (Some(rn), Some(gn), Some(bn)) = (&columns.rgb_r, &columns.rgb_g, &columns.rgb_b) {
        if df.column(rn).is_ok() && df.column(gn).is_ok() && df.column(bn).is_ok() {
            let r = get_u8_col(&df, rn)?;
            let g = get_u8_col(&df, gn)?;
            let b = get_u8_col(&df, bn)?;
            let n = pc.point_count();
            if r.len() == n && g.len() == n && b.len() == n {
                pc.set_attribute("red".to_string(), AttributeValue::U8(r))?;
                pc.set_attribute("green".to_string(), AttributeValue::U8(g))?;
                pc.set_attribute("blue".to_string(), AttributeValue::U8(b))?;
            }
        }
    }

    Ok(pc)
}

fn to_dataframe(
    pc: &HighPerformancePointCloud,
    column_names: TableColumnNames,
) -> Result<DataFrame> {
    let xyz = pc.get_xyz_vec();
    let x: Vec<f32> = xyz.iter().map(|p| p[0]).collect();
    let y: Vec<f32> = xyz.iter().map(|p| p[1]).collect();
    let z: Vec<f32> = xyz.iter().map(|p| p[2]).collect();

    let mut columns: Vec<Column> = vec![
        Column::new(PlSmallStr::from_str(&column_names.x), x),
        Column::new(PlSmallStr::from_str(&column_names.y), y),
        Column::new(PlSmallStr::from_str(&column_names.z), z),
    ];

    if let Some(name) = &column_names.intensity {
        if let Some(attr) = pc.get_attribute("intensity") {
            if let Some(v) = attr.as_f32() {
                columns.push(Column::new(PlSmallStr::from_str(name), v.clone()));
            }
        }
    }

    if let (Some(rn), Some(gn), Some(bn)) = (
        &column_names.rgb_r,
        &column_names.rgb_g,
        &column_names.rgb_b,
    ) {
        if let (Some(r_attr), Some(g_attr), Some(b_attr)) = (
            pc.get_attribute("red"),
            pc.get_attribute("green"),
            pc.get_attribute("blue"),
        ) {
            if let (Some(r), Some(g), Some(b)) = (r_attr.as_u8(), g_attr.as_u8(), b_attr.as_u8()) {
                let r_u32: Vec<u32> = r.iter().map(|&v| v as u32).collect();
                let g_u32: Vec<u32> = g.iter().map(|&v| v as u32).collect();
                let b_u32: Vec<u32> = b.iter().map(|&v| v as u32).collect();
                columns.push(Column::new(PlSmallStr::from_str(rn), r_u32));
                columns.push(Column::new(PlSmallStr::from_str(gn), g_u32));
                columns.push(Column::new(PlSmallStr::from_str(bn), b_u32));
            }
        }
    }

    DataFrame::new(columns).map_err(|e| PointCloudError::ParseError(e.to_string()))
}

fn get_f32_col(df: &DataFrame, name: &str) -> Result<Vec<f32>> {
    let series = df
        .column(name)
        .map_err(|_| PointCloudError::ParseError(format!("missing column: {}", name)))?;
    if let Ok(col) = series.f32() {
        return Ok(col.into_no_null_iter().collect());
    }
    if let Ok(col) = series.f64() {
        return Ok(col.into_no_null_iter().map(|v| v as f32).collect());
    }
    if let Ok(col) = series.i64() {
        return Ok(col.into_no_null_iter().map(|v| v as f32).collect());
    }
    if let Ok(col) = series.i32() {
        return Ok(col.into_no_null_iter().map(|v| v as f32).collect());
    }
    Err(PointCloudError::ParseError(format!(
        "unsupported column type: {}",
        name
    )))
}

fn get_u8_col(df: &DataFrame, name: &str) -> Result<Vec<u8>> {
    let series = df
        .column(name)
        .map_err(|_| PointCloudError::ParseError(format!("missing column: {}", name)))?;
    if let Ok(col) = series.u8() {
        return Ok(col.into_no_null_iter().collect());
    }
    if let Ok(col) = series.u16() {
        return Ok(col.into_no_null_iter().map(|v| (v >> 8) as u8).collect());
    }
    if let Ok(col) = series.u32() {
        return Ok(col.into_no_null_iter().map(|v| v.min(255) as u8).collect());
    }
    if let Ok(col) = series.i64() {
        return Ok(col
            .into_no_null_iter()
            .map(|v| v.clamp(0, 255) as u8)
            .collect());
    }
    if let Ok(col) = series.i32() {
        return Ok(col
            .into_no_null_iter()
            .map(|v| v.clamp(0, 255) as u8)
            .collect());
    }
    Err(PointCloudError::ParseError(format!(
        "unsupported column type: {}",
        name
    )))
}
