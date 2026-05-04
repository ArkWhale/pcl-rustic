use crate::utils::error::{PointCloudError, Result};

#[derive(Clone, Debug, PartialEq)]
pub enum AttrDType {
    F32,
    F64,
    U8,
    U16,
    U32,
    I32,
    I64,
    Bool,
    F32x6,
}

impl std::fmt::Display for AttrDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttrDType::F32 => write!(f, "float32"),
            AttrDType::F64 => write!(f, "float64"),
            AttrDType::U8 => write!(f, "uint8"),
            AttrDType::U16 => write!(f, "uint16"),
            AttrDType::U32 => write!(f, "uint32"),
            AttrDType::I32 => write!(f, "int32"),
            AttrDType::I64 => write!(f, "int64"),
            AttrDType::Bool => write!(f, "bool"),
            AttrDType::F32x6 => write!(f, "float32[6]"),
        }
    }
}

#[derive(Clone)]
pub enum AttributeValue {
    F32(Vec<f32>),
    F64(Vec<f64>),
    U8(Vec<u8>),
    U16(Vec<u16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Bool(Vec<bool>),
    F32x6(Vec<[f32; 6]>),
}

impl AttributeValue {
    pub fn len(&self) -> usize {
        match self {
            AttributeValue::F32(v) => v.len(),
            AttributeValue::F64(v) => v.len(),
            AttributeValue::U8(v) => v.len(),
            AttributeValue::U16(v) => v.len(),
            AttributeValue::U32(v) => v.len(),
            AttributeValue::I32(v) => v.len(),
            AttributeValue::I64(v) => v.len(),
            AttributeValue::Bool(v) => v.len(),
            AttributeValue::F32x6(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dtype(&self) -> AttrDType {
        match self {
            AttributeValue::F32(_) => AttrDType::F32,
            AttributeValue::F64(_) => AttrDType::F64,
            AttributeValue::U8(_) => AttrDType::U8,
            AttributeValue::U16(_) => AttrDType::U16,
            AttributeValue::U32(_) => AttrDType::U32,
            AttributeValue::I32(_) => AttrDType::I32,
            AttributeValue::I64(_) => AttrDType::I64,
            AttributeValue::Bool(_) => AttrDType::Bool,
            AttributeValue::F32x6(_) => AttrDType::F32x6,
        }
    }

    pub fn gather(&self, indices: &[usize]) -> Result<Self> {
        let len = self.len();
        for &idx in indices {
            if idx >= len {
                return Err(PointCloudError::InvalidParameter(format!(
                    "index {} out of bounds for attribute of length {}",
                    idx, len
                )));
            }
        }
        Ok(match self {
            AttributeValue::F32(v) => AttributeValue::F32(indices.iter().map(|&i| v[i]).collect()),
            AttributeValue::F64(v) => AttributeValue::F64(indices.iter().map(|&i| v[i]).collect()),
            AttributeValue::U8(v) => AttributeValue::U8(indices.iter().map(|&i| v[i]).collect()),
            AttributeValue::U16(v) => AttributeValue::U16(indices.iter().map(|&i| v[i]).collect()),
            AttributeValue::U32(v) => AttributeValue::U32(indices.iter().map(|&i| v[i]).collect()),
            AttributeValue::I32(v) => AttributeValue::I32(indices.iter().map(|&i| v[i]).collect()),
            AttributeValue::I64(v) => AttributeValue::I64(indices.iter().map(|&i| v[i]).collect()),
            AttributeValue::Bool(v) => {
                AttributeValue::Bool(indices.iter().map(|&i| v[i]).collect())
            }
            AttributeValue::F32x6(v) => {
                AttributeValue::F32x6(indices.iter().map(|&i| v[i]).collect())
            }
        })
    }

    pub fn select_mask(&self, mask: &[bool]) -> Result<Self> {
        if mask.len() != self.len() {
            return Err(PointCloudError::DimensionMismatch {
                expected: self.len(),
                actual: mask.len(),
            });
        }
        Ok(match self {
            AttributeValue::F32(v) => AttributeValue::F32(
                v.iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&val, _)| val)
                    .collect(),
            ),
            AttributeValue::F64(v) => AttributeValue::F64(
                v.iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&val, _)| val)
                    .collect(),
            ),
            AttributeValue::U8(v) => AttributeValue::U8(
                v.iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&val, _)| val)
                    .collect(),
            ),
            AttributeValue::U16(v) => AttributeValue::U16(
                v.iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&val, _)| val)
                    .collect(),
            ),
            AttributeValue::U32(v) => AttributeValue::U32(
                v.iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&val, _)| val)
                    .collect(),
            ),
            AttributeValue::I32(v) => AttributeValue::I32(
                v.iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&val, _)| val)
                    .collect(),
            ),
            AttributeValue::I64(v) => AttributeValue::I64(
                v.iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&val, _)| val)
                    .collect(),
            ),
            AttributeValue::Bool(v) => AttributeValue::Bool(
                v.iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&val, _)| val)
                    .collect(),
            ),
            AttributeValue::F32x6(v) => AttributeValue::F32x6(
                v.iter()
                    .zip(mask.iter())
                    .filter(|(_, &m)| m)
                    .map(|(&val, _)| val)
                    .collect(),
            ),
        })
    }

    pub fn as_f32(&self) -> Option<&Vec<f32>> {
        match self {
            AttributeValue::F32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<&Vec<f64>> {
        match self {
            AttributeValue::F64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u8(&self) -> Option<&Vec<u8>> {
        match self {
            AttributeValue::U8(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u16(&self) -> Option<&Vec<u16>> {
        match self {
            AttributeValue::U16(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_u32(&self) -> Option<&Vec<u32>> {
        match self {
            AttributeValue::U32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i32(&self) -> Option<&Vec<i32>> {
        match self {
            AttributeValue::I32(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<&Vec<i64>> {
        match self {
            AttributeValue::I64(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<&Vec<bool>> {
        match self {
            AttributeValue::Bool(v) => Some(v),
            _ => None,
        }
    }

    pub fn as_f32x6(&self) -> Option<&Vec<[f32; 6]>> {
        match self {
            AttributeValue::F32x6(v) => Some(v),
            _ => None,
        }
    }

    pub fn to_f32_vec(&self) -> Vec<f32> {
        match self {
            AttributeValue::F32(v) => v.clone(),
            AttributeValue::F64(v) => v.iter().map(|&x| x as f32).collect(),
            AttributeValue::U8(v) => v.iter().map(|&x| x as f32).collect(),
            AttributeValue::U16(v) => v.iter().map(|&x| x as f32).collect(),
            AttributeValue::U32(v) => v.iter().map(|&x| x as f32).collect(),
            AttributeValue::I32(v) => v.iter().map(|&x| x as f32).collect(),
            AttributeValue::I64(v) => v.iter().map(|&x| x as f32).collect(),
            AttributeValue::Bool(v) => v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect(),
            AttributeValue::F32x6(v) => v.iter().flat_map(|row| row.iter().copied()).collect(),
        }
    }

    pub fn bytes_per_element(&self) -> usize {
        match self {
            AttributeValue::F32(_) => 4,
            AttributeValue::F64(_) => 8,
            AttributeValue::U8(_) => 1,
            AttributeValue::U16(_) => 2,
            AttributeValue::U32(_) => 4,
            AttributeValue::I32(_) => 4,
            AttributeValue::I64(_) => 8,
            AttributeValue::Bool(_) => 1,
            AttributeValue::F32x6(_) => 24,
        }
    }

    pub fn memory_usage(&self) -> usize {
        self.len() * self.bytes_per_element()
    }

    pub fn concatenate(values: &[&AttributeValue]) -> Result<Self> {
        if values.is_empty() {
            return Err(PointCloudError::InvalidParameter(
                "cannot concatenate empty attribute list".to_string(),
            ));
        }
        let dtype = values[0].dtype();
        for v in values.iter().skip(1) {
            if v.dtype() != dtype {
                return Err(PointCloudError::InvalidParameter(format!(
                    "dtype mismatch in concatenate: expected {}, got {}",
                    dtype,
                    v.dtype()
                )));
            }
        }
        Ok(match &dtype {
            AttrDType::F32 => {
                let mut out = Vec::new();
                for v in values {
                    out.extend_from_slice(v.as_f32().unwrap());
                }
                AttributeValue::F32(out)
            }
            AttrDType::F64 => {
                let mut out = Vec::new();
                for v in values {
                    out.extend_from_slice(v.as_f64().unwrap());
                }
                AttributeValue::F64(out)
            }
            AttrDType::U8 => {
                let mut out = Vec::new();
                for v in values {
                    out.extend_from_slice(v.as_u8().unwrap());
                }
                AttributeValue::U8(out)
            }
            AttrDType::U16 => {
                let mut out = Vec::new();
                for v in values {
                    out.extend_from_slice(v.as_u16().unwrap());
                }
                AttributeValue::U16(out)
            }
            AttrDType::U32 => {
                let mut out = Vec::new();
                for v in values {
                    out.extend_from_slice(v.as_u32().unwrap());
                }
                AttributeValue::U32(out)
            }
            AttrDType::I32 => {
                let mut out = Vec::new();
                for v in values {
                    out.extend_from_slice(v.as_i32().unwrap());
                }
                AttributeValue::I32(out)
            }
            AttrDType::I64 => {
                let mut out = Vec::new();
                for v in values {
                    out.extend_from_slice(v.as_i64().unwrap());
                }
                AttributeValue::I64(out)
            }
            AttrDType::Bool => {
                let mut out = Vec::new();
                for v in values {
                    out.extend_from_slice(v.as_bool().unwrap());
                }
                AttributeValue::Bool(out)
            }
            AttrDType::F32x6 => {
                let mut out = Vec::new();
                for v in values {
                    out.extend_from_slice(v.as_f32x6().unwrap());
                }
                AttributeValue::F32x6(out)
            }
        })
    }

    pub fn zeros(dtype: &AttrDType, len: usize) -> Self {
        match dtype {
            AttrDType::F32 => AttributeValue::F32(vec![0.0; len]),
            AttrDType::F64 => AttributeValue::F64(vec![0.0; len]),
            AttrDType::U8 => AttributeValue::U8(vec![0; len]),
            AttrDType::U16 => AttributeValue::U16(vec![0; len]),
            AttrDType::U32 => AttributeValue::U32(vec![0; len]),
            AttrDType::I32 => AttributeValue::I32(vec![0; len]),
            AttrDType::I64 => AttributeValue::I64(vec![0; len]),
            AttrDType::Bool => AttributeValue::Bool(vec![false; len]),
            AttrDType::F32x6 => AttributeValue::F32x6(vec![[0.0; 6]; len]),
        }
    }
}

impl std::fmt::Debug for AttributeValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "AttributeValue({}, len={})", self.dtype(), self.len())
    }
}
