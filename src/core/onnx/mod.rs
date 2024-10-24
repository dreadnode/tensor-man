use super::{FileType, Inspection, TensorDescriptor};

mod graph;
mod inspect;
mod protos;

pub(crate) use graph::*;
pub(crate) use inspect::*;

#[inline]
fn data_type_bytes(dtype: i32) -> usize {
    match dtype {
        1 => 4,   // float
        2 => 1,   // uint8_t
        3 => 1,   // int8_t
        4 => 2,   // uint16_t
        5 => 2,   // int16_t
        6 => 4,   // int32_t
        7 => 8,   // int64_t
        8 => 1,   // string
        9 => 1,   // bool
        10 => 2,  // FLOAT16
        11 => 8,  // DOUBLE
        12 => 4,  // UINT32
        13 => 8,  // UINT64
        14 => 8,  // COMPLEX64
        15 => 16, // COMPLEX128
        16 => 2,  // BFLOAT16
        17 => 1,  // FLOAT8E4M3FN
        18 => 1,  // FLOAT8E4M3FNUZ
        19 => 1,  // FLOAT8E5M2
        20 => 1,  // FLOAT8E5M2FNUZ
        21 => 1,  // UINT4 (4-bit values are packed, returning 1 as a placeholder)
        22 => 1,  // INT4 (4-bit values are packed, returning 1 as a placeholder)
        23 => 1,  // FLOAT4E2M1 (4-bit values are packed, returning 1 as a placeholder)
        _ => panic!("Unsupported data type: {}", dtype),
    }
}

#[inline]
pub(crate) fn data_type_string(dtype: i32) -> &'static str {
    match dtype {
        1 => "FLOAT",
        2 => "UINT8",
        3 => "INT8",
        4 => "UINT16",
        5 => "INT16",
        6 => "INT32",
        7 => "INT64",
        8 => "STRING",
        9 => "BOOL",
        10 => "FLOAT16",
        11 => "DOUBLE",
        12 => "UINT32",
        13 => "UINT64",
        14 => "COMPLEX64",
        15 => "COMPLEX128",
        16 => "BFLOAT16",
        17 => "FLOAT8E4M3FN",
        18 => "FLOAT8E4M3FNUZ",
        19 => "FLOAT8E5M2",
        20 => "FLOAT8E5M2FNUZ",
        21 => "UINT4",
        22 => "INT4",
        23 => "FLOAT4E2M1",
        _ => "UNKNOWN",
    }
}
