use super::{FileType, Inspection, TensorDescriptor};

mod graph;
mod inspect;
mod protos;

pub(crate) use graph::*;
pub(crate) use inspect::*;

#[inline]
fn data_type_bits(dtype: i32) -> usize {
    match dtype {
        1 => 32,   // float
        2 => 8,    // uint8_t
        3 => 8,    // int8_t
        4 => 16,   // uint16_t
        5 => 16,   // int16_t
        6 => 32,   // int32_t
        7 => 64,   // int64_t
        8 => 8,    // string (assuming 8 bits per character)
        9 => 8,    // bool (typically 8 bits in most systems)
        10 => 16,  // FLOAT16
        11 => 64,  // DOUBLE
        12 => 32,  // UINT32
        13 => 64,  // UINT64
        14 => 64,  // COMPLEX64 (two 32-bit floats)
        15 => 128, // COMPLEX128 (two 64-bit floats)
        16 => 16,  // BFLOAT16
        17 => 8,   // FLOAT8E4M3FN
        18 => 8,   // FLOAT8E4M3FNUZ
        19 => 8,   // FLOAT8E5M2
        20 => 8,   // FLOAT8E5M2FNUZ
        21 => 4,   // UINT4
        22 => 4,   // INT4
        23 => 4,   // FLOAT4E2M1
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
