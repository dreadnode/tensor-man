mod inspect;

use std::path::{Path, PathBuf};

use gguf::GGMLType;
pub(crate) use inspect::*;

#[inline]
fn data_type_bits(dtype: GGMLType) -> usize {
    match dtype {
        GGMLType::F32 => 32,
        GGMLType::F16 => 16,
        GGMLType::Q4_0 => 4,
        GGMLType::Q4_1 => 4,
        GGMLType::Q5_0 => 5,
        GGMLType::Q5_1 => 5,
        GGMLType::Q8_0 => 8,
        GGMLType::Q8_1 => 8,
        GGMLType::Q2K => 2,
        GGMLType::Q3K => 3,
        GGMLType::Q4K => 4,
        GGMLType::Q5K => 5,
        GGMLType::Q6K => 6,
        GGMLType::Q8K => 8,
        GGMLType::I8 => 8,
        GGMLType::I16 => 16,
        GGMLType::I32 => 32,
        GGMLType::Count => 32, // Assuming Count is 32-bit, adjust if needed
    }
}

pub(crate) fn paths_to_sign(file_path: &Path) -> anyhow::Result<Vec<PathBuf>> {
    // GGUF are self contained
    Ok(vec![file_path.to_path_buf()])
}
